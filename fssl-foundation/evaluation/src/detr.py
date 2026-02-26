import torch
from torchvision import transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader, random_split
from transformers import DetrForObjectDetection, DetrConfig, DetrImageProcessor
from tqdm import tqdm
from torch.utils.data import Subset

import config

from load_backbone import *
from zod_dataset import *
from coco_evaluation import *


torch.manual_seed(42)
 
class DETR(nn.Module):
    def __init__(self, num_classes, backbone=None, pretrained=False):
        super().__init__()
        # pre-trained config
        if pretrained:
            config = DetrConfig.from_pretrained("facebook/detr-resnet-50")
        else: 
             # random weights config
            config = DetrConfig(use_pretrained_backbone=False)
        config.num_labels = num_classes
        # Initialize DETR with custom backbone if provided
        self.detr = DetrForObjectDetection(config)
        if backbone is not None:
            self.detr.model.backbone = backbone

    def forward(self, pixel_values, pixel_mask=None, labels=None):
        return self.detr(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)


class ODFarCrop():
    """Far ROI CROP"""
    def __init__(self, top=924, left=1284, height=384, width=1280):
        self.top = top
        self.left = left
        self.height = height
        self.width = width

    def __call__(self, img):
        return crop(img, self.top, self.left, self.height, self.width)


class ZODROIFar(Dataset):
    """Prepares ZOD dataset for object detection using the Far ROI. 
    Bounding boxes outside the crop are removed & the remaining ones are resized accordingly. 
    Depending on the *type* input, all frames, only train frames, or val frames are loaded."""
    def __init__(self, dataset_root, processor, version="full", type="val", transform=None, rescaled_size=(800, 800)):
        self.transform = transform        
        self.processor = processor
        self.rescaled_size = rescaled_size
        self.class_mapping = {"Vehicle": 0, "VulnerableVehicle": 1, "Pedestrian": 2} # for DETR, the mapping should start at 0
        self.crop = ODFarCrop()

        zod_frames = ZodFrames(dataset_root, version)
        
        if type == "all": # both train and val frames 
            self.frames = zod_frames
        elif type == "train": # only train frames
            train_indices = zod_frames.get_split(constants.TRAIN)
            self.frames = [zod_frames[idx] for idx in train_indices]
        elif type == "val":  # only val frames
            val_indices = zod_frames.get_split(constants.VAL)
            self.frames = [zod_frames[idx] for idx in val_indices]
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        if idx >= len(self.frames):
            raise IndexError(f"Index {idx} out of bounds")
        frame = self.frames[idx]
        metadata = frame.metadata
        frame_id = metadata.frame_id

        img = frame.info.get_key_camera_frame(Anonymization.BLUR).read() # change it to tensor

        img = Image.fromarray(img)
        img = self.crop(img)

        if self.transform:
            img = self.transform(img)
        annotations = frame.get_annotation(AnnotationProject.OBJECT_DETECTION)
        bboxes = [annotation.box2d.xyxy for annotation in annotations if annotation.name in self.class_mapping]

        labels = [self.class_mapping[annotation.name] for annotation in annotations if annotation.name in self.class_mapping]

        # resize & filter bounding boxes from cropped data
        bboxes, labels = rescale_filter_cropped(bboxes, labels, top=924, left=1284, cropped_width=1280, cropped_height=384, 
                                                new_size=self.rescaled_size)
        
        # convert to coco format
        coco_bboxes = []
        for box in bboxes:
            x_min, y_min, x_max, y_max = box
            width = x_max - x_min
            height = y_max - y_min
            coco_bboxes.append([x_min, y_min, width, height])
        
        target = {
            'image_id': int(frame_id),
            'orig_size': [img.size[1], img.size[0]],
            'annotations': [
                {
                    'bbox': bbox,
                    'category_id': label,
                    'area': bbox[2] * bbox[3],
                    'iscrowd': 0
                } for bbox, label in zip(coco_bboxes, labels)
            ]
        }

         # Process image and target using DETR processor
        encoding = self.processor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]

        return pixel_values, target, coco_bboxes

def generate_detr_coco_ground_truth(dataset, output_path="coco_ground_truth_pedestrian.json"):
    coco_ground_truth = {
        "images": [],
        "annotations": [],
        "categories": [
        {"id": 1, "name": "Vehicle"},
        {"id": 2, "name": "VulnerableVehicle"},
        {"id": 3, "name": "Pedestrian"},
        ]
    }
    annotation_id = 1
    for image, target, coco_bboxes in dataset:
        orig_height, orig_width = target["orig_size"].tolist()
        
        labels = target["class_labels"].tolist()
        image_id = target["image_id"]

        image = {
            "id": int(image_id),
            "width": orig_width,
            "height": orig_height
        }
        coco_ground_truth["images"].append(image)

        for i in range(len(labels)):
            bbox = [float(x) for x in coco_bboxes[i]] 
            annotation = {
                "id": annotation_id,            # unique id (for each annotation)
                "category_id": labels[i],       # == label
                "iscrowd": 0,                   # always 0
                "image_id": int(image_id),      # references image
                "area": float(bbox[2] * bbox[3]), # width * height
                "bbox": [float(x) for x in bbox]      # bounding box in coco format
            }
            coco_ground_truth["annotations"].append(annotation)
            annotation_id += 1
            
    with open(output_path, "w") as f:
        json.dump(coco_ground_truth, f)


def detr_predictions_to_coco(test_dataset, model, processor,threshold=0.5):
    generate_detr_coco_ground_truth(test_dataset, output_path="coco_ground_truth_pedestrian.json")
    
    test_loader = DataLoader(test_dataset, batch_size=4, collate_fn=lambda x: collate_fn(x, processor), shuffle=False)
    
    model.eval()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    coco_predictions = []

    with torch.no_grad():
        for batch in test_loader:
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]] # these are in DETR format, resized + normalized
            
            # forward pass
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
            # turn into a list of dictionaries (one item for each example in the batch)
            orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
            predictions = processor.post_process_object_detection(outputs, target_sizes=orig_target_sizes, threshold=0)

            for (target, prediction) in zip(labels, predictions):
                image_id = target["image_id"].item()
                bboxes = prediction["boxes"].tolist()
                labels = prediction["labels"].tolist()
                scores = prediction["scores"].tolist()

                sorted_predictions = sorted(zip(scores, bboxes, labels), key=lambda x: x[0], reverse=True)

                for score, box, label in sorted_predictions:
                    if score < threshold:
                        continue # skip predictions with low score 

                    x_min, y_min, x_max, y_max = box
                    box_width = x_max - x_min
                    box_height = y_max - y_min

                    pred_bbox = {
                        "image_id": image_id,
                        "category_id": label+1, # Convert from DETR (0-based) to COCO (1-based) format
                        "bbox": [x_min, y_min, box_width, box_height],
                        "score": score
                    }
                    coco_predictions.append(pred_bbox)

    # Write predictions to file
    with open("detr_predictions.json", "w") as f:
        json.dump(coco_predictions, f)

    # 4) load everything and use coco to evaluate
    coco_gt = COCO("coco_ground_truth_pedestrian.json")

    # Load the predictions (COCO format)
    coco_dt = coco_gt.loadRes("detr_predictions.json")

    # Initialize COCOeval object
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    
    log_coco_eval(coco_eval)


def collate_fn(batch, processor):
    pixel_values = [item[0] for item in batch]
    encoding = processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    coco_bboxes = [item[2] for item in batch] 
    batch = {}
    batch['pixel_values'] = encoding['pixel_values']
    batch['pixel_mask'] = encoding['pixel_mask']
    batch['labels'] = labels
    batch['coco_bboxes'] = coco_bboxes
    return batch


def train(model, train_loader, output_path="detr.pth", num_epochs=20):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("running on ", device)
    model.to(device)
    
    lr=1e-4 
    lr_backbone=1e-5
    weight_decay=1e-4

    # Create optimizer with different learning rates for backbone and other parameters
    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters() 
                        if "backbone" not in n and p.requires_grad]
        },
        {
            "params": [p for n, p in model.named_parameters() 
                        if "backbone" in n and p.requires_grad],
            "lr": lr_backbone,
        },
    ]

    optimizer = torch.optim.AdamW(param_dicts, lr=lr, weight_decay=weight_decay)
    model.train()

    loss_per_epoch = []
    global_progress = tqdm(range(0, num_epochs), desc=f'Training')
    for epoch in global_progress:
        epoch_loss = 0

        local_progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, batch in enumerate(local_progress):
            optimizer.zero_grad()

            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

            outputs = model(pixel_values=pixel_values, 
                            pixel_mask=pixel_mask, 
                            labels=labels)

            loss = outputs.loss
            epoch_loss += loss.item()
 
            loss.backward()
            optimizer.step()
            
        print(f"Epoch {epoch} average loss: {epoch_loss/len(train_loader):.4f}")
        loss_per_epoch.append(epoch_loss/len(train_loader))
    
    torch.save(model.state_dict(), output_path)
        

def eval(model, test_dataset, processor, score_threshold=0.0):
    detr_predictions_to_coco(test_dataset, model, processor, score_threshold)


if __name__=="__main__":
    # pretrained model (backbone + transformer)
    model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            revision="no_timm",
            num_labels=3, # 3 object classes
            ignore_mismatched_sizes=True
        )
    
    # pretrained detr (only backbone)
    # config = DetrConfig(use_pretrained_backbone=True,
    #                       num_queries=100,
    #                       num_labels=3)
    # model = DetrForObjectDetection(config)


    # random weights for backbone + transformer
    # config = DetrConfig(use_pretrained_backbone=False,
    #                 num_queries=100,
    #                 num_labels=3)
    # model = DetrForObjectDetection(config)

    # TODO: use own pre-trained resnet backbone

    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    processor.image_size = (800, 800)
    processor.image_mean = [0.326, 0.323, 0.330]
    processor.image_std = [0.113, 0.112, 0.117]
    
    dataset = ZODROIFar(dataset_root=config.datapath, processor=processor, type="val", 
                              rescaled_size=(800, 800))
    
    # NOTE: use small subset for debugging
    dataset = Subset(dataset, indices=range(1000))
    train_loader = DataLoader(dataset, collate_fn=lambda x: collate_fn(x, processor), batch_size=4, shuffle=True)
    test_dataset = dataset



    # train_size = int(0.8 * len(dataset))
    # test_size = len(dataset) - train_size
    # train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # train_loader = DataLoader(train_dataset, collate_fn=lambda x: collate_fn(x, processor), batch_size=4, shuffle=True)

    print("starting training")
    train(model, train_loader, config.output_path, config.num_epochs)

    print("starting evaluation")
    eval(model, test_dataset, processor, config.score_threshold)