import json 
import sys
import os
import torch
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def generate_coco_ground_truth(dataset, output_path="zod_ground_truth.json"): 
    """For a given ZOD dataset input, it generates the corresponding coco ground truth json file
    
    Expected output format:
    {
    "images": [
        {
        "id": 0,
        "file_name": "image1.jpg",
        "width": 640,
        "height": 480
        },
        {
        "id": 1,
        "file_name": "image2.jpg",
        "width": 640,
        "height": 480
        }
    ],
    "annotations": [
        {
        "id": 0,
        "image_id": 0,
        "category_id": 0,
        "bbox": [84.99, 119.99, 3.08, 8.43],
        "area": 26.01,
        "iscrowd": 0
        },
        {
        "id": 1,
        "image_id": 0,
        "category_id": 2,
        "bbox": [60.05, 119.63, 1.29, 14.62],
        "area": 18.87,
        "iscrowd": 0
        }
        // more annotations...
    ],
    "categories": [
        {
        "id": 1,
        "name": "category1"
        },
        {
        "id": 2,
        "name": "category2"
        },
        {
        "id": 3,
        "name": "category3"
        }
        // more categories...]
    }
    """

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
    for image, target in dataset:
        height, width = image.shape[1], image.shape[2]
        boxes = target["boxes"].tolist()
        labels = target["labels"].tolist()
        image_id = target["image_id"]

        image = {
            "id": int(image_id),
            "width": width,
            "height": height
        }
        coco_ground_truth["images"].append(image)

        for i in range(len(labels)):
            bbox = boxes[i] 
            # to coco format
            x_min, y_min, x_max, y_max = bbox
            box_width = x_max - x_min
            box_height = y_max - y_min
            coco_bbox = [x_min, y_min, box_width, box_height]

            annotation = {
                "id": annotation_id,            # unique id (for each annotation)
                "category_id": labels[i],       # == label
                "iscrowd": 0,                   # always 0
                "image_id": int(image_id),      # references image
                "area": box_width * box_height, # width * height
                "bbox": coco_bbox               # bounding box in coco format
            }
            coco_ground_truth["annotations"].append(annotation)
            annotation_id += 1
            
    with open(output_path, "w") as f:
        json.dump(coco_ground_truth, f)


def collate_fn(batch):
    return tuple(zip(*batch))


def predictions_to_coco(test_dataset, model, score_threshold=0.5):
    """
    Converts model predictions on the test dataset to COCO format, evaluates them, 
    and outputs COCO evaluation metrics.

    Args:
        test_dataset (torch.utils.data.Dataset): 
            The evaluation dataset (ZOD dataset) containing images and annotations.
        model (torch.nn.Module): 
            A trained Faster R-CNN model for object detection, used to make predictions on the dataset.
        score_threshold (float): 
            Confidence threshold for predictions. Only bounding boxes with confidence scores 
            equal to or greater than this value will be included in the output. Defaults to 0.5.    

            
    The generated coco predictions file (zod_predictions.json) file has the following format

    "predictions": [
    {
        "image_id": 0,
        "category_id": 0,
        "score": 0.95,
        "bbox": [84.98870849609375, 119.9878158569336, 3.0844573974609375, 8.432609558105469]
    },
    {
        "image_id": 0,
        "category_id": 2,
        "score": 0.89,
        "bbox": [60.04625701904297, 119.63187408447266, 1.2911643981933594, 14.616050720214844]
    },
    {
        "image_id": 0,
        "category_id": 1,
        "score": 0.85,
        "bbox": [7.367759704589844, 119.48949432373047, 10.993881225585938, 41.10425567626953]
    },
    {
        "image_id": 1,
        "category_id": 0,
        "score": 0.92,
        "bbox": [138.71774291992188, 144.53440856933594, 3.22613525390625, 5.0591888427734375]
    },
    ]
    """

    # 1) generate ground truth
    generate_coco_ground_truth(test_dataset, "zod_ground_truth.json")

    # 2) evaluate model on dataset and get predictions
    test_loader = DataLoader(test_dataset, batch_size=4, collate_fn=collate_fn, shuffle=False)
    model.eval()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for images, targets in test_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v for k, v in t.items()} for t in targets]
            # Get predictions from the model
            predictions = model(images)

            all_predictions.extend(predictions)
            all_targets.extend(targets)
    
    # 3) convert predictions to coco format
    coco_predictions = []
    for target, prediction in zip(all_targets, all_predictions):
        image_id = target["image_id"]
        bboxes = prediction["boxes"].tolist()
        labels = prediction["labels"].tolist()
        scores = prediction["scores"].tolist()


        sorted_predictions = sorted(zip(scores, bboxes, labels), key=lambda x: x[0], reverse=True)

        for score, box, label in sorted_predictions:
            if score < score_threshold:
                continue # skip predictions with low score 

            x_min, y_min, x_max, y_max = box
            box_width = x_max - x_min
            box_height = y_max - y_min

            pred_bbox = {
                "image_id": image_id,
                "category_id": label,
                "bbox": [x_min, y_min, box_width, box_height],
                "score": score
            }
            coco_predictions.append(pred_bbox)

    # Write predictions to file
    with open("zod_predictions.json", "w") as f:
        json.dump(coco_predictions, f)


    # 4) load everything and use coco to evaluate
    coco_gt = COCO("zod_ground_truth.json")

    # Load the predictions (COCO format)
    coco_dt = coco_gt.loadRes("zod_predictions.json")

    # Initialize COCOeval object
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")

    log_coco_eval(coco_eval)


def log_coco_eval(coco_eval_object, filename="eval_results.txt"):
    with open(filename, 'w') as f:
        original_stdout = sys.stdout  # Save a reference to the original standard output
        sys.stdout = f  # Change the standard output to the file

        # Run coco evaluation
        coco_eval_object.evaluate()
        coco_eval_object.accumulate()
        coco_eval_object.summarize()

        # Reset the standard output to its original value
        sys.stdout = original_stdout