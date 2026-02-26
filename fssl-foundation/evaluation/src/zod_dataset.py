from zod import ZodFrames
import zod.constants as constants
from zod.constants import Anonymization, AnnotationProject
from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision.transforms.functional import crop


class ODWideCrop():
    """Wide ROI Crop"""
    def __init__(self, top=428, left=4, width=3840, height=1152):
        self.top = top
        self.left = left
        self.width = width
        self.height = height

    def __call__(self, img):
        return crop(img, self.top, self.left, self.height, self.width)
    

class ODFarCrop():
    """Far ROI CROP"""
    def __init__(self, top=924, left=1284, height=384, width=1280):
        self.top = top
        self.left = left
        self.height = height
        self.width = width

    def __call__(self, img):
        return crop(img, self.top, self.left, self.height, self.width)


class ZODROIWide(Dataset):
    """Prepares ZOD dataset for object detection using the Wide ROI. 
    Bounding boxes outside the crop are removed & the remaining ones are resized accordingly. 
    Depending on the *type* input, all frames, only train frames, or val frames are loaded."""
    def __init__(self, dataset_root, version="full", type="val", transform=None, rescaled_size=(800, 800)):
        self.transform = transform
        self.rescaled_size = rescaled_size
        self.class_mapping = {"Vehicle": 1, "VulnerableVehicle": 2, "Pedestrian": 3}
        zod_frames = ZodFrames(dataset_root, version)
        self.crop = ODWideCrop()
        
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
        bboxes, labels = rescale_filter_cropped(bboxes, labels, top=428, left=4, cropped_width=3840, cropped_height=1152, 
                                                new_size=self.rescaled_size) 

        # handle empty annotations
        if len(bboxes) == 0:
            bboxes = torch.zeros((0, 4)) 
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            bboxes = torch.tensor(bboxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        
        target = {'boxes': bboxes,
                  'labels': labels, 
                  "image_id": int(frame_id)}

        return img, target


class ZODROIFar(Dataset):
    """Prepares ZOD dataset for object detection using the Far ROI. 
    Bounding boxes outside the crop are removed & the remaining ones are resized accordingly. 
    Depending on the *type* input, all frames, only train frames, or val frames are loaded."""
    def __init__(self, dataset_root, version="full", type="val", transform=None, rescaled_size=(800, 800)):
        self.transform = transform        
        self.rescaled_size = rescaled_size
        self.class_mapping = {"Vehicle": 1, "VulnerableVehicle": 2, "Pedestrian": 3}
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

        # handle empty annotations
        if len(bboxes) == 0:
            bboxes = torch.zeros((0, 4)) 
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            bboxes = torch.tensor(bboxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        
        target = {'boxes': bboxes,
                  'labels': labels, 
                  "image_id": int(frame_id)}

        return img, target


class ZODRescaled(Dataset):
    """Rescales the original ZOD dataset for object detection (no cropping). 
    Depending on the *type* input, all frames, only train frames, or val frames are loaded."""
    def __init__(self, dataset_root, version="full", type="val", transform=None, rescaled_size=(1000, 700)):
        self.transform = transform
        self.rescaled_size = rescaled_size
        self.class_mapping = {"Vehicle": 1, "VulnerableVehicle": 2, "Pedestrian": 3}
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

        if self.transform:
            img = self.transform(img)
        annotations = frame.get_annotation(AnnotationProject.OBJECT_DETECTION)
        bboxes = [annotation.box2d.xyxy for annotation in annotations if annotation.name in self.class_mapping]

        labels = [self.class_mapping[annotation.name] for annotation in annotations if annotation.name in self.class_mapping]

        # resize & filter bounding boxes from cropped data
        bboxes = resize_bboxes(bboxes, original_size=(3848, 2168), rescaled_size=self.rescaled_size) 

        # handle empty annotations
        if len(bboxes) == 0:
            bboxes = torch.zeros((0, 4)) 
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            bboxes = torch.tensor(bboxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        
        target = {'boxes': bboxes,
                  'labels': labels, 
                  "image_id": int(frame_id)}

        return img, target

def rescale_filter_cropped(bboxes, labels, top=428, left=4, cropped_width=3840, cropped_height=1152, new_size=(800, 800)):
    """Rescales bounding boxes from cropped + rescaled images & filters out boxes + labels outside of the crop."""
    scale_x = new_size[0] / cropped_width
    scale_y = new_size[1] / cropped_height

    bboxes_rescaled_to_keep = []
    labels_to_keep = []
    for bbox, label in zip(bboxes, labels):
        x_min, y_min, x_max, y_max = bbox

        # Adjust for the crop
        x_min_cropped = x_min - left
        y_min_cropped = y_min - top
        x_max_cropped = x_max - left
        y_max_cropped = y_max - top

        # Check if the bounding box is completely outside the cropped area
        if x_max_cropped <= 0 or y_max_cropped <= 0 or x_min_cropped >= cropped_width or y_min_cropped >= cropped_height:
            continue  # Skip this bounding box & label

        # Clip bounding boxes that are partially outside the crop
        x_min_clipped = max(0, x_min_cropped)
        y_min_clipped = max(0, y_min_cropped)
        x_max_clipped = min(cropped_width, x_max_cropped)
        y_max_clipped = min(cropped_height, y_max_cropped)

        # Only keep bounding boxes that still have some area after clipping
        if x_max_clipped > x_min_clipped and y_max_clipped > y_min_clipped:
            # Rescale the clipped bounding boxes to the new size
            x_min_rescaled = x_min_clipped * scale_x
            y_min_rescaled = y_min_clipped * scale_y
            x_max_rescaled = x_max_clipped * scale_x
            y_max_rescaled = y_max_clipped * scale_y

            bboxes_rescaled_to_keep.append([x_min_rescaled, y_min_rescaled, x_max_rescaled, y_max_rescaled])
            labels_to_keep.append(label)

    return bboxes_rescaled_to_keep, labels_to_keep


def resize_bboxes(bboxes, original_size=(3848, 2168), rescaled_size=(256, 256)):
    """Resized bounding boxes for downsampled images where no cropping was applied."""
    original_width, original_height = original_size
    new_width, new_height = rescaled_size

    scale_x = new_width / original_width
    scale_y = new_height / original_height

    resized_bboxes = []
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        x_min_resized = x_min * scale_x
        y_min_resized = y_min * scale_y
        x_max_resized = x_max * scale_x
        y_max_resized = y_max * scale_y
        resized_bboxes.append([x_min_resized, y_min_resized, x_max_resized, y_max_resized])
    
    return resized_bboxes