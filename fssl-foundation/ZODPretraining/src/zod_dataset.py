from zod import ZodFrames
import zod.constants as constants
from zod.constants import Anonymization
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms.functional import crop


class ZOD(Dataset):
    """ZOD dataset containing train frames"""

    def __init__(self, dataset_root, version="full", transform=None):
        self.transform = transform
        zod_frames = ZodFrames(dataset_root, version)
        # only use train frames for pre-training on ZOD
        train_indices = zod_frames.get_split(constants.TRAIN)
        self.frames = [zod_frames[idx] for idx in train_indices]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        img = frame.info.get_key_camera_frame(
            Anonymization.BLUR
        ).read()  # change it to tensor
        img = Image.fromarray(img)  # to pil image
        if self.transform:
            img = self.transform(img)
        return img


class ZODAll(Dataset):
    """ZOD dataset containing both train frames and all sequence frames"""

    def __init__(self, dataset_root, version="full", transform=None):
        self.transform = transform
        # sequences
        self.zod_sequences = ZodSequences(dataset_root, version)
        # training frames
        zod_frames = ZodFrames(dataset_root, version)
        train_indices = zod_frames.get_split(constants.TRAIN)
        self.zod_frames = [zod_frames[idx] for idx in train_indices]

        self.frame_references = []  # stores references to sequence and training frames

        # append train frames
        for frame in self.zod_frames:
            frame_obj = frame.info.get_key_camera_frame(Anonymization.BLUR)
            self.frame_references.append(frame_obj)

        # loop through sequences
        for seq in self.zod_sequences:
            seq_frames = seq.info.camera_frames[
                "front_blur"
            ]  # returns all frames in sequence
            for seq_frame in seq_frames:
                self.frame_references.append(seq_frame)

    def __len__(self):
        return len(self.frame_references)

    def __getitem__(self, idx):
        try:
            frame = self.frame_references[idx].read()
            img = Image.fromarray(frame)  # Convert to PIL image
            if self.transform:
                img = self.transform(img)
            return img
        except FileNotFoundError:
            frame = self.frame_references[0].read()  # use 0th image in this case
            img = Image.fromarray(frame)
            if self.transform:
                img = self.transform(img)
            return img


class ZOD160_000(Dataset):
    """ZOD dataset containing 160.000 frames, from both train frames and sequence frames.
    Every 4th sequence frame is used."""

    def __init__(self, dataset_root, version="full", transform=None):
        self.transform = transform
        # sequences
        self.zod_sequences = ZodSequences(dataset_root, version)
        # training frames
        zod_frames = ZodFrames(dataset_root, version)
        train_indices = zod_frames.get_split(constants.TRAIN)
        self.zod_frames = [zod_frames[idx] for idx in train_indices]

        self.frame_references = []  # stores references to sequence and training frames

        # append train frames
        for frame in self.zod_frames:
            frame_obj = frame.info.get_key_camera_frame(Anonymization.BLUR)
            self.frame_references.append(frame_obj)

        # loop through sequences
        for seq in self.zod_sequences:
            seq_frames = seq.info.camera_frames[
                "front_blur"
            ]  # returns all frames in sequence
            num_frames_in_seq = len(seq_frames)
            frames_idx_to_keep = list(
                range(0, num_frames_in_seq, 4)
            )  # Get indices of every 4th frame
            seq_frames = [seq_frames[i] for i in frames_idx_to_keep]

            for seq_frame in seq_frames:
                self.frame_references.append(seq_frame)

    def __len__(self):
        return len(self.frame_references)

    def __getitem__(self, idx):
        try:
            frame = self.frame_references[idx].read()
            img = Image.fromarray(frame)  # Convert to PIL image
            if self.transform:
                img = self.transform(img)
            return img
        except FileNotFoundError:
            frame = self.frame_references[0].read()  # use 0th image in this case
            img = Image.fromarray(frame)
            if self.transform:
                img = self.transform(img)
            return img
