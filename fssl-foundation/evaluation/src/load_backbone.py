import os
import torch
from torchvision.models import resnet50

from utils import *
from vision_transformer import *


def load_trained_resnet(dino_checkpoint_path="checkpoint.pth", network="student"):
    """loads resnet50 backbone from DINO checkpoint file"""
    model = resnet50()
    if os.path.isfile(dino_checkpoint_path):
        state_dict = torch.load(dino_checkpoint_path, map_location="cpu")
    else:
        raise (f"checkpoint file {dino_checkpoint_path} not found!")
    if network is not None and network in state_dict:
        print(f"Take key {network} in provided checkpoint dict")
        state_dict = state_dict[network]
    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    msg = model.load_state_dict(state_dict, strict=False)
    print(
        "Pretrained weights found at {} and loaded with msg: {}".format(
            dino_checkpoint_path, msg
        )
    )
    return model


def load_untrained_vit(model=vit_small()):
    return model


def load_trained_vit(
    model=vit_small(), dino_checkpoint_path="checkpoint.pth", network="student"
):
    """loads vit backbone from a DINO checkpoint file"""
    if os.path.isfile(dino_checkpoint_path):
        state_dict = torch.load(dino_checkpoint_path, map_location="cpu")
    else:
        raise (f"checkpoint file {dino_checkpoint_path} not found!")
    if network is not None and network in state_dict:
        print(f"Take key {network} in provided checkpoint dict")
        state_dict = state_dict[network]
    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    msg = model.load_state_dict(state_dict, strict=False)
    print(
        "Pretrained weights found at {} and loaded with msg: {}".format(
            dino_checkpoint_path, msg
        )
    )
    return model
