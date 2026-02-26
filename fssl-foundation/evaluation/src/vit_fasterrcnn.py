import torch
import torch.nn as nn
import math
from tqdm import tqdm

from vision_transformer import *
from fasterrcnn import *
from zod_dataset import *


class VitWithFPN(nn.Module):
    def __init__(self, vit_model, fpn_channels=256):
        super().__init__()
        self.vit = vit_model
        self.fpn_channels = fpn_channels
        self.out_channels = fpn_channels

        # Split transformer blocks into 4 stages
        num_blocks = len(self.vit.blocks)
        blocks_per_stage = num_blocks // 4
        self.stage1 = nn.ModuleList(self.vit.blocks[:blocks_per_stage])
        self.stage2 = nn.ModuleList(self.vit.blocks[blocks_per_stage:2*blocks_per_stage])
        self.stage3 = nn.ModuleList(self.vit.blocks[2*blocks_per_stage:3*blocks_per_stage])
        self.stage4 = nn.ModuleList(self.vit.blocks[3*blocks_per_stage:])
        
        # Projection layers for different scales
        embed_dim = self.vit.embed_dim
        self.proj1 = nn.Conv2d(embed_dim, fpn_channels, 1)  # stride 4
        self.proj2 = nn.Conv2d(embed_dim, fpn_channels, 1)  # stride 8
        self.proj3 = nn.Conv2d(embed_dim, fpn_channels, 1)  # stride 16
        self.proj4 = nn.Conv2d(embed_dim, fpn_channels, 1)  # stride 32
        
        # Upsampling and downsampling convolutions
        self.upsample = nn.ConvTranspose2d(fpn_channels, fpn_channels, kernel_size=2, stride=2)
        self.downsample = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=2, stride=2)

    def forward(self, x):
        # Initial patch embedding
        x = self.vit.prepare_tokens(x)
        B = x.shape[0]
        
        features = {}
        
        # Stage 1 (stride 4 - upsample)
        for block in self.stage1:
            x = block(x)
        features["layer1"] = self.upsample(self._reshape_and_project(x, self.proj1, B))
        
        # Stage 2 (stride 8)
        for block in self.stage2:
            x = block(x)
        features["layer2"] = self.upsample(self._reshape_and_project(x, self.proj2, B))
        
        # Stage 3 (stride 16 - no scaling needed)
        for block in self.stage3:
            x = block(x)
        features["layer3"] = self._reshape_and_project(x, self.proj3, B)
        
        # Stage 4 (stride 32 - downsample)
        for block in self.stage4:
            x = block(x)
        features["layer4"] = self.downsample(self._reshape_and_project(x, self.proj4, B))
        
        return features


    def _reshape_and_project(self, x, proj, batch_size):
        # Remove CLS token
        x = x[:, 1:]  # shape: [B, N, C]
        
        # Calculate grid size from sequence length
        N = x.shape[1]  # N = H*W = number of patches
        h = w = int(math.sqrt(N))  # Assume square feature map
        assert h * w == N, f"Mismatch: {h}x{w} != {N}"
        
        # Reshape to 2D feature map [B, C, H, W]
        x = x.transpose(1, 2).reshape(batch_size, -1, h, w)
        
        # Project to FPN channels
        return proj(x)
    

def create_vit_fasterrcnn(num_classes=4, vit_model=vit_small()):    
    backbone = VitWithFPN(vit_model)

    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 4
    )
    
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=['layer1', 'layer2', 'layer3', 'layer4'],
        output_size=7,
        sampling_ratio=2
    )
    
    # Create Faster R-CNN model
    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )
    return model


def vit_small(img_size=[800, 800], patch_size=16, **kwargs):
    model = VisionTransformer(
        img_size=img_size, patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def collate_fn(batch):
    return tuple(zip(*batch))

def load_trained_vit(model=vit_small(), dino_checkpoint_path="checkpoint.pth", network="student"):
    """loads vit backbone from a DINO checkpoint file"""
    if os.path.isfile(dino_checkpoint_path):
        state_dict = torch.load(dino_checkpoint_path, map_location="cpu")
    else:
        raise(f"checkpoint file {dino_checkpoint_path} not found!")
    if network is not None and network in state_dict:
        print(f"Take key {network} in provided checkpoint dict")
        state_dict = state_dict[network]
    
    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

    # Interpolate position embedding if needed
    if 'pos_embed' in state_dict:
        pos_embed_checkpoint = state_dict['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = 1  # cls token
        
        # Height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # Height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        
        if orig_size != new_size:
            print(f"Position interpolate from {orig_size}x{orig_size} to {new_size}x{new_size}")
            # Class token and all other tokens
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:].reshape(
                -1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((pos_embed_checkpoint[:, :num_extra_tokens], pos_tokens), dim=1)
            state_dict['pos_embed'] = new_pos_embed

    msg = model.load_state_dict(state_dict, strict=False)
    print('Pretrained weights found at {} and loaded with msg: {}'.format(dino_checkpoint_path, msg))
    return model


def train(model, train_loader, output_path="vit_fasterrcnn.pth", num_epochs=20):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("running on ", device)
    model.to(device)
 
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.003, momentum=0.9, weight_decay=0.0005)

    model.train()

    loss_per_epoch = []
    global_progress = tqdm(range(0, num_epochs), desc=f'Training')
    for epoch in global_progress:
        epoch_loss = 0

        local_progress=tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for images, targets in local_progress:
            
            optimizer.zero_grad()
            
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items() if isinstance(v, torch.Tensor)} for t in targets]

            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            epoch_loss += losses.item()

            losses.backward()

            optimizer.step()
        try:
            print(f"loss for epoch {epoch+1}: {epoch_loss / len(train_loader)}")
            loss_per_epoch.append(epoch_loss / len(train_loader))
        except Exception as e:
            print(e) 

    torch.save(model.state_dict(), output_path)
    

if __name__ == "__main__":
    # # Create model with random weights
    vit = vit_small(img_size=[1024, 1024], patch_size=16)
    model = create_vit_fasterrcnn(num_classes=4, vit_model=vit)

    # pretrained ViT
    # vit = vit_small(img_size=[1024, 1024], patch_size=16)
    # pretrained_vit = load_trained_vit(model=vit, dino_checkpoint_path="models/vit_small_400_frames.pth", network="student")
    # model = create_vit_fasterrcnn(num_classes=4, vit_model=pretrained_vit)

    transform = T.Compose([
        T.Resize((1024, 1024)), # height, width 
        T.ToTensor(),  
        T.Normalize(mean=[0.326, 0.323, 0.330],   # ZOD normalization
                     std=[0.113, 0.112, 0.117])  
        ])                                                  
    
    # try to see how much train data we need to get results ~like in paper
    dataset = ZODROIFar(dataset_root=config.datapath, type="val", transform=transform, rescaled_size=(1024, 1024)) 

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=config.num_workers, collate_fn=collate_fn, shuffle=False)

    if config.train: # train + evaluate
        print("starting training")
        train(model, train_loader, config.output_path, config.num_epochs)

        print("starting evaluation")
        eval(model, test_dataset, config.score_threshold)

    else: # only evaluate, load trained model (use output_path file name)
        print("starting evaluation")
        state_dict = torch.load(config.output_path)
        model.load_state_dict(state_dict)
        eval(model, test_dataset, config.score_threshold)