from zod.constants import Anonymization
from zod import ZodFrames
from visualize_attention import *
from vision_transformer import *


index = 1
checkpoint_path = "." # "trained_backbones/pretrained_vit-small-140epochs.pth" # "." # path to dino checkpoint
patch_size = 16
model_arch = vit_small(patch_size)

# get image path based on index
zod_frames = ZodFrames(dataset_root="../../../../../../mnt/ZODversions/ZODCropped/", version="full")
frame = zod_frames[index]
image_path = frame.info.get_key_camera_frame(Anonymization.BLUR).filepath
image_size = (256, 256) # (3848, 2168)
out_dir = "."
checkpoint_key = "student" # use trained student network
arch = "vit_small" # None # change to vit_small/vit_base and no checkpoint path when want to use facebook's pre-trained weights


if __name__=="__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # build model
    model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)

    # load local weights
    if os.path.isfile(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            state_dict = state_dict[checkpoint_key]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(checkpoint_path, msg))
    else:
        # load from torch hub
        url = None
        if arch == "vit_small" and patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif arch == "vit_small" and patch_size == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"  # model used for visualizations in our paper
        elif arch == "vit_base" and patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif arch == "vit_base" and patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        if url is not None:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            model.load_state_dict(state_dict, strict=True)
        else:
            print("There is no reference weights available for this model => We use random weights.")


    # load image
    if os.path.isfile(image_path):
        with open(image_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
    else:
        print(f"Provided image path {image_path} is non valid.")
        sys.exit(1)

    transform = pth_transforms.Compose([
        pth_transforms.Resize(image_size),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    img = transform(img)


    # make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
    img = img[:, :w, :h].unsqueeze(0)

    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size

    # attentions
    attentions = model.get_last_selfattention(img.to(device))

    nh = attentions.shape[1] # number of head

    # we keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1) # -> uses attention of cls token (0th position)

    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()

    # save attentions heatmaps
    os.makedirs(out_dir, exist_ok=True)
    torchvision.utils.save_image(torchvision.utils.make_grid(img, normalize=True, scale_each=True), os.path.join(out_dir, "img.png"))
    for j in range(nh):
        fname = os.path.join(out_dir, "attn-head" + str(j) + ".png")
        plt.imsave(fname=fname, arr=attentions[j], format='png')
        print(f"{fname} saved.")