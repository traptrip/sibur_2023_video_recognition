import torch
from pathlib import Path
from timm.data import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
import albumentations as A
from albumentations.pytorch import ToTensorV2

"""
eva_giant_patch14_224.clip_ft_in1k
beitv2_large_patch16_224.in1k_ft_in22k_in1k
vit_huge_patch14_clip_224.laion2b_ft_in12k_in1k
vit_large_patch14_clip_224.openai_ft_in12k_in1k
beit_large_patch16_224.in22k_ft_in22k_in1k
beitv2_large_patch16_224.in1k_ft_in1k
vit_large_patch14_clip_224.laion2b_ft_in1k
deit3_huge_patch14_224.fb_in22k_ft_in1k
deit3_large_patch16_224.fb_in22k_ft_in1k
maxvit_rmlp_base_rw_224.sw_in12k_ft_in1k
maxxvitv2_rmlp_base_rw_224.sw_in12k_ft_in1k
caformer_m36.sail_in22k_ft_in1k
beitv2_base_patch16_224.in1k_ft_in22k_in1k
swin_large_patch4_window7_224.ms_in22k_ft_in1k
vit_base_patch8_224.augreg2_in21k_ft_in1k
vit_base_patch16_clip_224.laion2b_ft_in12k_in1k
deit3_base_patch16_224.fb_in22k_ft_in1k
beitv2_base_patch16_224.in1k_ft_in1k
"""


class NetConfig:
    backbone_name = "eva_giant_patch14_224.clip_ft_in1k"
    projection_dims = 4
    pretrained = True
    grad_checkpointing = False


class Config:
    n_epochs = 3
    n_warm_epochs = 0
    n_warm_steps = 0
    batch_size = 128  # 112 512 64
    n_workers = 4
    use_amp = True
    clip_value = None

    backbone_lr = 1e-5
    head_lr = 1e-5

    # data
    data_dir = Path("data/sibur_data")

    # exp paths
    resume = False
    exp_dir = Path(f"runs/{NetConfig().backbone_name}__ls_0.1")

    pretrained_weights = None
    # pretrained_weights = torch.load(
    #     "path/to/weights.pth",
    #     map_location="cuda",
    # )["net_state"]

    # training params
    seed = 42
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = NetConfig()
    criterion = None
    optimizer = None
    scheduler = None

    transforms = {
        "train": A.Compose(
            [
                A.Resize(224, 224),
                A.HorizontalFlip(),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
                ),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),
                ToTensorV2(),
            ]
        ),
        "test": A.Compose(
            [
                A.Resize(256, 256),
                A.CenterCrop(size=(224, 224)),
                A.Normalize(mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),
                ToTensorV2(),
            ]
        ),
    }


cfg = Config()
