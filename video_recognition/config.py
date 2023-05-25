import torch
from pathlib import Path
from timm.data import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    IMAGENET_INCEPTION_MEAN,
    IMAGENET_INCEPTION_STD,
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
)
import albumentations as A
from albumentations.pytorch import ToTensorV2


class NetConfig:
    backbone_name = "levit_384"
    projection_dims = 4
    pretrained = True
    grad_checkpointing = False


class Config:
    n_epochs = 2
    n_warm_epochs = 0
    n_warm_steps = 0
    batch_size = 256  # 112 512 64
    n_workers = 4
    use_amp = True
    clip_value = None

    backbone_lr = 1e-3
    head_lr = 1e-3

    # data
    data_dir = Path("data/sibur_data")

    # exp paths
    resume = False
    exp_dir = Path("runs/levit_384")

    pretrained_weights = None
    # pretrained_weights = torch.load(
    #     "runs/sd__vit_large_336__m_0.5__negaug_0.1__adamw/model_2ep.pth",
    #     map_location="cuda",
    # )["net_state"]

    # training params
    seed = 42
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = NetConfig()
    criterion = None
    optimizer = None
    scheduler = None
    miner = None

    transforms = {
        "train": A.Compose(
            [
                A.Resize(224, 224),
                A.HorizontalFlip(),
                A.Normalize(mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),
                ToTensorV2(),
            ]
        ),
        "test": A.Compose(
            [
                A.Resize(224, 224),
                # transforms.CenterCrop(size=(224, 224)),
                A.Normalize(mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),
                ToTensorV2(),
            ]
        ),
    }


cfg = Config()
