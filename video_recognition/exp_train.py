from pathlib import Path

import torch
import timm

from video_recognition.training import Trainer
from video_recognition.config import Config

if __name__ == "__main__":
    cfg = Config()

    net: torch.nn.Module = timm.create_model(
        cfg.net.backbone_name, cfg.net.pretrained, num_classes=cfg.net.projection_dims
    )
    if cfg.pretrained_weights is not None:
        net.load_state_dict(cfg.pretrained_weights)

    if cfg.net.grad_checkpointing:
        net.set_grad_checkpointing()

    net.to(cfg.device)

    ### LOSS FUNC ###
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    ###

    ### OPTIMIZER ###
    classifier = net.get_classifier()

    if "levit" in cfg.net.backbone_name:
        param_groups = [
            {
                "params": list(net.parameters())[:-8],
                "lr": cfg.backbone_lr,
            },
            {
                "params": list(classifier[0].parameters())
                + list(classifier[1].parameters()),
                "lr": cfg.head_lr,
            },
        ]
    else:
        param_groups = [
            {
                "params": list(net.parameters())[:-2],
                "lr": cfg.backbone_lr,
            },
            {
                "params": list(classifier.parameters()) + list(criterion.parameters()),
                "lr": cfg.head_lr,
            },
        ]

    optimizer = torch.optim.AdamW(param_groups, cfg.head_lr)
    ###

    ### SCHEDULER ###
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.n_epochs, eta_min=1e-5
    )
    ###

    cfg.net = net
    cfg.criterion = criterion
    cfg.optimizer = optimizer
    cfg.scheduler = scheduler

    trainer = Trainer(cfg)
    Path(cfg.exp_dir).mkdir(exist_ok=True, parents=True)
    Path(cfg.exp_dir / "exp_train.py").write_text(Path(__file__).read_text())
    Path(cfg.exp_dir / "dataset.py").write_text(
        (Path(__file__).parent / "dataset.py").read_text()
    )
    Path(cfg.exp_dir / "training.py").write_text(
        (Path(__file__).parent / "training.py").read_text()
    )
    Path(cfg.exp_dir / "config.py").write_text(
        (Path(__file__).parent / "config.py").read_text()
    )

    print(f"EXP: {cfg.exp_dir.name}")
    trainer.train()
