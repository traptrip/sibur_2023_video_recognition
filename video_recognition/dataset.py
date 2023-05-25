import gc
from typing import Union, Optional
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms
from tqdm.auto import tqdm


def get_frames(video_path: Path, img_size=None, n_frames=None):
    cpr = cv2.VideoCapture(video_path.as_posix())
    has_frame = True
    frames = []

    while has_frame:
        has_frame, frame = cpr.read()
        if has_frame:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if img_size:
                frame = cv2.resize(frame, img_size)
            frames.append(frame)
    cpr.release()
    return np.array(frames)


class ScFramesDataset(Dataset):
    def __init__(
        self,
        root: Path,
        stage: str,
        transform: Optional[transforms.Compose] = None,
        base_img_size=(232, 232),
        frames=None,
        targets=None,
    ):

        if frames is None and targets is None:
            le = LabelEncoder()
            videos_paths = list(root.rglob("*.mp4"))
            clips = [
                get_frames(vp, img_size=base_img_size) for vp in tqdm(videos_paths)
            ]
            clips_targets = [vp.parent.name for vp in videos_paths]
            clips_targets = le.fit_transform(clips_targets)
            self.classes_ = le.classes_

            frames = np.concatenate(clips, axis=0)
            frames_targets = np.array(
                [cls for c, cls in zip(clips, clips_targets) for _ in c]
            )

            del clips
            gc.collect()

            # if stage != "train":  # if want to train all data
            stage_idxs = np.load(root / f"{stage}_idxs.npz")["idxs"]
            self.images = frames[stage_idxs]
            self.targets = frames_targets[stage_idxs]
        else:
            self.images = frames
            self.targets = targets

        self.transform = transform
        self.root = root
        self.stage = stage

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        target = self.targets[idx]

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, target
