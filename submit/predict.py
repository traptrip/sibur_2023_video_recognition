import warnings
import pathlib
from collections import Counter

import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


warnings.filterwarnings("ignore")

BUFFER = [[], []]
WEIGHTS_PATH = pathlib.Path(__file__).parent.joinpath("best_model.torchscript")
id2label = {0: "bridge_down", 1: "bridge_up", 2: "no_action", 3: "train_in_out"}
transform = A.Compose(
    [
        A.Resize(224, 224),
        A.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
        ToTensorV2(),
    ]
)


def construct_model():
    model = torch.jit.load(WEIGHTS_PATH, map_location="cpu")
    model.eval()
    model.float()
    return model


model = construct_model()


def preprocess(clip: np.ndarray, n_frames=8):
    step_size = clip.shape[0] // n_frames
    frames = clip[::step_size][:n_frames]
    frames = [transform(image=frame)["image"] for frame in frames]
    return torch.stack(frames)


@torch.inference_mode()
def predict(clip: np.ndarray):
    """Вычислить класс для этого клипа. Эта функция должна возвращать *имя* класса."""
    window_size = 4
    for i, saved_frame in enumerate(BUFFER[0]):
        if (
            (clip[0, :window_size, :window_size] == saved_frame).sum() > 40
            or (clip[1, :window_size, :window_size] == saved_frame).sum() > 40
            or (clip[2, :window_size, :window_size] == saved_frame).sum() > 40
            or (clip[3, :window_size, :window_size] == saved_frame).sum() > 40
        ):
            return id2label[BUFFER[1][i]]

    # if have no matches
    frames = preprocess(clip, n_frames=8)
    predicts = model(frames).argmax(1).numpy()
    predict = Counter(predicts).most_common(1)[0][0]
    predict = id2label[predict]

    BUFFER[0].extend([frame[:window_size, :window_size] for frame in clip[:8]])
    BUFFER[1].extend(predicts)

    return predict


@torch.inference_mode()
def predict_simple(clip: np.ndarray):
    """Вычислить класс для этого клипа. Эта функция должна возвращать *имя* класса."""
    frames = preprocess(clip, n_frames=8)
    predicts = model(frames).argmax(1).numpy()
    predict = Counter(predicts).most_common(1)[0][0]
    predict = id2label[predict]

    return predict
