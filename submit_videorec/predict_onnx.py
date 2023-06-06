import warnings
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


warnings.filterwarnings("ignore")

N_FRAMES = 4
MEAN = [0.48145466, 0.4578275, 0.40821073]
STD = [0.26862954, 0.26130258, 0.27577711]
WEIGHTS_PATH = Path(__file__).parent.joinpath("model/model.onnx")
id2label = {0: "bridge_down", 1: "bridge_up", 2: "no_action", 3: "train_in_out"}
labels = list(id2label.values())
offset = (232 - 224) // 2


def construct_model():
    ort_session = ort.InferenceSession(WEIGHTS_PATH)
    return ort_session


model = construct_model()


def process_frame(frame: np.ndarray):
    frame = cv2.resize(frame, (224, 224))
    # frame = frame[offset:-offset, offset:-offset]
    frame = frame.astype(np.float32)
    return frame


def process_clip(clip: np.ndarray):
    clip = np.concatenate([process_frame(frame)[None] for frame in clip])
    clip = clip / 255.0
    clip -= MEAN
    clip /= STD
    clip = clip.transpose(3, 0, 1, 2)[None]  # (seq,w,h,ch) -> (ch,seq,w,h)
    return clip


def preprocess(clip: np.ndarray, n_frames=8, step=4):
    start_idx, end_idx = 0, len(clip)
    indices = np.linspace(start_idx, end_idx, num=n_frames)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int32)
    clip = clip[indices]
    # start = 0
    # end = (start + n_frames) * step
    # clip = clip[start:end:step]
    inputs = process_clip(clip)
    return {"input": inputs}


def predict(clip: np.ndarray, BUFFER):
    """Вычислить класс для этого клипа. Эта функция должна возвращать *имя* класса."""
    inputs = preprocess(clip, n_frames=N_FRAMES)
    predict = model.run(None, inputs)[0].argmax(1)[0]
    return id2label[predict]
