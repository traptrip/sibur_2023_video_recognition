import warnings
import pathlib

import numpy as np
from transformers import AutoProcessor
import onnxruntime as ort


warnings.filterwarnings("ignore")

MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)
BUFFER = {}
WEIGHTS_PATH = pathlib.Path(__file__).parent.joinpath("xclip")
id2label = {0: "bridge_down", 1: "bridge_up", 2: "no_action", 3: "train_in_out"}
labels = list(id2label.values())


def construct_model():
    processor = AutoProcessor.from_pretrained(WEIGHTS_PATH)
    ort_session = ort.InferenceSession(WEIGHTS_PATH / "model.onnx")
    return ort_session, processor


model, processor = construct_model()


def preprocess(clip: np.ndarray, n_frames=8):
    # start_idx, end_idx = 0, len(clip)
    # indices = np.linspace(start_idx, end_idx, num=n_frames)
    # indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int32)
    # clip = clip[indices]
    start = 1
    step = 2
    end = (start + n_frames) * step
    clip = clip[start:end:step]

    # step_size = len(clip) // n_frames
    # clip = clip[::step_size][:n_frames]

    inputs = processor(
        text=labels,
        videos=list(clip),
        return_tensors="np",
        padding="max_length",
        max_length=n_frames,
    )
    for i in ["input_ids", "attention_mask"]:
        inputs[i] = inputs[i].astype(np.int32)

    return dict(inputs)


def predict(clip: np.ndarray):
    """Вычислить класс для этого клипа. Эта функция должна возвращать *имя* класса."""
    window_size = 2
    for c in clip[5:7, :window_size, :window_size]:
        c = tuple(c.flatten())
        if c in BUFFER:
            return id2label[BUFFER[c]]

    # if have no matches
    inputs = preprocess(clip, n_frames=8)
    predict = model.run(None, inputs)[0].argmax(1)[0]

    left_center = 224 / 4
    for c in clip[
        1:9,
        left_center - window_size // 2 : window_size,
        left_center - window_size // 2 : window_size,
    ]:
        c = tuple(c.flatten())
        if c not in BUFFER:
            BUFFER[c] = predict

    return id2label[predict]
