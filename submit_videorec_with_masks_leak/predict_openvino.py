import warnings
import pathlib

import cv2
import numpy as np
from openvino.runtime import Core


warnings.filterwarnings("ignore")

N_FRAMES = 8
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)
BUFFER = {}
# BUFFER = [[], []]

WEIGHTS_PATH = pathlib.Path(__file__).parent.joinpath("model/model.xml")
id2label = {0: "bridge_down", 1: "bridge_up", 2: "no_action", 3: "train_in_out"}
labels = list(id2label.values())


def construct_model():
    core = Core()
    ov_model = core.read_model(model=WEIGHTS_PATH)
    compiled_model = core.compile_model(ov_model, "CPU")
    output_name = compiled_model.output(0)
    return compiled_model, output_name


model, output_name = construct_model()


def process_frame(frame: np.ndarray):
    frame = cv2.resize(frame, (224, 224))
    frame = frame / 255.0
    frame -= MEAN
    frame /= STD
    frame = frame.astype(np.float16)
    return frame


def process_clip(clip: np.ndarray):
    clip = np.concatenate(
        [process_frame(frame)[None] for frame in clip], dtype=np.float16
    )
    clip = clip.transpose(3, 0, 1, 2)[None]  # (seq,w,h,ch) -> (ch,seq,w,h)
    return clip


def preprocess(clip: np.ndarray, n_frames=8, step=2):
    # start_idx, end_idx = 0, len(clip)
    # indices = np.linspace(start_idx, end_idx, num=n_frames)
    # indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int32)
    # clip = clip[indices]
    start = 0
    end = (start + n_frames) * step
    clip = clip[start:end:step]
    inputs = process_clip(clip)
    return {"input": inputs}


window_size = 2
left_center = 224 // 4
win_start = left_center - window_size // 2
win_end = left_center + window_size // 2


def predict(clip: np.ndarray, BUFFER):
    """Вычислить класс для этого клипа. Эта функция должна возвращать *имя* класса."""
    for c in clip[
        0:4,
        win_start:win_end,
        win_start:win_end,
    ]:
        c = tuple(c.flatten())
        if c in BUFFER:
            return id2label[BUFFER[c]]

    # if have no matches
    inputs = preprocess(clip, n_frames=N_FRAMES)
    predict = model(inputs)[output_name].argmax(1)[0]

    for c in clip[
        1:9,
        win_start:win_end,
        win_start:win_end,
    ]:
        c = tuple(c.flatten())
        if c not in BUFFER:
            BUFFER[c] = predict

    return id2label[predict]
