import warnings
from pathlib import Path

import numpy as np
from openvino.runtime import Core
import albumentations as A


warnings.filterwarnings("ignore")

N_FRAMES = 4
MEAN = [0.48145466, 0.4578275, 0.40821073]
STD = [0.26862954, 0.26130258, 0.27577711]
WEIGHTS_PATH = Path(__file__).parent.joinpath("model/model.xml")
id2label = {0: "bridge_down", 1: "bridge_up", 2: "no_action", 3: "train_in_out"}
labels = list(id2label.values())

transform = A.Compose(
    [A.Resize(256, 256), A.CenterCrop(224, 224), A.Normalize(MEAN, STD)]
)


def construct_model():
    core = Core()
    ov_model = core.read_model(model=WEIGHTS_PATH)
    compiled_model = core.compile_model(ov_model, "CPU")
    output_name = compiled_model.output(0)
    return compiled_model, output_name


model, output_name = construct_model()


def process_frame(frame: np.ndarray):
    frame = transform(image=frame)["image"]
    frame = frame.astype(np.float16)
    return frame


def process_clip(clip: np.ndarray):
    clip = np.concatenate([process_frame(frame)[None] for frame in clip])
    clip = clip.transpose(3, 0, 1, 2)[None]  # (seq,w,h,ch) -> (ch,seq,w,h)
    return clip


def preprocess(clip: np.ndarray, n_frames=8, step=2):
    start_idx, end_idx = 0, len(clip)
    indices = np.linspace(start_idx, end_idx, num=n_frames)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int32)
    clip = clip[indices]
    # start = 0
    # end = (start + n_frames) * step
    # clip = clip[start:end:step]
    inputs = process_clip(clip)
    return {"input": inputs}


def predict(clip: np.ndarray):
    """Вычислить класс для этого клипа. Эта функция должна возвращать *имя* класса."""
    inputs = preprocess(clip, n_frames=N_FRAMES)
    predict = model(inputs)[output_name].argmax(1)[0]
    return id2label[predict]
