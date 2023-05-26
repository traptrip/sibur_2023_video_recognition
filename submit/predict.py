import warnings
import pathlib
from collections import Counter

import cv2
import numpy as np
import onnxruntime as ort


warnings.filterwarnings("ignore")

MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)
BUFFER = [[], []]
WEIGHTS_PATH = pathlib.Path(__file__).parent.joinpath("model.onnx")
id2label = {0: "bridge_down", 1: "bridge_up", 2: "no_action", 3: "train_in_out"}


def construct_model():
    # model = torch.jit.load(WEIGHTS_PATH, map_location="cpu")
    # model.eval()
    # model.float()
    # return model
    ort_session = ort.InferenceSession(WEIGHTS_PATH)
    return ort_session


model = construct_model()


# def preprocess(clip: np.ndarray, n_frames=8):
#     # step_size = clip.shape[0] // n_frames
#     # frames = clip[::step_size][:n_frames]
#     if n_frames == 1:
#         return transform(image=clip[0])["image"][None]
#     else:
#         frames = clip[:n_frames]
#         frames = [transform(image=frame)["image"] for frame in frames]
#         return torch.stack(frames)


def process_frame(frame: np.ndarray):
    frame = cv2.resize(frame, (224, 224))
    frame = frame / 255.0
    frame -= MEAN
    frame /= STD
    frame = frame.astype(np.float32)
    return frame.transpose(2, 0, 1)


def preprocess(clip: np.ndarray, n_frames=None):
    if n_frames == 1:
        frame = process_frame(clip[0])
        ort_inputs = {model.get_inputs()[0].name: frame[None]}
        return ort_inputs
    else:
        # step_size = clip.shape[0] // n_frames
        # frames = clip[::step_size][:n_frames]
        frames = clip[:n_frames]
        frames = [process_frame(frame) for frame in frames]
        return np.stack(frames)


def predict(clip: np.ndarray):
    """Вычислить класс для этого клипа. Эта функция должна возвращать *имя* класса."""
    window_size = 2
    thresh = 10
    for i, saved_frame in enumerate(BUFFER[0]):
        if (
            (clip[0, :window_size, :window_size] == saved_frame).sum() > thresh
            or (clip[1, :window_size, :window_size] == saved_frame).sum() > thresh
            or (clip[2, :window_size, :window_size] == saved_frame).sum() > thresh
            or (clip[3, :window_size, :window_size] == saved_frame).sum() > thresh
        ):
            return id2label[BUFFER[1][i]]

    # if have no matches
    frames = preprocess(clip, n_frames=1)
    # predicts = model(frames).argmax(1).numpy()
    predicts = model.run(None, frames)[0].argmax(1)
    predict = Counter(predicts).most_common(1)[0][0]

    BUFFER[0].extend([frame[:window_size, :window_size] for frame in clip[:4]])
    BUFFER[1].extend([predict] * 4)

    return id2label[predict]
