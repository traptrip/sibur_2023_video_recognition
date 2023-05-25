import warnings
import pathlib
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from joblib import load



warnings.filterwarnings("ignore")


PROJECTOR_FILE = pathlib.Path(__file__).parent.joinpath("projector-v1.joblib")
CLASSIFIER_FILE = pathlib.Path(__file__).parent.joinpath("classifier-v1.joblib")


def construct_model():
    fts_extract = tf.keras.Sequential([
        tf.keras.layers.Resizing(96, 96, interpolation="bilinear"),
        tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1),
        hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v2_100_96/feature_vector/5", trainable=False)
    ])
    fts_extract.build([None, 240, 320, 3])
    return fts_extract


model = construct_model()
projector = load(PROJECTOR_FILE)
classifier = load(CLASSIFIER_FILE)


def predict(clip: np.ndarray):
    """Вычислить класс для этого клипа. Эта функция должна возвращать *имя* класса."""

    features = projector.transform(model(clip).numpy()).mean(axis=0, keepdims=True)
    return classifier.predict(features)[0]
