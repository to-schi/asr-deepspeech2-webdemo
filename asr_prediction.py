"""
Prediction Service for Deepspeech2-Keras
"""
# pylint: disable=C0301
import logging
import os

import streamlit as st
import tensorflow as tf
import tensorflow_io as tfio
from tensorflow.keras import backend as k
from tensorflow.keras.layers import StringLookup  # type: ignore

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

MODEL_PATH = "./model/DeepSpeech_RNN.h5"
characters = list("abcdefghijklmnopqrstuvwxyz' ")

string_lookup = StringLookup()  # type: ignore
char_to_num = string_lookup(vocabulary=characters, oov_token="")
num_to_char = string_lookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)


class PredictionService:
    """
    Service class for prediction
    """

    model = None
    instance = None

    @tf.function(experimental_relax_shapes=True)
    def encode_audio(self, file, stride=128, window=256, n_mels=256):
        """converts audio-data to spectrogram"""

        audio = tf.io.read_file(file)
        audio = tfio.audio.decode_wav(audio, dtype=tf.int16)
        audio = audio / tf.reduce_max(audio)
        audio = tf.squeeze(audio, axis=-1)
        audio = tf.cast(audio, tf.float32)
        spectrogram = tfio.audio.spectrogram(
            audio, nfft=2048, window=window, stride=stride
        )
        mel_spectrogram = tfio.audio.melscale(
            spectrogram, rate=16000, mels=n_mels, fmin=40, fmax=6000
        )
        dbscale_mel_spectrogram = tfio.audio.dbscale(mel_spectrogram, top_db=80)
        return dbscale_mel_spectrogram

    def decode_predictions(self, pred):
        """decodes model-predictions with ctc-decoder"""

        input_len = tf.ones(pred.shape[0]) * pred.shape[1]

        result = k.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]

        result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
        return result

    def make_prediction(self, file):
        """makes predictions with the pretrained model"""

        if self.model is None:
            self.model = tf.keras.models.load_model(
                MODEL_PATH, custom_objects={"ctc_loss": ctc_loss}
            )
        recording = self.encode_audio(file, 16000)
        recording = tf.expand_dims(recording, axis=0)
        output = self.model.predict(recording)
        output = self.decode_predictions(output)
        return output


@tf.function(experimental_relax_shapes=True)
def ctc_loss(y_true, y_pred):
    """adapted loss-function from keras.backend"""

    batch_len = tf.cast(tf.shape(y_true)[0], dtype=tf.int64)

    input_length = tf.cast(tf.shape(y_pred)[1], dtype=tf.int64)
    label_length = tf.cast(tf.shape(y_true)[1], dtype=tf.int64)

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype=tf.int64)
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype=tf.int64)

    input_length = tf.cast(tf.squeeze(input_length, axis=-1), tf.int32)
    label_length = tf.cast(tf.squeeze(label_length, axis=-1), tf.int32)

    sparse_labels = tf.cast(
        tf.keras.backend.ctc_label_dense_to_sparse(y_true, label_length), tf.int32
    )
    y_pred = tf.math.log(
        tf.transpose(y_pred, perm=[1, 0, 2]) + tf.keras.backend.epsilon()
    )

    loss = tf.expand_dims(
        tf.nn.ctc_loss(
            labels=sparse_labels,
            logits=y_pred,
            label_length=label_length,
            logit_length=input_length,
            blank_index=29,
        ),
        1,
    )
    return loss


@st.cache
def prediction_service():
    """starting prediction service"""

    if PredictionService.instance is None:
        PredictionService.instance = PredictionService()  # type: ignore
        PredictionService.model = tf.keras.models.load_model(
            MODEL_PATH, custom_objects={"ctc_loss": ctc_loss}
        )
    return PredictionService.instance


if __name__ == "__main__":
    PS = prediction_service()
    PS2 = prediction_service()

    assert PS is PS2
