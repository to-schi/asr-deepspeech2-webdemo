"""
Prediction Service for ASR-Deepspeech2-Tensorflow
"""
import logging
import os
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
import tensorflow_io as tfio
from pyctcdecode import build_ctcdecoder

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
HERE = Path(__file__).parent
MODEL_LOCAL_PATH = HERE / "model/deepspeech2-float16.tflite"
LM_MODEL_LOCAL_PATH = HERE / "model/kenlm_librispeech.binary"
UNIGRAMS_PATH = HERE / "vocab/vocab-600000.txt"
characters = list("abcdefghijklmnopqrstuvwxyz' ")
char_to_int = tf.keras.layers.StringLookup(vocabulary=characters, oov_token="")
int_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_int.get_vocabulary(), oov_token="", invert=True
)


class _Prediction_Service:

    interpreter = None
    _instance = None

    @tf.function(experimental_relax_shapes=True)
    def encode_audio(
        self, file, sr=16000, n_fft=2048, stride=128, window=256, n_mels=256
    ):
        audio = tf.io.read_file(file)
        audio = tfio.audio.decode_wav(audio, dtype=tf.int16)
        audio = audio / tf.reduce_max(audio)
        audio = tf.squeeze(audio, axis=-1)

        audio = tf.cast(audio, tf.float32)
        spectrogram = tfio.audio.spectrogram(
            audio, nfft=n_fft, window=window, stride=stride
        )
        mel_spectrogram = tfio.audio.melscale(
            spectrogram, rate=sr, mels=n_mels, fmin=40, fmax=6000
        )
        dbscale_mel_spectrogram = tfio.audio.dbscale(mel_spectrogram, top_db=80)
        return dbscale_mel_spectrogram

    @tf.function(experimental_relax_shapes=True)
    def ctc_decoding(self, y_pred, input_length):
        """
        Based on: https://github.com/keras-team/keras/blob/master/keras/backend.py
        """
        input_shape = tf.keras.backend.shape(y_pred)
        num_samples, num_steps = input_shape[0], input_shape[1]
        y_pred = tf.math.log(
            tf.transpose(y_pred, perm=[1, 0, 2]) + tf.keras.backend.epsilon()
        )
        input_length = tf.cast(input_length, tf.int32)

        (decoded, log_prob) = tf.nn.ctc_greedy_decoder(
            inputs=y_pred,
            sequence_length=input_length,
            blank_index=0,  # Default: num_classes - 1 , here="oov_token"=0
        )

        decoded_dense = []
        for sp_tensor in decoded:
            sp_tensor = tf.SparseTensor(
                sp_tensor.indices, sp_tensor.values, (num_samples, num_steps)
            )
            decoded_dense.append(
                tf.sparse.to_dense(sp_input=sp_tensor, default_value=None)
            )
        return (decoded_dense, log_prob)

    @st.cache
    def build_ctc_decoder(self):
        """
        Build the pyctc-decoder with the binary language model and 
        corresponding unigrams from the vocabulary file.
        """
        with open(str(UNIGRAMS_PATH)) as file:
            list_of_unigrams = [line.rstrip() for line in file]

        decoder = build_ctcdecoder(
            labels=char_to_int.get_vocabulary(),
            kenlm_model_path=str(LM_MODEL_LOCAL_PATH),  # either .arpa or .bin file
            unigrams=list_of_unigrams,
            alpha=0.6,  # 0.9,  # 0.7 tuned on a val set, 0.931289039105002 0.5  LM Weight: 0.75
            beta=1.2,  # 1.2,  # 1.0 tuned on a val set, 1.1834137581510284 1.0 # LM Usage Reward: 1.85
        )
        return decoder


    def decode_predictions(self, pred):
        """
        Takes prediction-logits and decodes to text.
        """
        input_len = tf.ones(pred.shape[0]) * pred.shape[1]
        result = self.ctc_decoding(pred, input_len)[0][0]
        result = tf.strings.reduce_join(int_to_char(result)).numpy().decode("utf-8")
        return result


    def decode_predictions_lm(self, logits, beam_width=100):
        """
        Takes prediction-logits and decodes to text with
        pyctcdecoder using a kenlm-model.
        """
        decoder = self.build_ctc_decoder()

        logits = np.squeeze(logits)
        text = decoder.decode(logits, beam_width=beam_width)
        return text


    def make_prediction(self, file, lm: bool):
        """
        Main prediction function.
        Args:
            file = filepath of input
            lm = bool for use of language model
        Returns:
            prediction as str
        """
        if self.interpreter is None:
            self.interpreter = tf.lite.Interpreter(MODEL_LOCAL_PATH)

        input_details = self.interpreter.get_input_details()[0]
        output_details = self.interpreter.get_output_details()[0]

        recording = self.encode_audio(file, 16000)
        recording = tf.expand_dims(recording, axis=0)
        self.interpreter.resize_tensor_input(input_details["index"], recording.shape)
        self.interpreter.allocate_tensors()
        self.interpreter.set_tensor(input_details["index"], recording)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(output_details["index"])

        if lm is True:
            output = self.decode_predictions_lm(output)
        else:
            output = self.decode_predictions(output)

        return output


@tf.function(experimental_relax_shapes=True)
def ctc_loss(y_true, y_pred):
    """
    Loss function based on:
    https://github.com/keras-team/keras/blob/253dc4604479b832dd254d0d348c0b3e7e53fe0f/keras/backend.py
    Adapted to 'blank_index = 0'
    """
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
            blank_index=0,
        ),
        1,
    )
    return loss


@st.cache
def Prediction_Service():
    if _Prediction_Service._instance is None:
        _Prediction_Service._instance = _Prediction_Service()
        _Prediction_Service.interpreter = tf.lite.Interpreter(str(MODEL_LOCAL_PATH))

    return _Prediction_Service._instance


if __name__ == "__main__":
    ps = Prediction_Service()
    ps2 = Prediction_Service()

    assert ps is ps2
