import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras
import logging
import streamlit as st
import tensorflow as tf
import tensorflow_io as tfio

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

MODEL_PATH = "./model/DeepSpeech_RNN.h5"
characters = [x for x in "abcdefghijklmnopqrstuvwxyz' "]
char_to_num = keras.layers.StringLookup(vocabulary=characters, oov_token="")
num_to_char = keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)

class _Prediction_Service:

    model = None
    _instance = None

    @tf.function(experimental_relax_shapes=True)
    def encode_audio(self, file, sr=16000, n_fft=2048, stride=128, window=256, n_mels=256):
        audio = tf.io.read_file(file)
        audio = tfio.audio.decode_wav(audio, dtype=tf.int16)

        # resample if necessary:
        audio_tensor = tfio.audio.AudioIOTensor(file)
        rate_in = audio_tensor.rate.numpy()
        if rate_in != sr:
            audio = tfio.audio.resample(audio, rate_in=rate_in, rate_out=sr)

        audio = audio / tf.reduce_max(audio)
        audio = tf.squeeze(audio, axis=-1)
        audio = tf.cast(audio, tf.float32)
        spectrogram = tfio.audio.spectrogram(
            audio, nfft=n_fft, window=window, stride=stride)
        mel_spectrogram = tfio.audio.melscale(
            spectrogram, rate=sr, mels=n_mels, fmin=40, fmax=6000)
        dbscale_mel_spectrogram = tfio.audio.dbscale(
            mel_spectrogram, top_db=80)
        return dbscale_mel_spectrogram


    def decode_predictions(self, pred):
        input_len = tf.ones(pred.shape[0]) * pred.shape[1]

        result = keras.backend.ctc_decode(
            pred, input_length=input_len, greedy=True)[0][0]

        result = tf.strings.reduce_join(
            num_to_char(result)).numpy().decode("utf-8")
        return result


    def make_prediction(self, file):
        if self.model == None:
            self.model = keras.models.load_model(
                MODEL_PATH, custom_objects={'ctc_loss': ctc_loss})
        recording = self.encode_audio(file, 16000)
        recording = tf.expand_dims(recording, axis=0)
        output = self.model.predict(recording)
        output = self.decode_predictions(output)
        return output


@tf.function(experimental_relax_shapes=True)
def ctc_loss(y_true, y_pred):
    # Compute the training-time loss value
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = keras.backend.ctc_batch_cost(
        y_true, y_pred, input_length, label_length)
    return loss

@st.cache
def Prediction_Service():
    if _Prediction_Service._instance is None:
        _Prediction_Service._instance = _Prediction_Service()
        _Prediction_Service.model = keras.models.load_model(MODEL_PATH, custom_objects={'ctc_loss': ctc_loss})
    return _Prediction_Service._instance

if __name__ == "__main__":
    ps = Prediction_Service()
    ps2 = Prediction_Service()

    assert ps is ps2
