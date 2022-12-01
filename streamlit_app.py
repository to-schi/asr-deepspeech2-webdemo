"""
Streamlit application for the demo of ASR Deepspeech2-Keras
"""

import logging
import logging.handlers
import os
from pathlib import Path

import streamlit as st
from pydub import AudioSegment

from asr_prediction import Prediction_Service
from model_downloader import download_file
from web_recorder import record_to_file

logger = logging.getLogger(__name__)

# Initialize session state for existence of model-file:
if "interpreter" not in st.session_state:
    st.session_state["interpreter"] = False

# file+path-variables:
HERE = Path(__file__).parent
MODEL_URL = (
    "https://www.dropbox.com/s/1fn1a6kam7peg8q/RNN_ctc2_6000_29out_float16.tflite?raw=1"
)
LM_MODEL_URL = "https://www.dropbox.com/s/ii8b5hrrd9sa13j/3gram_500k_p001.bin?raw=1"

MODEL_LOCAL_PATH = HERE / "model/deepspeech2-float16.tflite"
LM_MODEL_LOCAL_PATH = HERE / "model/kenlm_librispeech.binary"
RECORDED = HERE / "recordings/temp.wav"
UPLOADED = HERE / "recordings/uploaded.wav"
RESAMPLED = HERE / "recordings/resampled.wav"
EXAMPLE_1 = HERE / "recordings/destroy_him_my_robots.wav"
EXAMPLE_2 = HERE / "recordings/the_jungle_book2.wav"
EXAMPLE_3 = HERE / "recordings/the_jungle_book1.wav"

SAMPLERATE = 16000

# page-config:
st.set_page_config(page_title="Speech Recognition Demo", page_icon=":robot_face:")
st.title("Speech Recognition Demo")
st.markdown(
    """
This demo app is presenting an end-to-end speech recognition engine similar to 
[DeepSpeech2](https://arxiv.org/abs/1512.02595).
It was implemented in Tensorflow and trained on the [LibriSpeech](https://www.openslr.org/12) 
dataset with 960 hours of English speech. The word-error-rate on the clean test data is 10%.\n
With the use of the **language model** the performance can significantly be improved (6% wer). 
It works as a scorer for letter-sequence probabilities in the decoder. 
You can switch off the language model to see the transcriptions generated using only the 
information of the **accustic model**. A good microphone is recommended.\n
More information can be found [here](https://github.com/to-schi/ASR-Deepspeech2-Tensorflow).\n
"""
)
# color "st.buttons" in main page light blue:
st.markdown(
    """
 <style>
 div.stButton > button:first-child {
     background-color: rgb(0, 131, 184);
 }
 </style>""",
    unsafe_allow_html=True,
)
# hide menu
st.markdown(
    """ <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """,
    unsafe_allow_html=True,
)

# sidebar image
st.sidebar.image("./img/nummer5_w_input.svg")

### functions #########################################################################
def read_audio(file):
    """
    reads an audio-file to bytes
    """
    audio_file = open(str(file), "rb")
    audio_bytes = audio_file.read()
    return audio_bytes


def main():
    """
    Main function for defining streamlit page-behaviour
    """
    # set 3 pages to select in sidebar:
    page = st.sidebar.selectbox(
        "Choose an option:", ["Record speech", "Open wav-file", "Examples"]
    )

    if page == "Record speech":
        st.header("Record speech")
        record_to_file(RECORDED)

        if RECORDED.exists() is True:
            recording_info = st.info("Recording found.")
            st.audio(read_audio(RECORDED), format="audio/wav")

            lm_r = st.checkbox("Use language model", value=False, key=4)
            predict_rec = st.button("Transcribe recording")
            if predict_rec:
                prediction = ps.make_prediction(str(RECORDED), lm_r)
                recording_info.info(f"**Prediction:  '{prediction}'**")
                st.download_button("Save recorded file", read_audio(RECORDED), "audio")

            # Creates a button to delete the current recorded file from the server.
            delete_rec = st.button("Delete recording")
            if delete_rec:
                os.remove(RECORDED)
                recording_info.info("Recording deleted.")

    elif page == "Open wav-file":
        st.header("Open wav-file")
        # create upload button:
        uploaded_file = st.file_uploader("")
        lm_o = st.checkbox("Use language model", value=False, key=5)

        if uploaded_file is not None:
            # Uploaded file is saved to the server as "uploaded.wav":
            bytes_data = uploaded_file.getvalue()
            with open(str(UPLOADED), "wb") as file:
                file.write(bytes_data)

            # resample to 16000Hz and reduce channels to 1:
            audio_data = AudioSegment.from_wav(str(UPLOADED))
            audio_data = audio_data.set_channels(1)
            audio_data = audio_data.set_frame_rate(SAMPLERATE)
            audio_data.export(str(RESAMPLED), format="wav")

            # predict with PredictionService
            st.audio(read_audio(RESAMPLED), format="audio/wav")
            transcribing = st.info("Transcribing...")
            prediction = ps.make_prediction(str(RESAMPLED), lm_o)
            transcribing.info(f"**Prediction:  '{prediction}'**")

    else:
        # streamlit container-structure
        st.header("Examples")
        first = st.container()
        second = st.container()
        third = st.container()

        lm_e = first.checkbox("Use language model", value=True, key=6)
        first.audio(read_audio(EXAMPLE_1), format="audio/wav")
        first.write("Text: 'destroy him my robots' (Impossible Mission)")
        example1 = first.button("Transcribe", key=1)

        second.markdown("""---""")
        second.audio(read_audio(EXAMPLE_2), format="audio/wav")
        second.write(
            "Text: 'there was a long hush for no single wolf cared to fight' (The Jungle Book)"
        )
        example2 = second.button("Transcribe", key=2)

        third.markdown("""---""")
        third.audio(read_audio(EXAMPLE_3), format="audio/wav")
        third.write(
            "Text: 'and he grew very tired of saying the same thing over a hundred times' (The Jungle Book)"
        )
        example3 = third.button("Transcribe", key=3)

        if example1:
            first.write(
                f"**Prediction:  '{ps.make_prediction(str(EXAMPLE_1), lm_e)}'**"
            )
        if example2:
            second.write(
                f"**Prediction:  '{ps.make_prediction(str(EXAMPLE_2), lm_e)}'**"
            )
        if example3:
            third.write(
                f"**Prediction:  '{ps.make_prediction(str(EXAMPLE_3), lm_e)}'**"
            )


#########################################################################
if __name__ == "__main__":
    DEBUG = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
        "%(message)s",
        force=True,
    )
    logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)

    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.DEBUG)

    fsevents_logger = logging.getLogger("fsevents")
    fsevents_logger.setLevel(logging.WARNING)

    if st.session_state["interpreter"] is True:
        ps = Prediction_Service()

    else:
        # Download model-file if not existing and set session_state['model'] = True
        download_file(
            MODEL_URL, MODEL_LOCAL_PATH, expected_size=56304616
        )  # expected_size= 337407816 112500560 112496464
        download_file(LM_MODEL_URL, LM_MODEL_LOCAL_PATH, expected_size=377637008)
        ps = Prediction_Service()

    main()
