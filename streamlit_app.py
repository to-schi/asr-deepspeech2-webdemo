import logging
import logging.handlers
import queue
import urllib.request
from pathlib import Path
import pydub
import os
import streamlit as st
import queue
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from asr_prediction import Prediction_Service

logger = logging.getLogger(__name__)

# Initialize session state for existence of model-file:
if 'model' not in st.session_state:
    st.session_state['model'] = False
# file-variables:
HERE = Path(__file__).parent
MODEL_URL = "https://www.dropbox.com/s/woqi5hfh3bn5thk/DeepSpeech_RNN.h5?raw=1"  # noqa
MODEL_LOCAL_PATH = HERE / "model/DeepSpeech_RNN.h5"
RECORDED = HERE / "recordings/temp.wav"
UPLOADED = HERE / "recordings/uploaded.wav"
EXAMPLE_1 = HERE / "recordings/destroy_him_my_robots.wav"
EXAMPLE_2 = HERE / "recordings/the_jungle_book2.wav"
EXAMPLE_3 = HERE / "recordings/the_jungle_book1.wav"
# page-config:
st.set_page_config(page_title="Speech Recognition Demo",
                   page_icon=":robot_face:")
st.title('Speech Recognition Demo')
st.markdown(
    """
This demo app is using an simplified end-to-end speech recognition engine based on DeepSpeech2.
It was trained on the LibriSpeech 100 hours dataset. The word-error-rate is currently at 19%.
A good microphone is recommended.\n
More information can be found [here](https://github.com/to-schi).
"""
)
# coloring "st.buttons" in main page light blue:
m = st.markdown("""
 <style>
 div.stButton > button:first-child {
     background-color: rgb(0, 131, 184);
 }
 </style>""", unsafe_allow_html=True)

##########################################################
# file-download based on https://github.com/streamlit/demo-self-driving/blob/230245391f2dda0cb464008195a470751c01770b/streamlit_app.py#L48  # noqa: E501
def download_file(url, download_to: Path, expected_size=None):
    # Don't download the file twice.
    # (If possible, verify the download using the file length.)
    if download_to.exists():
        if expected_size:
            if download_to.stat().st_size == expected_size:
                st.session_state['model'] = True
                #st.write("Model found.")
                return
        else:
            st.info(f"{url} is already downloaded.")
            if not st.button("Download again?"):
                return

    download_to.parent.mkdir(parents=True, exist_ok=True)

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % url)
        progress_bar = st.progress(0)
        with open(download_to, "wb") as output_file:
            with urllib.request.urlopen(url) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning(
                        "Downloading %s... (%6.2f/%6.2f MB)"
                        % (url, counter / MEGABYTES, length / MEGABYTES)
                    )
                    progress_bar.progress(min(counter / length, 1.0))
    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()
        st.session_state['model'] = True

## recording-functions:####################################
def detect_leading_silence(sound, silence_threshold=-30.0, chunk_size=5):
    '''
    sound is a pydub.AudioSegment
    silence_threshold in dB
    chunk_size in ms
    iterate over chunks until you find the first one with sound
    '''
    trim_ms = 0  # ms
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold:
        trim_ms += chunk_size
    return trim_ms - chunk_size*40 # added 200ms

def trim_silence(sound):
    start_trim = detect_leading_silence(sound)
    end_trim = detect_leading_silence(sound.reverse())
    duration = len(sound)
    trimmed_sound = sound[start_trim:duration-end_trim]
    return trimmed_sound

# based on https://github.com/whitphx/streamlit-webrtc##########
def record_save(FILENAME):
    webrtc_ctx = webrtc_streamer(
        key="sendonly-audio",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=512,
        rtc_configuration={"iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": False,
                                  "audio": True,
                                  "noiseSuppression": True
                                  },
    )
    if "audio_buffer" not in st.session_state:
        st.session_state["audio_buffer"] = pydub.AudioSegment.empty()
    status_indicator = st.empty()

    while True:
        if webrtc_ctx.audio_receiver:
            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            except queue.Empty:
                status_indicator.write("Connecting...")
                continue

            status_indicator.write("Running. Say something!")

            sound_chunk = pydub.AudioSegment.empty()
            for audio_frame in audio_frames:
                sound = pydub.AudioSegment(
                    data=audio_frame.to_ndarray().tobytes(),
                    sample_width=audio_frame.format.bytes,
                    frame_rate=audio_frame.sample_rate,
                    channels=len(audio_frame.layout.channels),
                )
                sound_chunk += sound

            if len(sound_chunk) > 0:
                st.session_state["audio_buffer"] += sound_chunk
        else:
            status_indicator.write("Click 'start' and wait till 'running'.")
            break

    audio_buffer = st.session_state["audio_buffer"]

    if not webrtc_ctx.state.playing and len(audio_buffer) > 0:
        st.info("Writing wav to disk")
        audio_buffer = trim_silence(audio_buffer)
        #resample to 16000sr, 1ch, 16bit
        audio_buffer = audio_buffer.set_frame_rate(16000).set_channels(1)
        audio_buffer.export(str(FILENAME), format="wav")
        # Reset
        st.session_state["audio_buffer"] = pydub.AudioSegment.empty()

#### predict and display#################################################
def predict_file(file):
    prediction = ps.make_prediction(str(file))
    return prediction
    
def display_audio(file):
    audio_file = open(str(file), 'rb')
    audio_bytes = audio_file.read()
    return audio_bytes

def main():
    # Download model-file if not existing and set session_state['model'] = True
    download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=337431496)
    page = st.sidebar.selectbox(
        "Choose an option", ['Record speech', 'Open wav-file', 'Examples'])

    if page == 'Record speech':
        st.header('Make a new recording')
        record_save(RECORDED)
        if RECORDED.exists() == True:
            st.write("Recording found.")
            predict_rec = st.button('Predict recording')
            if predict_rec:
                st. audio(display_audio(RECORDED), format='audio/wav')
                st.write(f"**Prediction:  '{predict_file(RECORDED)}'**")
                st.download_button(
                    "Save recorded file", display_audio(RECORDED), 'audio')

    elif page == 'Open wav-file':
        st.header('Open wav-file')
        # create upload button:
        uploaded_file = st.file_uploader("")
        if uploaded_file is not None:
            # File is read, uploaded and saved to the server as "uploaded.wav":
            bytes_data = uploaded_file.getvalue()
            with open(str(UPLOADED), 'wb') as f:
                f.write(bytes_data)
            # predict with PredictionService
            st.audio(display_audio(UPLOADED), format='audio/wav')
            st.write(f"**Prediction:  '{predict_file(UPLOADED)}'**")

    else:
        # streamlit container-structure
        st.header('Examples')
        first = st.container()
        second = st.container()
        third = st.container()

        first.audio(display_audio(EXAMPLE_1), format='audio/wav')
        first.write("Text: 'destroy him my robots' (Impossible Mission)")
        example1 = first.button("Transcribe", key=1)

        second.markdown("""---""")
        second.audio(display_audio(EXAMPLE_2), format='audio/wav')
        second.write(
            "Text: 'there was a long hush for no single wolf cared to fight' (The Jungle Book)")
        example2 = second.button("Transcribe", key=2)

        third.markdown("""---""")
        third.audio(display_audio(EXAMPLE_3), format='audio/wav')
        third.write(
            "Text: 'and he grew very tired of saying the same thing over a hundred times' (The Jungle Book)")
        example3 = third.button("Transcribe", key=3)

        if example1:
            first.write(f"**Prediction:  '{predict_file(EXAMPLE_1)}'**")
        if example2:
            second.write(f"**Prediction:  '{predict_file(EXAMPLE_2)}'**")
        if example3:
            third.write(f"**Prediction:  '{predict_file(EXAMPLE_3)}'**")

#########################################################################
if __name__ == '__main__':
    DEBUG = os.environ.get("DEBUG", "false").lower() not in [
        "false", "no", "0"]

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

    if st.session_state['model'] == True:
        ps = Prediction_Service()

    main()

