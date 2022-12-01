"""
Module for webrtc stream-recording
"""
import queue

import pydub
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer


def detect_leading_silence(sound, silence_threshold=-30.0, chunk_size=5):
    """
    sound is a pydub.AudioSegment
    silence_threshold in dB
    chunk_size in ms
    iterate over chunks until you find the first one with sound
    """
    trim_ms = 0  # ms
    while sound[trim_ms : trim_ms + chunk_size].dBFS < silence_threshold:
        trim_ms += chunk_size
    return trim_ms - chunk_size * 40  # added 200ms


def trim_silence(sound):
    start_trim = detect_leading_silence(sound)
    end_trim = detect_leading_silence(sound.reverse())
    duration = len(sound)
    trimmed_sound = sound[start_trim : duration - end_trim]
    return trimmed_sound


def record_to_file(FILENAME):
    """
    based on https://github.com/whitphx/streamlit-webrtc
    """
    webrtc_ctx = webrtc_streamer(
        key="sendonly-audio",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=512,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={
            "video": False,
            "audio": True,
            "noiseSuppression": True,
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
        status_indicator.write("Writing wav to disk")  # st.info("Writing wav to disk")
        audio_buffer = trim_silence(audio_buffer)
        # resample to 16000sr, 1ch, 16bit
        audio_buffer = audio_buffer.set_frame_rate(16000).set_channels(1)
        audio_buffer.export(str(FILENAME), format="wav")
        # Reset
        st.session_state["audio_buffer"] = pydub.AudioSegment.empty()
        status_indicator.empty()
