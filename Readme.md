# Webdemo of a speech recognition engine with streamlit
[![StreamlitCloud](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/to-schi/asr-deepspeech2-webdemo/main)
This is a demo of an [end-to-end speech recognition engine](https://github.com/to-schi/speech-recognition-from-scratch) similar to DeepSpeech2. It was trained on the LibriSpeech dataset with 360 hours of English speech and is a work in progress. The word-error-rate on the test data is currently at 14%.

### Options:
1. Make an own recording directly from the browser
2. Upload a wav-file from your computer
3. Try out examples from the app

-> Let the engine transcribe your speech.

[![app](./img/screenshot.jpg)](https://share.streamlit.io/to-schi/asr-deepspeech2-webdemo/main)

The app is using [streamlit_webrtc](https://github.com/whitphx/streamlit-webrtc) for audio streaming.



