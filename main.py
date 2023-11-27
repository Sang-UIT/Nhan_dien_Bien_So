import streamlit as st
import livecam
import livecam_tesseract
from streamlit_webrtc import webrtc_streamer

#webrtc_streamer=webrtc_streamer(key="testing", video_transformer_factory=livecam.Detect)
webrtc_streamer=webrtc_streamer(key="testing", video_transformer_factory=livecam_tesseract.Detect)