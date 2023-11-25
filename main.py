import streamlit as st
import livecam
from streamlit_webrtc import webrtc_streamer


webrtc_streamer=webrtc_streamer(key="testing", video_processor_factory=livecam.Detect)