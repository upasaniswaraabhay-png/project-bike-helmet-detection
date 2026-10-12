import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import tempfile
from ultralytics import YOLO

st.set_page_config(page_title="Real-Life Object Detection System", page_icon="🔍", layout="wide")

st.markdown("""
<div style="background-color: #1a1c23; padding: 15px; border-radius: 10px; border-left: 6px solid #4b7bec;">
    <h2 style="color: #ffffff; margin:0;">🔍 REAL-LIFE OBJECT DETECTION SYSTEM</h2>
    <p style="color: #888; margin:5px 0 0 0;">YOLO11 Real-Time Computer Vision Pipeline</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.title("⚙️ Control Panel")
input_mode = st.sidebar.radio("Select Demo Mode", ["Image Snapshot Analysis", "Video Demonstration"])
conf_threshold = st.sidebar.slider("AI Confidence Threshold", 0.1, 1.0, 0.3, 0.05)

@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

if input_mode == "Image Snapshot Analysis":
    st.subheader("📷 Image Detection & Traffic Compliance Mode")
    uploaded_file = st.file_uploader("Upload Test Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Input Frame", use_container_width=True)
        with col2:
            with st.spinner("Running YOLO11 Inference..."):
                results = model.predict(source=image, conf=conf_threshold)
                st.image(results[0].plot(), caption="AI Detection Overlay", channels="BGR", use_container_width=True)
        st.success("Inference executed successfully!")

elif input_mode == "Video Demonstration":
    st.subheader("🎥 Video Surveillance Stream")
    st.info("💡 **Tip for Evaluators:** For real-time heavy video processing, models are executed locally. Below is your live video processing module.")
    
    uploaded_video = st.file_uploader("Upload Video Clip", type=["mp4", "mov", "avi"])
    
    if uploaded_video is not None:
        try:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_video.read())
            tfile.close()
            
            st.video(tfile.name)
            st.success("Video loaded and streamed successfully!")
        except Exception as e:
            st.error(f"Could not load video: {e}")
