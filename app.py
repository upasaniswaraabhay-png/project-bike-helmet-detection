import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import tempfile
from datetime import datetime
from ultralytics import YOLO

st.set_page_config(page_title="Real-Life Object Detection System", page_icon="🔍", layout="wide")

st.markdown("""
<div style="background-color: #1a1c23; padding: 15px; border-radius: 10px; border-left: 6px solid #4b7bec;">
    <h2 style="color: #ffffff; margin:0;">🔍 REAL-LIFE OBJECT DETECTION SYSTEM</h2>
    <p style="color: #888; margin:5px 0 0 0;">YOLO11 Real-Time Computer Vision Pipeline</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.title("⚙️ Control Panel")
input_mode = st.sidebar.radio("Select Input Mode", ["Image Snapshot", "Video Clip"])
conf_threshold = st.sidebar.slider("AI Confidence Threshold", 0.1, 1.0, 0.3, 0.05)

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

if input_mode == "Image Snapshot":
    uploaded_file = st.file_uploader("Upload Test Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Input", use_container_width=True)
        with col2:
            results = model.predict(source=image, conf=conf_threshold)
            st.image(results[0].plot(), caption="Detected Objects Overlay", channels="BGR", use_container_width=True)

elif input_mode == "Video Clip":
    uploaded_video = st.file_uploader("Upload Short Video Clip (MP4/MOV)", type=["mp4", "mov"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        
        cap = cv2.VideoCapture(tfile.name)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS) or 30)
        
        st.info("Processing video frames through YOLO11... Please wait.")
        progress_bar = st.progress(0)
        
        output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        out = cv2.VideoWriter(output_file.name, cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height))
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model.predict(source=frame, conf=conf_threshold, verbose=False)
            annotated_frame = results[0].plot()
            out.write(annotated_frame)
            frame_count += 1
            if total_frames > 0:
                progress_bar.progress(min(frame_count / total_frames, 1.0))
                
        cap.release()
        out.release()
        st.success("Video processing complete!")
        st.video(output_file.name)
