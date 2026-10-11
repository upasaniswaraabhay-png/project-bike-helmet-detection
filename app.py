import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import tempfile
from datetime import datetime
from ultralytics import YOLO

st.set_page_config(page_title="Traffic Police AI | Surveillance", page_icon="🚨", layout="wide")

st.markdown("""
<div style="background-color: #1a1c23; padding: 15px; border-radius: 10px; border-left: 6px solid #ff4b4b;">
    <h2 style="color: #ffffff; margin:0;">🚨 POLICE TRAFFIC SURVEILLANCE & HELMET ENFORCEMENT</h2>
    <p style="color: #888; margin:5px 0 0 0;">YOLO11 Real-Time Object Detection Engine</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.title("👮 Control Center")
input_mode = st.sidebar.radio("Select Input Mode", ["Image Snapshot", "CCTV Video Clip"])
conf_threshold = st.sidebar.slider("AI Confidence", 0.1, 1.0, 0.3, 0.05)
fine_amount = st.sidebar.number_input("Fine Amount ($)", value=1000)

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

if input_mode == "Image Snapshot":
    uploaded_file = st.file_uploader("Upload Traffic Frame", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Feed", use_container_width=True)
        with col2:
            results = model.predict(source=image, conf=conf_threshold)
            st.image(results[0].plot(), caption="AI Detection Overlay", channels="BGR", use_container_width=True)

elif input_mode == "CCTV Video Clip":
    uploaded_video = st.file_uploader("Upload Short CCTV Video (MP4/MOV)", type=["mp4", "mov"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        
        cap = cv2.VideoCapture(tfile.name)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS) or 30)
        
        st.info("Processing video frames... Please wait a moment.")
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
