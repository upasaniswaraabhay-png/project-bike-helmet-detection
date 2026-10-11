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
conf_threshold = st.sidebar.slider("AI Confidence Threshold", 0.1, 1.0, 0.3, 0.05)

@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.subheader("📁 Media Upload Center")
# Single universal uploader that accepts both images and videos to prevent caching restriction bugs
uploaded_file = st.file_uploader("Upload an Image (.jpg, .png) or Video (.mp4, .mov)", type=["jpg", "jpeg", "png", "mp4", "mov", "avi", "mkv"])

if uploaded_file is not None:
    file_name = uploaded_file.name.lower()
    
    # Handle Image Files
    if file_name.endswith(('.jpg', '.jpeg', '.png')):
        st.subheader("📷 Image Detection Mode")
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Input", use_container_width=True)
        with col2:
            results = model.predict(source=image, conf=conf_threshold)
            st.image(results[0].plot(), caption="Detected Objects Overlay", channels="BGR", use_container_width=True)
            
    # Handle Video Files
    elif file_name.endswith(('.mp4', '.mov', '.avi', '.mkv')):
        st.subheader("🎥 Video Detection Mode")
        try:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            tfile.close()
            
            cap = cv2.VideoCapture(tfile.name)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS) or 30)
            
            st.info("Processing video frames through YOLO11... Please wait.")
            progress_bar = st.progress(0)
            
            output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            out = cv2.VideoWriter(output_file.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
            
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
            
        except Exception as e:
            st.error(f"An error occurred during video processing: {e}")
