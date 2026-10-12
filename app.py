import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import tempfile
from ultralytics import YOLO

# Page Configuration
st.set_page_config(page_title="Real-Life Object Detection System", page_icon="🔍", layout="wide")

# Professional Traffic Enforcement Header
st.markdown("""
<div style="background-color: #1a1c23; padding: 15px; border-radius: 10px; border-left: 6px solid #4b7bec;">
    <h2 style="color: #ffffff; margin:0;">🔍 REAL-LIFE OBJECT DETECTION SYSTEM</h2>
    <p style="color: #888; margin:5px 0 0 0;">YOLO11 Real-Time Computer Vision Pipeline & Traffic Surveillance</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Control Panel
st.sidebar.title("⚙️ Control Panel")
app_mode = st.sidebar.radio("Select Operating Mode", ["Image Snapshot Analysis", "Video Surveillance Detection"])
conf_threshold = st.sidebar.slider("AI Confidence Threshold", 0.1, 1.0, 0.3, 0.05)

# Load YOLO11 Model (Cached)
@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading YOLO11 model ('best.pt'): {e}")
    st.stop()

# ==========================================
# 1. IMAGE SNAPSHOT ANALYSIS MODE
# ==========================================
if app_mode == "Image Snapshot Analysis":
    st.subheader("📷 Image Detection & Traffic Compliance")
    uploaded_image = st.file_uploader("Upload a traffic snapshot", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Original Input Image", use_container_width=True)
            
        with col2:
            with st.spinner("Running YOLO11 Inference..."):
                results = model.predict(source=image, conf=conf_threshold)
                annotated_img = results[0].plot()
                st.image(annotated_img, caption="AI Detection Overlay", channels="BGR", use_container_width=True)
                
        st.success("Image analysis completed successfully!")

# ==========================================
# 2. VIDEO SURVEILLANCE DETECTION MODE
# ==========================================
elif app_mode == "Video Surveillance Detection":
    st.subheader("🎥 Video Object Detection Pipeline")
    st.info("💡 Upload a short traffic clip (.mp4) to run frame-by-frame YOLO11 detection with browser playback.")
    
    uploaded_video = st.file_uploader("Upload a Video Clip", type=["mp4", "mov", "avi"])
    
    if uploaded_video is not None:
        try:
            # Save uploaded video to temporary file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_video.read())
            tfile.close()
            
            cap = cv2.VideoCapture(tfile.name)
            fps = int(cap.get(cv2.CAP_PROP_FPS) or 20)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Downscale resolution to protect cloud memory limits (640px width)
            target_w = 640
            target_h = int(height * (target_w / width)) if width > 0 else 480
            
            # Setup output video writer using browser-compatible mp4 format
            output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_file.name, fourcc, fps, (target_w, target_h))
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = 0
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("Processing video frames through YOLO11...")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resize and predict
                frame_resized = cv2.resize(frame, (target_w, target_h))
                results = model.predict(source=frame_resized, conf=conf_threshold, verbose=False)
                annotated_frame = results[0].plot()
                
                # Write frame
                out.write(annotated_frame)
                frame_count += 1
                
                if total_frames > 0:
                    progress_bar.progress(min(frame_count / total_frames, 1.0))
                    
            cap.release()
            out.release()
            
            status_text.text("Video processing complete!")
            st.success("YOLO11 successfully annotated the video stream!")
            
            # Play processed video natively in browser
            st.video(output_file.name)
            
        except Exception as e:
            st.error(f"An error occurred during video processing: {e}")
