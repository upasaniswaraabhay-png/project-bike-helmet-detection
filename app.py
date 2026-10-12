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
input_mode = st.sidebar.radio("Select Demo Mode", ["Image Snapshot Analysis", "Video Surveillance Detection"])
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

elif input_mode == "Video Surveillance Detection":
    st.subheader("🎥 Video Object Detection Pipeline")
    st.info("💡 Upload a short clip (.mp4) to run full YOLO11 frame-by-frame detection.")
    
    uploaded_video = st.file_uploader("Upload Video File", type=["mp4", "mov", "avi"])
    
    if uploaded_video is not None:
        try:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_video.read())
            tfile.close()
            
            cap = cv2.VideoCapture(tfile.name)
            fps = int(cap.get(cv2.CAP_PROP_FPS) or 20)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Downscale resolution to protect cloud memory limits
            target_w = 640
            target_h = int(height * (target_w / width)) if width > 0 else 480
            
            output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.avi')
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter(output_file.name, fourcc, fps, (target_w, target_h))
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = 0
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("Processing video through YOLO11...")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resize frame for performance
                frame_resized = cv2.resize(frame, (target_w, target_h))
                
                # Run YOLO11 prediction on the frame
                results = model.predict(source=frame_resized, conf=conf_threshold, verbose=False)
                annotated_frame = results[0].plot()
                
                out.write(annotated_frame)
                frame_count += 1
                
                if total_frames > 0:
                    progress_bar.progress(min(frame_count / total_frames, 1.0))
                    
            cap.release()
            out.release()
            status_text.text("Video processing complete!")
            st.success("YOLO11 successfully annotated the video stream!")
            st.video(output_file.name)
            
        except Exception as e:
            st.error(f"Video processing error: {e}")
