import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tempfile
from ultralytics import YOLO

st.set_page_config(page_title="Real-Life Object Detection System", page_icon="🔍", layout="wide")

st.markdown("""
<div style="background-color: #1a1c23; padding: 15px; border-radius: 10px; border-left: 6px solid #4b7bec;">
    <h2 style="color: #ffffff; margin:0;">🔍 REAL-LIFE OBJECT DETECTION SYSTEM</h2>
    <p style="color: #888; margin:5px 0 0 0;">YOLO11 Real-Time Computer Vision Pipeline & Traffic Surveillance</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.title("⚙️ Control Panel")
conf_threshold = st.sidebar.slider("AI Confidence Threshold", 0.01, 1.0, 0.25, 0.05)

@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading YOLO11 model ('best.pt'): {e}")
    st.stop()

st.subheader("📁 Universal Media Hub")
uploaded_file = st.file_uploader("Upload an Image (.jpg, .png) or Video (.mp4, .mov)", type=["jpg", "jpeg", "png", "mp4", "mov", "avi"])

if uploaded_file is not None:
    file_name = uploaded_file.name.lower()
    
    # 1. IMAGE HANDLING
    if file_name.endswith(('.jpg', '.jpeg', '.png')):
        st.subheader("📷 Image Detection Mode")
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Input Image", use_container_width=True)
        with col2:
            with st.spinner("Running YOLO11 Inference..."):
                results = model.predict(source=image, conf=conf_threshold)
                st.image(results[0].plot(), caption="AI Detection Overlay", channels="BGR", use_container_width=True)
        st.success("Image analysis completed successfully!")
        
    # 2. VIDEO HANDLING (Native Browser Player with Controls)
    elif file_name.endswith(('.mp4', '.mov', '.avi')):
        st.subheader("🎥 Video Surveillance & Detection Mode")
        
        try:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            tfile.close()
            
            cap = cv2.VideoCapture(tfile.name)
            fps = int(cap.get(cv2.CAP_PROP_FPS) or 20)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Safe downscaling for cloud memory constraints
            target_w = 640
            target_h = int(height * (target_w / width)) if width > 0 else 480
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("Processing frames through YOLO11...")
            
            output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            
            # Use standard mp4v container
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_file.name, fourcc, fps, (target_w, target_h))
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_resized = cv2.resize(frame, (target_w, target_h))
                results = model.predict(source=frame_resized, conf=conf_threshold, verbose=False)
                annotated_frame = results[0].plot()
                
                out.write(annotated_frame)
                frame_count += 1
                
                if total_frames > 0:
                    progress_bar.progress(min(frame_count / total_frames, 1.0))
                    
            cap.release()
            out.release()
            
            status_text.success("Video processing complete!")
            
            # Read the file bytes directly to force Streamlit's video component to render standard player controls
            video_file = open(output_file.name, 'rb')
            video_bytes = video_file.read()
            
            st.video(video_bytes, format="video/mp4")
            
        except Exception as e:
            st.error(f"An error occurred during video processing: {e}")
