import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tempfile
import time
from ultralytics import YOLO

st.set_page_config(page_title="Real-Life Object Detection System", page_icon="🔍", layout="wide")

# Professional Traffic Surveillance Header
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
st.info("💡 Upload any **Image** (.jpg, .png) or **Video** (.mp4, .mov, .avi) below, and the system will automatically detect and play it!")

# Single universal file uploader to avoid caching restriction bugs
uploaded_file = st.file_uploader("Upload Image or Video", type=["jpg", "jpeg", "png", "mp4", "mov", "avi"])

if uploaded_file is not None:
    file_name = uploaded_file.name.lower()
    
    # ==========================================
    # 1. IMAGE HANDLING
    # ==========================================
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
        
    # ==========================================
    # 2. VIDEO HANDLING (Smooth Frame-by-Frame Player)
    # ==========================================
    elif file_name.endswith(('.mp4', '.mov', '.avi')):
        st.subheader("🎥 Video Surveillance & Detection Mode")
        
        try:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            tfile.close()
            
            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text(f"Processing {total_frames} frames through YOLO11...")
            
            processed_frames = []
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resize for smooth performance and memory optimization on cloud
                frame_resized = cv2.resize(frame, (640, 360))
                
                # YOLO11 Inference
                results = model.predict(source=frame_resized, conf=conf_threshold, verbose=False)
                annotated_frame = results[0].plot()
                
                # Convert BGR to RGB for browser playback compatibility
                annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                processed_frames.append(annotated_rgb)
                
                frame_count += 1
                if total_frames > 0:
                    progress_bar.progress(min(frame_count / total_frames, 1.0))
                    
            cap.release()
            status_text.success("Processing complete! Streaming annotated video feed below:")
            
            # Auto-playing video stream container
            if processed_frames:
                video_placeholder = st.empty()
                st.info("🔄 Playing live detection stream...")
                
                # Smooth playback loop showing bounding boxes in action
                for f in processed_frames:
                    video_placeholder.image(f, channels="RGB", use_container_width=True)
                    time.sleep(0.03)  # Matches standard video frame rate
                    
        except Exception as e:
            st.error(f"An error occurred during video processing: {e}")
