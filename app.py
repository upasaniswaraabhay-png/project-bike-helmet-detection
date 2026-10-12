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
app_mode = st.sidebar.radio("Select Operating Mode", ["Image Snapshot Analysis", "Video Surveillance Detection"])
conf_threshold = st.sidebar.slider("AI Confidence Threshold", 0.1, 1.0, 0.3, 0.05)

@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading YOLO11 model ('best.pt'): {e}")
    st.stop()

# 1. IMAGE SNAPSHOT MODE
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
                st.image(results[0].plot(), caption="AI Detection Overlay", channels="BGR", use_container_width=True)
        st.success("Image analysis completed successfully!")

# 2. VIDEO SURVEILLANCE MODE (Codec-Free Frame Player)
elif app_mode == "Video Surveillance Detection":
    st.subheader("🎥 Video Object Detection Pipeline")
    st.info("💡 Upload a short traffic clip (.mp4) to run frame-by-frame YOLO11 detection.")
    
    uploaded_video = st.file_uploader("Upload a Video Clip", type=["mp4", "mov", "avi"])
    
    if uploaded_video is not None:
        try:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_video.read())
            tfile.close()
            
            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            st.info(f"Loaded video successfully! Processing {total_frames} frames...")
            
            frame_placeholder = st.empty()
            progress_bar = st.progress(0)
            
            processed_frames = []
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resize to protect memory
                frame_resized = cv2.resize(frame, (640, 360))
                
                # Run YOLO11 prediction
                results = model.predict(source=frame_resized, conf=conf_threshold, verbose=False)
                annotated_frame = results[0].plot()
                
                # Convert BGR (OpenCV) to RGB (Streamlit/PIL)
                annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                processed_frames.append(annotated_rgb)
                
                frame_count += 1
                if total_frames > 0:
                    progress_bar.progress(min(frame_count / total_frames, 1.0))
                    
            cap.release()
            st.success("Processing complete! Displaying annotated video stream:")
            
            # Interactive Playback Loop in Browser
            if st.button("▶ Play Processed Video Stream"):
                play_placeholder = st.empty()
                for f in processed_frames:
                    play_placeholder.image(f, channels="RGB", use_container_width=True)
            else:
                # Show final frame thumbnail by default
                st.image(processed_frames[0], channels="RGB", caption="Processed Frame Preview (Click 'Play' above to stream)", use_container_width=True)
                
        except Exception as e:
            st.error(f"An error occurred during video processing: {e}")
