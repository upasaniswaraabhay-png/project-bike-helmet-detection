import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Bike Helmet Detection AI", 
    page_icon="🚴‍♂️", 
    layout="centered"
)

st.title("🚴‍♂️ Real-Time Bike Helmet Detection System")
st.write("Powered by **YOLO11n** | Upload an image to analyze safety compliance (*With Helmet* vs. *Without Helmet*).")

# --- Load Model Weights ---
@st.cache_resource
def load_yolo_model():
    # Looks for best.pt inside weights folder or main directory
    model_path = "weights/best.pt" if os.path.exists("weights/best.pt") else "best.pt"
    return YOLO(model_path)

try:
    model = load_yolo_model()
    st.success("YOLO11 Model Loaded Successfully!")
except Exception as e:
    st.error(f"Error loading model weights: {e}")

# --- Image Upload Section ---
uploaded_file = st.file_uploader("Choose an image (JPG, PNG, JPEG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    input_image = Image.open(uploaded_file)
    st.image(input_image, caption="Uploaded Image", use_container_width=True)
    
    if st.button("Detect Helmets"):
        with st.spinner("Analyzing with YOLO11..."):
            # Save uploaded file to temp path for YOLO model prediction
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                input_image.save(tmp_file.name)
                tmp_path = tmp_file.name

            # Run prediction
            results = model.predict(source=tmp_path, conf=0.25)
            
            # Save result image
            res_image_path = results[0].save()
            
            # Display processed image
            st.image(res_image_path, caption="Detection Result", use_container_width=True)
            
            # Cleanup temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)