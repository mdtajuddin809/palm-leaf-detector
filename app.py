import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import numpy as np
from PIL import Image

# --- Page Config ---
st.set_page_config(page_title="Palm Leaf Detector", page_icon="🌿", layout="wide")

# --- Load Model (Cached for performance) ---
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# --- Sidebar ---
with st.sidebar:
    st.title("🌿 Palm Leaf Nutrient Detection")
    st.markdown("Upload a leaf image and adjust the detection settings.")
    
    # Confidence Threshold Slider
    conf_threshold = st.slider("Confidence Threshold", min_value=0.1, max_value=1.0, value=0.25, step=0.05)
    
    # File Uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# --- Main App ---
if uploaded_file is not None:
    # Convert the file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # Convert BGR to RGB for displaying the original image correctly
    original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Prediction with the selected confidence threshold
    with st.spinner("Analyzing the leaf... 🔍"):
        results = model(image, conf=conf_threshold)

    # Plot result
    result_img = results[0].plot()
    result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

    # --- Layout: Show Original vs Predicted side-by-side ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(original_rgb, use_column_width=True)
    with col2:
        st.subheader("Detection Output")
        st.image(result_rgb, use_column_width=True)

    # --- Download Button ---
    # Convert result back to PIL Image for downloading
    result_pil = Image.fromarray(result_rgb)
    with st.sidebar:
        st.download_button(
            label="⬇️ Download Annotated Image",
            data=cv2.imencode('.jpg', cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR))[1].tobytes(),
            file_name="detection_output.jpg",
            mime="image/jpeg"
        )

    # --- Detection Details ---
    st.markdown("---")
    st.subheader("📊 Detection Details:")
    
    boxes = results[0].boxes
    if len(boxes) == 0:
        st.warning("No nutrients/deficiencies detected. Try lowering the confidence threshold.")
    else:
        # Create columns for metrics based on number of detections
        cols = st.columns(min(len(boxes), 4)) # Max 4 columns per row
        
        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]
            
            # Display in neat metric boxes
            with cols[i % 4]:
                st.metric(label=f"🌿 {label}", value=f"{conf*100:.1f}%")
                
else:
    # Placeholder when no image is uploaded
    st.markdown(
        """
        <div style="text-align: center; margin-top: 100px;">
            <h2>Welcome to the Palm Leaf Nutrient Detector!</h2>
            <p>Please upload an image from the sidebar to get started.</p>
        </div>
        """, 
        unsafe_allow_html=True
    )