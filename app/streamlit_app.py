"""
streamlit_app.py

Author: Ayush Kr. Gupta
Roll Number: 113

This is the Streamlit web application for flower classification.
It loads a converted TFLite model, runs image classification on uploaded images,
displays predictions with confidence bars, and supports generating Grad-CAM heatmaps.
"""

import os
import sys
import tempfile
import numpy as np
import tensorflow as tf
from PIL import Image
import streamlit as st

# Add project root to python path to resolve scripts imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.grad_cam import generate_gradcam

# Page configuration for a premium look
st.set_page_config(
    page_title="FloraVision AI | Deep Learning Flower Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling Injection (Sleek glassmorphism/gradient vibes)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

/* Global Font Override */
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

/* Premium Header styling */
.main-title {
    background: linear-gradient(135deg, #10B981 0%, #3B82F6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    font-size: 2.8rem;
    text-align: center;
    margin-bottom: 0.5rem;
}

.subtitle {
    text-align: center;
    font-size: 1.1rem;
    color: #6B7280;
    margin-bottom: 2rem;
}

/* Glassmorphism Card Wrapper */
.premium-card {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    border-radius: 15px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    padding: 1.5rem;
    margin-top: 1rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
}

/* Horizontal Progress Bars styling */
.bar-container {
    margin-bottom: 15px;
}

.bar-label-row {
    display: flex;
    justify-content: space-between;
    font-weight: 500;
    font-size: 0.95rem;
    margin-bottom: 4px;
}

.bar-bg {
    background-color: rgba(255, 255, 255, 0.1);
    border-radius: 6px;
    height: 12px;
    width: 100%;
    overflow: hidden;
}

.bar-fill {
    background: linear-gradient(90deg, #10B981 0%, #3B82F6 100%);
    height: 100%;
    border-radius: 6px;
    transition: width 0.8s ease-in-out;
}
</style>
""", unsafe_allow_html=True)

# Setup Paths relative to project root
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TFLITE_MODEL_PATH = os.path.join(ROOT_DIR, "models", "model.tflite")
KERAS_MODEL_PATH = os.path.join(ROOT_DIR, "models", "best_model.keras")

# Supported flower class labels
CLASS_NAMES = ['Daisy', 'Dandelion', 'Rose', 'Sunflower', 'Tulip']

# Sidebar Header & Instructions
st.sidebar.markdown("<h2 style='text-align: center; color: #10B981;'>🌸 FloraVision AI</h2>", unsafe_allow_html=True)
st.sidebar.markdown("""
### About the Project
FloraVision AI uses state-of-the-art transfer learning models to classify flower species with high precision.

**Supported Categories:**
- 🌼 Daisy
- 🌾 Dandelion
- 🌹 Rose
- 🌻 Sunflower
- 🌷 Tulip

### How to use:
1. Upload a flower image (.jpg, .jpeg, or .png).
2. The model will run real-time inference using an optimized **TensorFlow Lite** backend.
3. View the prediction, confidence distributions, and generate a **Grad-CAM heatmap** to see what parts of the flower the model focused on.
""")

st.sidebar.info("Author: Ayush Kr. Gupta\n\nRoll Number: 113")

# Main Header
st.markdown("<h1 class='main-title'>FloraVision Flower Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Optimized Transfer Learning & Model Interpretability Dashboard</p>", unsafe_allow_html=True)

# Check if model exists
if not os.path.exists(TFLITE_MODEL_PATH):
    st.error(f"🚨 TFLite model not found at `{TFLITE_MODEL_PATH}`. Please run `convert_tflite.py` or place your trained `model.tflite` there.")
    st.stop()

# Load TFLite Model
@st.cache_resource
def load_tflite_interpreter(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_interpreter(TFLITE_MODEL_PATH)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Image Uploader
uploaded_file = st.file_uploader("Upload Flower Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🖼️ Input Image")
        st.image(image, use_column_width=True, caption="Uploaded Flower Image")
        
    with col2:
        st.markdown("### ⚡ AI Inference Output")
        
        # Preprocess the image for model (target size 224x224, normalized output uint8/float32 in 0-255)
        # Note: input_details[0]['shape'] is typically [1, 224, 224, 3]
        target_size = (input_details[0]['shape'][1], input_details[0]['shape'][2])
        img_resized = image.resize(target_size)
        img_array = tf.keras.utils.img_to_array(img_resized)
        input_data = np.expand_dims(img_array, axis=0).astype(np.float32)
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        
        # Calculate predicted class and confidence
        pred_idx = np.argmax(predictions)
        pred_label = CLASS_NAMES[pred_idx]
        confidence = predictions[pred_idx]
        
        # Prominent prediction display
        st.metric(label="Predicted Flower Species", value=pred_label, delta=f"{confidence * 100:.2f}% Confidence")
        st.success(f"Successfully classified image as: **{pred_label}**")
        
        # HTML styled progress bars
        st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
        st.markdown("<h4>Species Confidence Scores</h4>", unsafe_allow_html=True)
        
        for name, score in zip(CLASS_NAMES, predictions):
            score_percent = score * 100
            st.markdown(f"""
            <div class='bar-container'>
                <div class='bar-label-row'>
                    <span>{name}</span>
                    <span>{score_percent:.1f}%</span>
                </div>
                <div class='bar-bg'>
                    <div class='bar-fill' style='width: {score_percent}%;'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("</div>", unsafe_allow_html=True)

    # Divider & Grad-CAM Section
    st.markdown("---")
    st.markdown("### 🔍 Model Explainability & Interpretability")
    st.write("Grad-CAM (Gradient-weighted Class Activation Mapping) produces heatmaps highlighting the exact pixels the deep learning architecture analyzed to classify the flower.")
    
    if st.button("Generate Grad-CAM Heatmap"):
        # Display feedback loader
        with st.spinner("Analyzing convolutional feature maps gradients..."):
            if not os.path.exists(KERAS_MODEL_PATH):
                st.warning("⚠️ Keras model (`best_model.keras`) is required to generate the Grad-CAM gradients. Please ensure the file is saved in the `models/` directory.")
            else:
                # Save input image to a temporary file
                temp_img_dir = tempfile.mkdtemp()
                temp_img_path = os.path.join(temp_img_dir, "input_image.jpg")
                image.save(temp_img_path)
                
                # Output path for Grad-CAM
                output_heatmap_path = os.path.join(temp_img_dir, "grad_cam_output.png")
                
                try:
                    # Run Grad-CAM logic
                    generate_gradcam(
                        model_path=KERAS_MODEL_PATH,
                        img_path=temp_img_path,
                        output_path=output_heatmap_path,
                        last_conv_layer_name="top_activation"
                    )
                    
                    # Display the heatmap overlay
                    st.image(
                        output_heatmap_path, 
                        caption=f"Grad-CAM Heatmap overlay for predicted category '{pred_label}'",
                        use_column_width=True
                    )
                    
                except Exception as e:
                    st.error(f"Error running Grad-CAM: {e}")
                finally:
                    # Cleanup temp dir
                    if os.path.exists(temp_img_path):
                        os.remove(temp_img_path)
                    if os.path.exists(output_heatmap_path):
                        os.remove(output_heatmap_path)
                    if os.path.exists(temp_img_dir):
                        os.rmdir(temp_img_dir)
else:
    # Default State (No image uploaded)
    st.info("💡 Please upload a flower image (.jpg, .jpeg, or .png) in the uploader above or sidebar to get started.")
    
    # Showcase custom aesthetics card when empty
    st.markdown("""
    <div style='text-align: center; padding: 4rem 2rem; border: 2px dashed rgba(255,255,255,0.15); border-radius: 15px;'>
        <h3 style='color: #a3a3a3;'>Ready for Classification</h3>
        <p style='color: #737373;'>Predictions, confidence bars, and visual explanations will appear here once an image is uploaded.</p>
    </div>
    """, unsafe_allow_html=True)
