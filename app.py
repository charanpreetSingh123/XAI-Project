# app.py
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import cv2
from grad_cam import make_gradcam_heatmap, save_and_display_gradcam, get_img_array

# Set page config
st.set_page_config(layout="wide", page_title="X-Ray Pneumonia Diagnosis with XAI")

# Load Model
@st.cache_resource
def load_app_model():
    return load_model("pneumonia_classifier.keras")

model = load_app_model()
last_conv_layer_name = "block5_conv3" # This is the last conv layer in VGG16

st.title("🩺 Explainable AI for Pneumonia Detection")
st.write("Upload a chest X-ray image and the AI will predict if it shows signs of pneumonia. It will also show a heatmap highlighting the areas that most influenced its decision.")

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpeg", "jpg", "png"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp_img.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original X-Ray Image")
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)

    with col2:
        st.subheader("AI Analysis")
       # Get the image array and make predictions
        img_array = get_img_array("temp_img.jpg", size=(180, 180))

        # Make predictions using the raw image array.
        # The model will handle preprocessing internally.
        prediction = model.predict(img_array)[0][0]
        # Display prediction
        confidence = prediction * 100
        if confidence > 50:
            st.error(f"**Diagnosis: PNEUMONIA** (Confidence: {confidence:.2f}%)")
        else:
            st.success(f"**Diagnosis: NORMAL** (Confidence: {100 - confidence:.2f}%)")

        st.write("Generating explainability heatmap...")

        # Generate Grad-CAM
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
        
        # Display Grad-CAM
        gradcam_image = save_and_display_gradcam("temp_img.jpg", heatmap, alpha=0.6)
        gradcam_image_rgb = cv2.cvtColor(gradcam_image, cv2.COLOR_BGR2RGB) # Convert for display
        st.image(gradcam_image_rgb, caption="Grad-CAM Heatmap: Red areas show where the AI 'looked' to make its decision.", use_column_width=True)