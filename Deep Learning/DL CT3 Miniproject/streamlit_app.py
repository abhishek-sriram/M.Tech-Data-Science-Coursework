# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 21:11:37 2024

@author: abhis
"""

import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the saved model
model = tf.keras.models.load_model("beans_model.keras")

# Preprocess the uploaded image to the size MobileNet expects
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0  # Scale the image
    return np.expand_dims(image, axis=0)

# Streamlit app interface
st.title("Bean Disease Detection")
st.write("Upload an image of a bean leaf to detect if it's diseased.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Error handling for unsupported file types
if uploaded_file is not None:
    try:
        # Check the file extension
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension not in ['jpg', 'jpeg', 'png']:
            st.error("Unsupported file type. Please upload an image in JPG, JPEG, or PNG format.")
        else:
            # Load and preprocess the image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.write("Classifying...")

            processed_image = preprocess_image(image)
            predictions = model.predict(processed_image)
            class_names = ['angular_leaf_spot', 'bean_rust', 'healthy']  # Update with actual class names

            # Display the prediction
            score = tf.nn.softmax(predictions[0])
            st.write(f"Prediction: {class_names[np.argmax(score)]}")
            st.write(f"Confidence: {100 * np.max(score):.2f}%")

    except Exception as e:
        st.error(f"An error occurred while processing the image: {str(e)}")

else:
    st.write("Please upload an image of a bean leaf.")
