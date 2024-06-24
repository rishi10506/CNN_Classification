
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Load the trained model
try:
    model = load_model('cnn_model.keras')
    st.write("Model loaded successfully.")
except OSError as e:
    st.write(f"Error loading model: {e}")

# Define a function to preprocess the image
def preprocess_image(image):
    size = (32, 32)  # Resize the image to the required input size of the model
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image = np.asarray(image)
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)
    return image

# Streamlit interface
st.title("Image Classification with CNN")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    st.write("")
    st.write("Classifying...")

    image = preprocess_image(image)

    predictions = model.predict(image)
    class_idx = np.argmax(predictions)
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    st.write(f"Predicted class: {class_names[class_idx]}")
