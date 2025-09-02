import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# CIFAR-10 classes
class_names = ['Airplane','Car','Bird','Cat','Deer',
               'Dog','Frog','Horse','Ship','Truck']

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cifar10_model.h5")

model = load_model()

st.title("🖼️ CIFAR-10 Image Classifier")
st.write("Upload an image and the model will predict its class.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB").resize((32, 32))
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    preds = model.predict(img_array)[0]
    top_idx = np.argmax(preds)
    st.subheader(f"🎯 Prediction: {class_names[top_idx]}")
    st.write({class_names[i]: float(preds[i]) for i in range(10)})
