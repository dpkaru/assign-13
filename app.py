import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import tensorflowjs as tfjs
import os

st.title("Anomaly Detection using Teachable Machine Model")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)

    img = img.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    if "model" not in st.session_state:
        if not os.path.exists("converted_model"):
            st.error("Model not converted yet. Please convert TF.js to keras format using tfjs_converter.")
        else:
            st.session_state.model = tf.keras.models.load_model("converted_model")

    if "model" in st.session_state:
        predictions = st.session_state.model.predict(img_array)
        class_names = ["Normal", "Anomalous"]
        st.write("### Prediction:")
        st.write(f" {class_names[np.argmax(predictions)]} ({100 * np.max(predictions):.2f}%)")
