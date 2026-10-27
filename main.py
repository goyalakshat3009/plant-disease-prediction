
import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st


# Limit TensorFlow CPU thread usage
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Paths
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(
    working_dir,
    "trained_model",
    "plant_disease_prediction_model.h5"
)
class_indices_path = os.path.join(
    working_dir,
    "class_indices.json"
)


# Load model only once and reuse it across Streamlit reruns
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        model_path,
        compile=False
    )


@st.cache_data
def load_class_indices():
    with open(class_indices_path, "r") as file:
        return json.load(file)


# Load model and class names
try:
    model = load_model()
    class_indices = load_class_indices()
except Exception as e:
    st.error(f"Unable to load the model or class names: {e}")
    st.stop()


# Load and preprocess the uploaded image
def load_and_preprocess_image(image, target_size=(224, 224)):
    with Image.open(image) as img:
        img = img.convert("RGB")
        img = img.resize(target_size)

        img_array = np.asarray(img, dtype=np.float32)

    # Normalize pixel values to [0, 1]
    img_array /= 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# Predict the disease class
def predict_image_class(model, image, class_indices):
    preprocessed_img = load_and_preprocess_image(image)

    predictions = model.predict(preprocessed_img, verbose=0)

    predicted_class_index = int(np.argmax(predictions, axis=1)[0])
    predicted_class_name = class_indices[str(predicted_class_index)]

    confidence = float(np.max(predictions)) * 100

    return predicted_class_name, confidence


# Streamlit application
st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Plant Disease Classifier")
st.write("Upload a plant leaf image to predict its disease class.")

uploaded_image = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_image is not None:
    try:
        image = Image.open(uploaded_image).convert("RGB")

        st.image(
            image,
            caption="Uploaded plant leaf",
            width=250
        )

        if st.button("Classify", type="primary"):
            with st.spinner("Analyzing the plant leaf..."):
                prediction, confidence = predict_image_class(
                    model,
                    image,
                    class_indices
                )

            st.success(f"Prediction: {prediction}")
            st.write(f"Model confidence: {confidence:.2f}%")

    except Exception as e:
        st.error(f"Unable to process this image: {e}")
