import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Guava & Banana Freshness Classifier",
    page_icon="🍌",
    layout="centered"
)


# IMPORTANT:
# Change these names ONLY if your training class order was different.
CLASS_NAMES = [
    "ripe",
    "unripe",
    "spoiled"
]


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model_url = "https://github.com/abuobaidahmim/guava-banana-resnet50/releases/download/v1.0.0/ResNet50_Guava_Banana_Best.h5"

    model_path = tf.keras.utils.get_file(
        "ResNet50_Guava_Banana_Best.h5",
        model_url
    )

    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )

    return model


model = load_model()


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(image):

    image = image.convert("RGB")
    image = image.resize((224, 224))

    image_array = np.array(image, dtype=np.float32)

    # ResNet50 preprocessing
    image_array = tf.keras.applications.resnet50.preprocess_input(
        image_array
    )

    image_array = np.expand_dims(image_array, axis=0)

    return image_array


# ============================================================
# TITLE
# ============================================================

st.title("🍈🍌 Guava & Banana Freshness Classifier")

st.write(
    "Upload a guava or banana image to predict its freshness condition."
)


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("🔍 Predict"):

        with st.spinner("Analyzing image..."):

            processed_image = preprocess_image(image)

            predictions = model.predict(
                processed_image,
                verbose=0
            )

            predicted_index = int(
                np.argmax(predictions[0])
            )

            predicted_class = CLASS_NAMES[predicted_index]

            confidence = float(
                predictions[0][predicted_index]
            ) * 100

        st.success(
            f"Prediction: {predicted_class.upper()}"
        )

        st.info(
            f"Confidence: {confidence:.2f}%"
        )

        # Show probability for each class
        st.subheader("Class Probabilities")

        for i, class_name in enumerate(CLASS_NAMES):

            probability = float(
                predictions[0][i]
            ) * 100

            st.write(
                f"{class_name.capitalize()}: "
                f"{probability:.2f}%"
            )

            st.progress(
                min(int(probability), 100)
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "ResNet50-based Guava & Banana Freshness Classification Model"
)
