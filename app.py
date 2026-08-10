
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

st.set_page_config(
    page_title="Fruit Freshness Classifier",
    page_icon="🍈"
)

st.title("🍈 Fruit Freshness Classification")
st.write("ResNet50-based freshness classification")

MODEL_PATH = "ResNet50_Guava_Banana_Best.h5"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

model = load_model()

class_names = ["Ripe", "Spoiled", "Unripe"]

uploaded_file = st.file_uploader(
    "Upload a fruit image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Resize
    image_resized = image.resize((224, 224))

    # Convert to array
    img_array = np.array(image_resized)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # ResNet50 preprocessing
    img_array = preprocess_input(img_array)

    # Prediction
    prediction = model.predict(img_array, verbose=0)[0]

    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = prediction[predicted_index] * 100

    st.success(f"Prediction: {predicted_class}")
    st.metric(
        "Confidence",
        f"{confidence:.2f}%"
    )

    st.subheader("Class Probabilities")

    for class_name, probability in zip(class_names, prediction):
        st.write(
            f"**{class_name}:** {probability * 100:.2f}%"
        )
