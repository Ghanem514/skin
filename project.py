import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

# Load trained model
model = load_model("my_cnn_model2.h5")

# Updated class names
class_names = [
    "Eczema",
    "Melanoma",
    "Atopic Dermatitis",
    "Basal Cell Carcinoma",
    "Melanocytic Nevi"
]

# Confidence threshold for classifying as "Unknown"
CONFIDENCE_THRESHOLD = 0.50  # You can adjust this threshold based on your model's performance

# Streamlit app
st.set_page_config(page_title="Skin Disease Classifier", layout="centered")
st.title("🩺 AI Skin Disease Classifier")
st.write("Upload a skin image and get a prediction from 5 disease categories.")

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((64, 64))  # Resize to your model's input size
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)

    # Debugging the prediction output
    st.write("Raw prediction values:", prediction[0])
    st.write("Prediction shape:", prediction.shape)

    # Check if the prediction output matches the number of classes
    if prediction.shape[1] == len(class_names):
        pred_index = np.argmax(prediction)
        confidence = float(prediction[0][pred_index]) * 100

        # If confidence is below the threshold, classify as "Unknown"
        if confidence < CONFIDENCE_THRESHOLD * 100:
            pred_class = "Unknown"
            confidence = 0.0
        else:
            pred_class = class_names[pred_index]

        # Show result
        st.subheader("Prediction:")
        st.success(f"{pred_class} ({confidence:.2f}%)")

        # Show all probabilities
        st.subheader("All class probabilities:")
        for i, prob in enumerate(prediction[0]):
            st.write(f"{class_names[i]}: {prob*100:.2f}%")
    else:
        st.error("Prediction output shape doesn't match the number of class names.")
