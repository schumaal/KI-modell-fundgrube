import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
pip install transformers torch pillow


IMG_SIZE = 224

st.cache_resource
def load_model():
    return tf.keras.models.load_model("keras_model.h5")

model = load_model()

from transformers import AutoImageProcessor, AutoModelForObjectDetection
import torch

@st.cache_resource
def load_model():
    processor = AutoImageProcessor.from_pretrained("hustvl/yolos-tiny")
    model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-tiny")
    return processor, model

processor, model = load_model()

def load_labels():
    with open("labels.txt", "r") as f:
        return [line.strip() for line in f.readlines()]

class_names = load_labels()

# =========================================================
# =================== KI FUNKTION =========================
# =========================================================

def predict_image(image):
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    index = np.argmax(prediction)
    confidence = float(np.max(prediction))

    return class_names[index], confidence


# =========================================================
# =================== DEMO DATENBANK ======================
# =========================================================

data = pd.DataFrame([
    {"Name": "Schwarzes T-Shirt", "Kategorie": "T-Shirt", "Farbe": "schwarz"},
    {"Name": "Rote Flasche", "Kategorie": "Flasche", "Farbe": "rot"},
    {"Name": "Weißes Buch", "Kategorie": "Buch", "Farbe": "weiß"},
    {"Name": "Schwarzer Schuh", "Kategorie": "Schuh", "Farbe": "schwarz"},
    {"Name": "Grüne Jacke", "Kategorie": "Sonstiges", "Farbe": "grün"},
])


# =========================================================
# =================== STREAMLIT UI ========================
# =========================================================

st.title("🔎 Digitales Fundbüro")

# Suchleiste
search = st.text_input("Suche nach Gegenstand")

# Filter
category_filter = st.selectbox(
    "Kategorie wählen",
    ["Alle", "T-Shirt", "Flasche", "Buch", "Schuh", "Sonstiges"]
)

color_filter = st.selectbox(
    "Farbe wählen",
    ["Alle", "schwarz", "weiß", "rot", "blau", "grün"]
)

st.divider()

# =========================================================
# =================== BILD UPLOAD =========================
# =========================================================

st.subheader("📷 Bild hochladen zur KI-Erkennung")

uploaded_file = st.file_uploader("Bild auswählen", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Hochgeladenes Bild", use_container_width=True)

    prediction, confidence = predict_image(image)

    st.success(f"Erkannt: {prediction}")
    st.write(f"Sicherheit: {confidence*100:.2f} %")


# =========================================================
# =================== FILTER LOGIK ========================
# =========================================================

filtered_data = data.copy()

if search:
    filtered_data = filtered_data[
        filtered_data["Name"].str.contains(search, case=False)
    ]

if category_filter != "Alle":
    filtered_data = filtered_data[
        filtered_data["Kategorie"] == category_filter
    ]

if color_filter != "Alle":
    filtered_data = filtered_data[
        filtered_data["Farbe"] == color_filter
    ]

st.divider()

st.subheader("📦 Gefundene Gegenstände")
st.dataframe(filtered_data, use_container_width=True)
