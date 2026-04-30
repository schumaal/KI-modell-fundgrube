import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import torch

from ultralytics import YOLO


# =========================================================
# MODEL
# =========================================================

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")   # kleines, schnelles Modell

model = load_model()


# =========================================================
# YOLO PREDICTION
# =========================================================

def predict_image(image):
    # PIL → numpy
    img = np.array(image)

    # YOLO inference
    results = model(img)[0]

    detections = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        label = model.names[cls_id]

        detections.append({
            "label": label,
            "confidence": conf
        })

    return detections


# =========================================================
# DEMO DATENBANK
# =========================================================

data = pd.DataFrame([
    {"Name": "Schwarzes T-Shirt", "Kategorie": "T-Shirt", "Farbe": "schwarz"},
    {"Name": "Rote Flasche", "Kategorie": "Flasche", "Farbe": "rot"},
    {"Name": "Weißes Buch", "Kategorie": "Buch", "Farbe": "weiß"},
    {"Name": "Schwarzer Schuh", "Kategorie": "Schuh", "Farbe": "schwarz"},
    {"Name": "Grüne Jacke", "Kategorie": "Sonstiges", "Farbe": "grün"},
])


# =========================================================
# UI
# =========================================================

st.title("🔎 Digitales Fundbüro (YOLOv8)")

uploaded_file = st.file_uploader(
    "Bild hochladen",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Upload", use_container_width=True)

    detections = predict_image(image)

    st.subheader("Erkannte Objekte:")

    if len(detections) == 0:
        st.warning("Keine Objekte erkannt")
    else:
        for det in detections:
            st.write(f"👉 {det['label']} ({det['confidence']*100:.1f}%)")


# =========================================================
# SIMPLE FILTER (DEIN FUNDGRUBEN-TEIL BLEIBT)
# =========================================================

st.divider()

st.subheader("📦 Datenbank")

st.dataframe(data, use_container_width=True)
