import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection
import torch

# =========================================================
# =================== MODEL ===============================
# =========================================================

@st.cache_resource
def load_model():
    processor = AutoImageProcessor.from_pretrained("hustvl/yolos-tiny")
    model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-tiny")
    return processor, model

processor, model = load_model()

# =========================================================
# =================== KI FUNKTION =========================
# =========================================================

def predict_image(image):
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_object_detection(
        outputs,
        target_sizes=torch.tensor([image.size[::-1]])
    )[0]

    detections = []

    for score, label, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        detections.append({
            "label": model.config.id2label[label.item()],
            "confidence": float(score),
        })

    return detections

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
# =================== UI ========================
# =========================================================

st.title("🔎 Digitales Fundbüro")

uploaded_file = st.file_uploader("Bild auswählen", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Hochgeladenes Bild", use_container_width=True)

    detections = predict_image(image)

    st.subheader("Erkannte Objekte:")

    for det in detections:
        st.write(f"{det['label']} ({det['confidence']*100:.2f}%)")
