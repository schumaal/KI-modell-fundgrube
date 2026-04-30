import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO
from ultralytics import YOLO

# =========================================================
# =================== MODEL (YOLOv8) ======================
# =========================================================

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # kleines, schnelles Modell

model = load_model()

# =========================================================
# =================== KI FUNKTION =========================
# =========================================================

def predict_image(image):
    results = model(image)

    detections = []

    for r in results:
        for box in r.boxes:
            detections.append({
                "label": model.names[int(box.cls)],
                "confidence": float(box.conf)
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

# Mapping YOLO → deine Kategorien
label_mapping = {
    "bottle": "Flasche",
    "book": "Buch",
    "shoe": "Schuh",
    "backpack": "Sonstiges",
    "handbag": "Sonstiges",
    "person": "Sonstiges",
}

# =========================================================
# =================== STREAMLIT UI ========================
# =========================================================

st.title("🔎 Digitales Fundbüro mit YOLOv8")

uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Hochgeladenes Bild", use_container_width=True)

    detections = predict_image(image)

    st.subheader("Erkannte Objekte:")

    if not detections:
        st.warning("Kein Objekt erkannt")
    else:
        for det in detections:
            st.write(f"{det['label']} ({det['confidence']*100:.2f}%)")

        # =====================================================
        # AUTO FUND-BÜRO FILTER
        # =====================================================

        top_label = detections[0]["label"]

        if top_label in label_mapping:
            category = label_mapping[top_label]

            st.success(f"🔎 Automatisch erkannt: {category}")

            filtered_data = data[data["Kategorie"] == category]

            st.subheader("📦 Passende Fundstücke:")
            st.dataframe(filtered_data, use_container_width=True)

# =========================================================
# =================== MANUELLER FILTER ====================
# =========================================================

st.divider()

st.subheader("🔍 Manuelle Suche")

search = st.text_input("Suche nach Gegenstand")

category_filter = st.selectbox(
    "Kategorie wählen",
    ["Alle", "T-Shirt", "Flasche", "Buch", "Schuh", "Sonstiges"]
)

color_filter = st.selectbox(
    "Farbe wählen",
    ["Alle", "schwarz", "weiß", "rot", "blau", "grün"]
)

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

st.subheader("📦 Gefundene Gegenstände")
st.dataframe(filtered_data, use_container_width=True)
