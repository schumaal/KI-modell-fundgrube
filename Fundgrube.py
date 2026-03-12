import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from supabase import create_client
import uuid

IMG_SIZE = 224

# =========================================================
# =================== SUPABASE VERBINDUNG =================
# =========================================================

SUPABASE_URL = "https://ajybcyjgvvmcnwujsvnz.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFqeWJjeWpndnZtY253dWpzdm56Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzIxMDQ3NjksImV4cCI6MjA4NzY4MDc2OX0.oKNBirTY-E1aO1L-xmsXiG_IpHwuZjyv1pg3rFwuhjE"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# =========================================================
# =================== KI MODELL ===========================
# =========================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("keras_model.h5")

model = load_model()

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

    prediction = model.predict(img_array, verbose=0)

    index = np.argmax(prediction)
    confidence = float(np.max(prediction))

    return class_names[index], confidence

# =========================================================
# =================== SUPABASE FUNKTIONEN =================
# =========================================================

def upload_image(file):

    file_bytes = file.getvalue()
    filename = f"{uuid.uuid4()}.png"

    supabase.storage.from_("lost-items").upload(filename, file_bytes)

    image_url = supabase.storage.from_("lost-items").get_public_url(filename)

    return image_url


def save_item(name, category, color, description, labels, image_url):

    data = {
        "name": name,
        "category": category,
        "color": color,
        "description": description,
        "labels": labels,
        "image_url": image_url
    }

    supabase.table("items").insert(data).execute()


def load_items():

    response = supabase.table("items").select("*").order("created_at", desc=True).execute()

    return pd.DataFrame(response.data)

# =========================================================
# =================== STREAMLIT UI ========================
# =========================================================

st.title("🔎 Digitales Fundbüro")

# =========================================================
# =================== SUCHLEISTE ==========================
# =========================================================

search = st.text_input("Suche nach Gegenstand oder Label")

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
# =================== GEGENSTAND HOCHLADEN ================
# =========================================================

st.subheader("📷 Gegenstand hochladen")

name = st.text_input("Name des Gegenstands")

description = st.text_area("Beschreibung")

labels = st.text_input("Labels (z.B: schlüssel, metall, klein)")

category = st.selectbox(
    "Kategorie",
    ["T-Shirt", "Flasche", "Buch", "Schuh", "Sonstiges"]
)

color = st.selectbox(
    "Farbe",
    ["schwarz", "weiß", "rot", "blau", "grün"]
)

uploaded_file = st.file_uploader("Bild auswählen", type=["jpg","jpeg","png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Hochgeladenes Bild", use_container_width=True)

    prediction, confidence = predict_image(image)

    st.info(f"KI erkennt: {prediction} ({confidence*100:.1f} %)")

    if st.button("Gegenstand speichern"):

        image_url = upload_image(uploaded_file)

        save_item(
            name,
            category,
            color,
            description,
            labels,
            image_url
        )

        st.success("Gegenstand erfolgreich gespeichert!")

st.divider()

# =========================================================
# =================== ARCHIV ==============================
# =========================================================

st.subheader("📦 Archiv der gefundenen Gegenstände")

data = load_items()

filtered_data = data.copy()

if search:

    filtered_data = filtered_data[
        filtered_data["name"].str.contains(search, case=False) |
        filtered_data["labels"].str.contains(search, case=False) |
        filtered_data["description"].str.contains(search, case=False)
    ]

if category_filter != "Alle":

    filtered_data = filtered_data[
        filtered_data["category"] == category_filter
    ]

if color_filter != "Alle":

    filtered_data = filtered_data[
        filtered_data["color"] == color_filter
    ]

# =========================================================
# =================== ERGEBNISSE ANZEIGEN =================
# =========================================================

for index, row in filtered_data.iterrows():

    st.image(row["image_url"], width=250)

    st.write("**Name:**", row["name"])
    st.write("Kategorie:", row["category"])
    st.write("Farbe:", row["color"])
    st.write("Beschreibung:", row["description"])
    st.write("Labels:", row["labels"])
    st.write("Hochgeladen:", row["created_at"])

    st.divider()
