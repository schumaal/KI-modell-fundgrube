import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image

from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
image = Image.open("<IMAGE_PATH>").convert("RGB")

# resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# turn the image into a numpy array
image_array = np.asarray(image)

# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load the image into the array
data[0] = normalized_image_array

# Predicts the model
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

# Print prediction and confidence score
print("Class:", class_name[2:], end="")
print("Confidence Score:", confidence_score)


MODEL_PATH = "model/keras_model.h5"   # <-- HIER Pfad anpassen falls nötig
LABELS_PATH = "model/labels.txt"      # <-- HIER Pfad anpassen

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH, "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return model, labels

model, class_names = load_model()

IMG_SIZE = 224   # Falls dein Modell andere Größe nutzt → HIER ändern

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
