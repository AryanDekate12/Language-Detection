# app.py

import streamlit as st

# ✅ Set page config first before any other Streamlit commands
st.set_page_config(
    page_title="Language Recognition App",
    page_icon="🗣️",
    layout="centered"
)

import pandas as pd
import numpy as np
import pickle  # To load your saved model and vectorizer
import gzip
from pathlib import Path

# Determine directory where app.py is located
BASE_DIR = Path(__file__).resolve().parent

vectorizer_path = BASE_DIR / "count_vectorizer.pkl.gz"
model_path      = BASE_DIR / "language_detection_model.pkl.gz"
# --- 1. Load your pre-trained model and vectorizer ---
@st.cache_resource
def load_assets():
    try:
        with gzip.open(BASE_DIR / "count_vectorizer.pkl.gz", "rb") as f:
        vectorizer = pickle.load(f)
        with gzip.open(BASE_DIR / "language_detection_model.pkl.gz", "rb") as f:
        model = pickle.load(f)
        return loaded_cv, loaded_model
    except FileNotFoundError:
        st.error("Error: Model or vectorizer files not found. "
                 "Please ensure 'count_vectorizer.pkl' and 'language_detection_model.pkl' "
                 "are in the same directory as this 'app.py' file. "
                 "You need to run the saving code in your Jupyter notebook first.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading assets: {e}")
        st.stop()

cv, model = load_assets()

# --- 2. Streamlit UI ---
st.title("🗣️ Language Recognition App")
st.markdown("""
Welcome to the Language Recognition App! Enter any text, and I will try to identify its language.
""")

# Text input from the user
user_input = st.text_area(
    "Enter text here:",
    height=150
)

# Button to trigger detection
if st.button("Detect Language"):
    if user_input.strip():  # Check if the input is not empty or just whitespace
        # Transform the user input using the loaded CountVectorizer
        data = cv.transform([user_input]).toarray()

        # Make a prediction using the loaded model
        output = model.predict(data)

        # Display the detected language
        st.success(f"**Detected Language:** `{output[0]}`")
        st.balloons()  # Just for fun!
    else:
        st.warning("Please enter some text to detect its language.")

st.markdown("---")
st.markdown("Developed by Aryan Dekate using `scikit-learn` and `Streamlit`.")
st.markdown("This model was trained on a dataset containing 22 different languages.")
#You can now view your Streamlit app in your browser.
 # Local URL: http://localhost:8501
  #Network URL: http://172.20.10.13:8501# app.py

import streamlit as st
import pickle
import gzip
from pathlib import Path

# ✅ Page Configuration
st.set_page_config(
    page_title="Language Recognition App",
    page_icon="🗣️",
    layout="centered"
)

# ✅ File paths
BASE_DIR = Path(__file__).resolve().parent
vectorizer_path = BASE_DIR / "count_vectorizer.pkl.gz"
model_path = BASE_DIR / "language_detection_model.pkl.gz"

# ✅ Load model and vectorizer
@st.cache_resource
def load_assets():
    try:
        with gzip.open(vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)
        with gzip.open(model_path, "rb") as f:
            model = pickle.load(f)
        return vectorizer, model
    except FileNotFoundError:
        st.error("❌ Required files not found. Please ensure 'count_vectorizer.pkl.gz' and 'language_detection_model.pkl.gz' are in the same directory as app.py.")
        st.stop()
    except Exception as e:
        st.error(f"❌ An unexpected error occurred while loading model files: {e}")
        st.stop()

vectorizer, model = load_assets()

# ✅ Streamlit UI
st.title("🗣️ Language Recognition App")
st.markdown("""
Welcome to the **Language Recognition App**!  
Enter any text below and I'll try to detect its language using a machine learning model trained on 22 languages.
""")

user_input = st.text_area("Enter text here:", height=150)

# ✅ Prediction logic
if st.button("Detect Language"):
    if user_input.strip():
        try:
            input_data = vectorizer.transform([user_input])
            prediction = model.predict(input_data)
            st.success(f"**Detected Language:** `{prediction[0]}`")
            st.balloons()
        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")
    else:
        st.warning("⚠️ Please enter some text to detect the language.")

# ✅ Footer
st.markdown("---")
st.markdown("👨‍💻 Developed by Aryan Dekate using `scikit-learn` and `Streamlit`.")
st.markdown("📚 Model trained on a dataset containing 22 different languages.")
