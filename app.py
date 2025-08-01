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

# --- 1. Load your pre-trained model and vectorizer ---
@st.cache_resource
def load_assets():
    try:
        # Load the CountVectorizer
        with open('count_vectorizer.pkl', 'rb') as cv_file:
            loaded_cv = pickle.load(cv_file)

        # Load the trained MultinomialNB model
        with open('language_detection_model.pkl', 'rb') as model_file:
            loaded_model = pickle.load(model_file)

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
  #Network URL: http://172.20.10.13:8501