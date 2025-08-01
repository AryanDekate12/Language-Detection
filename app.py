# app.py

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

# ✅ Language code to name and emoji mapping
language_names = {
    "en": "English",     "fr": "French",      "de": "German",
    "es": "Spanish",     "it": "Italian",     "pt": "Portuguese",
    "nl": "Dutch",       "ru": "Russian",     "ar": "Arabic",
    "hi": "Hindi",       "ur": "Urdu",        "sw": "Swahili",
    "tr": "Turkish",     "ja": "Japanese",    "zh-cn": "Chinese",
    "ko": "Korean",      "pl": "Polish",      "vi": "Vietnamese",
    "ro": "Romanian",    "th": "Thai",        "fa": "Persian",
    "sv": "Swedish"
}

language_flags = {
    "en": "🇬🇧", "fr": "🇫🇷", "de": "🇩🇪", "es": "🇪🇸",
    "it": "🇮🇹", "pt": "🇵🇹", "nl": "🇳🇱", "ru": "🇷🇺",
    "ar": "🇸🇦", "hi": "🇮🇳", "ur": "🇵🇰", "sw": "🇰🇪",
    "tr": "🇹🇷", "ja": "🇯🇵", "zh-cn": "🇨🇳", "ko": "🇰🇷",
    "pl": "🇵🇱", "vi": "🇻🇳", "ro": "🇷🇴", "th": "🇹🇭",
    "fa": "🇮🇷", "sv": "🇸🇪"
}

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
        st.error("❌ Required files not found. Please ensure '.pkl.gz' files are in the same folder as app.py.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        st.stop()

vectorizer, model = load_assets()

# ✅ UI Header
st.title("🗣️ Language Recognition App")
st.markdown("""
Enter text in any language, and this app will try to predict which language it is written in.
""")

# ✅ Text Input
user_input = st.text_area("Enter text here:", height=150)

# ✅ Detect Button
if st.button("Detect Language"):
    if user_input.strip():
        try:
            input_vector = vectorizer.transform([user_input])
            prediction = model.predict(input_vector)
            code = prediction[0]
            name = language_names.get(code, "Unknown")
            flag = language_flags.get(code, "")
            st.success(f"{flag} **Detected Language:** {name} (`{code}`)")
            st.balloons()
        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")
    else:
        st.warning("⚠️ Please enter some text to detect the language.")

# ✅ Footer
st.markdown("---")
st.markdown("👨‍💻 Developed by Aryan Dekate using `scikit-learn` and `Streamlit`.")
st.markdown("📚 Model trained on a dataset containing 22 different languages.")
