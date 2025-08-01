# app.py

import streamlit as st
import pickle
import gzip
import pandas as pd
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

# ✅ Language mappings
language_names = {
    "en": "English", "fr": "French", "de": "German", "es": "Spanish",
    "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "ru": "Russian",
    "ar": "Arabic", "hi": "Hindi", "ur": "Urdu", "sw": "Swahili",
    "tr": "Turkish", "ja": "Japanese", "zh-cn": "Chinese", "ko": "Korean",
    "pl": "Polish", "vi": "Vietnamese", "ro": "Romanian", "th": "Thai",
    "fa": "Persian", "sv": "Swedish"
}

language_flags = {
    "en": "🇬🇧", "fr": "🇫🇷", "de": "🇩🇪", "es": "🇪🇸", "it": "🇮🇹", "pt": "🇵🇹",
    "nl": "🇳🇱", "ru": "🇷🇺", "ar": "🇸🇦", "hi": "🇮🇳", "ur": "🇵🇰", "sw": "🇰🇪",
    "tr": "🇹🇷", "ja": "🇯🇵", "zh-cn": "🇨🇳", "ko": "🇰🇷", "pl": "🇵🇱",
    "vi": "🇻🇳", "ro": "🇷🇴", "th": "🇹🇭", "fa": "🇮🇷", "sv": "🇸🇪"
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
        st.error("❌ Required model/vectorizer files not found.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        st.stop()

vectorizer, model = load_assets()

# ✅ Preprocessing
def preprocess_text(text):
    return text.strip().lower()

# ✅ UI Header
st.title("🗣️ Language Recognition App")
st.markdown("Enter text in any language, and this app will try to predict which language it is written in.")

# ✅ Text Input
user_input = st.text_area("Enter text here:", height=150)

# ✅ Predict Button
if st.button("Detect Language"):
    if user_input.strip():
        try:
            processed = preprocess_text(user_input)
            input_vector = vectorizer.transform([processed])
            probs = model.predict_proba(input_vector)[0]
            top_indices = probs.argsort()[-3:][::-1]
            top_confidence = probs[top_indices[0]] * 100

            # 🔐 Set a minimum confidence threshold (e.g., 40%)
            threshold = 40.0

            if top_confidence < threshold:
                st.warning("🤖 The model is unsure about the input. It might not match any trained language.")
            else:
                st.subheader("🔍 Top 3 Language Predictions")

                for i, idx in enumerate(top_indices):
                    lang_code = model.classes_[idx]
                    confidence = probs[idx] * 100
                    name = language_names.get(lang_code, "Unknown")
                    flag = language_flags.get(lang_code, "")
                    st.markdown(f"{i+1}. {flag} **{name}** (`{lang_code}`) — {confidence:.2f}%")

                # Optional bar chart
                top_langs = [model.classes_[i] for i in top_indices]
                lang_labels = [f"{language_flags.get(code, '')} {language_names.get(code, code)}" for code in top_langs]
                df = pd.DataFrame({
                    "Language": lang_labels,
                    "Confidence (%)": [probs[i] * 100 for i in top_indices]
                })
                st.bar_chart(df.set_index("Language"))

                st.balloons()

        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")
    else:
        st.warning("⚠️ Please enter some text to detect the language.")

# ✅ Footer
st.markdown("---")
st.markdown("👨‍💻 Developed by Aryan Dekate using `scikit-learn` and `Streamlit`.")
st.markdown("📚 Model trained on a dataset containing 22 different languages.")
