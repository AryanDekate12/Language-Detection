# app.py

import streamlit as st
import pickle
import gzip
from pathlib import Path

# ✅ Page Configuration
st.set_page_config(
    page_title="Language Code Predictor",
    page_icon="🔤",
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
        st.error("❌ Required .pkl.gz files not found. Please ensure they are in the same folder as app.py.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model/vectorizer: {e}")
        st.stop()

vectorizer, model = load_assets()

# ✅ UI Header
st.title("🔤 Language Code Predictor")
st.markdown("""
This app uses a machine learning model to predict the **language code** (e.g., `en`, `hi`, `fr`) based on your text input.
""")

# ✅ User input
user_input = st.text_area("Enter text to detect language code:", height=150)

# ✅ Predict Button
if st.button("Predict Language Code"):
    if user_input.strip():
        try:
            input_vector = vectorizer.transform([user_input])
            prediction = model.predict(input_vector)
            code = prediction[0]
            st.success(f"**Predicted Language Code:** `{code}`")
            st.balloons()
        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")
    else:
        st.warning("⚠️ Please enter text before predicting.")

# ✅ Footer
st.markdown("---")
st.markdown("👨‍💻 Built by Aryan Dekate using `scikit-learn` and `Streamlit`.")
st.markdown("📚 Model predicts raw ISO language codes (like `en`, `fr`, `hi`, etc.).")
