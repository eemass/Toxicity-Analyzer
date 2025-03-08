import streamlit as st
import numpy as np
import pandas as pd
from model import load_or_train_model
from data_loader import get_vectorizer

model = load_or_train_model()
vectorizer = get_vectorizer()

CATEGORIES = ["Toxic", "Severe Toxic", "Obscene", "Threat", "Insult", "Identity Hate"]

st.set_page_config(page_title="Toxicity Analysis", page_icon="🧪", layout="wide")

st.title("🔍 Toxicity Analysis Model")
st.write(
    "This tool analyzes text to detect different forms of toxicity. "
    "Enter a comment below and check the toxicity levels."
)

text_input = st.text_area("💬 **Enter your text below:**", "")

if st.button("Analyze"):
    if text_input.strip():
        input_vector = vectorizer([text_input])
        input_vector = np.array(input_vector)
        input_vector = np.reshape(input_vector, (1, 1800))

        raw_predictions = model.predict(input_vector)[0]

        binary_results = ["✅ Yes" if p >= 0.5 else "❌ No" for p in raw_predictions]
        percentage_results = [f"{p * 100:.2f}%" for p in raw_predictions]

        st.subheader("📊 **Toxicity Analysis Results**")

        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            for i in range(3):
                st.markdown(f"**{CATEGORIES[i]}**")
                st.markdown(
                    f"<h3 style='margin:0px;'>{binary_results[i]}</h3>",
                    unsafe_allow_html=True,
                )

        with col2:
            for i in range(3, 6):
                st.markdown(f"**{CATEGORIES[i]}**")
                st.markdown(
                    f"<h3 style='margin:0px;'>{binary_results[i]}</h3>",
                    unsafe_allow_html=True,
                )

        st.subheader("📈 **Toxicity Score Breakdown**")
        for i, category in enumerate(CATEGORIES):
            st.write(f"**{category}:** {percentage_results[i]}")
            st.progress(float(raw_predictions[i]))

    else:
        st.warning("⚠️ Please enter a comment before analyzing.")
