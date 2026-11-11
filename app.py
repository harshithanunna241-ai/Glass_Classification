import streamlit as st
import pandas as pd
import joblib


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Glass Classification",
    page_icon="🔬",
    layout="centered"
)

st.title("🔬 Glass Classification")
st.write("Predict the type of glass using AdaBoost or Gradient Boosting.")


# -----------------------------
# Load models
# -----------------------------
@st.cache_resource
def load_models():
    adaboost_model = joblib.load("adaboost_model.pkl")
    gradient_model = joblib.load("gradient_boosting_model.pkl")

    return adaboost_model, gradient_model


adaboost_model, gradient_model = load_models()


# -----------------------------
# Model selection
# -----------------------------
st.subheader("Select Machine Learning Model")

model_name = st.selectbox(
    "Choose Model",
    ["AdaBoost", "Gradient Boosting"]
)


# -----------------------------
# Input fields
# -----------------------------
st.subheader("Enter Glass Properties")

RI = st.number_input(
    "Refractive Index (RI)",
    value=1.52,
    format="%.5f"
)

Na = st.number_input(
    "Sodium (Na)",
    value=13.0,
    format="%.5f"
)

Mg = st.number_input(
    "Magnesium (Mg)",
    value=3.0,
    format="%.5f"
)

Al = st.number_input(
    "Aluminum (Al)",
    value=1.0,
    format="%.5f"
)

Si = st.number_input(
    "Silicon (Si)",
    value=72.0,
    format="%.5f"
)

K = st.number_input(
    "Potassium (K)",
    value=0.5,
    format="%.5f"
)

Ca = st.number_input(
    "Calcium (Ca)",
    value=8.0,
    format="%.5f"
)

Ba = st.number_input(
    "Barium (Ba)",
    value=0.0,
    format="%.5f"
)

Fe = st.number_input(
    "Iron (Fe)",
    value=0.1,
    format="%.5f"
)


# -----------------------------
# Create input dataframe
# -----------------------------
input_data = pd.DataFrame({
    "RI": [RI],
    "Na": [Na],
    "Mg": [Mg],
    "Al": [Al],
    "Si": [Si],
    "K": [K],
    "Ca": [Ca],
    "Ba": [Ba],
    "Fe": [Fe]
})


# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Glass Type"):

    if model_name == "AdaBoost":
        prediction = adaboost_model.predict(input_data)
    else:
        prediction = gradient_model.predict(input_data)

    predicted_type = prediction[0]

    st.success(
        f"Predicted Glass Type: {predicted_type}"
    )

    st.write("### Input Values")
    st.dataframe(input_data)
