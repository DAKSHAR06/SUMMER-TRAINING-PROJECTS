import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
    page_title="Wine Quality Prediction",
    page_icon="🍷",
    layout="wide"
)

with open("annregg12.pkl", "rb") as file:
    model = pickle.load(file)

st.title("🍷 Wine Quality Prediction")
st.subheader("ANN-Based Wine Quality Prediction")

st.write("Enter the chemical properties of the wine to predict its quality.")

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    fixed_acidity = st.number_input(
        "Fixed Acidity",
        min_value=0.0,
        value=7.4,
        step=0.1
    )

    volatile_acidity = st.number_input(
        "Volatile Acidity",
        min_value=0.0,
        value=0.70,
        step=0.01
    )

    citric_acid = st.number_input(
        "Citric Acid",
        min_value=0.0,
        value=0.00,
        step=0.01
    )

    residual_sugar = st.number_input(
        "Residual Sugar",
        min_value=0.0,
        value=1.9,
        step=0.1
    )

with col2:
    chlorides = st.number_input(
        "Chlorides",
        min_value=0.0,
        value=0.076,
        step=0.001,
        format="%.3f"
    )

    free_sulfur_dioxide = st.number_input(
        "Free Sulfur Dioxide",
        min_value=0.0,
        value=11.0,
        step=1.0
    )

    total_sulfur_dioxide = st.number_input(
        "Total Sulfur Dioxide",
        min_value=0.0,
        value=34.0,
        step=1.0
    )

    density = st.number_input(
        "Density",
        min_value=0.0,
        value=0.9978,
        step=0.0001,
        format="%.4f"
    )

with col3:
    ph = st.number_input(
        "pH",
        min_value=0.0,
        max_value=14.0,
        value=3.51,
        step=0.01
    )

    sulphates = st.number_input(
        "Sulphates",
        min_value=0.0,
        value=0.56,
        step=0.01
    )

    alcohol = st.number_input(
        "Alcohol",
        min_value=0.0,
        value=9.4,
        step=0.1
    )

st.divider()

if st.button("🍷 Predict Wine Quality", use_container_width=True):

    input_data = pd.DataFrame({
        "fixed acidity": [fixed_acidity],
        "volatile acidity": [volatile_acidity],
        "citric acid": [citric_acid],
        "residual sugar": [residual_sugar],
        "chlorides": [chlorides],
        "free sulfur dioxide": [free_sulfur_dioxide],
        "total sulfur dioxide": [total_sulfur_dioxide],
        "density": [density],
        "pH": [ph],
        "sulphates": [sulphates],
        "alcohol": [alcohol]
    })

    prediction = model.predict(input_data, verbose=0)

    predicted_quality = float(prediction[0][0])
    st.success(f"Predicted Wine Quality: {predicted_quality:.2f}")