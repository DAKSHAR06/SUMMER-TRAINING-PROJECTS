import streamlit as st
import numpy as np
import joblib

st.set_page_config(
    page_title="ANNR Wine Quality Predictor",
    page_icon="🍷",
    layout="centered"
)

model = joblib.load("annregg.pkl")


st.title("🍷 ANNR Wine Quality Prediction")
st.write("Enter the wine properties below to predict its quality.")

fixed_acidity = st.number_input("Fixed Acidity", value=7.40)
volatile_acidity = st.number_input("Volatile Acidity", value=0.70)
citric_acid = st.number_input("Citric Acid", value=0.00)
residual_sugar = st.number_input("Residual Sugar", value=1.90)
chlorides = st.number_input("Chlorides", value=0.076, format="%.4f")
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", value=11.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", value=34.0)
density = st.number_input("Density", value=0.9978, format="%.4f")
ph = st.number_input("pH", value=3.51)
sulphates = st.number_input("Sulphates", value=0.56)
alcohol = st.number_input("Alcohol", value=9.40)

if st.button("Predict Wine Quality"):

    input_data = np.array([[
        fixed_acidity,
        volatile_acidity,
        citric_acid,
        residual_sugar,
        chlorides,
        free_sulfur_dioxide,
        total_sulfur_dioxide,
        density,
        ph,
        sulphates,
        alcohol
    ]], dtype=np.float32)

    prediction = model.predict(input_data, verbose=0)

    predicted_quality = float(prediction[0][0])

    st.write(
        f"Predicted Quality Score: {predicted_quality:.2f}"
    )

    if predicted_quality >= 6:
        st.success("Good Quality Wine")
    else:
        st.error("Bad Quality Wine")
