import streamlit as st
import pandas as pd
import joblib

model = joblib.load("knn_regressor.pkl")
scaler = joblib.load("scalerKNNr.pkl")

st.set_page_config(
    page_title="Salary Prediction using KNN Regressor",
    page_icon="💼",
    layout="centered"
)

st.title("💼 Salary Prediction using KNN Regressor")

experience = st.number_input(
    "Years of Experience",
    min_value=0.0,
    max_value=50.0,
    value=2.0,
    step=0.1
)

if st.button("Predict Salary"):

    data = pd.DataFrame({
        "YearsExperience": [experience]
    })

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)

    st.success(f"💰 Predicted Salary: ₹ {prediction[0]:,.2f}")