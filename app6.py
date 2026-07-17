import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("randomforestregressor.pkl")

# Page Config
st.set_page_config(
    page_title="Random Forest House Price Predictor",
    page_icon="🏡",
    layout="centered"
)

st.title("🏡 Random Forest House Price Prediction")

st.write("Enter the details below:")

avg_income = st.number_input("Average Income", value=50000.0)
avg_area_house_age = st.number_input("Average Area House Age", value=5.0)
avg_area_num_rooms = st.number_input("Average Area Number of Rooms", value=7.0)
avg_bedrooms = st.number_input("Average Bedrooms", value=4.0)
avg_population = st.number_input("Average Population", value=30000.0)

if st.button("Predict House Price"):

    data = pd.DataFrame({
        "avg_income": [avg_income],
        "avg_area_house_age": [avg_area_house_age],
        "avg_area_num_rooms": [avg_area_num_rooms],
        "avg_bedrooms": [avg_bedrooms],
        "avg_population": [avg_population]
    })

    prediction = model.predict(data)

    st.success(f"🏠 Predicted House Price: ${prediction[0]:,.2f}")