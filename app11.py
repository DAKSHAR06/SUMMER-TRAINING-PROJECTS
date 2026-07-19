import streamlit as st
import pandas as pd
import joblib

model = joblib.load("svrp.pkl")
x_scaler = joblib.load("xscalersvr1.pkl")
y_scaler = joblib.load("yscalersvr2.pkl")
fuel_encoder = joblib.load("fuelencoder.pkl")

st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗"
)

st.title("🚗 Car Price Prediction using SVR")
st.write("Enter the car details to predict the estimated price.")

citympg = st.number_input(
    "City Mileage (MPG)",
    min_value=1.0,
    value=25.0
)

highwaympg = st.number_input(
    "Highway Mileage (MPG)",
    min_value=1.0,
    value=30.0
)

horsepower = st.number_input(
    "Horsepower",
    min_value=1.0,
    value=120.0
)

fueltype = st.selectbox(
    "Fuel Type",
    list(fuel_encoder.classes_)
)

symboling = st.selectbox(
    "Symboling",
    [-2, -1, 0, 1, 2, 3]
)

if st.button("Predict Car Price"):

    fuel_encoded = fuel_encoder.transform([fueltype])[0]

    data = pd.DataFrame(
        [[
            citympg,
            highwaympg,
            horsepower,
            fuel_encoded,
            symboling
        ]],
        columns=[
            "citympg",
            "highwaympg",
            "horsepower",
            "fueltype",
            "symboling"
        ]
    )

    data_scaled = x_scaler.transform(data)

    prediction_scaled = model.predict(data_scaled)

    prediction = y_scaler.inverse_transform(
        prediction_scaled.reshape(-1, 1)
    )[0][0]

    st.success(
        f"💰 Predicted Car Price: ${prediction:,.2f}"
    )