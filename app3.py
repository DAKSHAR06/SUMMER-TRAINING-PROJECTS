import streamlit as st
import pickle
import pandas as pd

with open("housemodellinearreg.pkl", "rb") as file:
    model = pickle.load(file)
    
# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="House Price Prediction Model",
    layout="centered"
)

# -----------------------------
# Title
# -----------------------------
st.title("🌳 House Price Prediction Model")
st.write("Enter your average income ")

# -----------------------------
# User Input
# -----------------------------
income = st.number_input(
    "Average Income",
    min_value=0.0,
    step=1000.0,
    value=500000.0,
    
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict House Price"):

    data = pd.DataFrame(
        [[income]],
        columns=["avg_income"]   # Must match the training dataset
    )

    prediction = model.predict(data)

    st.success(f"Predicted House Price: ₹ {prediction[0]:,.2f}")