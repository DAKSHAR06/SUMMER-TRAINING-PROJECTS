import streamlit as st
import pickle
import pandas as pd

with open("decisiontreeregressor.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Salary Predictor")


experience = st.number_input(
    "Years of Experience",
    min_value=0.0,
    max_value=50.0,
    value=5.0
)

if st.button("Predict Salary"):
    data = pd.DataFrame(
        [[experience]],
        columns=["YearsExperience"]
    )

    prediction = model.predict(data)

    st.success(f"Predicted Salary: ₹{prediction[0]:,.2f}")
    
