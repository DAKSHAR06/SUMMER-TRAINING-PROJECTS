import streamlit as st
import pickle
import pandas as pd

# Load the trained model
with open("synthetic_logisticregression.pkl", "rb") as file:
    model = pickle.load(file)

# Page settings
st.set_page_config(
    page_title="Logistic Regression Predictor",
    layout="centered"

)

# Title
st.title("Logistic Regression Predictor")
st.write("Enter values for Feature1 and Feature2")

# Input boxes
feature1 = st.number_input("Feature1", value=0.0)
feature2 = st.number_input("Feature2", value=0.0)

# Prediction
if st.button("Predict"):

    input_data = pd.DataFrame(
        [[feature1, feature2]],
        columns=["Feature1", "Feature2"]
    )

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    st.success(f"Predicted Class: {prediction[0]}")

    st.write("Probability")
    st.write(f"Class 0 : {probability[0][0]:.4f}")
    st.write(f"Class 1 : {probability[0][1]:.4f}")