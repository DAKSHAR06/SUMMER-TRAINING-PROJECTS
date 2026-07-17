import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("decisiontreeclassifier.pkl")

# Page configuration
st.set_page_config(
    page_title="Decision Tree Classifier for Iris Species Prediction",
    layout="centered"
)

st.title("🌳 Decision Tree Classifier")
st.write("### Iris Flower Species Prediction")

# User Inputs
sepal_length = st.number_input(
    "Sepal Length (cm)",
    min_value=4.0,
    max_value=8.0,
    value=5.1
)

sepal_width = st.number_input(
    "Sepal Width (cm)",
    min_value=2.0,
    max_value=5.0,
    value=3.5
)

petal_length = st.number_input(
    "Petal Length (cm)",
    min_value=1.0,
    max_value=7.0,
    value=1.4
)

petal_width = st.number_input(
    "Petal Width (cm)",
    min_value=0.1,
    max_value=3.0,
    value=0.2
)

# Predict Button
if st.button("Predict"):

    data = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=[
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width"
        ]
    )

    prediction = model.predict(data)

    species = {
        0: "Iris Setosa 🌸",
        1: "Iris Versicolor 🌼",
        2: "Iris Virginica 🌺"
    }

    st.success(f"Predicted Species: {species[prediction[0]]}")