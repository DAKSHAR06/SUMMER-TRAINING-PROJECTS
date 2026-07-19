import streamlit as st
import pandas as pd
import joblib

model=joblib.load ("SVC.pkl")

st.set_page_config(
    page_title="Iris Flower Classification",
    page_icon="🌸",
    layout="centered"
)

st.title("🌸 Iris Flower Classification")
st.write("Support Vector Classifier (SVC)")

sepal_length = st.number_input(
    "Sepal Length",
    min_value=0.0,
    value=5.1,
    step=0.1
)

sepal_width = st.number_input(
    "Sepal Width",
    min_value=0.0,
    value=3.5,
    step=0.1
)

petal_length = st.number_input(
    "Petal Length",
    min_value=0.0,
    value=1.4,
    step=0.1
)

petal_width = st.number_input(
    "Petal Width",
    min_value=0.0,
    value=0.2,
    step=0.1
)

if st.button("Predict Species"):

    data = pd.DataFrame(
        [[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]],
        columns=[
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width"
        ]
    )

    prediction = model.predict(data)[0]

    species = {
        0: "Setosa",
        1: "Versicolor",
        2: "Virginica"
    }

    predicted_species = species.get(
        prediction,
        str(prediction)
    )

    st.success(
        f"🌸 Predicted Iris Species: {predicted_species}"
    )