import streamlit as st
import numpy as np
import pickle
from keras.models import load_model
st.set_page_config(
    page_title="ANN Breast Cancer Classifier",
    page_icon="🧠",
    layout="wide"
)

st.title("ANN Breast Cancer Classifier")

model = load_model("ann_classifier.keras")

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("feature_names.pkl", "rb") as f:
    features = pickle.load(f)

values = []

for feature in features:
    values.append(st.number_input(feature, value=0.0))

if st.button("Predict"):
    x = np.array([values])
    x = scaler.transform(x)

    probability = model(x, training=False).numpy()[0][0]

    if probability >= 0.5:
        st.success("Benign")
    else:
        st.error("Malignant")

    st.write("Probability:", float(probability))