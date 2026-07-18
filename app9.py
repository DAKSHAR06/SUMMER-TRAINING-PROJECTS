import streamlit as st
import joblib
import numpy as np

model = joblib.load("clusterkmeans.pkl")
scaler = joblib.load("scalerkmeans.pkl")
encoder = joblib.load("genderencoderkmeans.pkl")

st.title("Customer Segmentation using K-Means Clustering")

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=1, max_value=100, value=25)
income = st.number_input("Annual Income (k$)", min_value=1, value=40)
score = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50)

if st.button("Predict Cluster"):

    gender = encoder.transform([gender])[0]

    data = np.array([[gender, age, income, score]])

    data = scaler.transform(data)

    cluster = model.predict(data)

    st.success(f"Customer belongs to Cluster {cluster[0]}")