import streamlit as st
import pickle
import pandas as pd



with open("pca.pkl", "rb") as file:
    pca = pickle.load(file)

with open("pca_scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

st.set_page_config(
    page_title="CUSTOMER ANALYSIS USING PCA",
    page_icon="📊",
    layout="centered"

)


st.title("Customer Analysis Using PCA")

st.write(
    "Enter the customer's details below. "
    "PCA will reduce the 3 customer features into 2 principal components."
)


age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=25
)

income = st.number_input(
    "Annual Income (k$)",
    min_value=0.0,
    value=60.0
)

spending_score = st.slider(
    "Spending Score",
    min_value=1,
    max_value=100,
    value=50
)




if st.button("Apply PCA"):

    # Create input in the same format used during training
    customer = pd.DataFrame(
        [[age, income, spending_score]],
        columns=[
            "Age",
            "Annual Income (k$)",
            "Spending Score (1-100)"
        ]
    )

    
    customer_scaled = scaler.transform(customer)

   
    customer_pca = pca.transform(customer_scaled)

    pc1 = customer_pca[0][0]
    pc2 = customer_pca[0][1]

    st.success("PCA Transformation Completed")

    st.subheader("PCA Result")

    st.write("Principal Component 1:", round(pc1, 3))
    st.write("Principal Component 2:", round(pc2, 3))

    st.info(
        "The original 3 customer features — Age, Annual Income, "
        "and Spending Score — have been reduced to 2 principal components."
    )



st.divider()

st.subheader("About PCA")

st.write(
    "Principal Component Analysis (PCA) is a dimensionality reduction "
    "technique. It transforms multiple features into a smaller number "
    "of principal components while retaining as much information as possible."
)