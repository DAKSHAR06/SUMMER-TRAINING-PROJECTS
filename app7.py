import streamlit as st
import pandas as pd
import joblib

model = joblib.load("movie_recommender.pkl")
movie_names = joblib.load("movie_names.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")
st.set_page_config(
    page_title="Movie Recommendation using KNN Classifier",
    page_icon="🎬",
    layout="centered"
)


df = pd.read_csv("Data for repository.csv")

movie_name = st.selectbox("Select Movie", sorted(movie_names.tolist()))

st.title("🎬 Movie Recommendation System")
st.write("Find 5 Similar Movies Using KNN")

if st.button("Recommend Movies"):

    movie_index = movie_names[movie_names == movie_name].index[0]

    data = df.drop("Movie Name", axis=1).copy()

    for column in data.columns:
        if column in encoders:
            data[column] = encoders[column].transform(data[column])

    scaled_data = scaler.transform(data)

    distance, indices = model.kneighbors([scaled_data[movie_index]])

    st.subheader("Recommended Movies")

    count = 1

    for i in indices[0][1:]:
        st.write(f"{count}. {movie_names.iloc[i]}")
        count += 1