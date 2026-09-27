import os
import pickle

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv


# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()


# -----------------------------
# Fetch movie poster
# -----------------------------
def fetch_poster(movie_id):

    api_key = os.getenv("TMDB_API_KEY")

    if not api_key:
        st.error("TMDB_API_KEY is missing from .env file")
        return None

    url = (
        f"https://api.themoviedb.org/3/movie/{movie_id}"
        f"?api_key={api_key}"
    )

    try:
        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            return None

        data = response.json()

        poster_path = data.get("poster_path")

        if not poster_path:
            return None

        return "https://image.tmdb.org/t/p/w500" + poster_path

    except requests.RequestException:
        return None


# -----------------------------
# Recommendation function
# -----------------------------
def recommend(movie, movies, similarity):

    titles = movies["title"].astype(str)

    matching_movies = movies[titles == str(movie)]

    if matching_movies.empty:
        return [], []

    # Position of selected movie
    movie_index = matching_movies.index[0]
    movie_position = movies.index.get_loc(movie_index)

    # Similarity scores
    distances = similarity[movie_position]

    # Get top 5 recommendations
    movie_list = sorted(
        enumerate(distances),
        key=lambda x: x[1],
        reverse=True
    )[1:6]

    recommended_names = []
    recommended_posters = []

    for index, score in movie_list:

        movie_id = movies.iloc[index]["movie_id"]
        movie_name = movies.iloc[index]["title"]

        recommended_names.append(movie_name)

        poster = fetch_poster(movie_id)
        recommended_posters.append(poster)

    return recommended_names, recommended_posters


# -----------------------------
# Load movie data
# -----------------------------
with open("movie_dict.pkl", "rb") as file:
    movies = pickle.load(file)


with open("similarity.pkl", "rb") as file:
    similarity = pickle.load(file)


# -----------------------------
# Convert to DataFrame
# -----------------------------
if isinstance(movies, dict):
    movies = pd.DataFrame(movies)


# -----------------------------
# Validate data
# -----------------------------
required_columns = ["title", "movie_id"]

for column in required_columns:

    if column not in movies.columns:

        st.error(
            f"Missing required column: {column}"
        )

        st.write(
            "Available columns:",
            movies.columns.tolist()
        )

        st.stop()


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🎬 Movie Recommender System")


movie_titles = movies["title"].astype(str).tolist()


selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_titles
)


# -----------------------------
# Recommendation button
# -----------------------------
if st.button("Show Recommendation"):

    names, posters = recommend(
        selected_movie,
        movies,
        similarity
    )

    if not names:

        st.error("No recommendations found.")

    else:

        cols = st.columns(5)

        for col, name, poster in zip(
            cols,
            names,
            posters
        ):

            with col:

                st.write(name)

                if poster:

                    st.image(
                        poster,
                        use_container_width=True
                    )

                else:

                    st.write("Poster unavailable")