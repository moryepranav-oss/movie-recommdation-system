
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
import os


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")


# ============================================================
# LOAD MOVIE DATA
# ============================================================

@st.cache_resource
def load_movies():

    with open("movie_dict.pkl", "rb") as file:
        movie_dict = pickle.load(file)

    return pd.DataFrame(movie_dict)


movies = load_movies()


# ============================================================
# LOAD OPTIMIZED SIMILARITY MATRIX
# ============================================================

@st.cache_resource
def load_similarity():

    with open("similarity_optimized.pkl", "rb") as file:
        similarity_data = pickle.load(file)

    # Check whether this is our optimized format
    if isinstance(similarity_data, dict):

        if similarity_data.get("type") == "symmetric_matrix_float16":

            shape = similarity_data["shape"]
            n = shape[0]

            # Create empty matrix
            similarity = np.zeros(
                shape,
                dtype=np.float16
            )

            # Get upper-triangle indexes
            indices = np.triu_indices(n)

            # Put stored values into upper triangle
            similarity[indices] = similarity_data["data"]

            # Mirror upper triangle to lower triangle
            similarity[
                (indices[1], indices[0])
            ] = similarity_data["data"]

            return similarity

    # Fallback if normal similarity matrix
    return similarity_data


similarity = load_similarity()


# ============================================================
# FETCH POSTER FROM TMDB
# ============================================================

@st.cache_data(show_spinner=False)
def fetch_poster(movie_id):

    if not TMDB_API_KEY:
        return None

    url = f"https://api.themoviedb.org/3/movie/{movie_id}"

    params = {
        "api_key": TMDB_API_KEY
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        # TMDB request failed
        if response.status_code != 200:
            return None

        data = response.json()

        poster_path = data.get("poster_path")

        if poster_path:

            return (
                "https://image.tmdb.org/t/p/w500"
                + poster_path
            )

        return None

    except requests.exceptions.RequestException:

        # Internet/TMDB timeout
        return None

    except Exception:

        return None


# ============================================================
# RECOMMENDATION FUNCTION
# ============================================================

def recommend(movie):

    # Find selected movie index
    matching_movies = movies[
        movies["title"] == movie
    ]

    if matching_movies.empty:
        return [], []

    movie_index = matching_movies.index[0]

    # Get similarity scores
    distances = similarity[movie_index]

    # Get top 5 movies
    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []
    recommended_posters = []

    for index, score in movie_list:

        movie_title = movies.iloc[index]["title"]

        # Get movie ID
        movie_id = movies.iloc[index]["movie_id"]

        recommended_movies.append(
            movie_title
        )

        # Get poster
        poster = fetch_poster(movie_id)

        recommended_posters.append(
            poster
        )

    return (
        recommended_movies,
        recommended_posters
    )


# ============================================================
# STREAMLIT UI
# ============================================================

st.title("🎬 Movie Recommendation System")

st.write(
    "Select a movie and get 5 similar movie recommendations."
)

st.divider()


# ============================================================
# MOVIE SELECTION
# ============================================================

movie_list = movies["title"].values

selected_movie = st.selectbox(
    "🎥 Select a movie",
    movie_list
)


# ============================================================
# RECOMMEND BUTTON
# ============================================================

if st.button(
    "🚀 Recommend Movies",
    use_container_width=True
):

    with st.spinner("Finding similar movies..."):

        names, posters = recommend(
            selected_movie
        )

    if not names:

        st.error(
            "Movie not found."
        )

    else:

        st.subheader(
            "🍿 Recommended Movies"
        )

        # Create 5 columns
        cols = st.columns(5)

        for col, name, poster in zip(
            cols,
            names,
            posters
        ):

            with col:

                # Display poster if available
                if poster:

                    st.image(
                        poster,
                        use_container_width=True
                    )

                else:

                    st.info(
                        "Poster unavailable"
                    )

                # Movie title
                st.write(
                    f"**{name}**"
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🎬 Movie Recommendation System | "
    "Built with Python, Streamlit, NumPy and TMDB"
)
