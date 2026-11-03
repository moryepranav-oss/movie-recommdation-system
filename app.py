import streamlit as st
import pickle
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
import os
from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# Load .env from the same folder as app.py (works locally)
load_dotenv(Path(__file__).parent / ".env")


def get_api_key():
    # 1) Local: .env / environment variable
    key = os.getenv("TMDB_API_KEY")

    # 2) Streamlit Cloud: Settings -> Secrets
    if not key:
        try:
            key = st.secrets["TMDB_API_KEY"]
        except Exception:
            key = None

    if key:
        return key.strip().strip('"').strip("'")
    return None


TMDB_API_KEY = get_api_key()


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

@st.cache_resource
def get_session():
    """One shared HTTP session that retries on temporary failures."""
    session = requests.Session()

    retries = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504]
    )

    session.mount("https://", HTTPAdapter(max_retries=retries))

    return session


@st.cache_data(show_spinner=False)
def _fetch_poster_path(movie_id):
    """
    Raises an exception on failure, so failures are NOT cached.
    Only successful results are remembered.
    """
    session = get_session()

    url = f"https://api.themoviedb.org/3/movie/{int(movie_id)}"

    headers = {}
    params = {}

    # v4 token is a long JWT starting with "eyJ"; v3 key is 32 characters
    if TMDB_API_KEY.startswith("eyJ"):
        headers["Authorization"] = f"Bearer {TMDB_API_KEY}"
    else:
        params["api_key"] = TMDB_API_KEY

    response = session.get(
        url,
        params=params,
        headers=headers,
        timeout=10
    )

    response.raise_for_status()

    return response.json().get("poster_path")


def fetch_poster(movie_id):

    if not TMDB_API_KEY:
        st.session_state["poster_error"] = (
            "TMDB_API_KEY not found. Add it to .env (local) "
            "or Settings -> Secrets (Streamlit Cloud)."
        )
        return None

    try:
        poster_path = _fetch_poster_path(movie_id)

        if poster_path:
            return "https://image.tmdb.org/t/p/w500" + poster_path

        # Movie genuinely has no poster on TMDB
        return None

    except Exception as e:
        st.session_state["poster_error"] = f"{type(e).__name__}: {e}"
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

    # Clear any old poster error before a new search
    st.session_state.pop("poster_error", None)

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

        # Show the real reason if any poster failed
        if "poster_error" in st.session_state:
            st.warning(
                f"Poster problem: {st.session_state.pop('poster_error')}"
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🎬 Movie Recommendation System | "
    "Built with Python, Streamlit, NumPy and TMDB"
)
