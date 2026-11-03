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

# Load .env from the same folder as app.py (not the terminal's folder)
load_dotenv(Path(__file__).parent / ".env")

def get_api_key():
    key = os.getenv("TMDB_API_KEY")
    if not key:
        try:
            key = st.secrets["TMDB_API_KEY"]   # for Streamlit Cloud
        except Exception:
            key = None
    return key.strip().strip('"').strip("'") if key else None

TMDB_API_KEY = get_api_key()


# ============================================================
# FETCH POSTER FROM TMDB
# ============================================================

@st.cache_resource
def get_session():
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
    """Raises on failure, so failures are NOT cached."""
    session = get_session()
    url = f"https://api.themoviedb.org/3/movie/{int(movie_id)}"

    headers = {}
    params = {}
    # v4 token is a long JWT starting with "eyJ"; v3 key is 32 characters
    if TMDB_API_KEY.startswith("eyJ"):
        headers["Authorization"] = f"Bearer {TMDB_API_KEY}"
    else:
        params["api_key"] = TMDB_API_KEY

    response = session.get(url, params=params, headers=headers, timeout=10)
    response.raise_for_status()
    return response.json().get("poster_path")


def fetch_poster(movie_id):
    if not TMDB_API_KEY:
        st.session_state["poster_error"] = "TMDB_API_KEY not found."
        return None
    try:
        poster_path = _fetch_poster_path(movie_id)
        if poster_path:
            return "https://image.tmdb.org/t/p/w500" + poster_path
        return None   # movie genuinely has no poster
    except Exception as e:
        st.session_state["poster_error"] = f"{type(e).__name__}: {e}"
        return None
