import streamlit as st
import pandas as pd
import joblib
import hashlib
import sqlite3
from surprise import Dataset, Reader
import warnings

warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")

# ---------------- SESSION STATE ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "rated_movies" not in st.session_state:
    st.session_state.rated_movies = None
if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = False

# ---------------- DATABASE ----------------
DB_PATH = "recommender.db"

def get_db_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

conn = get_db_connection()
conn.execute('''
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password TEXT NOT NULL
    )
''')
conn.execute('''
    CREATE TABLE IF NOT EXISTS user_ratings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        movie_id TEXT,
        rating INTEGER,
        FOREIGN KEY (username) REFERENCES users(username)
    )
''')
conn.commit()

# ---------------- LOAD MOVIE METADATA ----------------
@st.cache_data
def load_movies():
    df = pd.read_csv(r"C:\Users\DJ COMPUTERS\OneDrive\Desktop\Movie Recommender App\u.item", sep='|', encoding='latin-1', header=None, usecols=[0, 1])
    df.columns = ['movie_id', 'title']
    return df

movies_df = load_movies()
movie_dict = dict(zip(movies_df.movie_id.astype(str), movies_df.title))

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

# ---------------- HELPERS ----------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password):
    try:
        with conn:
            conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hash_password(password)))
        return True
    except sqlite3.IntegrityError:
        return False

def verify_user(username, password):
    cur = conn.cursor()
    cur.execute("SELECT password FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    return row and row[0] == hash_password(password)

def store_user_ratings(username, ratings):
    with conn:
        for movie_id, rating in ratings.items():
            conn.execute(
                "INSERT INTO user_ratings (username, movie_id, rating) VALUES (?, ?, ?)",
                (username, movie_id, rating)
            )

def get_user_ratings(username):
    df = pd.read_sql_query("SELECT username, movie_id, rating FROM user_ratings WHERE username = ?", conn, params=(username,))
    return df

def recommend_top_n(model, user_df, username, N=5):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(user_df[['username', 'movie_id', 'rating']], reader)
    trainset = data.build_full_trainset()
    model.fit(trainset)

    rated = set(user_df['movie_id'])
    all_movies = set(movies_df['movie_id'].astype(str))
    unrated = list(all_movies - rated)

    predictions = [model.predict(username, movie_id) for movie_id in unrated]
    top_n = sorted(predictions, key=lambda x: x.est, reverse=True)[:N]
    return [(movie_dict.get(pred.iid, pred.iid), round(pred.est, 2)) for pred in top_n]

# ---------------- LOGIN UI ----------------
def login_ui():
    st.markdown("## 🔐 Login or Register")
    col1, col2, col3 = st.columns([1.5, 2, 1.5])
    with col2:
        action = st.radio("Choose", ["Login", "Register"], horizontal=True)
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if action == "Register":
            if st.button("Register"):
                if register_user(username, password):
                    st.success("✅ Registered successfully. You can now log in.")
                else:
                    st.error("⚠️ Username already exists.")
        else:
            if st.button("Login"):
                if verify_user(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.rated_movies = movies_df.sample(5)
                    st.session_state.feedback_given = False
                    st.success("✅ Welcome, " + username)
                    st.rerun()
                else:
                    st.error("❌ Invalid login")

# ---------------- MAIN APP UI ----------------
def main_ui():
    st.markdown(f"## 🎬 Welcome, {st.session_state.username}")
    tabs = st.tabs(["⭐ Rate Movies", "🎯 Get Recommendations"])

    # ---------- Rate Movies ----------
    with tabs[0]:
        st.subheader("Rate these 5 movies (or skip with 0)")

        static_movies = st.session_state.rated_movies
        ratings = {}

        if not st.session_state.feedback_given:
            for _, row in static_movies.iterrows():
                rating = st.slider(
                    label=row["title"],
                    min_value=0, max_value=5,
                    value=0,
                    help="0 = Not Watched",
                    key=f"slider_{row['movie_id']}"
                )
                if rating != 0:
                    ratings[str(row["movie_id"])] = rating

            if st.button("✅ Submit Ratings"):
                if ratings:
                    store_user_ratings(st.session_state.username, ratings)
                    st.session_state.feedback_given = True
                    st.success("🎉 Thank you! Your feedback is saved.")
                else:
                    st.warning("⚠️ Please rate at least one movie.")
        else:
            st.info("✅ You've already submitted ratings. Go to the next tab for recommendations.")

    # ---------- Get Recommendations ----------
    with tabs[1]:
        st.subheader("🎯 Recommended for you")
        user_df = get_user_ratings(st.session_state.username)

        if not user_df.empty:
            model = load_model()
            recommendations = recommend_top_n(model, user_df, st.session_state.username)

            if recommendations:
                for title, est in recommendations:
                    st.markdown(f"⭐ **{title}** — _Predicted rating: {est}_")
            else:
                st.warning("⚠️ Not enough ratings to recommend. Please rate more movies.")
        else:
            st.warning("⚠️ No ratings found. Please go to the first tab and rate some movies.")

# ---------------- APP START ----------------
if not st.session_state.logged_in:
    login_ui()
else:
    main_ui()
