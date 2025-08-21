import streamlit as st
import pickle
import pandas as pd
import tensorflow as tf
import numpy as np
import random

movies = pd.read_csv(
    "movies.dat",
    sep="::",
    engine="python",  # Needed because "::" is multi-character separator
    header=None,      # No header row in file
    names=["movieId", "title", "genres"],  # Assign column names
    encoding="latin-1"
)

movie_dict = dict(zip(movies["movieId"], movies["title"]))





# Load model + mappings

@st.cache_resource
def load_model_and_mappings():
    model = tf.keras.models.load_model("ncf_model.h5")  # update path if needed
    with open("user2id.pkl", "rb") as f:
        user2id = pickle.load(f)
    with open("movie2id.pkl", "rb") as f:
        movie2id = pickle.load(f)
    with open("id2movie.pkl", "rb") as f:
        id2movie = pickle.load(f)
    return model, user2id, movie2id, id2movie

model, user2id, movie2id, id2movie = load_model_and_mappings()




# Recommendation function

def recommend_movies(user_id, top_n=10):
    if user_id not in user2id:
        return None

    uid = user2id[user_id]
    all_movie_ids = list(movie2id.values())

    user_array = np.full(len(all_movie_ids), uid)
    movie_array = np.array(all_movie_ids)

    preds = model.predict([user_array, movie_array], verbose=0).flatten()
    top_indices = preds.argsort()[-top_n:][::-1]

    recommended_movie_ids = movie_array[top_indices]

    # 🔹 map movieId → title using movie_dict
    recommended_titles = [
        movie_dict[mid] if mid in movie_dict else f"Movie ID {mid}"
        for mid in recommended_movie_ids
    ]

    return recommended_titles


# Streamlit UI

st.title("🎬 Movie Recommender System (NCF)")
st.write("Enter your User ID to get personalized recommendations.")

user_id_input = st.text_input("Enter User ID:")

if st.button("Get Recommendations"):
    if user_id_input.strip():
        try:
            user_id_int = int(user_id_input)
            recommendations = recommend_movies(user_id_int)
            if recommendations:
                st.subheader("Top Recommendations for you:")
                for i, movie in enumerate(recommendations, 1):
                    st.write(f"{i}. {movie}")
            else:
                st.warning("User ID not found. Please try again or sign up as a new user.")
        except ValueError:
            st.error("Please enter a valid numeric User ID.")
    else:
        st.warning("Please enter a User ID first.")




# Cold Start Recommendation

def cold_start_recommendations(selected_movies, ratings, top_n=10):
    """
    For new users: use rated movies to compute a weighted preference
    and recommend similar ones.
    """
    if not selected_movies:
        return []

    # Map selected movies → internal IDs
    valid_ids = [movie2id[movie] for movie in selected_movies if movie in movie2id]

    all_movie_ids = list(movie2id.values())
    user_array = []
    movie_array = []

    # Fake "new user embedding" by repeating liked movies
    for mid, rating in zip(valid_ids, ratings):
        weight = int(rating)  # stronger weight if rating higher
        user_array.extend([0] * weight)  # 0 = placeholder new user ID
        movie_array.extend([mid] * weight)

    user_array = np.array(user_array)
    movie_array = np.array(movie_array)

    preds = model.predict([user_array, movie_array], verbose=0).mean(axis=0)

    all_preds = []
    for mid in all_movie_ids:
        score = model.predict([np.array([0]), np.array([mid])], verbose=0).flatten()[0]
        all_preds.append((mid, score))

    all_preds.sort(key=lambda x: x[1], reverse=True)
    recommended_movie_ids = [mid for mid, _ in all_preds[:top_n]]

    return [movie_dict[mid] if mid in movie_dict else f"Movie ID {mid}" for mid in recommended_movie_ids]


# Cold Start UI

st.subheader("✨ New User? Rate some movies to get recommendations!")

if st.checkbox("I’m a new user"):
    random_movies = movies.sample(5, random_state=random.randint(0, 1000))
    ratings = {}
    for _, row in random_movies.iterrows():
        ratings[row["movieId"]] = st.slider(f"Rate {row['title']} ({row['genres']})", 1, 5, 3)

    if st.button("Get Cold Start Recommendations"):
        selected_movies = list(ratings.keys())
        selected_ratings = list(ratings.values())
        recs = cold_start_recommendations(selected_movies, selected_ratings)
        st.subheader("Your Cold Start Recommendations:")
        for i, movie in enumerate(recs, 1):
            st.write(f"{i}. {movie}")