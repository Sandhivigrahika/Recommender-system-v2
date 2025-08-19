import streamlit as st
import pickle
import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np




# ---------------------------
# Load model + mappings
# ---------------------------
@st.cache_resource
def load_artifacts():
    model = load_model("ncf.h5")

    with open("user2id.pkl", "rb") as f:
        user2id = pickle.load(f)
    with open("movie2id.pkl", "rb") as f:
        movie2id = pickle.load(f)
    with open("id2movie.pkl", "rb") as f:
        id2movie = pickle.load(f)

    movies_df = pd.read_csv("movies.csv")  # must have at least movieId + title
    return model, user2id, movie2id, id2movie, movies_df


model, user2id, movie2id, id2movie, movies_df = load_artifacts()



# ---------------------------
# Recommendation function
# ---------------------------
def recommend(user_id, k=10):
    if user_id not in user2id:
        return ["User not found. Please onboard as new user."]

    uid = user2id[user_id]
    all_movie_ids = list(movie2id.keys())

    # Build input arrays
    user_input = np.array([uid] * len(all_movie_ids))
    item_input = np.array([movie2id[mid] for mid in all_movie_ids])

    # Predict scores
    preds = model.predict([user_input, item_input], verbose=0).flatten()

    # Sort top-K
    top_k_idx = preds.argsort()[-k:][::-1]
    top_movie_ids = [all_movie_ids[i] for i in top_k_idx]

    # Map back to titles
    results = movies_df[movies_df["movieId"].isin(top_movie_ids)][["title", "genres"]]
    return results


# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🎬 Movie Recommender (NCF - Keras)")

user_type = st.radio("Are you an existing user?", ["Yes", "No"])

if user_type == "Yes":
    user_id = st.text_input("Enter your User ID:")
    if st.button("Get Recommendations"):
        if user_id.strip():
            recs = recommend(user_id, k=10)
            st.write(recs)
else:
    st.write("Onboarding new user...")
    fav_genres = st.multiselect("Pick your favorite genres:",
                                ["Action", "Comedy", "Drama", "Romance", "Sci-Fi", "Thriller"])
    if st.button("Get Recommendations"):
        # Fallback logic for cold start (e.g., top movies from those genres)
        if fav_genres:
            recs = movies_df[movies_df["genres"].str.contains("|".join(fav_genres))]
            recs = recs.sample(10)[["title", "genres"]]
            st.write(recs)
        else:
            st.warning("Please pick at least one genre!")