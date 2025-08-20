import streamlit as st
import pickle
import pandas as pd
import tensorflow as tf
import numpy as np




# ---------------------------
# Load model + mappings
# ---------------------------
# ---------------------------
# Load model and mappings
# ---------------------------
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



# ---------------------------
# Recommendation function
# --------------------------
def recommend_movies(user_id, top_n=10):
    if user_id not in user2id:
        return None #user not found
    uid = user2id[user_id]
    all_movie_ids = list(movie2id.values())

    user_array = np.full(len(all_movie_ids),uid)
    movie_array = np.array(all_movie_ids)

    preds = model.predict([user_array,movie_array], verbose=0).faltten()
    top_indices = preds.argsort()[-top_n:][::-1]

    recommended_movie = [id2movie[mid] for mid in movie_array[top_indices]]

    return recommended_movie

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