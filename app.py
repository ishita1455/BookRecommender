# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches

# -------------------------------
# 🧠 Load Model and Data
# -------------------------------

st.set_page_config(page_title="📚 Book Recommender", layout="wide")

@st.cache_resource
def load_model():
    return SentenceTransformer("all-mpnet-base-v2")

@st.cache_data
def load_dataset():
    return pd.read_csv("clustered_dataset.csv")

@st.cache_data
def load_embeddings():
    return np.load("embeddings.npy")

model = load_model()
dataset = load_dataset()
embeddings = load_embeddings()

# -------------------------------
# 🔍 Recommendation Functions
# -------------------------------

from difflib import get_close_matches
from sklearn.metrics.pairwise import cosine_similarity

def recommend_similar_books(title, dataset, embeddings, top_n=10, min_rating=4.0, 
                            alpha=0.7, beta=0.6, gamma=0.4):

    
    matches = dataset[dataset['title'].str.lower() == title.lower()]
    if matches.empty:
        
        partial_matches = dataset[dataset['title'].str.lower().str.contains(title.lower())]
        if partial_matches.empty:
           
            close_matches = get_close_matches(title, dataset['title'].tolist(), n=5, cutoff=0.6)
            return None, close_matches
        idx = partial_matches.index[0]
    else:
        idx = matches.index[0]

  
    input_emb = embeddings[idx].reshape(1, -1)
    content_sim = cosine_similarity(input_emb, embeddings)[0]

    
    input_cluster = dataset.loc[idx, 'cluster']
    cluster_bonus = (dataset['cluster'] == input_cluster).astype(int)

   
    rating_norm = (dataset['rating'] - dataset['rating'].min()) / (dataset['rating'].max() - dataset['rating'].min() + 1e-6)
    numRatings_norm = (dataset['numRatings'] - dataset['numRatings'].min()) / (dataset['numRatings'].max() - dataset['numRatings'].min() + 1e-6)
    popularity_score = 0.5 * rating_norm + 0.5 * numRatings_norm

   
    total_weight = alpha + beta + gamma
    combined_score = (alpha * content_sim + beta * cluster_bonus + gamma * popularity_score) / total_weight

    
    dataset = dataset.copy()
    dataset['combined_score'] = combined_score

   
    recommended = dataset[(dataset.index != idx) & (dataset['rating'] >= min_rating)]

  
    recommended = recommended.sort_values(by='combined_score', ascending=False)

    cols = ['title', 'author', 'rating', 'numRatings', 'description', 'combined_score']
    return recommended[cols].head(top_n), None



def recommend_by_genre_popularity(genre, top_n=10, min_rating=4.0):
   
    matches = dataset[dataset['genres'].str.lower().str.contains(genre.lower(), na=False)]
    if matches.empty:
        return None

   
    filtered = matches[matches['rating'] >= min_rating]
    if filtered.empty:
        return None

   
    rating_min, rating_max = filtered['rating'].min(), filtered['rating'].max()
    numRatings_min, numRatings_max = filtered['numRatings'].min(), filtered['numRatings'].max()

    rating_norm = (filtered['rating'] - rating_min) / (rating_max - rating_min + 1e-6)
    numRatings_norm = (filtered['numRatings'] - numRatings_min) / (numRatings_max - numRatings_min + 1e-6)


    alpha = 0.2
    popularity_score = alpha * rating_norm + (1 - alpha) * numRatings_norm

    filtered = filtered.copy()
    filtered['popularity_score'] = popularity_score

   
    recommended = filtered.sort_values(by='popularity_score', ascending=False)

    return recommended[['title', 'author', 'rating', 'numRatings', 'likedPercent', 'description']].head(top_n)

# -------------------------------
# 🎨 Streamlit UI
# -------------------------------

st.title("📚 Book Recommender System")
st.write("Get personalized book recommendations using a title or genre.")

st.markdown("---")

option = st.radio("Choose input method:", ("🔤 By Book Title", "📚 By Genre"))

if option == "🔤 By Book Title":
    user_input = st.text_input("Enter a book title:")
    if st.button("Recommend"):

        if user_input.strip():

            recommended, close_matches = recommend_similar_books(user_input.strip(), dataset, embeddings=embeddings,  top_n=10,
    min_rating=4.0,
    alpha=0.6,  
    beta=0.2,    
    gamma=0.2)
            
            if recommended is not None:
                st.subheader("✅ Recommended Books:")
                st.dataframe(recommended.style.set_properties(**{'white-space': 'normal'}), use_container_width=True, hide_index=True)
            else:
                st.warning(f"❌ Book '{user_input}' not found.")
                if close_matches:
                    st.info("Did you mean:")
                    for title in close_matches:
                        st.markdown(f"- **{title}**")
        else:
            st.warning("⚠️ Please enter a book title.")

else:  
    user_input = st.text_input("Enter a genre:")
    if st.button("Recommend"):
        if user_input.strip():
            recommended = recommend_by_genre_popularity(user_input.strip())
            if recommended is not None and not recommended.empty:
                st.subheader("✅ Recommended Books:")
                st.dataframe(recommended.style.set_properties(**{'white-space': 'normal'}), use_container_width=True, hide_index=True)
            else:
                st.warning(f"❌ No books found for genre '{user_input}'.")
        else:
            st.warning("⚠️ Please enter a genre.")
