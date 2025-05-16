import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import umap.umap_ as umap
from difflib import get_close_matches

# Load dataset
url = "https://raw.githubusercontent.com/scostap/goodreads_bbe_dataset/main/Best_Books_Ever_dataset/books_1.Best_Books_Ever.csv"
dataset = pd.read_csv(url)
dataset['description'] = dataset['description'].fillna('')
dataset = dataset[dataset['description'].str.len() > 50].reset_index(drop=True)

# Load embeddings
embeddings = np.load('embeddings.npy')

# Reduce dimensions
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
reduced_embeddings = reducer.fit_transform(embeddings)

# Clustering
n_clusters = 20
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
labels = kmeans.fit_predict(reduced_embeddings)
dataset['cluster'] = labels


def recommend_books_by_title(input_title, top_n=10, min_rating=4.0, alpha=0.8):
    # substring match ignoring case
    matches = dataset[dataset['title'].str.lower().str.contains(input_title.lower())]
    if matches.empty:
        close = get_close_matches(input_title, dataset['title'].tolist(), n=5, cutoff=0.6)
        return None, close

    idx = matches.index[0]
    input_embedding = embeddings[idx].reshape(1, -1)
    input_cluster = dataset.loc[idx, 'cluster']
    content_sim = cosine_similarity(input_embedding, embeddings)[0]
    cluster_bonus = (dataset['cluster'] == input_cluster).astype(int)
    hybrid_score = alpha * content_sim + (1 - alpha) * cluster_bonus
    dataset['hybrid_score'] = hybrid_score
    recommended = dataset[(dataset.index != idx) & (dataset['rating'] >= min_rating)]
    recommended = recommended.sort_values(by='hybrid_score', ascending=False)

    return recommended[['title', 'author', 'rating', 'description']].head(top_n), None


def recommend_books_by_description(input_description, top_n=10, min_rating=4.0):
    model = SentenceTransformer('all-mpnet-base-v2')
    input_embedding = model.encode([input_description])
    similarity = cosine_similarity(input_embedding, embeddings)[0]
    dataset['similarity'] = similarity
    recommended = dataset[dataset['rating'] >= min_rating]
    recommended = recommended.sort_values(by='similarity', ascending=False)

    # Return columns including description but exclude similarity
    return recommended[['title', 'author', 'rating', 'description']].head(top_n)

# Streamlit UI starts here
st.set_page_config(page_title="📚 Book Recommender", layout="wide")

st.markdown("""
<style>
    .stDataFrame div[data-testid="stTable"] table {
        border-collapse: collapse;
        width: 100%;
    }
    .stDataFrame div[data-testid="stTable"] th, td {
        border: 1px solid #ddd;
        padding: 8px;
    }
    .stDataFrame div[data-testid="stTable"] tr:nth-child(even) {
        background-color: #f9f9f9;
    }
    .stDataFrame div[data-testid="stTable"] th {
        background-color: #4CAF50;
        color: white;
        text-align: left;
    }
</style>
""", unsafe_allow_html=True)

st.title("📚 Book Recommender System")
st.write("Get book recommendations by title or by writing your own description!")

option = st.radio("Choose input type:", ("By Book Title", "By Custom Description"))

if option == "By Book Title":
    user_input = st.text_input("Enter a book title:")
    if st.button("Recommend"):
        if user_input:
            recommended, close_matches = recommend_books_by_title(user_input)
            if recommended is not None:
                st.subheader("Recommended Books:")
                st.dataframe(recommended.style.set_properties(**{'white-space': 'normal'}), use_container_width=True, hide_index=True)
            else:
                st.warning(f"Book '{user_input}' not found.")
                if close_matches:
                    st.write("Did you mean:")
                    for title in close_matches:
                        st.write(f"- {title}")
        else:
            st.warning("Please enter a book title.")

else:
    user_input = st.text_area("Enter a custom book description:")
    if st.button("Recommend"):
        if user_input:
            recommended = recommend_books_by_description(user_input)
            st.subheader("Recommended Books:")
            st.dataframe(recommended.style.set_properties(**{'white-space': 'normal'}), use_container_width=True, hide_index=True)
        else:
            st.warning("Please enter a description.")
