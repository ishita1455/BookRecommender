import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import umap.umap_ as umap
from difflib import get_close_matches


url = "https://raw.githubusercontent.com/scostap/goodreads_bbe_dataset/main/Best_Books_Ever_dataset/books_1.Best_Books_Ever.csv"
dataset = pd.read_csv(url)

dataset['description'] = dataset['description'].fillna('')
dataset = dataset[dataset['description'].str.len() > 50].reset_index(drop=True)

titles = dataset['title'].tolist()
summaries = dataset['description'].tolist()

def get_bert_embeddings(texts, model_name='all-mpnet-base-v2', batch_size=32):
    print("Generating BERT embeddings...")
    model = SentenceTransformer(model_name, device='cpu')
    return model.encode(texts, show_progress_bar=True, batch_size=batch_size)

embeddings = get_bert_embeddings(summaries)

print("Reducing dimensions with UMAP...")
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
reduced_embeddings = reducer.fit_transform(embeddings)


n_clusters = 20
print("Clustering with KMeans...")
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
labels = kmeans.fit_predict(reduced_embeddings)

sil_score = silhouette_score(reduced_embeddings, labels)
print(f"Silhouette Score: {sil_score:.2f}")

dataset['cluster'] = labels


def recommend_books_by_title(input_title, top_n=10, min_rating=4.0, alpha=0.8):
    matches = dataset[dataset['title'].str.lower() == input_title.lower()]
    
    if matches.empty:
        print(f"\nBook '{input_title}' not found.")
        close = get_close_matches(input_title, dataset['title'].tolist(), n=5, cutoff=0.6)
        if close:
            print("Did you mean:")
            for title in close:
                print(f" - {title}")
        return pd.DataFrame()

    idx = matches.index[0]
    input_embedding = embeddings[idx].reshape(1, -1)
    input_cluster = dataset.loc[idx, 'cluster']

    content_sim = cosine_similarity(input_embedding, embeddings)[0]
    cluster_bonus = (dataset['cluster'] == input_cluster).astype(int)
    
    hybrid_score = alpha * content_sim + (1 - alpha) * cluster_bonus
    dataset['hybrid_score'] = hybrid_score

    recommended = dataset[(dataset.index != idx) & (dataset['rating'] >= min_rating)]
    recommended = recommended.sort_values(by='hybrid_score', ascending=False)

    return recommended[['title', 'author', 'rating', 'hybrid_score']].head(top_n)


def recommend_books_by_description(input_description, top_n=10, min_rating=4.0):
    input_embedding = get_bert_embeddings([input_description])
    similarity = cosine_similarity(input_embedding, embeddings)[0]
    dataset['similarity'] = similarity

    recommended = dataset[dataset['rating'] >= min_rating]
    recommended = recommended.sort_values(by='similarity', ascending=False)

    return recommended[['title', 'author', 'rating', 'similarity']].head(top_n)


plt.figure(figsize=(12, 7))
sns.scatterplot(x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1], hue=labels, palette='tab10', s=50)
plt.title("Book Clusters by Genre (UMAP + KMeans)", fontsize=16)
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

def get_user_input():
    print("Choose an option:")
    print("1: Enter Book Title:")
    print("2: Enter Custom Description:")

    option = input("Enter 1 or 2: ")

    if option == "1":
        title = input("Enter the book title: ")
        recommended = recommend_books_by_title(title)
        if not recommended.empty:
            print("\nHybrid Recommended Books:")
            print(recommended)

    elif option == "2":
        description = input("Enter a custom book description: ")
        recommended = recommend_books_by_description(description)
        print("\nRecommended Books (by Description):")
        print(recommended)

    else:
        print("Invalid option. Please enter 1 or 2.")

get_user_input()
