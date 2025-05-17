# preprocess.py

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import umap.umap_ as umap

url = "https://raw.githubusercontent.com/scostap/goodreads_bbe_dataset/main/Best_Books_Ever_dataset/books_1.Best_Books_Ever.csv"
df = pd.read_csv(url)
df['description'] = df['description'].fillna('')
df = df[df['description'].str.len() > 50].reset_index(drop=True)

embeddings = np.load("embeddings.npy")
if embeddings.shape[0] != len(df):
    raise ValueError(f"⚠️ Mismatch: embeddings rows ({embeddings.shape[0]}) ≠ dataset rows ({len(df)})")

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
reduced = reducer.fit_transform(embeddings)
kmeans = KMeans(n_clusters=20, random_state=42, n_init='auto')
df['cluster'] = kmeans.fit_predict(reduced)

df.to_csv("clustered_dataset.csv", index=False)
print("✅ Preprocessing complete. Files saved: clustered_dataset.csv")


