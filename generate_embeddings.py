import pandas as pd
from sentence_transformers import SentenceTransformer

url = "https://raw.githubusercontent.com/scostap/goodreads_bbe_dataset/main/Best_Books_Ever_dataset/books_1.Best_Books_Ever.csv"
dataset = pd.read_csv(url)
summaries = dataset['description'].fillna("").tolist()

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(summaries, show_progress_bar=True, batch_size=32)


import numpy as np
np.save('embeddings.npy', embeddings)

print("✅ Saved embeddings to embeddings.npy")
