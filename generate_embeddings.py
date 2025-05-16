import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

url = "https://raw.githubusercontent.com/scostap/goodreads_bbe_dataset/main/Best_Books_Ever_dataset/books_1.Best_Books_Ever.csv"
dataset = pd.read_csv(url)

dataset['description'] = dataset['description'].fillna('')
dataset = dataset[dataset['description'].str.len() > 50].reset_index(drop=True)

summaries = dataset['description'].tolist()

model = SentenceTransformer('all-MiniLM-L6-v2')  
embeddings = model.encode(summaries, show_progress_bar=True, batch_size=32)

np.save('embeddings.npy', embeddings)

print("✅ Saved embeddings.npy matching filtered dataset (len =", len(embeddings), ")")

