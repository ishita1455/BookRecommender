# 📚 Book Recommender System using BERT, UMAP & Streamlit

A hybrid book recommendation system that combines **semantic similarity using BERT embeddings** with **unsupervised clustering** (UMAP + KMeans) to recommend books based on title or a custom description.

## 🚀 Live Demo

👉 [Try it](https://bookrecommender-abc.streamlit.app)

---

## ✨ Features

* 🔤 **Title-Based Recommendations**
  Enter the title of a book you like to get similar suggestions.

* ✍️ **Custom Description-Based Recommendations**
  Describe the kind of book you're looking for, and get relevant matches using sentence-level semantic search.

* 🤖 **BERT Embeddings**
  Uses `all-mpnet-base-v2` from [Sentence-Transformers](https://www.sbert.net/) for meaningful semantic representations.

* 📊 **UMAP + KMeans Clustering**
  Groups books into clusters of similar content for improved relevance and diversity.

* ⚡ **Fast Performance**
  Precomputed embeddings and clusters loaded with caching for near-instant results.

* 🖥️ **Responsive UI**
  Built with Streamlit for a clean, interactive experience.

---

## 🛠️ Technologies Used

| Category                 | Tools / Libraries                           |
| ------------------------ | ------------------------------------------- |
| Language                 | Python 3.12                                 |
| Embeddings               | `sentence-transformers (all-mpnet-base-v2)` |
| Dimensionality Reduction | `UMAP`                                      |
| Clustering               | `KMeans (Scikit-learn)`                     |
| UI                       | `Streamlit`                                 |
| Data Source              | Goodreads BBE Dataset                       |

---

## 🧠 How It Works

1. **Data Cleaning:** Loads and filters the Goodreads dataset.
2. **Embeddings:** Precomputed using `all-mpnet-base-v2` on book descriptions.
3. **Clustering:** UMAP reduces dimensions, and KMeans identifies clusters.
4. **Hybrid Recommendation:** Uses a weighted combination of cosine similarity and cluster proximity.
5. **Streamlit UI:** Lets users get book suggestions based on title or description.

