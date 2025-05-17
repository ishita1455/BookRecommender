import pandas as pd
import numpy as np

df = pd.read_csv("books.csv")
embeddings = np.load("embeddings.npy")


print(df.columns)