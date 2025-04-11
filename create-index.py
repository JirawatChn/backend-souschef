import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle


embedder = SentenceTransformer("BAAI/bge-m3")

# โหลด + encode
df = pd.read_csv("thai_recipes_processed.csv")
texts = df["context"].dropna().tolist()
embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)
faiss.normalize_L2(embeddings)

# สร้าง + บันทึก
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, "thai_recipes.index")

with open("texts.pkl", "wb") as f:
    pickle.dump(texts, f)

print("✅ Index + texts saved.")