# -*- coding: utf-8 -*-
"""
Rebuild FAISS index using paraphrase-multilingual-MiniLM-L12-v2.
Run once: python rebuild_index.py
Output: thai_recipes_minilm.index
"""
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

OUTPUT_INDEX = "thai_recipes_minilm.index"
BATCH_SIZE = 256

print("Loading child_texts.pkl...")
child_texts = pickle.load(open("child_texts.pkl", "rb"))
print(f"Total child chunks: {len(child_texts)}")

print("Loading model paraphrase-multilingual-MiniLM-L12-v2...")
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

print("Embedding all chunks (this may take a few minutes)...")
vecs = embedder.encode(
    child_texts,
    batch_size=BATCH_SIZE,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True,
)

vecs = vecs.astype(np.float32)
dim = vecs.shape[1]

print(f"Building FAISS IndexFlatIP (dim={dim})...")
index = faiss.IndexFlatIP(dim)
index.add(vecs)

faiss.write_index(index, OUTPUT_INDEX)
print(f"Done! Saved to {OUTPUT_INDEX} (vectors={index.ntotal})")
