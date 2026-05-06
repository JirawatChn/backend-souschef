# -*- coding: utf-8 -*-
import os, json, pickle
import numpy as np
from typing import List, Dict, Any
import faiss
from sentence_transformers import SentenceTransformer
from google import genai
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY env variable")

client = genai.Client(api_key=GEMINI_API_KEY)

embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def embed_query(text: str) -> np.ndarray:
    vec = embedder.encode([text], convert_to_numpy=True, normalize_embeddings=True)
    return vec.astype(np.float32)

index = faiss.read_index("thai_recipes_minilm.index")
child_texts: List[str] = pickle.load(open("child_texts.pkl", "rb"))
child_meta: List[Dict[str, Any]] = pickle.load(open("child_meta.pkl", "rb"))

docstore: Dict[str, Dict[str, Any]] = {}
with open("menus_docstore.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        docstore[row["parent_id"]] = row

DAILY_GUIDE = {
    "energy_kcal":     2000.0,
    "fat_g_max":         65.0,
    "satfat_g_max":      20.0,
    "transfat_g_max":     2.0,
    "sugar_g_max":       50.0,
    "sugar_g_ideal":     25.0,
    "sodium_mg_max":   2000.0,
    "fiber_g_min":       25.0,
    "potassium_mg_min": 3500.0,
    "protein_g_min":     50.0,
    "protein_g_max":     75.0,
}
