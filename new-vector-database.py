import os
import json
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
import time

# -----------------------------
# 1) ตั้งค่าโมเดลและไฟล์
# -----------------------------
print("🚀 เริ่มต้นโปรเซสสร้าง Vector Database...\n")

t0 = time.time()

embedder = SentenceTransformer("BAAI/bge-m3")
print("✅ โหลดโมเดลสำเร็จ: BAAI/bge-m3\n")

PREFERRED_CSV = "thai_recipes_with_nutrition.csv"
FALLBACK_CSV  = "thai_recipes_processed.csv"
csv_path = PREFERRED_CSV if os.path.exists(PREFERRED_CSV) else FALLBACK_CSV

INDEX_PATH = "thai_recipes_bge_m3.index"
TEXTS_PATH = "texts.pkl"
META_PATH  = "meta.pkl"

print(f"📄 ใช้ไฟล์ CSV: {csv_path}\n")

# -----------------------------
# 2) โหลดข้อมูล
# -----------------------------
print("📥 กำลังโหลดข้อมูลจาก CSV ...")
df = pd.read_csv(csv_path)
print(f"✅ โหลดข้อมูลสำเร็จ ({len(df)} แถว)\n")

# เดาชื่อคอลัมน์เมนู (ถ้ามี)
def detect_name_col(columns):
    keys = ["menu", "name", "title", "dish", "recipe", "item",
            "เมนู", "ชื่อเมนู", "ชื่ออาหาร", "อาหาร"]
    for k in keys:
        for c in columns:
            if k in c.lower():
                return c
    for c in columns:
        if df[c].dtype == "object":
            return c
    return columns[0]

name_col = detect_name_col(df.columns)
print(f"🔍 ตรวจพบคอลัมน์ชื่อเมนู: '{name_col}'\n")

# -----------------------------
# 3) ฟังก์ชันแปลง nutrition(JSON) -> ข้อความไทยอ่านง่าย
# -----------------------------
def nutrition_to_th_text(nutri_value):
    if pd.isna(nutri_value):
        return ""
    try:
        nutri = json.loads(nutri_value) if isinstance(nutri_value, str) else dict(nutri_value)
    except Exception:
        return ""
    parts = []
    if "calories_kcal" in nutri: parts.append(f"พลังงาน {nutri['calories_kcal']} kcal")
    if "protein_g"    in nutri: parts.append(f"โปรตีน {nutri['protein_g']} g")
    if "fat_g"        in nutri: parts.append(f"ไขมัน {nutri['fat_g']} g")
    if "carbs_g"      in nutri: parts.append(f"คาร์บ {nutri['carbs_g']} g")
    if "sodium_mg"    in nutri: parts.append(f"โซเดียม {nutri['sodium_mg']} mg")
    return "ข้อมูลโภชนาการ: " + ", ".join(parts) if parts else ""

# -----------------------------
# 4) สร้างข้อความที่จะฝัง (context + name + nutrition)
# -----------------------------
print("🧩 กำลังรวมข้อความ context + nutrition ...")
contexts, metas = [], []

for i, row in df.iterrows():
    name_txt = str(row.get(name_col, "")).strip()
    ctx_txt  = str(row.get("context", "")).strip()
    nutri_txt = nutrition_to_th_text(row.get("nutrition", None))

    combined = " || ".join([t for t in [name_txt, ctx_txt, nutri_txt] if t]).strip()
    if not combined:
        combined = name_txt or ctx_txt or nutri_txt or ""

    contexts.append(combined)
    metas.append({
        "row_index": i,
        "name": name_txt,
        "nutrition_raw": row.get("nutrition", None),
        "source_csv": csv_path
    })

print(f"✅ รวมข้อความสำเร็จ: {len(contexts)} records\n")

# -----------------------------
# 5) ทำ Embedding
# -----------------------------
print("⚙️ เริ่มสร้าง embeddings (อาจใช้เวลาสักครู่)...")
embeddings = embedder.encode(
    contexts,
    convert_to_numpy=True,
    show_progress_bar=True,
    normalize_embeddings=False
)
print("✅ สร้าง embeddings เสร็จสิ้น\n")

# -----------------------------
# 6) Normalize และสร้าง Index
# -----------------------------
print("📊 กำลัง normalize L2 และสร้าง FAISS Index ...")
faiss.normalize_L2(embeddings)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)
print("✅ สร้าง Index สำเร็จ\n")

# -----------------------------
# 7) บันทึกไฟล์ทั้งหมด
# -----------------------------
print("💾 กำลังบันทึก index และข้อมูลเสริม ...")
faiss.write_index(index, INDEX_PATH)
with open(TEXTS_PATH, "wb") as f:
    pickle.dump(contexts, f)
with open(META_PATH, "wb") as f:
    pickle.dump(metas, f)
print("✅ บันทึกไฟล์ทั้งหมดเรียบร้อย\n")

# -----------------------------
# 8) สรุปผล
# -----------------------------
elapsed = time.time() - t0
print("🎯 ดำเนินการเสร็จสิ้น")
print(f"📦 Index: {INDEX_PATH}")
print(f"📝 Texts: {TEXTS_PATH}")
print(f"📘 Meta : {META_PATH}")
print(f"📄 CSV  : {csv_path}")
print(f"⏱️ ใช้เวลา: {elapsed:.2f} วินาที\n")
