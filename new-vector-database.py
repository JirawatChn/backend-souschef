# ingest_build_index.py
# Robust ingest สำหรับ dataset ที่บางแถวไม่มี ingredients/method
import os, re, json, time, pickle
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

# -----------------------------
# 0) ตั้งค่าไฟล์/โมเดล + พารามิเตอร์
# -----------------------------
CSV_PATH   = "thai_recipes_2.csv"     # <-- ไฟล์ใหม่
INDEX_PATH = "thai_recipes_bge_m3.index"
TEXTS_PATH = "child_texts.pkl"
META_PATH  = "child_meta.pkl"
DOCSTORE   = "menus_docstore.jsonl"   # parent (ระเบียนเต็ม) ต่อแถว
REPORT_CSV = "ingest_report.csv"      # รายงานคุณภาพข้อมูล

EMBED_MODEL = "BAAI/bge-m3"

# พฤติกรรมเมื่อข้อมูลขาด:
# - ถ้า TRUE: ข้ามแถวที่ "ไม่มีทั้ง ingredients และ method"
# - ถ้า FALSE: ยังเก็บ parent + child จาก field อื่น (เช่น name/tags/nutrition) เพื่อไม่เสียเมนูไป
SKIP_IF_NO_ING_AND_METHOD = False

print("🚀 Build Vector DB from columns: menu_name, ingredients, method, nutrition, tags")
t0 = time.time()

# -----------------------------
# 1) โหลด CSV
# -----------------------------
assert os.path.exists(CSV_PATH), f"ไม่พบไฟล์ {CSV_PATH}"
df = pd.read_csv(CSV_PATH)

# ---- จับคู่ชื่อคอลัมน์ (ยืดหยุ่น) ----
cols = {c.lower(): c for c in df.columns}
def pick(*cands):
    for c in cands:
        if c in cols: return cols[c]
    for c in df.columns:
        name = c.lower()
        if any(k in name for k in cands): return c
    return None

menu_col = pick("menu_name","menu","ชื่อเมนู","ชื่ออาหาร","dish","title","recipe")
ing_col  = pick("ingredients","วัตถุดิบ","ส่วนผสม")
meth_col = pick("method","วิธีทำ","ขั้นตอน","instructions","วิธีการทำ")
nut_col  = pick("nutrition","โภชนาการ","สารอาหาร")
tag_col  = pick("tags","tag","หมวดหมู่","ประเภท")

print("🧭 column mapping:", {
    "menu_name":menu_col, "ingredients":ing_col, "method":meth_col, "nutrition":nut_col, "tags":tag_col
})

# -----------------------------
# 2) ฟังก์ชันทำความสะอาด/ฟอร์แมต
# -----------------------------
HTML_TAG_RE = re.compile(r"<[^>]+>")
BULLET_RE = re.compile(r"^[\-\•\●\·\*]\s*", flags=re.MULTILINE)

def strip_html(s: str) -> str:
    return HTML_TAG_RE.sub("", s)

def norm_str(x):
    if pd.isna(x): return ""
    s = str(x)
    s = strip_html(s)
    s = BULLET_RE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def fmt_ingredients(v) -> str:
    if pd.isna(v): return ""
    if isinstance(v, str):
        t = v.strip()
        # JSON (list/dict) ?
        if (t.startswith("[") and t.endswith("]")) or (t.startswith("{") and t.endswith("}")):
            try:
                obj = json.loads(t)
                if isinstance(obj, dict): obj = obj.get("ingredients", obj)
                if isinstance(obj, list):
                    parts = [norm_str(str(x)) for x in obj if norm_str(str(x))]
                    if parts: return "ส่วนผสม: " + ", ".join(parts)
            except Exception:
                pass
        # ปกติ: แยกด้วย , | ; หรือขึ้นบรรทัด
        parts = re.split(r"[,\|\;\n]+", t)
        parts = [norm_str(p) for p in parts if norm_str(p)]
        return "ส่วนผสม: " + (", ".join(parts) if parts else norm_str(t))
    if isinstance(v, (list, tuple)):
        parts = [norm_str(str(x)) for x in v if norm_str(str(x))]
        return "ส่วนผสม: " + ", ".join(parts)
    return f"ส่วนผสม: {norm_str(str(v))}"

def fmt_method(v) -> str:
    s = norm_str(v)
    return ("วิธีทำ: " + s) if s else ""

def fmt_tags(v) -> str:
    s = norm_str(v)
    if not s: return ""
    parts = re.split(r"[,\|\;]+", s)
    parts = [p.strip() for p in parts if p.strip()]
    return "แท็ก: " + (", ".join(parts) if parts else s)

def fmt_nutrition(v) -> str:
    if pd.isna(v): return ""
    def from_dict(obj: Dict[str, Any]) -> str:
        keys_map = {
            "energy_kcal":["energy_kcal","calories_kcal","kcal","พลังงาน"],
            "protein_g":["protein_g","protein","โปรตีน"],
            "fat_g":["fat_g","fat","ไขมัน"],
            "carbs_g":["carbs_g","carb_g","carbohydrate_g","คาร์บ","คาร์โบไฮเดรต"],
            "sugar_g":["sugar_g","sugars_g","น้ำตาล"],
            "sodium_mg":["sodium_mg","sodium","โซเดียม"]
        }
        out = {}
        for std, aliases in keys_map.items():
            for a in aliases:
                if a in obj:
                    out[std] = obj[a]
                    break
        parts=[]
        if "energy_kcal" in out: parts.append(f"kcal {out['energy_kcal']}")
        if "protein_g" in out: parts.append(f"โปรตีน {out['protein_g']}g")
        if "fat_g" in out: parts.append(f"ไขมัน {out['fat_g']}g")
        if "carbs_g" in out: parts.append(f"คาร์บ {out['carbs_g']}g")
        if "sugar_g" in out: parts.append(f"น้ำตาล {out['sugar_g']}g")
        if "sodium_mg" in out: parts.append(f"โซเดียม {out['sodium_mg']}mg")
        return "โภชนาการ/ที่: " + ", ".join(parts) if parts else ""

    if isinstance(v, dict):
        return from_dict(v)

    if isinstance(v, str):
        t = v.strip()
        # JSON?
        if (t.startswith("{") and t.endswith("}")) or (t.startswith("[") and t.endswith("]")):
            try:
                obj = json.loads(t)
                if isinstance(obj, dict):
                    return from_dict(obj)
            except Exception:
                pass
        return "โภชนาการ/ที่: " + norm_str(t)

    return "โภชนาการ/ที่: " + norm_str(str(v))

# -----------------------------
# 3) สร้าง child chunks + parent docstore + รายงานคุณภาพ
# -----------------------------
child_texts: List[str] = []
child_meta:  List[Dict[str, Any]] = []
docstore_rows: List[Dict[str, Any]] = []
report_rows: List[Dict[str, Any]] = []

def add_child(text, parent_id, field):
    text = norm_str(text)
    if not text: return
    child_texts.append(text)
    child_meta.append({"parent_id": parent_id, "field": field})

skipped = 0
for i, row in df.iterrows():
    pid = f"ROW_{i}"
    name_raw = row.get(menu_col, "") if menu_col else ""
    ing_raw  = row.get(ing_col, "") if ing_col else ""
    mth_raw  = row.get(meth_col, "") if meth_col else ""
    nut_raw  = row.get(nut_col, "") if nut_col else ""
    tag_raw  = row.get(tag_col, "") if tag_col else ""

    # ฟอร์แมต
    name = norm_str(name_raw)
    ing  = fmt_ingredients(ing_raw) if str(ing_raw).strip() else ""
    mth  = fmt_method(mth_raw)       if str(mth_raw).strip() else ""
    nut  = fmt_nutrition(nut_raw)    if str(nut_raw).strip() else ""
    tag  = fmt_tags(tag_raw)         if str(tag_raw).strip() else ""

    # คุณภาพข้อมูล
    missing_ing = (ing == "")
    missing_mth = (mth == "")
    missing_any = {
        "missing_name": (name == ""),
        "missing_ingredients": missing_ing,
        "missing_method": missing_mth,
        "missing_nutrition": (nut == ""),
        "missing_tags": (tag == "")
    }

    # เงื่อนไขข้ามทั้งแถวถ้าไม่มีทั้ง ingredients และ method
    if SKIP_IF_NO_ING_AND_METHOD and missing_ing and missing_mth:
        skipped += 1
        report_rows.append({"parent_id": pid, "menu_name": name, **missing_any, "skipped": True})
        continue

    # เก็บ parent/ระเบียนเต็มใน docstore (แม้บางฟิลด์ว่าง)
    parent = {
        "parent_id": pid,
        "menu_name": name,
        "ingredients": ing.replace("ส่วนผสม: ","") if ing else "",
        "method": mth.replace("วิธีทำ: ","") if mth else "",
        "nutrition_text": nut.replace("โภชนาการ/ที่: ","") if nut else "",
        "tags": tag.replace("แท็ก: ","") if tag else "",
        "raw": {
            "menu_name": name_raw,
            "ingredients": ing_raw,
            "method": mth_raw,
            "nutrition": nut_raw,
            "tags": tag_raw,
        }
    }
    docstore_rows.append(parent)

    # child: เติมเฉพาะ field ที่มีข้อมูลจริง
    if name: add_child(f"ชื่อเมนู: {name}", pid, "name")
    if ing:  add_child(ing, pid, "ingredients")
    if mth:  add_child(mth, pid, "method")
    if nut:  add_child(nut, pid, "nutrition")
    if tag:  add_child(tag, pid, "tags")

    report_rows.append({"parent_id": pid, "menu_name": name, **missing_any, "skipped": False})

print(f"🧱 child chunks = {len(child_texts)} (parents={len(docstore_rows)}), skipped={skipped}")

# -----------------------------
# 4) สร้าง embedding + FAISS
# -----------------------------
print("⚙️ โหลดโมเดล:", EMBED_MODEL)
embedder = SentenceTransformer(EMBED_MODEL)
X = embedder.encode(child_texts, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=False)

# cosine similarity -> normalize L2 + IndexFlatIP
faiss.normalize_L2(X)
index = faiss.IndexFlatIP(X.shape[1])
index.add(X)

# -----------------------------
# 5) บันทึกไฟล์ทั้งหมด + รายงานคุณภาพ
# -----------------------------
faiss.write_index(index, INDEX_PATH)
with open(TEXTS_PATH, "wb") as f: pickle.dump(child_texts, f)
with open(META_PATH,  "wb") as f: pickle.dump(child_meta,  f)
with open(DOCSTORE,  "w",  encoding="utf-8") as f:
    for r in docstore_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# รายงาน: บอกว่าเมนูไหนขาดฟิลด์อะไร/ถูก skip หรือไม่
pd.DataFrame(report_rows).to_csv(REPORT_CSV, index=False, encoding="utf-8-sig")

elapsed = time.time() - t0
print("✅ DONE")
print(f"📦 Index   : {INDEX_PATH}")
print(f"📝 Texts   : {TEXTS_PATH}")
print(f"📘 Meta    : {META_PATH}")
print(f"📚 Docstore: {DOCSTORE}")
print(f"🧾 Report  : {REPORT_CSV}")
print(f"⏱️ Time    : {elapsed:.2f}s")
