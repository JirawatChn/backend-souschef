# test_retrieval_like_llm.py
# ดึงจากเวกเตอร์ แล้วประกอบคำตอบ "สไตล์ LLM" โดยไม่เรียก LLM
# เน้น ingredients > tags (ingredients=1.5, tags=0.5)
# และแสดงรายละเอียดเมนูสำรอง (ส่วนผสม/วิธีทำ/โภชนาการ) พร้อมตัวเลือกตัดความยาว

import os, json, pickle, sys, re
from typing import List, Dict, Any, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import argparse

# ---- พาธไฟล์ RAG ----
INDEX_PATH = "thai_recipes_minilm.index"
TEXTS_PATH = "child_texts.pkl"
META_PATH  = "child_meta.pkl"
DOCSTORE   = "menus_docstore.jsonl"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# ---- น้ำหนักฟิลด์ (boost) ----
FIELD_WEIGHTS = {
    "ingredients": 1.50,  # เน้นส่วนผสม
    "method": 1.10,
    "name": 1.00,
    "tags": 0.50,         # ลดบทบาทแท็ก
    "nutrition": 0.90,
}
DEFAULT_FIELD_WEIGHT = 1.0

def load_resources():
    assert os.path.exists(INDEX_PATH), f"ไม่พบไฟล์ {INDEX_PATH}"
    assert os.path.exists(TEXTS_PATH), f"ไม่พบไฟล์ {TEXTS_PATH}"
    assert os.path.exists(META_PATH),  f"ไม่พบไฟล์ {META_PATH}"
    assert os.path.exists(DOCSTORE),   f"ไม่พบไฟล์ {DOCSTORE}"

    index = faiss.read_index(INDEX_PATH)
    child_texts: List[str] = pickle.load(open(TEXTS_PATH, "rb"))
    child_meta:  List[Dict[str, Any]] = pickle.load(open(META_PATH,  "rb"))

    docstore: Dict[str, Dict[str, Any]] = {}
    with open(DOCSTORE, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            docstore[row["parent_id"]] = row

    embedder = SentenceTransformer(EMBED_MODEL)
    return index, child_texts, child_meta, docstore, embedder

def retrieve_parents_with_match(
    query: str,
    index,
    child_texts: List[str],
    child_meta: List[Dict[str, Any]],
    docstore: Dict[str, Dict[str, Any]],
    embedder,
    k_children: int = 40,
    top_parents: int = 3,
) -> List[Dict[str, Any]]:
    """คืน parent พร้อม child ที่ให้คะแนนสูงสุด (มี weighted score)"""
    q = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=False)
    faiss.normalize_L2(q)
    D, I = index.search(q, k_children)

    parent_best: Dict[str, Tuple[float, int, float]] = {}  # pid -> (weighted, child_idx, raw)
    for rank, idx in enumerate(I[0]):
        meta = child_meta[idx]
        pid = meta["parent_id"]
        field = meta.get("field")
        raw_cos = float(D[0][rank])
        w = FIELD_WEIGHTS.get(field, DEFAULT_FIELD_WEIGHT)
        weighted = raw_cos * w
        if (pid not in parent_best) or (weighted > parent_best[pid][0]):
            parent_best[pid] = (weighted, idx, raw_cos)

    items = sorted(parent_best.items(), key=lambda kv: kv[1][0], reverse=True)[:top_parents]

    results = []
    for pid, (weighted, idx, raw_cos) in items:
        parent = docstore.get(pid, {})
        cm = child_meta[idx]
        results.append({
            "parent_id": pid,
            "score": raw_cos,
            "weighted_score": weighted,
            "best_child_field": cm.get("field"),
            "best_child_text": child_texts[idx],
            "parent": parent
        })
    return results

# ------------ Utilities for composing output ------------
KW_GROUPS = {
    "คลีน": ["คลีน","clean","สุขภาพ","นึ่ง","ย่าง","โฮลวีต","ต้มจืด","อกไก่","โยเกิร์ต","ธัญพืช","คีนัว"],
    "ไม่เผ็ด": ["ไม่เผ็ด","จืด","เด็กกินได้","ซุปใส","ต้มจืด"],
    "เผ็ด": ["เผ็ด","ต้มยำ","แกงเผ็ด","พริก","แซ่บ","น้ำตก","พะแนง","เขียวหวาน","ผัดพริก"],
    "อาหารเช้า": ["อาหารเช้า","โจ๊ก","ข้าวต้ม","แซนด์วิช","ไข่กระทะ","โยเกิร์ต","แพนเค้ก","วาฟเฟิล","ซีเรียล"],
    "โปรตีนสูง": ["โปรตีนสูง","อกไก่","เต้าหู้","ไข่","ปลา","ทูน่า","แซลมอน"],
    "ซีฟู้ด": ["ซีฟู้ด","กุ้ง","ปลาหมึก","หอย","ปลา","แซลมอน","ทูน่า"],
}

def extract_intent_keywords(query: str) -> List[str]:
    q = query.lower()
    hits = []
    for label, kws in KW_GROUPS.items():
        if any(k.lower() in q for k in kws):
            hits.append(label)
    return hits

def truncate(text: str, limit: int) -> str:
    if not limit or limit <= 0:
        return text or "-"
    t = (text or "-").strip()
    return (t[:limit] + "…") if len(t) > limit else t

def compose_answer_like_llm(query: str, results: List[Dict[str, Any]], alts_count: int = 2, alt_limit: int = 0) -> str:
    """alt_limit=0 คือไม่ตัด; >0 จะตัด chars ต่อฟิลด์ของเมนูสำรอง"""
    if not results:
        return f"🔎 คำค้น: {query}\nไม่พบเมนูที่เกี่ยวข้องในฐานข้อมูล"

    intents = extract_intent_keywords(query)
    top = results[0]
    alts = results[1:1+alts_count]

    p = top["parent"]
    field = top.get("best_child_field")
    reason_bits = []
    if field == "ingredients":
        reason_bits.append("ส่วนผสมสอดคล้องกับคำค้น")
    elif field == "method":
        reason_bits.append("วิธีทำสอดคล้องกับคำค้น")
    elif field == "tags":
        reason_bits.append("แท็กของเมนูเข้ากับโจทย์")
    elif field == "name":
        reason_bits.append("ชื่อเมนูตรงกับคำค้น")
    elif field == "nutrition":
        reason_bits.append("เงื่อนไขโภชนาการสอดคล้อง")
    if intents:
        reason_bits.append("ตรงกับเจตนาคำค้น: " + ", ".join(intents))
    reason = " / ".join(reason_bits) if reason_bits else "คะแนนความใกล้เคียงสูงสุดจากฐานข้อมูล"

    lines = []
    lines.append(f"🔎 คำค้น: {query}\n")
    lines.append("✅ เมนูที่แนะนำ")
    lines.append(f"ชื่อ: {p.get('menu_name','-')}")
    lines.append(
        f"เหตุผลที่เลือก: {reason} "
        f"(cosine={top['score']:.4f}, weighted={top['weighted_score']:.4f}, match field={field})"
    )
    lines.append(f"แท็ก: {p.get('tags','-')}")
    lines.append(f"ส่วนผสม:\n{p.get('ingredients','-') or '-'}")
    lines.append(f"\nวิธีทำ:\n{p.get('method','-') or '-'}")
    lines.append(f"\nโภชนาการต่อ 1 ที่: {p.get('nutrition_text','-') or 'ไม่มีข้อมูล'}")
    lines.append(f"SOURCE_ID: {top.get('parent_id','-')}\n")

    if alts:
        lines.append("🟰 ตัวเลือกสำรอง")
        for i, r in enumerate(alts, 1):
            pp = r["parent"]
            fld = r.get("best_child_field","-")
            reason2 = f"match={fld}, cosine={r['score']:.4f}, weighted={r['weighted_score']:.4f}"
            lines.append(f"\n{i}. {pp.get('menu_name','-')}  | แท็ก: {pp.get('tags','-')}  | {reason2}")
            # แสดงรายละเอียด (ตัดความยาวตาม alt_limit ถ้ากำหนด)
            lines.append(f"ส่วนผสม:\n{truncate(pp.get('ingredients','-'), alt_limit)}")
            lines.append(f"\nวิธีทำ:\n{truncate(pp.get('method','-'), alt_limit)}")
            lines.append(f"\nโภชนาการต่อ 1 ที่: {truncate(pp.get('nutrition_text','-'), alt_limit)}")
            lines.append(f"SOURCE_ID: {r.get('parent_id','-')}")
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="*", help="คำค้นหา (ภาษาไทย/อังกฤษ)")
    parser.add_argument("--json", action="store_true", help="พิมพ์ผลในรูป JSON สำหรับโปรแกรมอ่านต่อ")
    parser.add_argument("--k_children", type=int, default=40, help="จำนวน child ที่ดึงมาก่อนคัด parent")
    parser.add_argument("--top_parents", type=int, default=3, help="จำนวน parent ทั้งหมด (รวมเมนูหลัก+สำรอง)")
    parser.add_argument("--alts", type=int, default=2, help="จำนวนเมนูสำรองที่ต้องการแสดง")
    parser.add_argument("--alt_limit", type=int, default=0, help="จำกัดอักขระต่อฟิลด์ในเมนูสำรอง (0=ไม่ตัด)")
    args = parser.parse_args()

    index, child_texts, child_meta, docstore, embedder = load_resources()
    query = " ".join(args.query) if args.query else "เมนูอกไก่ คลีน อาหารเช้า ไม่เผ็ด"

    results = retrieve_parents_with_match(
        query, index, child_texts, child_meta, docstore, embedder,
        k_children=args.k_children, top_parents=args.top_parents
    )

    if args.json:
        payload = {"query": query, "results": results}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(compose_answer_like_llm(query, results, alts_count=args.alts, alt_limit=args.alt_limit))

if __name__ == "__main__":
    main()
