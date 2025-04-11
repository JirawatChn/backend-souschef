import pandas as pd
import mysql.connector
from sentence_transformers import SentenceTransformer
import json
import numpy as np

# โหลดโมเดล
embedder = SentenceTransformer("BAAI/bge-m3")

# ฟังก์ชันเพิ่มข้อมูลลงใน MySQL
def add_document(conn, text):
    cur = conn.cursor()
    embedding = embedder.encode(text).tolist()
    embedding_str = "[" + ",".join(map(str, embedding)) + "]"

    sql = f"""
    INSERT INTO documents (document, embedding)
    VALUES (%s, CAST(%s AS VECTOR(1024)))
    """
    cur.execute(sql, (text, embedding_str))
    conn.commit()
    cur.close()

def cosine_similarity(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def query_db(conn, query_text, k=5):
    query_vec = embedder.encode(query_text).tolist()

    cur = conn.cursor()
    cur.execute("SELECT document, embedding FROM documents")
    rows = cur.fetchall()
    cur.close()

    results = []
    for doc, vec in rows:
        # ตรวจสอบว่า vec เป็น list แล้ว ถ้าเก็บแบบ JSON string ต้องแปลง
        if isinstance(vec, str):
            try:
                vec = json.loads(vec)
            except:
                continue  # ข้ามรายการที่พัง
        if len(vec) != len(query_vec):
            continue  # ข้ามเวกเตอร์ที่ผิดมิติ
        score = cosine_similarity(query_vec, vec)
        results.append((doc, score))

    # เรียงจากคะแนนมาก → น้อย (คล้ายที่สุด)
    results.sort(key=lambda x: x[1], reverse=True)

    print(f"\n🔍 Top {k} matches for: \"{query_text}\"")
    for i, (doc, score) in enumerate(results[:k], 1):
        print(f"{i}. (Score: {score:.4f}) → {doc[:80]}...")

    return results[:k]

# เชื่อมต่อฐานข้อมูล
conn = mysql.connector.connect(
    host = "gateway01.ap-southeast-1.prod.aws.tidbcloud.com",
    port = 4000,
    user = "jLeDsgMKVeyUMrJ.root",
    password = "37dji5x6wrBl1NSN",
    database = "souschef_db",
    ssl_ca = "isrgrootx1.pem",
    ssl_verify_cert = True,
    ssl_verify_identity = True
)

# # # โหลดไฟล์ CSV
# df = pd.read_csv("thai_recipes_processed.csv")

# # Loop ใส่เฉพาะ column 'context' พร้อม print log
# for i, context in enumerate(df['context'].dropna(), start=1):
#     add_document(conn, context)
#     print(f"[{i}/{len(df)}] Inserted context: {context[:60]}...")  # แสดงแค่ 60 ตัวอักษรแรก

# conn.close()
# print("✅ Done inserting all documents.")

result = query_db(conn, "ข้าวผัด", k=5)
print(result)