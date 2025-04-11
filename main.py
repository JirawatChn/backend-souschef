from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal, Optional

import faiss
import pickle
import numpy as np
import subprocess
from sentence_transformers import SentenceTransformer

# โหลดโมเดล & FAISS index
embedder = SentenceTransformer("BAAI/bge-m3")
index = faiss.read_index("thai_recipes.index")
with open("texts.pkl", "rb") as f:
    texts = pickle.load(f)


# 🔍 ดึง context
def retrieve_context(query, k=3):
    query_vec = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_vec)
    D, I = index.search(query_vec, k)
    return "\n".join([texts[i] for i in I[0]])


# 🔥 FastAPI setup
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔑 Persona profiles
personality_profiles = {
    "souschef": """คุณคือผู้ช่วยเชฟผู้เชี่ยวชาญ ที่ตอบคำถามเกี่ยวกับอาหารไทยด้วยความสุภาพ รอบรู้ และให้คำอธิบายที่ชัดเจนราวกับอยู่ในครัวจริง""",
    "buddy": """คุณคือเพื่อนสนิทที่ช่วยแนะนำเมนูอาหารให้ผู้ใช้แบบง่าย ๆ เป็นกันเอง พูดเล่นบ้างได้ ช่วยให้เขารู้สึกสบายใจในการทำอาหาร""",
    "chef-ian": """คุณคือเชฟเอียนจากรายการ MasterChef Thailand เชฟผู้ชายระดับมืออาชีพที่มีบุคลิกนิ่ง สุภาพ และเฉียบขาด  
คุณให้คำแนะนำเรื่องอาหารไทยอย่างจริงจัง ใส่ใจรสชาติ เทคนิค และการจัดจาน  
คุณพูดตรงไปตรงมา มีเหตุผล ไม่เยินยอ และไม่พูดเล่น  
หากอาหารยังไม่ถึงมาตรฐาน คุณจะบอกอย่างชัดเจนโดยไม่อ้อมค้อม  
คุณใช้สรรพนามว่า “ผม” และลงท้ายด้วย “ครับ” เสมอ""",
}


# 🧠 สร้าง prompt
def generate_response(prompt: str, personality: str = "souschef"):
    context = retrieve_context(prompt)
    persona_text = personality_profiles.get(personality, "")

    template = f"""[INST]
{persona_text}

คุณคือผู้ช่วยแนะนำอาหารไทยผ่านแชตอย่างมืออาชีพ

- เมื่อตอบคำถามเกี่ยวกับเมนูอาหาร ให้ตอบทั้งชื่อเมนู, รายการส่วนผสม และขั้นตอนการทำโดยละเอียด
- อย่าตอบเมนูอื่นถ้าไม่ได้รับการถามใหม่
- ใช้ข้อมูลจาก Context เพื่อช่วยให้คำแนะนำละเอียดและแม่นยำ
- ถ้าไม่มี context เพียงพอ ให้ตอบเมนูที่เกี่ยวข้องและให้สูตรอย่างชัดเจน

เมื่อเหมาะสม ให้ถามกลับ เช่น  
“อยากรู้เมนูอื่นเพิ่มเติมไหมครับ?” หรือ “มีวัตถุดิบอื่นที่อยากใช้เพิ่มเติมไหม?”

---

Context:
{context}

คำถาม:
{prompt}
[/INST]"""


    result = subprocess.run(
        ["ollama", "run", "llama3.1", template],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


# 🟢 API รับแค่ string + personality
class ChatRequest(BaseModel):
    question: str
    personality: Optional[str] = "souschef"


@app.post("/ask")
async def ask_question(request: ChatRequest):
    return {
        "answer": generate_response(request.question, request.personality or "souschef")
    }
