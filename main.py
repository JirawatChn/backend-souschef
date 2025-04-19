from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal, Optional
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from pythainlp.spell import correct
import ollama


def preprocess_input(user_input: str) -> str:
    return correct(user_input.strip())


# โหลดโมเดล & FAISS index
embedder = SentenceTransformer("BAAI/bge-m3")
index = faiss.read_index("thai_recipes.index")
with open("texts.pkl", "rb") as f:
    texts = pickle.load(f)


# ดึง context
def retrieve_context(query, k=3):
    query_vec = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_vec)
    D, I = index.search(query_vec, k)
    return "\n".join([texts[i] for i in I[0]])


# FastAPI setup
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Persona profiles
personality_profiles = {
    "souschef": """คุณคือผู้ช่วยเชฟที่สุภาพ รอบรู้เรื่องอาหารไทย  
อธิบายขั้นตอนชัดเจน เข้าใจง่าย แนะนำเมนูตามวัตถุดิบ พร้อมเทคนิคทำให้อร่อยมั่นใจ 
ใช้คำว่า “ฉัน” ทุกคำแนะนำ""",
    "buddy": """คุณคือเพื่อนซี้สายทำอาหาร พูดตรง ขี้แซว แต่จริงใจ  
คุยแบบเป็นกันเอง พร้อมบ่นแต่ก็ช่วยเต็มที่  
ให้สูตรเข้าใจง่าย มีทริคแถม และพูดเหมือนยืนหน้ากระทะด้วยกัน ใช้คำว่า “ฉัน” ทุกคำแนะนำ""",
    "chef-ian": """คุณคือเชฟเอียนมืออาชีพ พูดนิ่ง สุภาพ จริงจัง  
แนะนำแบบตรงไปตรงมา ใส่ใจรสชาติและเทคนิค  
ใช้คำว่า “ผม” และลงท้ายด้วย “ครับ” ทุกคำแนะนำ""",
}


def generate_response(prompt: str, personality: str = "souschef", lang: str = "th"):
    context = retrieve_context(prompt)
    persona_text = personality_profiles.get(personality, "")

    tone_map = {
        "souschef": "ภาษาสุภาพ เรียบง่าย เหมือนครูใจดี",
        "buddy": "เป็นกันเอง ขำได้ เข้าใจง่าย เหมือนเพื่อน",
        "chef-ian": "นิ่ง สุภาพ จริงจัง แทนตัวเองด้วย 'ผม' ตลอด",
    }
    tone_instruction = tone_map.get(personality, "สุภาพ ชัดเจน เข้าใจง่าย")

    if lang == "en":
        language_instruction = f"""
Reply in English only.

- Include dish name, ingredients, and steps.
- Suggest only related Thai dishes if context is unclear.
- Suggest only one menu per question.

Tone: {tone_instruction}
"""
    else:
        language_instruction = f"""
ตอบไทยเท่านั้น

- ระบุชื่อเมนู, ส่วนผสม, วิธีทำชัดเจน
- ไม่แนะนำเมนูอื่นถ้าไม่ได้ถาม
- ใช้ context ที่มี หรือแนะนำเมนูใกล้เคียง
- ตอบทีละหนึ่งเมนู

โทน: {tone_instruction}
"""

    system_message = f"{persona_text.strip()}\n{language_instruction.strip()}"
    user_message = f"Context:\n{context}\n\nQuestion:\n{prompt}"

    try:
        response = ollama.chat(
            model="gemma3",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
        )
        return response["message"]["content"].strip()
    except Exception as e:
        return f"เกิดข้อผิดพลาด: {str(e)}"


# 🟢 API รับแค่ string + personality
class ChatRequest(BaseModel):
    question: str
    personality: Optional[str] = "souschef"
    lang: Literal["th", "en"] = "th"


@app.post("/ask")
async def ask_question(request: ChatRequest):
    print(request.question)
    corrected_question = preprocess_input(request.question)
    print(corrected_question)
    return {
        "answer": generate_response(
            corrected_question,
            request.personality or "souschef",
            request.lang or "th",
        )
    }
