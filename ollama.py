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

# โหลดโมเดลฝังเวกเตอร์และ FAISS index
embedder = SentenceTransformer("BAAI/bge-m3")
index = faiss.read_index("thai_recipes.index")
with open("texts.pkl", "rb") as f:
    texts = pickle.load(f)

# ดึง context ที่ใกล้เคียง
def retrieve_context(query, k=3):
    query_vec = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_vec)
    D, I = index.search(query_vec, k)
    return "\n".join([texts[i] for i in I[0]])

# personality profiles
personality_profiles = {
    "souschef": """คุณคือผู้ช่วยเชฟที่สุภาพ รอบรู้เรื่องอาหารไทย  
อธิบายขั้นตอนชัดเจน เข้าใจง่าย แนะนำเมนูตามวัตถุดิบ พร้อมเทคนิคทำให้อร่อยมั่นใจ 
ใช้คำว่า “ฉัน” ทุกคำแนะนำ""",
    "buddy": """คุณคือเพื่อนซี้สายทำอาหาร พูดตรง ขี้แซว แต่จริงใจ  
คุยแบบเป็นกันเอง พร้อมบ่นแต่ก็ช่วยเต็มที่  
ให้สูตรเข้าใจง่าย มีทริคแถม และพูดเหมือนยืนหน้ากระทะด้วยกัน ใช้คำว่า “ฉัน” ทุกคำแนะนำ""",
    "chef-ian": """คุณคือเชฟเอียน Masterchef พูดนิ่ง สุภาพ จริงจัง  
แนะนำแบบตรงไปตรงมา ใส่ใจรสชาติและเทคนิค  
ใช้คำว่า “ผม” และลงท้ายด้วย “ครับ” ทุกคำแนะนำ""",
}

def generate_response(prompt: str, personality: str = "souschef", lang: str = "th"):
    context = retrieve_context(prompt, k=1) 
    print(f"[DEBUG] Context token length: {len(''.join(context))}") 

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
ตอบภาษาไทยเท่านั้น
คุณคือผู้ช่วยที่จะคอยแนะนำเมนูอาหารไทยที่เหมาะสมกับคำถามของผู้ใช้

**ตอบเฉพาะเรื่องอาหารเท่านั้น**

- ตอบเฉพาะเมนูอาหารที่เกี่ยวข้องกับคำถามเท่านั้น
- ห้ามเสนอตัวเลือกอื่น หรือแนะนำเมนูอื่นถ้าไม่ได้ถามโดยตรง
- ห้ามใช้ภาษาคลุมเครือ ให้ระบุชื่อเมนู, ส่วนผสม, และวิธีทำอย่างชัดเจน
- ถ้า context ไม่ชัดเจน ให้เลือกเมนูอาหารที่ใกล้เคียงที่สุด และต้องอธิบายเหตุผลอย่างมีข้อมูลอ้างอิง
- ตอบทีละหนึ่งเมนูเท่านั้น ห้ามตอบหลายเมนู
- ไม่ตอบนอกเรื่องโดยเด็ดขาด

**หากไม่สามารถตอบโดยยึดตามเงื่อนไขข้างต้นได้ ให้ตอบว่า "ไม่มีข้อมูลเกี่ยวกับคำถามนี้"**

โทน: {tone_instruction}
"""

    system_message = f"{persona_text.strip()}\n{language_instruction.strip()}"
    
    # เพิ่มข้อความที่ทำให้ LLM เข้าใจว่า context คือข้อมูลที่ให้คำตอบเกี่ยวกับเมนูอาหาร
    user_message = f"ข้อมูลที่เกี่ยวข้อง: {''.join(context)}\n\nคำถาม: {prompt}"
    print(lang)
    print(language_instruction)

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



# สร้าง FastAPI instance
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# รูปแบบคำขอจาก frontend
class ChatRequest(BaseModel):
    question: str
    personality: Optional[str] = "souschef"
    lang: Literal["th", "en"] = "th"

@app.post("/ask")
async def ask_question(request: ChatRequest):
    corrected_question = correct(request.question)
    return {
        "answer": generate_response(
            corrected_question,
            request.personality or "souschef",
            request.lang or "th",
        )
    }
