from fastapi import FastAPI
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal, Optional
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from pythainlp.spell import correct
from google import genai

# โหลด API KEY จาก .env
client = genai.Client(api_key="AIzaSyCai56noKNPuWd87W1F2slA7ABOS7hrNh8")

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

# สร้าง FastAPI instance
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173","https://jirawatchn.github.io"],  # เปลี่ยนตาม frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# รูปแบบคำขอจาก frontend
class ChatRequest(BaseModel):
    question: str
    personality: Optional[str] = "souschef"
    lang: Literal["th", "en"] = "th"

chat_sessions = {}
chat_histories = {}

@app.post("/ask")
async def ask_question(request: ChatRequest, raw_request: Request):
    session_id = raw_request.headers.get("X-Session-ID")
    if not session_id:
        return {"error": "Missing session ID"}

    if session_id not in chat_sessions:
        chat_sessions[session_id] = client.chats.create(model="gemini-2.0-flash")
    chat = chat_sessions[session_id]

    # ใช้ตัวสะกดอัตโนมัติ
    corrected_question = correct(request.question)

    # ดึง context จาก RAG (ใช้ k=2 หรือ ปรับตามต้องการ)
    context = retrieve_context(corrected_question, k=2)
    
    # เตรียมข้อความ prompt ผสม personality + tone + context + คำถาม
    persona_text = personality_profiles.get(request.personality or "souschef", "")
    tone_map = {
        "souschef": "ภาษาสุภาพ เรียบง่าย เหมือนครูใจดี",
        "buddy": "เป็นกันเอง ขำได้ เข้าใจง่าย เหมือนเพื่อน",
        "chef-ian": "นิ่ง สุภาพ จริงจัง แทนตัวเองด้วย 'ผม' ตลอด",
    }
    tone_instruction = tone_map.get(request.personality or "souschef", "สุภาพ ชัดเจน เข้าใจง่าย")

    if request.lang == "en":
        language_instruction = f"""
** Translate Menu, Tone, Personality and Reply in English only **
You are a cooking assistant that recommends appropriate Thai dishes based on the user's question.

**Only answer food-related questions**
- Recommend only one Thai dish that matches the user's question.
- Do not offer multiple options or alternative dishes unless asked directly.
- Be specific: include the dish name, ingredients, and preparation steps clearly.
- If the context is unclear, suggest the most relevant dish and justify your choice based on facts or logical relevance.
- Do NOT respond with vague or general language.
- If the question is like "What should I eat?" or "Any suggestions?", directly recommend one Thai dish and explain how to make it.

**If you cannot comply with the above rules, respond with "No information available for this question."**

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
- ถ้า "เมนูอาหารที่เกี่ยวข้อง" ไม่ชัดเจน ให้เลือกเมนูอาหารที่ใกล้เคียงที่สุด และต้องอธิบายเหตุผลอย่างมีข้อมูลอ้างอิง
- ตอบทีละหนึ่งเมนูเท่านั้น ห้ามตอบหลายเมนู
- ไม่ตอบนอกเรื่องโดยเด็ดขาด

** หากคำถามเป็นแนว "ทำอะไรกินดี" หรือ "มีเมนูแนะนำไหม" ให้ตอบเมนูอาหารตรง ๆ และวิธีทำ โดยไม่กล่าวถึงบริบทที่ให้ไป**

**หากไม่สามารถตอบโดยยึดตามเงื่อนไขข้างต้นได้ ให้ตอบว่า "ไม่มีข้อมูลเกี่ยวกับคำถามนี้"**

โทน: {tone_instruction}
"""

    full_prompt = f"""{persona_text.strip()}

{language_instruction.strip()}

เมนูอาหารที่เกี่ยวข้องกับคำถามของผู้ใช้:
{context}

คำถาม:
{corrected_question}
"""

    print(f"[DEBUG] Session ID: {session_id}")
    print(f"[DEBUG] Prompt to Gemini:\n{full_prompt}")

    # ส่ง prompt เข้า chat session ของ Gemini
    response = chat.send_message(full_prompt)

    print(f"[DEBUG] AI answer: {response.text.strip()}")

    return {
        "answer": response.text.strip()
    }
