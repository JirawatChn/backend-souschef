import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama

# === CONFIG ===
MODEL_NAME = "llama3.1"

# === APP SETUP ===
app = FastAPI()

# === LOAD DATA ===
df = pd.read_csv("thai_recipes_processed.csv")

def create_full_context(df: pd.DataFrame) -> str:
    context_list = []
    for index, row in df.iterrows():
        entry = f"ชื่อเมนู: {row.get('menu_name', f'unknown_{index}')}\n"
        entry += f"วัตถุดิบ: {row.get('ingredients_tokens', 'ไม่มีข้อมูล')}\n"
        entry += f"วิธีทำ: {row.get('method_tokens', 'ไม่มีข้อมูล')}\n"
        context_list.append(entry)
    return "\n\n".join(context_list)

FULL_CONTEXT = create_full_context(df)

# === LOAD LLM ===
llm = Ollama(model=MODEL_NAME)

# === PROMPT ===
prompt = PromptTemplate.from_template("""
คุณคือผู้ช่วยด้านอาหารไทย ฉันจะให้ข้อมูลเกี่ยวกับสูตรอาหารทั้งหมดแก่คุณในรูปแบบด้านล่าง

{context}

ตอนนี้ให้ตอบคำถามนี้โดยใช้ข้อมูลข้างต้นเท่านั้น:

{question}
""")

# === INPUT MODEL ===
class QuestionInput(BaseModel):
    question: str

# === ROUTE ===
@app.post("/ask")
async def ask_question(input: QuestionInput):
    full_prompt = prompt.format(context=FULL_CONTEXT, question=input.question)
    answer = llm.invoke(full_prompt)
    return {"answer": answer}
