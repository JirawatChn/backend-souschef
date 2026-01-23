# -*- coding: utf-8 -*-
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal, Optional, List, Dict, Any
import os, json, pickle, re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pythainlp.spell import correct
from google import genai
from dotenv import load_dotenv

# =========================================================
# 0) โหลด Gemini + โมเดล embed + RAG (FAISS) + Docstore
# =========================================================
# ใช้ env var: GEMINI_API_KEY
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY env variable")

client = genai.Client(api_key=GEMINI_API_KEY)

EMBED_MODEL = "BAAI/bge-m3"
embedder = SentenceTransformer(EMBED_MODEL)

INDEX_PATH = "thai_recipes_bge_m3.index"
TEXTS_PATH = "child_texts.pkl"
META_PATH  = "child_meta.pkl"
DOCSTORE   = "menus_docstore.jsonl"

# FAISS index + child views
index = faiss.read_index(INDEX_PATH)
child_texts: List[str] = pickle.load(open(TEXTS_PATH, "rb"))
child_meta: List[Dict[str, Any]] = pickle.load(open(META_PATH,  "rb"))

# parent docstore: {parent_id: parent_json}
docstore: Dict[str, Dict[str, Any]] = {}
with open(DOCSTORE, "r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        docstore[row["parent_id"]] = row


# =========================================================
# 1) Retrieval: child -> unique parent (จำกัด 3 เมนู)
# =========================================================
def retrieve_parents(query: str, k_children: int = 40, top_parents: int = 3) -> List[Dict[str, Any]]:
    q = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=False)
    faiss.normalize_L2(q)
    D, I = index.search(q, k_children)

    seen, parents = set(), []
    for idx in I[0]:
        pid = child_meta[idx]["parent_id"]
        if pid in seen:
            continue
        seen.add(pid)
        par = docstore.get(pid)
        if par:
            parents.append(par)
        if len(parents) >= top_parents:
            break
    return parents


def build_context_blocks(parents: List[Dict[str, Any]]) -> str:
    """จัดรูปแบบ context: แสดงเต็ม ทั้งส่วนผสม/วิธีทำ/โภชนาการ (ไม่ย่อ) — ไม่มี SOURCE_ID แล้ว"""
    blocks = []
    for p in parents:
        name = p.get("menu_name", "")
        ing  = p.get("ingredients", "")
        mth  = p.get("method", "")
        nut  = p.get("nutrition_text", "")
        tags = p.get("tags", "")
        block = (
f"""[MENU]
- ชื่อ: {name}
- แท็ก: {tags or "-"}
- ส่วนผสมหลัก: {ing if ing else "-"}
- วิธีทำ: {mth if mth else "-"}
- โภชนาการ: {nut or "ไม่มีข้อมูล"}"""
        )
        blocks.append(block)
    return "\n\n".join(blocks) if blocks else ""


# =========================================================
# 2) บุคลิก/โทน (persona) และ mapping โทนภาษา
# =========================================================
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
tone_map = {
    "souschef": "ภาษาสุภาพ เรียบง่าย เหมือนครูใจดี",
    "buddy": "เป็นกันเอง ขำได้ เข้าใจง่าย เหมือนเพื่อน",
    "chef-ian": "นิ่ง สุภาพ จริงจัง แทนตัวเองด้วย 'ผม' ตลอด",
}


# =========================================================
# 3) Nutrition: WHO daily guideline + Parser + Evaluator
# =========================================================
DAILY_GUIDE = {
    "energy_kcal": 2000.0,        # เป้าพลังงานต่อวัน
    "fat_g_max": 65.0,            # ไขมันรวม ≤ 30% ≈ 65g
    "satfat_g_max": 20.0,         # ไขมันอิ่มตัว ≤ 10% ≈ 20g (ถ้าข้อมูลมี)
    "transfat_g_max": 2.0,        # ไขมันทรานส์ < 1% ≈ 2g (ถ้าข้อมูลมี)
    "sugar_g_max": 50.0,          # น้ำตาลอิสระ ≤ 10% = 50g
    "sugar_g_ideal": 25.0,        # เป้าดีที่สุด <5% = 25g
    "sodium_mg_max": 2000.0,      # โซเดียม ≤ 2000 mg/วัน
    "fiber_g_min": 25.0,          # ใยอาหาร ≥ 25g (ถ้าข้อมูลมี)
    "potassium_mg_min": 3500.0,   # โพแทสเซียม ≥ 3500 mg (ถ้าข้อมูลมี)
    "protein_g_min": 50.0,        # โปรตีนช่วง ~50–75g (ประเมินขั้นต่ำ)
    "protein_g_max": 75.0,
}

def parse_nutrition_text(nut_text: str) -> Dict[str, Optional[float]]:
    """แยกค่า kcal, protein_g, fat_g, carb_g, sugar_g, sodium_mg จากข้อความโภชนาการ"""
    if not nut_text:
        return {
            "kcal": None, "protein_g": None, "fat_g": None, "carb_g": None,
            "sugar_g": None, "sodium_mg": None
        }
    text = nut_text.lower()

    def find_number(patterns, unit_multiplier=1.0):
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                try:
                    return float(m.group(1)) * unit_multiplier
                except:
                    continue
        return None

    kcal = find_number([
        r"kcal\s*([0-9]+(?:\.[0-9]+)?)",
        r"พลังงาน\s*([0-9]+(?:\.[0-9]+)?)\s*kcal",
        r"พลังงาน\s*([0-9]+(?:\.[0-9]+)?)"
    ])
    protein_g = find_number([
        r"โปรตีน\s*([0-9]+(?:\.[0-9]+)?)\s*g",
        r"protein\s*([0-9]+(?:\.[0-9]+)?)\s*g",
        r"โปรตีน\s*([0-9]+(?:\.[0-9]+)?)"
    ])
    fat_g = find_number([
        r"ไขมัน\s*([0-9]+(?:\.[0-9]+)?)\s*g",
        r"fat\s*([0-9]+(?:\.[0-9]+)?)\s*g",
        r"ไขมัน\s*([0-9]+(?:\.[0-9]+)?)"
    ])
    carb_g = find_number([
        r"คาร์บ(?:โฮไฮเดรต)?\s*([0-9]+(?:\.[0-9]+)?)\s*g",
        r"carb(?:ohydrate)?s?\s*([0-9]+(?:\.[0-9]+)?)\s*g",
        r"คาร์บ(?:โฮไฮเดรต)?\s*([0-9]+(?:\.[0-9]+)?)"
    ])
    sugar_g = find_number([
        r"น้ำตาล\s*([0-9]+(?:\.[0-9]+)?)\s*g",
        r"sugar\s*([0-9]+(?:\.[0-9]+)?)\s*g",
        r"น้ำตาล\s*([0-9]+(?:\.[0-9]+)?)"
    ])
    sodium_mg = find_number([
        r"โซเดียม\s*([0-9]+(?:\.[0-9]+)?)\s*mg",
        r"sodium\s*([0-9]+(?:\.[0-9]+)?)\s*mg"
    ], unit_multiplier=1.0)
    if sodium_mg is None:
        sodium_mg = find_number([
            r"โซเดียม\s*([0-9]+(?:\.[0-9]+)?)\s*g",
            r"sodium\s*([0-9]+(?:\.[0-9]+)?)\s*g"
        ], unit_multiplier=1000.0)

    return {
        "kcal": kcal,
        "protein_g": protein_g,
        "fat_g": fat_g,
        "carb_g": carb_g,
        "sugar_g": sugar_g,
        "sodium_mg": sodium_mg
    }

def evaluate_against_guidelines(nut: Dict[str, Optional[float]]) -> Dict[str, Dict[str, Any]]:
    """เทียบค่าที่ parse ได้กับเกณฑ์ WHO รายวัน → คืนเปอร์เซ็นต์/ช่วง/สถานะ"""
    out: Dict[str, Dict[str, Any]] = {}
    if nut.get("kcal") is not None:
        pct = (nut["kcal"] / DAILY_GUIDE["energy_kcal"]) * 100.0
        out["energy_kcal"] = {"value": round(nut["kcal"], 2), "percent_of_daily": round(pct, 1), "status": "OK" if pct <= 100 else "HIGH"}
    if nut.get("fat_g") is not None:
        pct = (nut["fat_g"] / DAILY_GUIDE["fat_g_max"]) * 100.0
        out["fat_g"] = {"value": round(nut["fat_g"], 2), "percent_of_daily_max": round(pct, 1), "status": "OK" if pct <= 100 else "HIGH"}
    if nut.get("protein_g") is not None:
        v = nut["protein_g"]
        status = "LOW" if v < DAILY_GUIDE["protein_g_min"] else ("HIGH" if v > DAILY_GUIDE["protein_g_max"] else "OK")
        out["protein_g"] = {"value": round(v, 2), "range_reco_g": [DAILY_GUIDE["protein_g_min"], DAILY_GUIDE["protein_g_max"]], "status": status}
    if nut.get("sugar_g") is not None:
        v = nut["sugar_g"]
        pct10 = (v / DAILY_GUIDE["sugar_g_max"]) * 100.0
        pct5  = (v / DAILY_GUIDE["sugar_g_ideal"]) * 100.0
        status = "GREAT" if v <= DAILY_GUIDE["sugar_g_ideal"] else ("OK" if v <= DAILY_GUIDE["sugar_g_max"] else "HIGH")
        out["sugar_g"] = {"value": round(v, 2), "percent_of_10pct_cap": round(pct10, 1), "percent_of_5pct_ideal": round(pct5, 1), "status": status}
    if nut.get("sodium_mg") is not None:
        pct = (nut["sodium_mg"] / DAILY_GUIDE["sodium_mg_max"]) * 100.0
        out["sodium_mg"] = {"value": round(nut["sodium_mg"], 2), "percent_of_daily_max": round(pct, 1), "status": "OK" if pct <= 100 else "HIGH"}
    return out


# =========================================================
# 4) Language-specific system instruction (TH/EN/cn)
#    — ไม่มี Source อีกต่อไป + เพิ่ม Comparison section
# =========================================================
def make_language_instruction(lang: str, tone_instruction: str) -> str:
    if lang == "en":
        return f"""
Answer in English only.

ROLE
You are a Thai cooking assistant. Choose exactly ONE dish from the provided CONTEXT.

STRICT RULES
- Use ONLY the given CONTEXT and the NUTRITION_GUIDE block below. Do NOT invent new facts.
- Always include NUTRITION. If any value is missing, write "Not available" (do not guess).
- Compute a comparison against the NUTRITION_GUIDE: show % of daily cap/range and flag each item as OK/HIGH/LOW/GREAT.
- If the user’s question does NOT ask for cooking steps (e.g. doesn’t contain words like “how to cook”, “steps”, “recipe”, or “method”), OMIT the “Steps” section completely.
- Output sections in this order:
  1) Dish: <name>
  2) Ingredients: bullet list (from CONTEXT only)
  3) [OPTIONAL] Steps: numbered steps (only if requested)
  4) Nutrition: kcal, protein(g), fat(g), carbs(g), sugar(g), sodium(mg); mark "Not available" if missing.
  5) Comparison to daily recommendations (WHO): energy, fat, sugar, sodium, protein → show percentage and status.

STYLE
- {tone_instruction}
- Be concise but complete.
"""

    # Chinese (strict Chinese-only)
    if lang.lower() in ("cn", "zh", "zh-cn", "zh-tw"):
        return f"""
仅使用中文回答（除了数字、单位、百分号、括号与常见拉丁计量缩写外，不得出现任何非中文字符）。
若 CONTEXT 中含有泰文或英文，必须**翻译为中文**后再输出，不得保留原文。

角色
你是泰国菜烹饪助手。从提供的 CONTEXT 中**只选择一道**最相关的菜。

严格规则
- 仅使用 CONTEXT 与下方 NUTRITION_GUIDE，不要臆造任何新事实。
- 必须给出“营养”。若缺少，写“无数据”，不要猜测。
- 依据 NUTRITION_GUIDE 计算对比：给出每日上限/范围百分比，并标注 OK/HIGH/LOW/GREAT。
- 若用户问题**未提及做法或步骤**（没有“做法/步骤/recipe/how to cook”等词），请完全省略“步骤”部分。
- 输出顺序：
  1) 菜名：<name>（用中文表述）
  2) 原料：项目符号（仅来自 CONTEXT；若为泰文/英文，请翻译成中文）
  3) 【可选】步骤：编号步骤（仅在问题涉及做法时；内容来自 CONTEXT，翻译为中文）
  4) 营养：kcal、蛋白质(g)、脂肪(g)、碳水(g)、糖(g)、钠(mg)；缺失写“无数据”
  5) 与每日建议（WHO）的对比：能量、脂肪、糖、钠、蛋白质 → 百分比 + 状态

语言自检（务必执行）
- 在输出前自查：不得出现任何泰文字母（U+0E00–U+0E7F）或非必要的外文。
- 若检测到非中文文本，须立即改写为中文后再输出。

文风
- {tone_instruction}
- 简洁但信息完整。
"""

    # default: Thai
    return f"""
ตอบภาษาไทยเท่านั้น

บทบาท
คุณคือผู้ช่วยทำอาหารไทย เลือกเมนูที่ตรงที่สุดจาก CONTEXT เพียง **หนึ่งเมนู**

กติกาเคร่งครัด
- ใช้ข้อมูลจาก CONTEXT และบล็อก NUTRITION_GUIDE ด้านล่างเท่านั้น ห้ามแต่งหรือเดา
- ต้องแสดงโภชนาการ ถ้าค่าบางตัวไม่มี ให้เขียนว่า “ไม่มีข้อมูล”
- คำนวณการเปรียบเทียบกับเกณฑ์รายวัน (WHO): แสดงเปอร์เซ็นต์เมื่อเทียบกับเพดาน/ช่วง และระบุสถานะ OK/HIGH/LOW/GREAT ต่อรายการ
- ถ้าคำถามของผู้ใช้ **ไม่ได้ถามถึงวิธีทำหรือขั้นตอน** (เช่น ไม่มีคำว่า “วิธีทำ”, “ทำยังไง”, “how to cook”, “recipe”, “做法”, “步骤”) ให้ละเว้นส่วน “วิธีทำ” ออก
- รูปแบบคำตอบ:
  1) ชื่อเมนู: <name>
  2) ส่วนผสม: หัวข้อย่อย (จาก CONTEXT เท่านั้น)
  3) [ถ้ามีการถาม] วิธีทำ: ลำดับข้อ (จาก CONTEXT เท่านั้น)
  4) โภชนาการ: kcal, โปรตีน(g), ไขมัน(g), คาร์บ(g), น้ำตาล(g), โซเดียม(mg); ถ้าไม่มีให้เขียน “ไม่มีข้อมูล”
  5) เปรียบเทียบกับคำแนะนำรายวัน (WHO): พลังงาน, ไขมัน, น้ำตาล, โซเดียม, โปรตีน → ใส่เปอร์เซ็นต์และสถานะ

โทนภาษา
- {tone_instruction}
- กะทัดรัด แต่ครบถ้วน
"""

# =========================================================
# 5) FastAPI App + Schemas + Session
# =========================================================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://jirawatchn.github.io"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str
    personality: Optional[str] = "souschef"
    lang: Literal["th", "en", "cn"] = "th"

chat_sessions: Dict[str, Any] = {}  # session_id -> genai chat handle


# =========================================================
# 6) Endpoint: /ask
# =========================================================
@app.post("/ask")
async def ask_question(request: ChatRequest, raw_request: Request):
    session_id = raw_request.headers.get("X-Session-ID")
    if not session_id:
        return {"error": "Missing session ID"}

    # lazy create Gemini session
    if session_id not in chat_sessions:
        chat_sessions[session_id] = client.chats.create(model="gemini-2.0-flash")
    chat = chat_sessions[session_id]

    # auto-correct spell (ไทย): ภาษาอื่นจะไม่ได้รับผลกระทบ
    corrected_question = correct(request.question) if request.lang == "th" else request.question

    # ===== RAG: ดึง 3 เมนูแบบ parent & สร้าง context =====
    parents = retrieve_parents(corrected_question, k_children=40, top_parents=3)
    context_blocks = build_context_blocks(parents)
    if not context_blocks:
        context_blocks = "ไม่พบเมนูที่เกี่ยวข้องในฐานข้อมูล / No related menu found."

    # ===== Parse & Evaluate nutrition ต่อ parent (ไว้ใช้ใน UI/ตรวจสอบ) =====
    parsed_and_eval = []
    for p in parents:
        pid = p.get("parent_id", "")
        nut_text = p.get("nutrition_text", "") or p.get("nutrition", "") or ""
        parsed = parse_nutrition_text(nut_text)
        evaled = evaluate_against_guidelines(parsed)
        parsed_and_eval.append({
            "parent_id": pid,
            "menu_name": p.get("menu_name", ""),
            "nutrition_text_raw": nut_text,
            "nutrition_parsed": parsed,
            "nutrition_eval": evaled
        })

    # persona & tone
    persona_text = personality_profiles.get(request.personality or "souschef", "")
    tone_instruction = tone_map.get(request.personality or "souschef", "สุภาพ ชัดเจน เข้าใจง่าย")

    # language-specific rules
    language_instruction = make_language_instruction(request.lang, tone_instruction)

    # ===== Inject NUTRITION_GUIDE ให้ LLM ใช้ในการเทียบ =====
    guide_block = f"""
NUTRITION_GUIDE (Adults, WHO — daily targets/caps):
- Energy: {DAILY_GUIDE["energy_kcal"]} kcal/day (contextual)
- Fat (total): ≤ {DAILY_GUIDE["fat_g_max"]} g/day
- Saturated fat: ≤ {DAILY_GUIDE["satfat_g_max"]} g/day (if available)
- Trans fat: < {DAILY_GUIDE["transfat_g_max"]} g/day (if available)
- Free sugar: ≤ {DAILY_GUIDE["sugar_g_max"]} g/day (ideal < {DAILY_GUIDE["sugar_g_ideal"]} g)
- Sodium: ≤ {DAILY_GUIDE["sodium_mg_max"]} mg/day
- Protein: {DAILY_GUIDE["protein_g_min"]}–{DAILY_GUIDE["protein_g_max"]} g/day
"""

    # Final prompt
    full_prompt = f"""{persona_text.strip()}

{language_instruction.strip()}

{guide_block.strip()}

CONTEXT (Top 3 candidates from database):
{context_blocks}

USER QUESTION:
{corrected_question}
"""

    response = chat.send_message(full_prompt)

    return {
        "answer": response.text.strip(),
        "context_n_menus": len(parents),
        # ส่งผล parse/eval ให้ฝั่ง UI ทำป้ายสี/กราฟได้ด้วยความแม่นยำเชิงตัวเลข
        "nutrition_analysis": parsed_and_eval,
        "guideline_used": {
            "energy_kcal": DAILY_GUIDE["energy_kcal"],
            "fat_g_max": DAILY_GUIDE["fat_g_max"],
            "sugar_g_max": DAILY_GUIDE["sugar_g_max"],
            "sugar_g_ideal": DAILY_GUIDE["sugar_g_ideal"],
            "sodium_mg_max": DAILY_GUIDE["sodium_mg_max"],
            "protein_g_range": [DAILY_GUIDE["protein_g_min"], DAILY_GUIDE["protein_g_max"]],
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
