# -*- coding: utf-8 -*-
import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal, Optional, Dict, Any
from pythainlp.spell import correct

from config import client, DAILY_GUIDE
from retrieval import retrieve_parents, build_context_blocks
from nutrition import parse_nutrition_text, evaluate_against_guidelines
from prompts import personality_profiles, tone_map, make_language_instruction

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

chat_sessions: Dict[str, Any] = {}


@app.post("/ask")
async def ask_question(request: ChatRequest, raw_request: Request):
    session_id = raw_request.headers.get("X-Session-ID")
    if not session_id:
        return {"error": "Missing session ID"}

    if session_id not in chat_sessions:
        chat_sessions[session_id] = client.chats.create(model="gemini-2.5-flash")
    chat = chat_sessions[session_id]

    corrected_question = correct(request.question) if request.lang == "th" else request.question

    parents = retrieve_parents(corrected_question, k_children=40, top_parents=3)
    context_blocks = build_context_blocks(parents)
    if not context_blocks:
        context_blocks = "ไม่พบเมนูที่เกี่ยวข้องในฐานข้อมูล / No related menu found."

    parsed_and_eval = []
    for p in parents:
        nut_text = p.get("nutrition_text", "") or p.get("nutrition", "") or ""
        parsed = parse_nutrition_text(nut_text)
        evaled = evaluate_against_guidelines(parsed)
        parsed_and_eval.append({
            "parent_id": p.get("parent_id", ""),
            "menu_name": p.get("menu_name", ""),
            "nutrition_text_raw": nut_text,
            "nutrition_parsed": parsed,
            "nutrition_eval": evaled,
        })

    persona_text = personality_profiles.get(request.personality or "souschef", "")
    tone_instruction = tone_map.get(request.personality or "souschef", "สุภาพ ชัดเจน เข้าใจง่าย")
    language_instruction = make_language_instruction(request.lang, tone_instruction)

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
        "nutrition_analysis": parsed_and_eval,
        "guideline_used": {
            "energy_kcal":    DAILY_GUIDE["energy_kcal"],
            "fat_g_max":      DAILY_GUIDE["fat_g_max"],
            "sugar_g_max":    DAILY_GUIDE["sugar_g_max"],
            "sugar_g_ideal":  DAILY_GUIDE["sugar_g_ideal"],
            "sodium_mg_max":  DAILY_GUIDE["sodium_mg_max"],
            "protein_g_range": [DAILY_GUIDE["protein_g_min"], DAILY_GUIDE["protein_g_max"]],
        },
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
