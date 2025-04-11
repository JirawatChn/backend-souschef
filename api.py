from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Literal, Optional, Union

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    sender: Literal["user", "bot"]
    message: str

class ChatRequest(BaseModel):
    question: Optional[str] = None
    messages: Optional[Union[str, List[Message]]] = None

@app.post("/ask")
async def ask_question(request: ChatRequest):
    if request.messages:
        if isinstance(request.messages, str):
            last_message = request.messages
        else:
            last_message = request.messages[-1].message
    elif request.question:
        last_message = request.question
    else:
        return {"answer": "ไม่ได้ส่งคำถามมานะ 🧐"}

    answer = generate_response(last_message)  # 💥 ใช้จริง
    return {"answer": answer}

