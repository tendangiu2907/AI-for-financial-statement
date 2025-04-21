from fastapi import APIRouter, UploadFile, Form
from uuid import uuid4
from utils import create_chatbot_service
from chatbot_sessions import chatbot_sessions 
from core.config import llm, UPLOAD_DIR 

router = APIRouter()

@router.post("/init_chat/")
async def init_chat(file: UploadFile = None):
    user_id = str(uuid4())
    file_path = f"{UPLOAD_DIR}/{user_id}_{file.filename}"

    with open(file_path, "wb") as f:
        f.write(await file.read())

    chatbot = create_chatbot_service(file_path, llm)
    chatbot_sessions[user_id] = chatbot

    return {
        "message": "Chatbot initialized successfully.",
        "user_id": user_id
    }

@router.post("/ask/")
async def ask(user_id: str = Form(...), query: str = Form(...)):
    chatbot = chatbot_sessions.get(user_id)
    if not chatbot:
        return {"error": "Chatbot session not found. Please initialize it first."}

    response = chatbot.process_query(query)
    return {"response": response}
