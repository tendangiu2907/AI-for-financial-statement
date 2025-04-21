from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from uuid import uuid4
from utils import create_chatbot_service
from core.config import llm, UPLOAD_DIR 
import chatbot_state  # Import module chứ không chỉ biến

router = APIRouter()

@router.post("/upload_extracted_path")
async def upload_extracted_path(extracted_file_path: str = Form(...)):
    try:
        if not extracted_file_path:
            raise HTTPException(status_code=400, detail="Missing 'extracted_file_path'.")
        extracted_file_path = extracted_file_path.lstrip("/")
        chatbot = create_chatbot_service(extracted_file_path, llm)
        chatbot_state.chatbot_instance = chatbot

        return JSONResponse(content={"message": "Chatbot created successfully."})

    except Exception as e:
        print("Lỗi tại upload_extracted_path:", e)
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/ask")
async def ask(query: str = Form(...)):
    if chatbot_state.chatbot_instance is None:
        return {"response": "Chatbot chưa đọc file dữ liệu của bạn"}
    try:
        response = chatbot_state.chatbot_instance.process_query(query)
        return {"response": response}

    except Exception as e:
        print("Lỗi khi xử lý câu hỏi:", e)
        raise HTTPException(status_code=500, detail=str(e))
