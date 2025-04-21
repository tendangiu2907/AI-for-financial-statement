from dotenv import load_dotenv
import os
import json
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI

MODEL_SIGNATURE_PATH = r"T:\Intern-Orient\AI-for-financial-statement\backend\model\Nhận diện chữ ký.pt"
MODEL_TABLE_TITLE_PATH = r"T:\Intern-Orient\AI-for-financial-statement\backend\model\best_model_YoLo.pt"
POPPLER_PATH = r"T:\Intern-Orient\dem\poppler-24.08.0\Library\bin"
DEVICE = "cpu"

EXTRACTED_FOLDER = "extracted-files"
UPLOAD_DIR = "uploads"
ALLOWED_EXTENSIONS = {"pdf"}

financial_tables = [
    "Bảng cân đối kế toán",
    "Báo cáo KQHĐKD",
    "Báo cáo lưu chuyển tiền tệ"
]
financial_tables_general = [
    "Bảng cân đối kế toán",
    "Báo cáo kết quả hoạt động kinh doanh",
    "Báo cáo lưu chuyển tiền tệ",
    "Bảng cân đối tài chính",
    "Báo cáo tình hình tài chính",
    "Báo cáo lãi lỗ",
    "Báo cáo dòng tiền",
    "Báo cáo lưu chuyển tiền",
]
model = "gemini-2.0-flash"

SERVER_ADDRESS = "localhost"
SERVER_PORT = 8080

# Load api key from .env
env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)
api_keys = json.loads(os.getenv("API_KEYS"))
chatbot_api_key = os.getenv("CHATBOT_API_KEY")

# LLM instance
llm = ChatGoogleGenerativeAI(
    model=model, google_api_key=chatbot_api_key, temperature=0.4
)
