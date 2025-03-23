MODEL_SIGNATURE_PATH = r"T:\Intern-Orient\AI-for-financial-statement\model\Nhận diện chữ ký.pt"
MODEL_TABLE_TITLE_PATH = r"T:\Intern-Orient\AI-for-financial-statement\model\best_model_YoLo.pt"
POPPLER_PATH = r"T:\Intern-Orient\Aun\dem\poppler-24.08.0\Library\bin"
DEVICE = "cpu"

EXTRACTED_FOLDER = "extracted-files"
UPLOAD_DIR = "uploads"
ALLOWED_EXTENSIONS = {"pdf"}

financial_tables = [
    "Bảng cân đối kế toán",
    "Báo cáo kết quả hoạt động kinh doanh",
    "Báo cáo lưu chuyển tiền tệ",
]
models = ["gemini-2.0-pro-exp-02-05", "gemini-2.0-flash-thinking-exp-01-21"]

SERVER_ADDRESS = "localhost"
SERVER_PORT = 8080