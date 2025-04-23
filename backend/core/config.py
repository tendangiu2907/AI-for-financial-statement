MODEL_SIGNATURE_PATH = r"T:\Intern-Orient\AI-for-financial-statement\backend\model\Nhận diện chữ ký.pt"
MODEL_TABLE_TITLE_PATH = r"T:\Intern-Orient\AI-for-financial-statement\backend\model\best_model_YoLo.pt"
POPPLER_PATH = r"T:\Intern-Orient\Aun\dem\poppler-24.08.0\Library\bin"
DEVICE = "cpu"

EXTRACTED_FOLDER = "extracted-files"
UPLOAD_DIR = "uploads"
ALLOWED_EXTENSIONS = {"pdf"}

financial_tables = [
    "Bảng cân đối kế toán",
    "Báo cáo KQHĐKD",
    "Báo cáo lưu chuyển tiền tệ"
]
financial_tables_general=["Bảng cân đối kế toán",
"Báo cáo kết quả hoạt động kinh doanh",
"Báo cáo lưu chuyển tiền tệ","Bảng cân đối tài chính","Báo cáo tình hình tài chính","Báo cáo lãi lỗ","Báo cáo dòng tiền","Báo cáo lưu chuyển tiền"
]
model = "gemini-2.0-flash"

SERVER_ADDRESS = "localhost"
SERVER_PORT = 8080
