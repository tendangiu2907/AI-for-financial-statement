from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from api.v1.detect_table_router import router as detect_router
import os
from core.config import EXTRACTED_FOLDER, UPLOAD_DIR, SERVER_ADDRESS, SERVER_PORT

# Đảm bảo các thư mục tồn tại
os.makedirs(EXTRACTED_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)


app = FastAPI()

# Đăng ký router
app.include_router(detect_router, prefix="/api/v1")

app.mount(f"/{EXTRACTED_FOLDER}", StaticFiles(directory=EXTRACTED_FOLDER), name=EXTRACTED_FOLDER)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=SERVER_ADDRESS, port=SERVER_PORT)