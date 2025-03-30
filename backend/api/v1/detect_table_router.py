from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from services.table_detection_service import TableDetectService
from core.config import UPLOAD_DIR
from utils import save_temp_pdf, allowed_file
import orjson

router = APIRouter()

table_service = TableDetectService()


"""
Cấu trúc của 1 route gồm:
- Định nghĩa phương thức http (GET, POST, DELETE, PUT/PATCH) và đường dẫn route
- hàm để thực thi
"""
@router.post("/detect_table")
async def detect_table(
    file: UploadFile = File(...),  # nhận file từ client
):
    try:
        if not allowed_file(file.filename):
            raise HTTPException(status_code=400, detail="Chỉ chấp nhận file PDF.")

        pdf_path = save_temp_pdf(file, UPLOAD_DIR)

        # Gọi hàm xử lý từ service
        dfs_dict, extracted_file_path = table_service.detect_table(pdf_path, file.filename)

        # Gộp kết quả thành dictionary
        result = {
            "tables": {
                key: df.to_dict(orient="records") for key, df in dfs_dict.items()
            },
            "extracted_file_path": extracted_file_path,
        }

        # Trả về kết quả
        return JSONResponse(content=orjson.loads(orjson.dumps(result, option=orjson.OPT_SERIALIZE_NUMPY)))

    except Exception as e:
        print("Router detect_table lỗi: ", e)
        raise HTTPException(status_code=500, detail=str(e))
