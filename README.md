# AI-for-financial-statement

# Cách chạy dự án
## Yêu cầu môi trường:
- Python 3.11 [Downloaf link](https://www.python.org/downloads/release/python-3110/)
## Ở thư mục gốc:
- Tạo file .env và thêm nội dung với định dạng
```
# streamlit in docker call fastapi via container name
# SERVER_ADDRESS=http://backend # for dockerize
SERVER_ADDRESS=http://localhost # for run manually
SERVER_PORT=8080

# Map to local port, you could change these
STREAMLIT_PORT_MAP_OUT=8501
FASTAPI_PORT_MAP_OUT=8080

API_KEYS='
    {
        "api_1": {
            "title": "api_key",
            "table": "api_key"
        }
    }
'

CHATBOT_API_KEY='api_key'
```
## Backend:
- Tạo môi trường ảo riêng cho backend
- Kích hoạt môi trường: ./venv/scripts/activate
- Chạy file requirements.txt để cài đặt thư viện: pip install -r requirements.txt
- Chỉnh sửa các đường dẫn trong file core/config.py
- Chạy lệnh trên môi trường vừa kích hoạt: 
```
python main.py
```
## Frontend
- Tạo môi trường ảo riêng cho frontend
- Kích hoạt môi trường: ./venv/scripts/activate
- Chạy file requirements.txt để cài đặt thư viện: pip install -r requirements.txt
- Chạy lệnh trên môi trường vừa kích hoạt: 
```
streamlit run streamlit_app.py
```
