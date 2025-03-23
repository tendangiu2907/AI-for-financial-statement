# AI-for-financial-statement

# Cách chạy dự án
## Backend:
- Tạo môi trường ảo riêng cho backend
- Kích hoạt môi trường: ./venv/scripts/activate
- Chạy file requirements.txt để cài đặt thư viện: pip install -r requirements.txt
- Chỉnh sửa các đường dẫn trong file core/config.py
- Tạo file secret.py, sau đó thêm nội dung với định dạng
```
api_keys = {
    "api_1": {
        "title": "api-key",
        "table": "api-key",
    },
    "api_2": {
        "title": "api-key",
        "table": "api-key",
    },
    "api_3": {
        "title": "api-key",
        "table": "api-key",
    },
    "api_4": {
        "title": "api-key",
        "table": "api-key",
    },
    "api_5": {
        "title": "api-key",
        "table": "api-key",
    },
    "api_6": {
        "title": "api-key",
        "table": "api-key",
    },
    "api_7": {
        "title": "api-key",
        "table": "api-key",
    },
    "api_8": {
        "title": "api-key",
        "table": "api-key",
    },
    "api_9": {
        "title": "api-key",
        "table": "api-key",
    },
    "api_10": {
        "title": "api-key",
        "table": "api-key",
    },
}
```
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
