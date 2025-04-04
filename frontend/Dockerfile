# Sử dụng image nhỏ hơn để tối ưu kích thước
FROM python:3.9-slim

# Đặt thư mục làm việc
WORKDIR /app

# Sao chép và cài đặt các dependencies trước để tận dụng cache
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ mã nguồn sau khi cài dependencies
COPY . /app

# Mở cổng mặc định của Streamlit
EXPOSE 8501

# Đặt ENTRYPOINT để chạy Streamlit
ENTRYPOINT ["streamlit", "run"]

# CMD sẽ chỉ định script và các biến môi trường cho Streamlit
CMD ["streamlit_app.py"]
