FROM python:3.9-slim

WORKDIR /app

# Cài đặt các gói cần thiết, bao gồm poppler-utils
RUN apt-get update && apt-get install -y \
    poppler-utils \
    ffmpeg \
    libsm6 \
    libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Sao chép file requirements.txt và cài đặt dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép mã nguồn ứng dụng
COPY . /app

EXPOSE 8080

# Chạy ứng dụng bằng Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
