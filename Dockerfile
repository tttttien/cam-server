FROM python:3.10-slim

WORKDIR /app

# Copy requirements trước để tận dụng cache
COPY requirements.txt .

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && pip install --no-cache-dir -r requirements.txt \
    && rm -rf /var/lib/apt/lists/*

# Copy toàn bộ source
COPY . .

EXPOSE 8888

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]
