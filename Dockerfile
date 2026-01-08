FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install necessary system libraries for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set environment variables
ENV PORT=8000
ENV RECORDER_TEMP_DIR=/app/recorder_temp

# Create temp directory for recorder
RUN mkdir -p /app/recorder_temp

# Expose the server port
EXPOSE $PORT

# Run the application
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]
