FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy source files
COPY distilled_student_model_weights.weights.h5 .
COPY main.py .
COPY requirements.txt .
COPY model/segment.py ./model/segment.py

# Install necessary system libraries
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port for Render
EXPOSE 8888

# Run the application using uvicorn
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]
