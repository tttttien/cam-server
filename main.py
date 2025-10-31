from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from starlette.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import asyncio
import time

from model.segment import segment_image, load_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model khi khởi động
@app.on_event("startup")
def startup_event():
    load_model()

# Trạng thái AI
detection_enabled = True

# Frame mới nhất + thời gian
latest_frame = None
latest_frame_time = 0.0
FRAME_COOLDOWN = 0.8  # Chỉ gửi frame mới nếu cách >0.8s

# Trang chủ
@app.get("/")
def read_root():
    return {"message": "Welcome to Fire Predictor API"}

# Bật/tắt AI
@app.post("/toggle_detection")
def toggle_detection(enable: bool):
    global detection_enabled
    detection_enabled = enable
    return {"detection_enabled": detection_enabled}

# Nhận ảnh từ ESP32 (raw JPEG)
@app.post("/upload_frame")
async def upload_frame(request: Request):
    global latest_frame, latest_frame_time, detection_enabled

    contents = await request.body()
    if not contents:
        return {"status": "error", "msg": "Empty body"}

    # Giải mã ảnh
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"status": "error", "msg": "Invalid JPEG"}

    # Xử lý AI
    result = segment_image(img) if detection_enabled else img

    # Nén nhẹ để mượt app (chất lượng 70 → ~6-8KB)
    _, jpeg = cv2.imencode('.jpg', result, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    latest_frame = jpeg.tobytes()
    latest_frame_time = time.time()  # Cập nhật thời gian

    return {"status": "ok", "size_kb": round(len(contents)/1024, 1)}

# Stream MJPEG
async def generate_video():
    global latest_frame, latest_frame_time
    last_sent_time = 0.0

    while True:
        current_time = time.time()
        if (latest_frame and 
            current_time - latest_frame_time < 5.0 and  # Frame không quá cũ
            current_time - last_sent_time >= FRAME_COOLDOWN):  # Tránh spam

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + latest_frame + b"\r\n")
            last_sent_time = current_time

        await asyncio.sleep(0.1)  # Kiểm tra 10 lần/giây

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        generate_video(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
