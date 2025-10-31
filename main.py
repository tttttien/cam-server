from fastapi import FastAPI, Request  # THAY UploadFile, File → Request
from fastapi.responses import StreamingResponse
from starlette.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import asyncio

from model.segment import segment_image, load_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    load_model()

detection_enabled = True
latest_frame = None

@app.get("/")
def read_root():
    return {"message": "Welcome to Fire Predictor API"}

@app.post("/toggle_detection")
def toggle_detection(enable: bool):
    global detection_enabled
    detection_enabled = enable
    return {"detection_enabled": detection_enabled}

# Dùng request.body() 
@app.post("/upload_frame")
async def upload_frame(request: Request):
    global latest_frame, detection_enabled

    contents = await request.body()  # Nhận raw JPEG

    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"status": "error", "msg": "Invalid image"}

    result = segment_image(img) if detection_enabled else img

    _, jpeg = cv2.imencode('.jpg', result)
    latest_frame = jpeg.tobytes()

    return {"status": "ok", "size": len(contents)}

# Streaming 
async def generate_video():
    global latest_frame
    while True:
        if latest_frame:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + latest_frame + b"\r\n")
        await asyncio.sleep(0.05)

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_video(), media_type="multipart/x-mixed-replace; boundary=frame")
