from fastapi import FastAPI, UploadFile, File
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

# 🔥 Load model khi server khởi động
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

@app.post("/upload_frame")
async def upload_frame(file: UploadFile = File(...)):
    global latest_frame, detection_enabled
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if detection_enabled:
        result = segment_image(img)
    else:
        result = img

    _, jpeg = cv2.imencode('.jpg', result)
    latest_frame = jpeg.tobytes()

    return {"status": "ok"}

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
