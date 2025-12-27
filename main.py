from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import StreamingResponse
from starlette.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from datetime import datetime
from collections import deque
from typing import Optional, Dict, Any
from uuid import UUID

import asyncio
import tempfile
import os

import cv2
import numpy as np
import asyncpg

# -----------------------------------------------------------------------------
# Optional boto3
# -----------------------------------------------------------------------------
try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
except Exception:
    boto3 = None
    BotoCoreError = Exception
    ClientError = Exception

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
S3_ENDPOINT_URL = "https://bwmqzqgnouisgshuprhh.storage.supabase.co/storage/v1/s3"
AWS_ACCESS_KEY_ID = "0ce4e6b6d05b9bf274d7a554d1cee534"
AWS_SECRET_ACCESS_KEY = "c596dda78c2c7dfdd351680b19b30b72fa8965700999460686abc7d7e66894d2"
S3_BUCKET_NAME = "fire"

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres.bwmqzqgnouisgshuprhh:1512200011032003Dac@aws-1-ap-southeast-1.pooler.supabase.com:6543/postgres",
)

RECORD_FPS = 15.0
RECORD_AFTER_FIRE_STOPS_SEC = 5
PRE_ROLL_BUFFER_SEC = 5
# Temporary directory for video recording
RECORDER_TEMP_DIR = "/Users/lebadac/Desktop/KLTN/Code/camera_server/recorder_temp"
os.makedirs(RECORDER_TEMP_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
from model.segment import segment_image, load_model

# -----------------------------------------------------------------------------
# GLOBAL STATE
# -----------------------------------------------------------------------------
detection_enabled = True

# Dictionary to store state for each camera: {camera_id: {"latest_frame": bytes, "event": asyncio.Event, "lock": asyncio.Lock}}
camera_states: Dict[int, Dict[str, Any]] = {}

def get_camera_state(camera_id: int) -> Dict[str, Any]:
    """Helper to get or create state for a specific camera."""
    if camera_id not in camera_states:
        camera_states[camera_id] = {
            "latest_frame": None,
            "event": asyncio.Event(),
            "lock": asyncio.Lock()
        }
    return camera_states[camera_id]

frame_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
pre_roll_buffer = deque(maxlen=int(RECORD_FPS * PRE_ROLL_BUFFER_SEC))

_db_pool: Optional[asyncpg.Pool] = None
_s3_client = None

# -----------------------------------------------------------------------------
# S3
# -----------------------------------------------------------------------------
def get_s3_client():
    global _s3_client
    if _s3_client:
        return _s3_client
    if boto3 is None:
        return None
    _s3_client = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )
    return _s3_client


def upload_file_to_s3(path: str, object_name: str, content_type: str, metadata=None):
    client = get_s3_client()
    if client is None:
        return False
    try:
        client.upload_file(
            path,
            S3_BUCKET_NAME,
            object_name,
            ExtraArgs={
                "ContentType": content_type,
                "Metadata": metadata or {}
            },
        )
        return True
    except Exception as e:
        print(f"❌ S3 upload error Details: {type(e).__name__} - {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# -----------------------------------------------------------------------------
# DB
# -----------------------------------------------------------------------------
async def init_db():
    global _db_pool
    _db_pool = await asyncpg.create_pool(DATABASE_URL, statement_cache_size=0)
    async with _db_pool.acquire() as conn:
        # Create table with INTEGER for camera_id
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id SERIAL PRIMARY KEY,
                camera_id INTEGER NOT NULL,
                object_name TEXT NOT NULL,
                event_type TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT now()
            );
            """
        )
        # Ensure it's INTEGER (migration path)
        try:
            await conn.execute("ALTER TABLE events ALTER COLUMN camera_id TYPE INTEGER USING camera_id::integer;")
            print("Database migration: camera_id column altered to INTEGER.")
        except Exception:
            pass


async def insert_event(camera_id: int, object_name: str, event_type: str):
    async with _db_pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO events(camera_id, object_name, event_type)
            VALUES ($1, $2, $3)
            """,
            camera_id,
            object_name,
            event_type,
        )


async def list_events(camera_id: int):
    async with _db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT object_name, event_type, created_at
            FROM events
            WHERE camera_id = $1
            ORDER BY created_at DESC
            """,
            camera_id,
        )
        return [dict(r) for r in rows]


# -----------------------------------------------------------------------------
# VIDEO RECORDER
# -----------------------------------------------------------------------------
async def video_recorder_worker():
    """Background worker that handles pre-roll and writes/uploads videos."""
    is_recording = False
    video_writer = None
    video_path = None
    video_object = None
    current_camera_id = None
    last_fire_time = None

    async def finish_and_upload_video():
        nonlocal is_recording, video_writer, video_path, video_object, current_camera_id
        if video_writer:
            await asyncio.to_thread(video_writer.release)
            print(f"☁️ [Camera {current_camera_id}] Đang upload video cuối cùng lên S3...")
            ok = await asyncio.to_thread(upload_file_to_s3, video_path, video_object, "video/mp4", {"fire": "true"})
            if ok:
                print(f"🚀 [Camera {current_camera_id}] Upload video cuối cùng THÀNH CÔNG!")
                await insert_event(current_camera_id, video_object, "video")
                try: os.remove(video_path)
                except: pass
            else:
                print(f"❌ [Camera {current_camera_id}] Không thể upload video cuối cùng.")
        
        is_recording = False
        video_writer = None
        video_path = None
        video_object = None
    
    print(f"DEBUG: Video recorder worker started. Buffer size: {pre_roll_buffer.maxlen}")

    while True:
        try:
            # Đợi frame mới với timeout 1 giây
            try:
                item = await asyncio.wait_for(frame_queue.get(), timeout=1.0)
                frame, is_fire, camera_id = item
                now = datetime.now()
            except asyncio.TimeoutError:
                # Nếu không có frame mới trong 1 giây, kiểm tra xem có đang ghi video dở không
                if is_recording and video_writer:
                    now = datetime.now()
                    # Nếu đã quá thời gian cooldown mà không có frame mới -> Kết thúc video luôn
                    if last_fire_time and (now - last_fire_time).total_seconds() > RECORD_AFTER_FIRE_STOPS_SEC:
                        print(f"⚠️ [Camera {current_camera_id}] Không nhận được dữ liệu mới. Đang chốt video...")
                        await finish_and_upload_video()
                continue

            # LOGIC XỬ LÝ FRAME
            if is_fire:
                last_fire_time = now
                if not is_recording:
                    is_recording = True
                    current_camera_id = camera_id
                    timestamp = now.strftime("%Y%m%d_%H%M%S")
                    video_object = f"video/camera_{camera_id}/fire-event-{timestamp}.mp4"
                    video_path = os.path.join(RECORDER_TEMP_DIR, f"event-{camera_id}-{timestamp}.mp4")
                    
                    h, w, _ = frame.shape
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    print(f"🎬 [Camera {camera_id}] Phát hiện lửa! Bắt đầu ghi video: {video_path}")
                    video_writer = await asyncio.to_thread(cv2.VideoWriter, video_path, fourcc, RECORD_FPS, (w, h))
                    
                    if len(pre_roll_buffer) > 0:
                        for f in list(pre_roll_buffer):
                            await asyncio.to_thread(video_writer.write, f)
                        pre_roll_buffer.clear()

                if video_writer:
                    await asyncio.to_thread(video_writer.write, frame)

            elif is_recording:
                if video_writer:
                    await asyncio.to_thread(video_writer.write, frame)

                if last_fire_time and (now - last_fire_time).total_seconds() > RECORD_AFTER_FIRE_STOPS_SEC:
                    print(f"✅ [Camera {camera_id}] Lửa đã tắt. Đang hoàn thiện video...")
                    await finish_and_upload_video()
            else:
                pre_roll_buffer.append(frame)

        except asyncio.CancelledError:
            print("🛑 Video worker đang dừng... Đang kiểm tra video dở dang...")
            if is_recording:
                await finish_and_upload_video()
            break
        except Exception as e:
            print(f"💥 LỖI trong video_recorder_worker: {e}")
            is_recording = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load_model is synchronous, run it in a thread to avoid blocking startup
    await asyncio.to_thread(load_model)
    await init_db()
    task = asyncio.create_task(video_recorder_worker())
    yield
    task.cancel()
    await _db_pool.close()


# -----------------------------------------------------------------------------
# FASTAPI
# -----------------------------------------------------------------------------
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/upload_frame")
async def upload_frame(
    file: UploadFile = File(...),
    camera_id: int = Query(...)
):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = await asyncio.to_thread(cv2.imdecode, nparr, cv2.IMREAD_COLOR)

    if detection_enabled:
        result_img, is_fire = await asyncio.to_thread(segment_image, img)
    else:
        result_img, is_fire = img, False

    await frame_queue.put((img, is_fire, camera_id))

    # update latest frame for /video_feed
    state = get_camera_state(camera_id)
    # encode result image for streaming
    encode_success, jpeg_buffer = await asyncio.to_thread(cv2.imencode, ".jpg", result_img)
    if encode_success:
        frame_bytes = jpeg_buffer.tobytes()
        async with state["lock"]:
            state["latest_frame"] = frame_bytes
        state["event"].set()
        state["event"].clear()

    if is_fire:
        name = f"frame/camera_{camera_id}/fire-{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        tmp = os.path.join(RECORDER_TEMP_DIR, f"alert_{camera_id}.jpg")
        
        await asyncio.to_thread(cv2.imwrite, tmp, result_img)
        
        # upload_file_to_s3 is sync
        ok = await asyncio.to_thread(upload_file_to_s3, tmp, name, "image/jpeg", {"fire": "true"})
        if ok:
            await insert_event(camera_id, name, "frame")
            try:
                os.remove(tmp)
            except Exception:
                pass

    return {"fire_detected": is_fire}


@app.get("/events/{camera_id}")
async def get_events(camera_id: int):
    rows = await list_events(camera_id)
    for r in rows:
        r["created_at"] = r["created_at"].isoformat()
    return {"camera_id": camera_id, "events": rows}


@app.get("/video_feed/{camera_id}")
def video_feed(camera_id: int):
    state = get_camera_state(camera_id)
    async def gen():
        while True:
            await state["event"].wait()
            async with state["lock"]:
                if state["latest_frame"] is None:
                    continue
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + state["latest_frame"] + b"\r\n"

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")
