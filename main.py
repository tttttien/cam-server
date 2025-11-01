from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from starlette.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from collections import deque
from typing import Optional, Dict, Any

import asyncio
import tempfile
import os
import io
import json

import cv2
import numpy as np
import asyncpg

# boto3 is optional in dev environment
try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
except Exception:
    boto3 = None
    BotoCoreError = Exception
    ClientError = Exception

# --- Configuration / Constants ---------------------------------------------
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
RECORDER_TEMP_DIR = tempfile.gettempdir()

# --- Model import (kept as-is) ---------------------------------------------
from model.segment import segment_image, load_model

# --- Global state ----------------------------------------------------------
detection_enabled = True
latest_frame: Optional[bytes] = None
frame_lock = asyncio.Lock()
new_frame_event = asyncio.Event()

frame_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
frame_buffer_size = int(RECORD_FPS * PRE_ROLL_BUFFER_SEC)
pre_roll_buffer = deque(maxlen=frame_buffer_size)

_db_pool: Optional[asyncpg.pool.Pool] = None

# --- S3 / Supabase Storage helpers ----------------------------------------
_s3_client = None

def get_s3_client():
    """Lazily construct and return an S3 client for the Supabase storage endpoint."""
    global _s3_client
    if _s3_client is not None:
        return _s3_client

    if boto3 is None:
        print("boto3 not available; S3 uploads are disabled.")
        return None

    try:
        _s3_client = boto3.client(
            "s3",
            endpoint_url=S3_ENDPOINT_URL,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        )
        return _s3_client
    except Exception as e:
        print(f"Could not create S3 client: {e}")
        _s3_client = None
        return None


def _write_temp_file(prefix: str, data: bytes) -> str:
    """Write bytes to a temp file and return its path."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    fname = f"{prefix}-{ts}"
    path = os.path.join(RECORDER_TEMP_DIR, fname)
    with open(path, "wb") as f:
        f.write(data)
    return path


def upload_file_to_s3(local_path_or_file, object_name: str, content_type: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
    """Upload a path or file-like object to the configured S3 endpoint.

    This function is synchronous by design; call it with asyncio.to_thread when used in async code.
    """
    client = get_s3_client()
    if client is None:
        raise RuntimeError("S3 client not configured. Set AWS keys and endpoint.")

    try:
        # Normalise metadata to strings
        meta = {k: str(v) for k, v in (metadata or {}).items()} if metadata else None

        if hasattr(local_path_or_file, "read"):
            try:
                local_path_or_file.seek(0)
            except Exception:
                pass
            body = local_path_or_file.read()
            kwargs = {"Bucket": S3_BUCKET_NAME, "Key": object_name, "Body": body}
            if content_type:
                kwargs["ContentType"] = content_type
            if meta:
                kwargs["Metadata"] = meta
            client.put_object(**kwargs)
        else:
            extra_args = {}
            if content_type:
                extra_args["ContentType"] = content_type
            if meta:
                extra_args["Metadata"] = meta
            if extra_args:
                client.upload_file(local_path_or_file, S3_BUCKET_NAME, object_name, ExtraArgs=extra_args)
            else:
                client.upload_file(local_path_or_file, S3_BUCKET_NAME, object_name)

        print(f"S3 upload succeeded: {object_name}")
        return True
    except ClientError as e:
        print(f"S3 ClientError: {e}")
        return False
    except BotoCoreError as e:
        print(f"S3 BotoCoreError: {e}")
        return False
    except Exception as e:
        print(f"S3 UPLOAD ERROR: {e}")
        return False

# Backwards compatibility
upload_file_to_supabase = upload_file_to_s3


def upload_alert_frame(image_bytes: bytes, camera_id: Optional[int] = None) -> Optional[str]:
    """Upload an alert frame and return the object name on success."""
    try:
        prefix = f"camera_{camera_id}/" if camera_id is not None else ""
        object_name = f"frame/{prefix}fire-alert-{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        tmp_path = _write_temp_file("alert", image_bytes)

        ok = upload_file_to_supabase(tmp_path, object_name, content_type="image/jpeg", metadata={"fire": "true"})
        if ok:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            print(f"Successfully uploaded alert frame: {object_name}")
            return object_name
        else:
            print("Failed to upload alert frame to Supabase.")
            return None
    except Exception as e:
        print(f"GENERAL ALERT UPLOAD ERROR: {e}")
        return None

# --- Database helpers -----------------------------------------------------
async def init_db():
    global _db_pool
    if _db_pool is None:
        _db_pool = await asyncpg.create_pool(DATABASE_URL)
    async with _db_pool.acquire() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id SERIAL PRIMARY KEY,
                camera_id INTEGER,
                object_name TEXT NOT NULL,
                event_type TEXT NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
            );
            """
        )


async def close_db():
    global _db_pool
    if _db_pool is not None:
        await _db_pool.close()


async def insert_event(camera_id: Optional[int], object_name: str, event_type: str, created_at: Optional[datetime] = None) -> bool:
    """Insert a event row into Postgres. Returns True on success."""
    global _db_pool
    if _db_pool is None:
        print("DB pool not initialized; cannot insert event.")
        return False
    try:
        async with _db_pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO events(camera_id, object_name, event_type, created_at) VALUES($1,$2,$3,COALESCE($4, now()))",
                camera_id, object_name, event_type, created_at,
            )
        return True
    except Exception as e:
        print(f"DB insert failed: {e}")
        return False


async def list_event_for_camera(camera_id: int, date_str: Optional[str] = None):
    global _db_pool
    if _db_pool is None:
        return []
    async with _db_pool.acquire() as conn:
        if date_str:
            rows = await conn.fetch(
                "SELECT object_name, event_type, created_at FROM events WHERE camera_id=$1 AND DATE(created_at) = $2 ORDER BY created_at",
                camera_id,
                date_str,
            )
        else:
            rows = await conn.fetch(
                "SELECT object_name, event_type, created_at FROM events WHERE camera_id=$1 ORDER BY created_at",
                camera_id,
            )
        return [dict(r) for r in rows]

# --- Video recorder worker ------------------------------------------------
async def video_recorder_worker():
    """Background worker that consumes frame_queue, handles pre-roll and writes/upload videos."""
    is_recording = False
    video_writer = None
    current_video_path = None
    current_object_name = None
    current_camera_id = None
    last_fire_time: Optional[datetime] = None
    
    print(f"Video recorder worker started. Pre-roll buffer: {frame_buffer_size} frames.")

    while True:
        try:
            item = await frame_queue.get()
            # support items with/without camera_id
            if len(item) == 3:
                frame_np, is_fire, camera_id = item
            else:
                frame_np, is_fire = item
                camera_id = None

            now = datetime.now()

            if is_fire:
                last_fire_time = now

                if not is_recording:
                    # start recording
                    is_recording = True
                    timestamp = now.strftime("%Y%m%d_%H%M%S")
                    prefix = f"camera_{camera_id}/" if camera_id is not None else ""
                    current_object_name = f"video/{prefix}fire-event-{timestamp}.mp4"
                    temp_file_name = f"event-{timestamp}.mp4"
                    current_video_path = os.path.join(RECORDER_TEMP_DIR, temp_file_name)
                    current_camera_id = camera_id
                    height, width, _ = frame_np.shape
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    print(f"🔥 Bắt đầu ghi video (Lưu vào: {current_object_name})")
                    video_writer = await asyncio.to_thread(
                        cv2.VideoWriter, current_video_path, fourcc, RECORD_FPS, (width, height)
                    )

                    # dump pre-roll frames
                    for pre_frame in pre_roll_buffer:
                        await asyncio.to_thread(video_writer.write, pre_frame)
                    pre_roll_buffer.clear()

                # write current fire frame
                if video_writer:
                    await asyncio.to_thread(video_writer.write, frame_np)

            elif not is_fire and is_recording:
                # still write frames until RECORD_AFTER_FIRE_STOPS_SEC passes
                if video_writer:
                    await asyncio.to_thread(video_writer.write, frame_np)

                if last_fire_time and (now - last_fire_time).total_seconds() > RECORD_AFTER_FIRE_STOPS_SEC:
                    print("✅ Hoàn tất video. Đang upload lên Supabase...")
                    await asyncio.to_thread(video_writer.release)

                    # Upload video in thread
                    print(f"Uploading {current_video_path} to {current_object_name}...")
                    upload_ok = await asyncio.to_thread(
                        upload_file_to_supabase, current_video_path, current_object_name, "video/mp4", {"fire": "true"}
                    )
                    if upload_ok:
                        print("Upload hoàn tất.")
                        try:
                            asyncio.create_task(insert_event(current_camera_id, current_object_name, "video", now))
                        except Exception:
                            pass
                        try:
                            os.remove(current_video_path)
                        except Exception:
                            pass
                    else:
                        print("Upload thất bại, giữ file tạm để thử lại hoặc kiểm tra logs.")

                    # reset state
                    is_recording = False
                    video_writer = None
                    current_video_path = None
                    current_object_name = None
                    current_camera_id = None
                    last_fire_time = None

            else:
                # not recording - populate pre-roll buffer
                pre_roll_buffer.append(frame_np)

        except Exception as e:
            print(f"LỖI trong video_recorder_worker: {e}")
            try:
                if video_writer:
                    await asyncio.to_thread(video_writer.release)
            except Exception:
                pass
            is_recording = False
            video_writer = None

# --- Application lifespan -------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Application starting up...")
    # start background worker
    recorder_task = asyncio.create_task(video_recorder_worker())

    try:
        # load ML model
        await asyncio.to_thread(load_model)
        print("Model loaded successfully.")

        # initialize DB
        try:
            await init_db()
            print("Database initialized.")
        except Exception as e:
            print(f"DB init error: {e}")

        # check S3 bucket availability
        client = get_s3_client()
        if client is not None:
            try:
                client.head_bucket(Bucket=S3_BUCKET_NAME)
                print(f"Bucket '{S3_BUCKET_NAME}' is accessible via S3 endpoint.")
            except ClientError as e:
                print(f"Bucket access/check error: {e}")
        else:
            print("S3 client not configured; skipping bucket access check.")

    except Exception as e:
        print(f"GENERAL ERROR: Could not connect or init: {e}")

    yield

    print("Application shutting down.")
    recorder_task.cancel()
    try:
        await close_db()
    except Exception:
        pass

# --- FastAPI app and endpoints --------------------------------------------
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to Fire Predictor API"}

@app.post("/toggle_detection")
def toggle_detection(enable: bool):
    global detection_enabled
    detection_enabled = enable
    return {"detection_enabled": detection_enabled}

@app.post("/upload_frame")
async def upload_frame(file: UploadFile = File(...), camera_id: Optional[int] = None):
    """Receive a frame, run segmentation, enqueue for video recorder and upload alert if fire detected."""
    global latest_frame

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = await asyncio.to_thread(cv2.imdecode, nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"status": "error", "message": "Invalid image data"}

    is_fire_detected = False
    result_img = img

    if detection_enabled:
        try:
            result_img, is_fire_detected = await asyncio.to_thread(segment_image, img)
        except Exception as e:
            print(f"MODEL SEGMENTATION ERROR: {e}. Returning original image.")
            is_fire_detected = False
            result_img = img

    # push original frame into recorder queue (with optional camera_id)
    try:
        if camera_id is not None:
            frame_queue.put_nowait((img, is_fire_detected, camera_id))
        else:
            frame_queue.put_nowait((img, is_fire_detected))
    except asyncio.QueueFull:
        print("CẢNH BÁO: Hàng đợi video recorder bị đầy.")

    # encode result image for streaming and alert upload
    encode_success, jpeg_buffer = await asyncio.to_thread(cv2.imencode, ".jpg", result_img)
    if not encode_success:
        return {"status": "error", "message": "Failed to encode result image"}

    frame_bytes = jpeg_buffer.tobytes()

    # update latest frame for /video_feed
    async with frame_lock:
        latest_frame = frame_bytes
    new_frame_event.set()
    new_frame_event.clear()

    # upload alert frame if fire detected
    if is_fire_detected:
        object_name = f"frame/{'camera_' + str(camera_id) + '/' if camera_id is not None else ''}fire-alert-{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        tmp_path = _write_temp_file('alert', frame_bytes)
        upload_ok = await asyncio.to_thread(upload_file_to_supabase, tmp_path, object_name, content_type="image/jpeg", metadata={"fire": "true"})
        if upload_ok:
            try:
                asyncio.create_task(insert_event(camera_id, object_name, 'frame'))
            except Exception:
                pass
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            return {"status": "ok", "fire_detected": True, "object_name": object_name}
        else:
            print("Failed to upload alert frame to Supabase.")
            return {"status": "ok", "fire_detected": True, "object_name": None}

    return {"status": "ok", "fire_detected": is_fire_detected}

@app.get("/events/{camera_id}")
async def get_event(camera_id: int, date: Optional[str] = None):
    rows = await list_event_for_camera(camera_id, date)
    for r in rows:
        if isinstance(r.get('created_at'), datetime):
            r['created_at'] = r['created_at'].isoformat()
    return {"camera_id": camera_id, "event": rows}

async def generate_video():
    global latest_frame
    while True:
        try:
            await new_frame_event.wait()
            local_frame = None
            async with frame_lock:
                if latest_frame is None:
                    continue
                local_frame = latest_frame

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + local_frame + b"\r\n")

        except asyncio.CancelledError:
            print("Video feed client disconnected.")
            break
        except Exception as e:
            print(f"Error in video generator: {e}")
            break

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_video(), media_type="multipart/x-mixed-replace; boundary=frame")