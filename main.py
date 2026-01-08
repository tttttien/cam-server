import asyncio
import os
import cv2
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Query, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from starlette.middleware.cors import CORSMiddleware

from config import RECORDER_TEMP_DIR, S3_BUCKET_NAME
from model.segment import segment_image, load_model
from utils.s3_client import upload_file_to_s3
from utils.database import (
    init_db, close_db, insert_event, list_events, 
    register_device_token, delete_device_token, get_tokens_for_camera_owner
)
from utils.notifier import init_firebase, send_fire_notification
from services.video_recorder import video_recorder_worker, frame_queue

# Global state for camera streams
detection_enabled = True
camera_states: Dict[int, Dict[str, Any]] = {}

def get_camera_state(camera_id: int) -> Dict[str, Any]:
    """Helper to get or create state for a specific camera."""
    if camera_id not in camera_states:
        camera_states[camera_id] = {
            "latest_frame": None,
            "event": asyncio.Event(),
            "lock": asyncio.Lock(),
            "is_processing": False,
            "detection_enabled": True  # Default to ON
        }
    return camera_states[camera_id]

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application lifecycle.
    - Startup: Loads the segmentation model, initializes the database, and starts the video recorder worker.
    - Shutdown: Cancels the recorder worker and closes the database connection.
    """
    # Startup
    os.makedirs(RECORDER_TEMP_DIR, exist_ok=True)
    from config import RECORD_FPS
    print(f"🎬 Video recording synchronized at {RECORD_FPS} FPS")
    await asyncio.to_thread(load_model)
    await init_db()
    await asyncio.to_thread(init_firebase) # Initialize Firebase
    recorder_task = asyncio.create_task(video_recorder_worker())
    
    yield
    
    # Shutdown
    recorder_task.cancel()
    await close_db()

app = FastAPI(title="Fire Detection Camera Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload_frame")
async def upload_frame(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    camera_id: int = Query(...)
):
    """
    Receives an image frame via HTTP POST, performs detection, and records in background.
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = await asyncio.to_thread(cv2.imdecode, nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Invalid image"}

    is_fire = await process_frame(img, camera_id, background_tasks)
    return {"fire_detected": is_fire}

async def upload_alert_task(original_img: np.ndarray, segmented_img: np.ndarray, camera_id: int):
    """
    Background task to handle S3 upload and DB logging for fire alerts.
    
    Args:
        original_img: Original image without segmentation overlay (for Gemini AI analysis)
        segmented_img: Image with segmentation overlay (for S3 storage and user viewing)
        camera_id: Camera identifier
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    object_name = f"frame/camera_{camera_id}/fire-{timestamp}.jpg"
    tmp_path_original = os.path.join(RECORDER_TEMP_DIR, f"alert_original_{camera_id}_{timestamp}.jpg")
    tmp_path_segmented = os.path.join(RECORDER_TEMP_DIR, f"alert_segmented_{camera_id}_{timestamp}.jpg")
    
    try:
        # Save segmented image locally for upload
        await asyncio.to_thread(cv2.imwrite, tmp_path_segmented, segmented_img)
        
        # Upload SEGMENTED image to S3 (for user viewing)
        ok = await asyncio.to_thread(upload_file_to_s3, tmp_path_segmented, object_name, "image/jpeg", {"fire": "true"})
        if ok:
            # Log to DB
            await insert_event(camera_id, object_name, "frame")
            # Cleanup temp file
            if os.path.exists(tmp_path_segmented):
                os.remove(tmp_path_segmented)
            
            return True
    except Exception as e:
        print(f"❌ Error in upload_alert_task: {e}")
        # Cleanup on error
        if os.path.exists(tmp_path_segmented):
            os.remove(tmp_path_segmented)
        return False

async def process_frame(img: np.ndarray, camera_id: int, background_tasks: Optional[BackgroundTasks] = None) -> bool:
    """
    Processes a single frame: AI segmentation, queueing for recording, and updating feed.
    S3 uploads are handled as background tasks to prevent blocking.
    """
    state = get_camera_state(camera_id)
    if state["detection_enabled"]:
        result_img, is_fire = await asyncio.to_thread(segment_image, img)
    else:
        result_img, is_fire = img, False

    # 1. Send to video recorder worker (in-memory queue, very fast)
    await frame_queue.put((img, is_fire, camera_id))

    # 2. Update latest frame for real-time /video_feed stream
    state = get_camera_state(camera_id)
    encode_success, jpeg_buffer = await asyncio.to_thread(cv2.imencode, ".jpg", result_img)
    if encode_success:
        frame_bytes = jpeg_buffer.tobytes()
        async with state["lock"]:
            state["latest_frame"] = frame_bytes
        state["event"].set()
        state["event"].clear()

    # 3. Handle fire alerts in the background if detected
    if is_fire:
        # Upload alert to S3 in background
        if background_tasks:
            background_tasks.add_task(upload_alert_task, img, result_img, camera_id)
        else:
            asyncio.create_task(upload_alert_task(img, result_img, camera_id))
        
        # Trigger Push Notifications
        async def notify_all():
            tokens = await get_tokens_for_camera_owner(camera_id)
            if tokens:
                await send_fire_notification(tokens, camera_id)
        
        asyncio.create_task(notify_all())
    
    return is_fire

@app.websocket("/ws/upload/{camera_id}")
async def websocket_upload(websocket: WebSocket, camera_id: int):
    """
    Handles real-time binary frame uploads via WebSocket.
    Uses asyncio.create_task for frame processing to avoid blocking the receive loop.
    """
    await websocket.accept()
    print(f"🔌 [Camera {camera_id}] WebSocket connected.")
    try:
        while True:
            # Receive binary frame (this is the only blocking part we want)
            data = await websocket.receive_bytes()
            
            # Offload decoding and processing to not block reception of the next frame
            asyncio.create_task(handle_websocket_frame(websocket, data, camera_id))
            
    except WebSocketDisconnect:
        print(f"🔌 [Camera {camera_id}] WebSocket disconnected.")
    except Exception as e:
        print(f"💥 WebSocket error: {e}")

async def handle_websocket_frame(websocket: WebSocket, data: bytes, camera_id: int):
    """Decodes and processes a frame received via WebSocket with frame-dropping logic."""
    state = get_camera_state(camera_id)
    
    # --- SERVER-SIDE FRAME DROPPING ---
    # If we are already processing a frame for this camera, drop the new one 
    # to maintain real-time performance and prevent a backlog.
    if state["is_processing"]:
        return

    try:
        state["is_processing"] = True
        
        nparr = np.frombuffer(data, np.uint8)
        img = await asyncio.to_thread(cv2.imdecode, nparr, cv2.IMREAD_COLOR)
        
        if img is not None:
            is_fire = await process_frame(img, camera_id)
            try:
                await websocket.send_json({"fire_detected": is_fire})
            except:
                pass
    except Exception as e:
        print(f"❌ Error processing WS frame: {e}")
    finally:
        state["is_processing"] = False

@app.post("/toggle_detection/{camera_id}")
@app.get("/toggle_detection/{camera_id}")
async def toggle_detection(
    camera_id: int, 
    enabled: Optional[bool] = None,
    enable: Optional[bool] = None  # Alternative parameter name for backward compatibility
):
    """
    Toggles fire detection for a specific camera.
    Supports both GET and POST methods.
    Accepts 'enabled' or 'enable' query params (enable takes precedence if both provided).
    If no param is provided, it flips the current state.
    """
    state = get_camera_state(camera_id)
    
    # Use 'enable' if provided, otherwise use 'enabled'
    toggle_value = enable if enable is not None else enabled
    
    if toggle_value is not None:
        state["detection_enabled"] = toggle_value
    else:
        state["detection_enabled"] = not state["detection_enabled"]
    
    status = "ON" if state["detection_enabled"] else "OFF"
    print(f"⚙️ [Camera {camera_id}] Fire detection switched to: {status}")
    return {"camera_id": camera_id, "detection_enabled": state["detection_enabled"]}

@app.post("/toggle_detection")
@app.get("/toggle_detection")
async def toggle_detection_legacy(
    camera_id: Optional[int] = Query(None),
    enabled: Optional[bool] = None,
    enable: Optional[bool] = None
):
    """
    Legacy endpoint for backward compatibility.
    Accepts camera_id as optional query parameter instead of path parameter.
    If camera_id is not provided, toggles detection for ALL active cameras.
    """
    # Use 'enable' if provided, otherwise use 'enabled'
    toggle_value = enable if enable is not None else enabled
    
    if camera_id is not None:
        # Toggle specific camera
        return await toggle_detection(camera_id, enabled, enable)
    else:
        # Toggle ALL cameras
        results = []
        if not camera_states:
            return {"message": "No active cameras", "cameras": []}
        
        for cam_id in camera_states.keys():
            state = get_camera_state(cam_id)
            
            if toggle_value is not None:
                state["detection_enabled"] = toggle_value
            else:
                state["detection_enabled"] = not state["detection_enabled"]
            
            status = "ON" if state["detection_enabled"] else "OFF"
            print(f"⚙️ [Camera {cam_id}] Fire detection switched to: {status}")
            results.append({"camera_id": cam_id, "detection_enabled": state["detection_enabled"]})
        
        return {"message": f"Toggled detection for {len(results)} cameras", "cameras": results}

@app.get("/status/{camera_id}")
async def get_status(camera_id: int):
    """Returns the current status of a specific camera."""
    state = get_camera_state(camera_id)
    return {
        "camera_id": camera_id,
        "detection_enabled": state["detection_enabled"],
        "is_processing": state["is_processing"]
    }

@app.get("/events/{camera_id}")
async def get_events(camera_id: int):
    """
    Lists recent fire detection events for a specific camera.

    Args:
        camera_id (int): The camera ID to query.

    Returns:
        Dict[str, Any]: Lists of event objects.
    """
    rows = await list_events(camera_id)
    for r in rows:
        if isinstance(r.get("created_at"), datetime):
            r["created_at"] = r["created_at"].isoformat()
    return {"camera_id": camera_id, "events": rows}

@app.post("/register_token")
async def register_token(data: Dict[str, Any]):
    """
    Registers an Android device token for Firebase Push Notifications.
    Example body: {"token": "YOUR_DEVICE_REGISTRATION_TOKEN", "user_id": "550e8400-e29b-41d4-a716-446655440000"}
    
    Note: user_id must be a valid UUID from auth.users table.
    """
    # Debug logging
    print(f"📥 Register token request received: {data}")
    
    token = data.get("token")
    user_id = data.get("user_id")
    
    print(f"🔍 token={token[:20] if token else None}..., user_id={user_id}")
    
    if not token or not user_id:
        print(f"❌ Missing required fields: token={bool(token)}, user_id={bool(user_id)}")
        raise HTTPException(status_code=400, detail="token and user_id are required")
    
    await register_device_token(token, user_id)
    print(f"📱 Device token registered for User {user_id[:8]}...: {token[:20]}...")
    return {"status": "success", "message": f"Token registered for user {user_id}"}

@app.post("/unregister_token")
async def unregister_token(data: Dict[str, Any]):
    """
    Unregisters a device token (e.g., on user logout).
    Example body: {"token": "YOUR_DEVICE_REGISTRATION_TOKEN"}
    
    This prevents notification leaks when users switch accounts on the same device.
    """
    token = data.get("token")
    
    if not token:
        raise HTTPException(status_code=400, detail="token is required")
    
    await delete_device_token(token)
    print(f"🚪 Device token unregistered: {token[:20]}...")
    return {"status": "success", "message": "Token unregistered"}


@app.get("/video_feed/{camera_id}")
async def video_feed(camera_id: int):
    """
    Provides a real-time MJPEG stream of processed frames from a camera.

    Args:
        camera_id (int): The camera ID to stream.

    Returns:
        StreamingResponse: An MJPEG multipart response.
    """
    state = get_camera_state(camera_id)
    
    async def frame_generator():
        while True:
            await state["event"].wait()
            async with state["lock"]:
                if state["latest_frame"] is not None:
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n" + state["latest_frame"] + b"\r\n")

    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
