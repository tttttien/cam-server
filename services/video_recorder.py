import asyncio
import os
import cv2
from datetime import datetime
from collections import deque
from config import (
    RECORD_FPS, RECORD_AFTER_FIRE_STOPS_SEC, PRE_ROLL_BUFFER_SEC, 
    RECORDER_TEMP_DIR
)
from utils.s3_client import upload_file_to_s3
from utils.database import insert_event

# Global pre-roll buffer and queue
frame_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
pre_roll_buffer = deque(maxlen=int(RECORD_FPS * PRE_ROLL_BUFFER_SEC))

async def finish_and_upload_video(video_writer, video_path, video_object, camera_id):
    """
    Stops a video recording, releases the writer, and uploads the video file to S3.

    Args:
        video_writer (cv2.VideoWriter): The OpenCV VideoWriter instance to release.
        video_path (str): The local temporary path of the recorded video file.
        video_object (str): The destination object name (key) in the S3 bucket.
        camera_id (int): The ID of the camera associated with the video.
    """
    if video_writer:
        await asyncio.to_thread(video_writer.release)
        print(f"☁️ [Camera {camera_id}] Đang upload video cuối cùng lên S3...")
        ok = await asyncio.to_thread(upload_file_to_s3, video_path, video_object, "video/mp4", {"fire": "true"})
        if ok:
            print(f"🚀 [Camera {camera_id}] Upload video cuối cùng THÀNH CÔNG!")
            await insert_event(camera_id, video_object, "video")
            try: 
                os.remove(video_path)
            except Exception: 
                pass
        else:
            print(f"❌ [Camera {camera_id}] Không thể upload video cuối cùng.")

async def video_recorder_worker():
    """
    Background worker that continuously monitors a queue for video frames.
    
    Logic:
    - Buffers frames for 'pre-roll' before a detection occurs.
    - If 'is_fire' is detected, starts/continues recording into a local file.
    - Stops recording and uploads to S3 after a cooldown period ('RECORD_AFTER_FIRE_STOPS_SEC').
    - Handles frame processing in separate threads (asyncio.to_thread) for non-blocking I/O.
    """
    is_recording = False
    video_writer = None
    video_path = None
    video_object = None
    current_camera_id = None
    last_fire_time = None

    print(f"DEBUG: Video recorder worker started. Buffer size: {pre_roll_buffer.maxlen}")

    while True:
        try:
            try:
                # Wait for a new frame item with a timeout
                item = await asyncio.wait_for(frame_queue.get(), timeout=1.0)
                frame, is_fire, camera_id = item
                now = datetime.now()
            except asyncio.TimeoutError:
                # Check for idle recording cooldown during inactivity
                if is_recording and video_writer:
                    now = datetime.now()
                    if last_fire_time and (now - last_fire_time).total_seconds() > RECORD_AFTER_FIRE_STOPS_SEC:
                        print(f"⚠️ [Camera {current_camera_id}] Không nhận được dữ liệu mới. Đang chốt video...")
                        await finish_and_upload_video(video_writer, video_path, video_object, current_camera_id)
                        is_recording = False
                        video_writer = None
                continue

            if is_fire:
                last_fire_time = now
                if not is_recording:
                    # Initialize a new recording
                    is_recording = True
                    current_camera_id = camera_id
                    timestamp = now.strftime("%Y%m%d_%H%M%S")
                    video_object = f"video/camera_{camera_id}/fire-event-{timestamp}.mp4"
                    video_path = os.path.join(RECORDER_TEMP_DIR, f"event-{camera_id}-{timestamp}.mp4")
                    
                    h, w, _ = frame.shape
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    print(f"🎬 [Camera {camera_id}] Phát hiện lửa! Bắt đầu ghi video: {video_path}")
                    video_writer = await asyncio.to_thread(cv2.VideoWriter, video_path, fourcc, RECORD_FPS, (w, h))
                    
                    # Prepend pre-roll buffer to the new recording
                    if len(pre_roll_buffer) > 0:
                        for f in list(pre_roll_buffer):
                            await asyncio.to_thread(video_writer.write, f)
                        pre_roll_buffer.clear()

                if video_writer:
                    await asyncio.to_thread(video_writer.write, frame)

            elif is_recording:
                # Continue recording during the cooldown period
                if video_writer:
                    await asyncio.to_thread(video_writer.write, frame)

                # Check if fire has stopped and cooldown has passed
                if last_fire_time and (now - last_fire_time).total_seconds() > RECORD_AFTER_FIRE_STOPS_SEC:
                    print(f"✅ [Camera {camera_id}] Lửa đã tắt. Đang hoàn thiện video...")
                    await finish_and_upload_video(video_writer, video_path, video_object, camera_id)
                    is_recording = False
                    video_writer = None
            else:
                # Maintain pre-roll buffer during normal operation
                pre_roll_buffer.append(frame)

        except asyncio.CancelledError:
            print("🛑 Video worker đang dừng... Đang kiểm tra video dở dang...")
            if is_recording:
                await finish_and_upload_video(video_writer, video_path, video_object, current_camera_id)
            break
        except Exception as e:
            print(f"💥 LỖI trong video_recorder_worker: {e}")
            is_recording = False
            if video_writer:
                video_writer.release()
                video_writer = None
