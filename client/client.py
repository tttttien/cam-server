import cv2
import requests
import time
import logging
import numpy as np
import threading
import json
import os
import sys
from typing import Optional
from websocket import create_connection

# ---------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger("CAMERA_CLIENT")

class CameraClient:
    """
    A multi-threaded camera client that captures frames from a video source 
    and sends them to a remote fire detection server via WebSockets.
    """
    def __init__(self, config_path: str = "camera_config.json"):
        """
        Initializes the CameraClient.

        Args:
            config_path (str): Path to the JSON configuration file.
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        self.camera_id = self.config.get("camera_id")
        self.api_url = self.config.get("api_url")
        self.video_source = self.config.get("video_source", 0)
        self.capture_fps = self.config.get("capture_fps", 30)
        self.detection_fps = self.config.get("detection_fps", 10)
        
        self.resize_scale = self.config.get("resize_scale")
        self.jpeg_quality = self.config.get("jpeg_quality", 80)
        
        # Derive WebSocket URL
        self.ws_url = self.api_url.replace("http", "ws") + f"/ws/upload/{self.camera_id}"
        
        if not self.camera_id or not self.api_url:
            logger.error("❌ camera_id or api_url missing in config")
            sys.exit(1)
            
        self.frame_lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.stop_event = threading.Event()
        self.frame_index = 0
        self.is_sending = False # Flag for adaptive send loop
        
        self.capture_thread: Optional[threading.Thread] = None
        self.send_thread: Optional[threading.Thread] = None

    def _load_config(self) -> dict:
        """Loads configuration from JSON file."""
        try:
            with open(self.config_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"❌ Could not load {self.config_path}: {e}")
            sys.exit(1)

    def _capture_loop(self):
        """Captures frames at source FPS."""
        logger.info(f"📷 Starting capture thread (Source: {self.video_source}, FPS: {self.capture_fps})")
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            logger.error(f"❌ Cannot open video source: {self.video_source}")
            self.stop_event.set()
            return

        delay = 1.0 / self.capture_fps
        while not self.stop_event.is_set():
            start_time = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                logger.error("❌ Failed to read frame from source")
                self.stop_event.set()
                break

            with self.frame_lock:
                self.latest_frame = frame
            
            elapsed = time.perf_counter() - start_time
            sleep_time = delay - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        cap.release()

    def _send_loop(self):
        """Sends frames via WebSocket at detection FPS with optional resizing and adaptive dropping."""
        delay = 1.0 / self.detection_fps
        logger.info(f"🚀 Starting optimized send thread (WebSocket: {self.ws_url}, Rate: {self.detection_fps} FPS)")
        
        if self.resize_scale:
            logger.info(f"📐 Scale-based resizing enabled: {self.resize_scale}x")

        ws = None
        while not self.stop_event.is_set():
            # --- ADAPTIVE CLIENT-SIDE FRAME DROPPING ---
            # If the previous frame is still being encoded or sent, drop this one.
            if self.is_sending:
                self.stop_event.wait(0.01)
                continue

            try:
                if ws is None:
                    ws = create_connection(self.ws_url, timeout=5)
                    logger.info("🔌 WebSocket connected")

                self.is_sending = True
                start_time = time.perf_counter()
                frame_to_send = None

                with self.frame_lock:
                    if self.latest_frame is not None:
                        frame_to_send = self.latest_frame.copy()

                if frame_to_send is not None:
                    self.frame_index += 1
                    
                    # --- OPTIMIZATION: SCALE RESIZE ---
                    if self.resize_scale and self.resize_scale != 1.0:
                        h, w = frame_to_send.shape[:2]
                        new_w, new_h = int(w * self.resize_scale), int(h * self.resize_scale)
                        frame_to_send = cv2.resize(frame_to_send, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

                    # Encode with optimized quality
                    ok, buffer = cv2.imencode(".jpg", frame_to_send, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
                    if ok:
                        ws.send_binary(buffer.tobytes())
                        
                        # Non-blocking receive for detection results
                        try:
                            orig_timeout = ws.gettimeout()
                            ws.settimeout(0.001) 
                            response = ws.recv()
                            ws.settimeout(orig_timeout)
                            
                            result = json.loads(response)
                            if result.get("fire_detected"):
                                logger.warning(f"🔥 FIRE DETECTED | frame #{self.frame_index}")
                        except Exception:
                            try: ws.settimeout(5)
                            except: pass

                elapsed = time.perf_counter() - start_time
                sleep_time = delay - elapsed
                if sleep_time > 0:
                    self.stop_event.wait(sleep_time)

            except Exception as e:
                logger.error(f"🔌 WebSocket link down: {e}. Retrying in 3 seconds...")
                if ws:
                    try: ws.close()
                    except: pass
                    ws = None
                self.stop_event.wait(3)
            finally:
                self.is_sending = False

        if ws:
            ws.close()

    def run(self):
        """Runs the capture and send threads."""
        logger.info(f"🎬 Camera client started (ID: {self.camera_id})")
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.send_thread = threading.Thread(target=self._send_loop, daemon=True)
        self.capture_thread.start()
        self.send_thread.start()

        try:
            while not self.stop_event.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop_event.set()

        self.capture_thread.join(timeout=2)
        self.send_thread.join(timeout=2)
        logger.info("✅ Client stopped cleanly")

if __name__ == "__main__":
    # Priority: Command line argument > Environment variable > Default
    config_path = "camera_config.json"
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    elif os.getenv("CAMERA_CONFIG"):
        config_path = os.getenv("CAMERA_CONFIG")
        
    client = CameraClient(config_path)
    client.run()
