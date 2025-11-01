import cv2
import requests
import threading
import queue
import time
import sys
import signal
import json
import numpy as np 

from typing import Optional

# =========================
# Config
# =========================
SERVER_BASE_URL = "http://localhost:8888" # Cập nhật từ 127.0.0.1 để tăng khả năng tương thích
UPLOAD_ENDPOINT = f"{SERVER_BASE_URL}/upload_frame"
VIDEO_FEED_URL = f"{SERVER_BASE_URL}/video_feed"
TOGGLE_ENDPOINT = f"{SERVER_BASE_URL}/toggle_detection"

CAMERA_INDEX = 0 # 0 = default webcam, hoặc đường dẫn video file
TARGET_FPS = 15  # tần số upload khung hình lên server
MAX_QUEUE = 10 # hàng đợi JPEG gửi đi
SHOW_LOCAL_PREVIEW = False # nếu True sẽ show ảnh gốc từ camera bên cạnh

# Cập nhật: Thêm cấu hình để bỏ qua proxy cho kết nối nội bộ
NO_PROXY_CONFIG = {
    'http': None, 
    'https': None
}

# =========================
# Globals
# =========================
STOP = False
jpeg_queue: "queue.Queue[bytes]" = queue.Queue(maxsize=MAX_QUEUE)
last_fire_flag_lock = threading.Lock()
last_fire_detected: Optional[bool] = None
last_upload_latency_ms: float = 0.0

# =========================
# Helpers
# =========================
def graceful_exit(code=0):
    """Đảm bảo dừng tất cả các thread và đóng cửa sổ OpenCV."""
    global STOP
    STOP = True
    time.sleep(0.2)
    try:
        # Đóng tất cả cửa sổ OpenCV một cách an toàn
        cv2.destroyAllWindows() 
    except Exception:
        pass
    sys.exit(code)

def signal_handler(sig, frame):
    """Xử lý tín hiệu Ctrl+C."""
    print("Ctrl+C detected, shutting down…")
    graceful_exit(0)

signal.signal(signal.SIGINT, signal_handler)

def toggle_detection(enable: bool):
    """Gửi yêu cầu POST để bật/tắt tính năng phát hiện trên server."""
    try:
        r = requests.post(
            TOGGLE_ENDPOINT, 
            params={"enable": "true" if enable else "false"}, 
            timeout=3,
            proxies=NO_PROXY_CONFIG 
        )
        print("Toggle:", r.status_code, r.text)
    except Exception as e:
        print("Toggle error:", e)

# =========================
# Camera Reader Thread
# =========================
def camera_reader():
    """Đọc khung hình từ camera và chuyển chúng thành JPEG để upload."""
    global STOP
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print(f"[Reader] Cannot open camera/video: {CAMERA_INDEX}")
        graceful_exit(1)

    print("[Reader] Started.")
    frame_interval = 1.0 / max(1, TARGET_FPS)
    last_push = 0.0

    while not STOP:
        ok, frame = cap.read()
        if not ok:
            print("[Reader] End of stream or read error.")
            break

        # encode JPEG
        now = time.time()
        if now - last_push >= frame_interval:
            last_push = now
            # Đảm bảo khung hình có kích thước trước khi encode
            if frame is not None and frame.size > 0:
                ok2, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if ok2:
                    try:
                        jpeg_queue.put(enc.tobytes(), timeout=0.05)
                    except queue.Full:
                        # nếu đầy, bỏ qua frame này để giữ độ trễ thấp
                        pass

        if SHOW_LOCAL_PREVIEW:
            cv2.imshow("Local Camera (raw)", frame)
            # Dùng waitKey để xử lý ESC nếu preview bật
            if cv2.waitKey(1) & 0xFF == 27:# ESC
                break

    cap.release()
    print("[Reader] Stopped.")
    # Nếu Reader dừng, yêu cầu thoát toàn bộ chương trình
    if not STOP:
        graceful_exit(0)

# =========================
# Uploader Thread (Modified to log to console)
# =========================
def uploader():
    """Lấy JPEG từ hàng đợi, upload lên server, và in kết quả ra console."""
    global STOP, last_fire_detected, last_upload_latency_ms
    s = requests.Session()
    s.proxies = NO_PROXY_CONFIG 
    
    print("[Uploader] Started.")
    while not STOP:
        try:
            jpeg_bytes = jpeg_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        files = {"file": ("frame.jpg", jpeg_bytes, "image/jpeg")}
        t0 = time.perf_counter()
        try:
            r = s.post(
                UPLOAD_ENDPOINT, 
                files=files, 
                timeout=5,
            )
            dt_ms = (time.perf_counter() - t0) * 1000.0
            last_upload_latency_ms = dt_ms
            
            if r.status_code == 200:
                data = r.json()
                fire_status = bool(data.get("fire_detected")) # Lưu trạng thái trước khi khóa
                with last_fire_flag_lock:
                    last_fire_detected = fire_status

                # CẬP NHẬT: In trạng thái và độ trễ ra console
                status_msg = "!!! FIRE DETECTED !!!" if fire_status else "Clear"
                print(f"[Uploader] Status: {status_msg:<18} | Latency: {dt_ms:.1f} ms | Queue: {jpeg_queue.qsize()} frames")

            else:
                print(f"[Uploader] HTTP {r.status_code}: {r.text[:200]}")

        except requests.exceptions.RequestException as e:
            # Gỡ lỗi: Giảm độ trễ nếu có lỗi kết nối
            print(f"[Uploader] Error: Connection failed ({e.__class__.__name__})")
            time.sleep(1) 
        except json.JSONDecodeError:
            print("[Uploader] Bad JSON response")
        except Exception as e:
            print(f"[Uploader] Unexpected error: {e}")

    print("[Uploader] Stopped.")

# Đã xóa: _iter_mjpeg() và viewer()

# =========================
# Main
# =========================
if __name__ == "__main__":
    
    print("=== Fire Segmentation Client ===")
    print("Keys: [ESC]=quit (if preview is on), [1]=enable detection, [0]=disable detection")

    reader = threading.Thread(target=camera_reader, name="Reader", daemon=True)
    sender = threading.Thread(target=uploader, name="Uploader", daemon=True)
    # View thread đã bị loại bỏ

    reader.start()
    sender.start()

    print("Client running. Press Ctrl+C to stop.")
    
    # Giữ thread chính chạy cho đến khi STOP được thiết lập (bởi Ctrl+C hoặc camera_reader dừng)
    try:
        while not STOP:
            # Đặt độ trễ nhỏ để tránh lãng phí CPU
            time.sleep(0.1) 
    except KeyboardInterrupt:
        # Đây là fallback nếu signal_handler không được gọi ngay lập tức
        pass 
    
    # Đảm bảo các thread khác dừng lại
    STOP = True
    print("Waiting for worker threads to finish...")
    reader.join(timeout=1.0)
    sender.join(timeout=1.0)
    
    # Dọn dẹp cuối cùng (thường được xử lý bởi graceful_exit nếu Ctrl+C)
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    
    print("Bye.")
