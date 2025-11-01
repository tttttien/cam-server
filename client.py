import cv2
import requests
import time
import signal
import sys
import logging
import numpy as np
import threading # CẢI THIỆN: Dùng đa luồng

# --- Cấu hình giữ nguyên ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
API_URL = "http://forceless-josette-unluckier.ngrok-free.dev/upload_frame"
VIDEO_SOURCE = 0 
TARGET_FPS = 15 

# --- CẢI THIỆN: Dùng biến toàn cục an toàn cho luồng ---
# Dùng Lock để bảo vệ khi đọc/ghi `latest_frame`
frame_lock = threading.Lock()
latest_frame = None
# Dùng Event để báo hiệu dừng cho các luồng
stop_event = threading.Event()
frame_index = 0

def send_frame(session: requests.Session, frame: np.ndarray, current_index: int):
    """
    Mã hóa và gửi frame bằng requests.Session (hiệu quả hơn).
    """
    try:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)
        frame_bytes = buffer.tobytes()
    except cv2.error as e:
        logger.error(f"Lỗi mã hóa frame #{current_index}: {e}")
        return

    files = {'file': ('frame.jpg', frame_bytes, 'image/jpeg')}
    start_time = time.perf_counter()
    
    try:
        # CẢI THIỆN: Dùng session.post thay vì requests.post
        response = session.post(API_URL, files=files, timeout=5)
        response.raise_for_status() 
        
        end_time = time.perf_counter()
        request_time_ms = (end_time - start_time) * 1000
        result = response.json()
        fire_detected = result.get("fire_detected", False)

        if fire_detected:
            logger.warning(
                f"🚨 PHÁT HIỆN LỬA trong frame #{current_index}! | Độ trễ: {request_time_ms:.1f} ms"
            )
        else:
            logger.info(
                f"✅ Frame #{current_index} đã xử lý. Không có lửa. | Độ trễ: {request_time_ms:.1f} ms"
            )

    except requests.exceptions.RequestException as e:
        logger.error(f"Lỗi HTTP/Mạng cho frame #{current_index}: {e}")
    except Exception as e:
        logger.error(f"Lỗi không mong muốn khi xử lý frame #{current_index}: {e}")

def capture_thread_func():
    """
    CẢI THIỆN: Luồng này CHỈ đọc frame từ camera nhanh nhất có thể.
    Nó cập nhật biến `latest_frame` toàn cục.
    """
    global latest_frame, frame_lock, stop_event, VIDEO_SOURCE
    
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        logger.error(f"Không thể mở nguồn video: {VIDEO_SOURCE}. Thoát.")
        stop_event.set() # Báo cho luồng khác cũng dừng
        return

    logger.info("Luồng Capture đã bắt đầu...")
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            logger.info("Luồng video kết thúc, dừng capture.")
            stop_event.set()
            break
            
        # Cập nhật frame mới nhất một cách an toàn
        with frame_lock:
            global latest_frame
            latest_frame = frame
            
    cap.release()
    logger.info("Luồng Capture đã tắt.")

def send_thread_func():
    """
    CẢI THIỆN: Luồng này CHỈ gửi frame.
    Nó chạy theo nhịp TARGET_FPS, lấy frame mới nhất và gửi đi.
    """
    global latest_frame, frame_lock, stop_event, TARGET_FPS, frame_index
    
    delay_s = 1.0 / TARGET_FPS
    
    logger.info(f"Luồng Send đã bắt đầu, gửi đến: {API_URL}")

    # CẢI THIỆN: Tạo 1 Session để tái sử dụng kết nối
    with requests.Session() as session:
        while not stop_event.is_set():
            loop_start_time = time.perf_counter()
            
            frame_to_send = None
            # Lấy frame mới nhất một cách an toàn
            with frame_lock:
                if latest_frame is not None:
                    # Phải copy() để tránh luồng capture ghi đè
                    # lên frame khi luồng send đang mã hóa
                    frame_to_send = latest_frame.copy() 
            
            if frame_to_send is not None:
                frame_index += 1
                send_frame(session, frame_to_send, frame_index)
            
            # --- Điều chỉnh tốc độ (Throttle) ---
            processing_time = time.perf_counter() - loop_start_time
            sleep_time = delay_s - processing_time
            
            if sleep_time > 0:
                # CẢI THIỆN: Dùng event.wait() thay cho time.sleep()
                # Nó sẽ ngủ đúng sleep_time, nhưng nếu stop_event 
                # được set, nó sẽ thức dậy ngay lập tức.
                stop_event.wait(sleep_time)

    logger.info("Luồng Send đã tắt.")


def signal_handler(sig, frame):
    """Xử lý Ctrl+C, set Event để các luồng dừng lại an toàn."""
    global stop_event
    logger.info("Tín hiệu Ctrl+C được nhận. Đang dừng các luồng...")
    stop_event.set()

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    
    logger.info("Bắt đầu client... (Nhấn Ctrl+C để thoát)")

    # Khởi tạo và chạy 2 luồng
    capture_thread = threading.Thread(target=capture_thread_func)
    sender_thread = threading.Thread(target=send_thread_func)
    
    capture_thread.start()
    sender_thread.start()
    
    # Chờ 2 luồng chạy xong
    capture_thread.join()
    sender_thread.join()
    
    logger.info("Client đã tắt thành công.")