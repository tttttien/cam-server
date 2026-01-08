# Fire Detection Camera System 🎥🔥

A modular and optimized fire detection system using FastAPI, OpenCV, and Deep Learning. Features a real-time WebSocket communication layer between camera clients and the server for ultra-low latency.

## 🚀 Features

- **Optimized Communication**: Uses WebSockets for real-time binary frame transmission.
- **Extreme Latency Optimization**: 
    - **Client-side Adaptive Dropping**: Skips encoding/sending frames if the network or CPU is busy.
    - **Server-side Frame Dropping**: AI skip frames if previous detection is still running.
    - **Adjustable Quality**: Scale-based resizing and JPEG quality control.
- **Multi-Camera Support**: Run multiple camera instances from a single codebase using unique config files.
- **Modular Architecture**: Clean separation of concerns (DB, S3, Video Recording, AI Model).
- **Secure Configuration**: Environment-based configuration (no hardcoded credentials).
- **Automatic Recording**: Automatically records fire events, adds pre-roll buffer, and uploads to S3.
- **Real-time Monitoring**: Provides an MJPEG stream for remote viewing of processed frames.

---

## 🛠 Setup & Installation

### 1. Prerequisites
- Python 3.10+
- (Optional) Docker & Docker Compose

### 2. Install & Setup (Using uv)
`uv` is the fastest way to run this project. If you haven't installed it, run: `curl -LsSf https://astral.sh/uv/install.sh | sh`

```bash
# 1. Install all dependencies into a virtual environment
uv pip install -r requirements.txt

# 2. Setup your environment variables
cp .env.example .env
# Open .env and fill in your DB/S3 credentials
```

---

## 🖥 How to Run

### 1. Start the Server
```bash
uv run python main.py
```
*Note: `uv run` will automatically use the virtual environment.*

### 2. Start the Camera Client(s)
```bash
# Run with default config
uv run python client/client.py

# Run with specific config for another camera
uv run python client/client.py client/cam2_config.json
```

---

## ⚙️ Client Performance Tuning

Edit `camera_config.json` to optimize for your network:
- `capture_fps`: Local camera capture rate (smooth local display).
- `detection_fps`: Target rate to send to AI server.
- `resize_scale`: Scale down image (e.g., `0.3` for 30% size).
- `jpeg_quality`: JPEG compression (e.g., `50` for ultra-light packets).

---

## 📡 Monitoring

- **MJPEG Stream**: `http://localhost:8000/video_feed/{camera_id}`
- **Recent Events**: `http://localhost:8000/events/{camera_id}`
- **Detection Toggle**: `http://localhost:8000/toggle_detection/{camera_id}` (Call this to turn ON/OFF AI)
- **Camera Status**: `http://localhost:8000/status/{camera_id}`
- **FCM Registration**: `POST /register_token` (Android app sends its FCM token here)

---

## 📦 Large File Management (Git LFS)
The AI weights file (`u_kan_lstm_mobilenetv2.weights.h5`) exceeds 100MB. If you cannot push to GitHub, follow these steps:
1. Install [Git LFS](https://git-lfs.github.com/).
2. Run: `git lfs install`
3. Run: `git lfs track "*.weights.h5"`
4. Add `.gitattributes`, then commit and push again.

---

## 🏗 Modular Structure

- `main.py`: Entry point and API/WebSocket routes.
- `config.py`: Configuration loader.
- `utils/`: Database and S3 helper modules.
- `services/`: Background video recording service.
- `model/`: AI model definition and segmentation logic.
- `client/`: Camera client implementation with WebSocket support.

---
*Created as part of the Clean Code Refactor project. Optimized for Real-time Fire Safety.*
