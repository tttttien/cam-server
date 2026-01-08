import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# S3 Configuration
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "fire")

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL")

# Recorder Configuration
RECORD_FPS = int(os.getenv("RECORD_FPS", "10"))
RECORD_AFTER_FIRE_STOPS_SEC = int(os.getenv("RECORD_AFTER_FIRE_STOPS_SEC", "5"))
PRE_ROLL_BUFFER_SEC = int(os.getenv("PRE_ROLL_BUFFER_SEC", "5"))
RECORDER_TEMP_DIR = os.getenv("RECORDER_TEMP_DIR", "recorder_temp")

# Model Configuration
MODEL_INPUT_SHAPE = (288, 288, 3)
MODEL_WEIGHTS_PATH = 'distilled_student_model_weights.weights.h5'
FIRE_THRESHOLD = 0.6
MIN_FIRE_RATIO = 0.003
