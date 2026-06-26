from pathlib import Path
import sys
import os

# -----------------------------
# Project Root
# -----------------------------
FILE = Path(__file__).resolve()
ROOT = FILE.parent

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# -----------------------------
# Source Types
# -----------------------------
IMAGE = "Image"
VIDEO = "Video"
WEBCAM = "Webcam"
RTSP = "RTSP"
YOUTUBE = "YouTube"
MULTI_CAMERA = "Multi Camera"

SOURCES_LIST = [
    IMAGE,
    VIDEO,
    WEBCAM,
    RTSP,
    YOUTUBE,
    MULTI_CAMERA
]

# -----------------------------
# Image Paths
# -----------------------------
IMAGES_DIR = ROOT / "images"

DEFAULT_IMAGE = IMAGES_DIR / "img1.jpg"
DEFAULT_DETECT_IMAGE = IMAGES_DIR / "img1_det.jpg"

# -----------------------------
# Video Paths
# -----------------------------
VIDEO_DIR = ROOT / "videos"

VIDEO_1_PATH = VIDEO_DIR / "drowning_1.mp4"
VIDEO_2_PATH = VIDEO_DIR / "test.mp4"
VIDEO_3_PATH = VIDEO_DIR / "12727733-preview.mp4"

VIDEOS_DICT = {
    "Video 1": VIDEO_1_PATH,
    "Video 2": VIDEO_2_PATH,
    "Video 3": VIDEO_3_PATH,
}

# -----------------------------
# Audio
# -----------------------------
AUDIO_PATH = "distress.mp3"

# -----------------------------
# Model
# -----------------------------
MODEL_DIR = ROOT / "weights"

DETECTION_MODEL = MODEL_DIR / "best.pt"

# -----------------------------
# Webcam
# -----------------------------
WEBCAM_PATH = 0

# -----------------------------
# Multi Camera
# -----------------------------
CAMERA_CONFIG = ROOT / "camera_config.json"

MAX_CAMERAS = 16

FRAME_WIDTH = 640

FRAME_HEIGHT = 480

# -----------------------------
# Detection
# -----------------------------
CONFIDENCE = 0.25

TIMEOUT = 6

ALERT_COOLDOWN = 30

# -----------------------------
# Recording
# -----------------------------
RECORD_SECONDS = 15

FPS = 20

VIDEO_CODEC = "mp4v"

# -----------------------------
# Evidence
# -----------------------------
EVIDENCE_DIR = ROOT / "evidence"

IMAGE_DIR = EVIDENCE_DIR / "images"

VIDEO_SAVE_DIR = EVIDENCE_DIR / "videos"

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(VIDEO_SAVE_DIR, exist_ok=True)

# -----------------------------
# Logs
# -----------------------------
LOG_DIR = ROOT / "logs"

LOG_FILE = LOG_DIR / "detections.csv"

os.makedirs(LOG_DIR, exist_ok=True)

# -----------------------------
# Dashboard
# -----------------------------
SHOW_FPS = True

SHOW_CAMERA_STATUS = True

SHOW_CAMERA_LOCATION = True

SHOW_CONFIDENCE = True

SHOW_TIME = True

# -----------------------------
# Twilio
# -----------------------------
account_sid = ""

auth_token = ""

from_ = ""

to_ = ""

# -----------------------------
# ImgBB
# -----------------------------
imgbb_api = ""

# -----------------------------
# Alert
# -----------------------------
alertmsg = """
🚨 DROWNING ALERT 🚨

Someone is drowning.

Please respond immediately.
"""