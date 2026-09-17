"""
config.py
Central configuration for the Snake Detection System.
Edit CAMERAS to add/remove CCTV feeds. Start with ONE camera on a 4GB RAM
machine, confirm it is stable, then uncomment/add more.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# CAMERAS
# ---------------------------------------------------------------------------
# Cameras are now normally added from the dashboard itself (pick a source -
# laptop webcam / phone camera / custom RTSP-IP URL - and press "Run").
# This list is only for OPTIONAL cameras you want auto-started every time
# the server launches, e.g. a permanently installed CCTV camera. Leave it
# empty if you only plan to add cameras from the dashboard.
#
# rtsp_url examples:
#   "rtsp://username:password@192.168.1.10:554/stream1"
#   "0"  (a numeric string means "use local webcam index 0")
CAMERAS = [
    # {
    #     "id": "cam1",
    #     "name": "Main Gate",
    #     "rtsp_url": "rtsp://admin:password@192.168.1.10:554/stream1",
    #     "enabled": True,
    # },
]

# ---------------------------------------------------------------------------
# YOLO MODEL
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")   # <-- put your trained weights here
CONFIDENCE_THRESHOLD = 0.50          # minimum confidence to count as a detection
DEVICE = "cpu"                       # "cpu" or "cuda" (use "cpu" on a 4GB laptop unless you have a GPU)

# Class name(s) in your best.pt that correspond to a snake.
# If your model has only one class this can stay as-is; otherwise list all
# class names that should trigger an alert.
SNAKE_CLASS_NAMES = ["snake"]

# ---------------------------------------------------------------------------
# PERFORMANCE (important for 4GB RAM machines)
# ---------------------------------------------------------------------------
FRAME_WIDTH = 640                    # resize frames before inference to save RAM/CPU
FRAME_HEIGHT = 480
PROCESS_EVERY_N_FRAMES = 5           # run YOLO only every Nth frame (rest are just displayed)
MAX_QUEUE_SIZE = 2                   # frame buffer per camera - keep small to avoid RAM buildup
RECONNECT_DELAY_SECONDS = 5          # wait time before retrying a dropped RTSP stream

# ---------------------------------------------------------------------------
# DETECTION BEHAVIOUR
# ---------------------------------------------------------------------------
DETECTION_COOLDOWN_SECONDS = 15      # don't log/alarm again for the same camera within this window
SAVE_SCREENSHOT_ON_DETECTION = True

# ---------------------------------------------------------------------------
# ALARM
# ---------------------------------------------------------------------------
ALARM_ENABLED = True
ALARM_SOUND_PATH = os.path.join(BASE_DIR, "static", "alarm", "alarm.wav")
ALARM_COOLDOWN_SECONDS = 15          # min gap between alarm sounds (system-wide)

# ---------------------------------------------------------------------------
# STORAGE
# ---------------------------------------------------------------------------
SCREENSHOT_DIR = os.path.join(BASE_DIR, "static", "screenshots")
DATABASE_PATH = os.path.join(BASE_DIR, "detections.db")

# ---------------------------------------------------------------------------
# WEB DASHBOARD
# ---------------------------------------------------------------------------
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
DASHBOARD_REFRESH_SECONDS = 5        # auto-refresh interval on the web page
