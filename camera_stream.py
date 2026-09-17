"""
camera_stream.py
One CameraWorker per CCTV camera. Each runs in its own thread:
  - connects to the RTSP stream (or webcam index for testing)
  - continuously reads frames
  - auto-reconnects if the stream drops
  - runs YOLO every Nth frame (PROCESS_EVERY_N_FRAMES) to save CPU/RAM
  - on a confirmed snake detection: saves a screenshot, logs to DB, triggers alarm
  - keeps the latest annotated frame available for the web dashboard (MJPEG)
"""

import os
import time
import threading
from datetime import datetime

import cv2

import config
import database
import alarm


class CameraWorker:
    def __init__(self, camera_cfg, detector):
        self.id = camera_cfg["id"]
        self.name = camera_cfg["name"]
        self.rtsp_url = camera_cfg["rtsp_url"]
        self.detector = detector

        self._running = False
        self._thread = None
        self._cap = None

        self._latest_frame = None
        self._frame_lock = threading.Lock()

        self._frame_count = 0
        self._last_detection_time = 0
        self.status = "starting"  # starting | online | reconnecting | offline

    # -- lifecycle ---------------------------------------------------------

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._cap:
            self._cap.release()

    # -- frame access for the dashboard -------------------------------------

    def get_latest_jpeg(self):
        with self._frame_lock:
            if self._latest_frame is None:
                return None
            ok, buf = cv2.imencode(".jpg", self._latest_frame)
            return buf.tobytes() if ok else None

    # -- main loop -----------------------------------------------------------

    def _connect(self):
        source = self.rtsp_url
        # Allow integer webcam indices for local testing (e.g. 0)
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimize latency/RAM buildup
        return cap

    def _run(self):
        database.update_camera_status(self.id, self.name, "starting")

        while self._running:
            self._cap = self._connect()

            if not self._cap or not self._cap.isOpened():
                self.status = "offline"
                database.update_camera_status(self.id, self.name, "offline")
                database.increment_reconnect(self.id)
                print(f"[{self.name}] Could not connect. Retrying in {config.RECONNECT_DELAY_SECONDS}s...")
                time.sleep(config.RECONNECT_DELAY_SECONDS)
                continue

            self.status = "online"
            database.update_camera_status(self.id, self.name, "online")
            print(f"[{self.name}] Connected.")

            # inner read loop - runs until stream drops
            while self._running:
                ok, frame = self._cap.read()
                if not ok or frame is None:
                    print(f"[{self.name}] Stream dropped. Reconnecting...")
                    self.status = "reconnecting"
                    database.update_camera_status(self.id, self.name, "reconnecting")
                    database.increment_reconnect(self.id)
                    break  # go back out to reconnect

                frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
                self._frame_count += 1

                display_frame = frame
                if self._frame_count % config.PROCESS_EVERY_N_FRAMES == 0:
                    display_frame = self._process_frame(frame)

                with self._frame_lock:
                    self._latest_frame = display_frame

                database.update_camera_status(self.id, self.name, "online")

            if self._cap:
                self._cap.release()
            if self._running:
                time.sleep(config.RECONNECT_DELAY_SECONDS)

        self.status = "offline"
        database.update_camera_status(self.id, self.name, "offline")

    # -- detection -----------------------------------------------------------

    def _process_frame(self, frame):
        try:
            detections = self.detector.detect(frame)
        except Exception as e:
            print(f"[{self.name}] Detection error: {e}")
            return frame

        if detections:
            annotated = self.detector.draw_boxes(frame.copy(), detections)
            self._handle_detection(annotated, detections)
            return annotated

        return frame

    def _handle_detection(self, annotated_frame, detections):
        now = time.time()
        if now - self._last_detection_time < config.DETECTION_COOLDOWN_SECONDS:
            return  # still in cooldown, skip logging/alarm/screenshot spam
        self._last_detection_time = now

        best = max(detections, key=lambda d: d["confidence"])
        confidence = best["confidence"]

        image_path = None
        if config.SAVE_SCREENSHOT_ON_DETECTION:
            image_path = self._save_screenshot(annotated_frame)

        database.log_detection(self.id, self.name, confidence, image_path)
        database.update_camera_status(self.id, self.name, "online", detection=True)
        alarm.trigger_alarm(self.name, confidence)

    def _save_screenshot(self, frame):
        os.makedirs(config.SCREENSHOT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{self.id}_{timestamp}.jpg"
        filepath = os.path.join(config.SCREENSHOT_DIR, filename)
        cv2.imwrite(filepath, frame)
        # store relative path (for use in Flask static URL)
        return f"screenshots/{filename}"
