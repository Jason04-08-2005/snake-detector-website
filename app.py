"""
app.py
Entry point. Loads the YOLO model once at startup, then lets the dashboard
add, start, and stop cameras dynamically (laptop webcam / phone camera /
custom RTSP-IP URL) via a small JSON API, rather than requiring every
camera to be predefined in config.py.

Cameras listed in config.CAMERAS (if any) are still auto-started on launch
as "pre-configured" cameras - useful for a fixed permanent installation -
but this is now optional. The dashboard's "Run" button is the primary way
to bring a camera online.

Run:
    python app.py
Then open:
    http://<this-machine-ip>:5000
"""

import time
import uuid
import threading

from flask import Flask, Response, render_template, jsonify, request

import config
import database
from detector import SnakeDetector
from camera_stream import CameraWorker

app = Flask(__name__)

workers = {}          # camera_id -> CameraWorker
workers_lock = threading.Lock()
detector = None        # set once in start_system()


def start_system():
    global detector
    database.init_db()

    print("[System] Loading YOLO model (this can take a moment)...")
    detector = SnakeDetector()
    print("[System] Model loaded. Ready to accept cameras.")

    # Optional: auto-start any cameras predefined in config.py
    active_cams = [c for c in config.CAMERAS if c.get("enabled", True)]
    for cam_cfg in active_cams:
        _start_worker(cam_cfg["id"], cam_cfg["name"], cam_cfg["rtsp_url"])
        time.sleep(1)  # stagger startup


def _build_source(source_type, source_value):
    """
    Translates a dashboard source selection into the value CameraWorker
    understands (an RTSP/HTTP URL string, or a webcam index as a string).
    """
    source_type = (source_type or "").strip().lower()
    source_value = (source_value or "").strip()

    if source_type == "webcam":
        # source_value is a device index e.g. "0", "1"; default to 0
        return source_value if source_value != "" else "0"

    if source_type in ("phone", "custom"):
        if not source_value:
            raise ValueError("A camera URL is required for this source type.")
        return source_value

    raise ValueError(f"Unknown source_type '{source_type}'.")


def _start_worker(camera_id, name, source):
    worker = CameraWorker({"id": camera_id, "name": name, "rtsp_url": source}, detector)
    with workers_lock:
        workers[camera_id] = worker
    worker.start()
    print(f"[System] Started worker '{name}' ({camera_id}) -> {source}")
    return worker


# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------

@app.route("/")
def dashboard():
    return render_template(
        "dashboard.html",
        refresh_seconds=config.DASHBOARD_REFRESH_SECONDS,
    )


@app.route("/video_feed/<camera_id>")
def video_feed(camera_id):
    worker = workers.get(camera_id)
    if worker is None:
        return "Camera not found", 404

    def generate():
        while True:
            if camera_id not in workers:
                break  # camera was stopped - end the stream
            jpeg = worker.get_latest_jpeg()
            if jpeg is not None:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
            time.sleep(0.05)  # ~20 fps cap on stream sent to browser

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/cameras", methods=["GET"])
def api_cameras_list():
    """Returns every camera currently known (running or recently running)."""
    statuses = {row["camera_id"]: row for row in database.get_all_camera_status()}
    out = []
    with workers_lock:
        camera_ids = list(workers.keys())
    for cid in camera_ids:
        row = statuses.get(cid, {})
        out.append({
            "id": cid,
            "name": row.get("camera_name", cid),
            "status": row.get("status", "starting"),
            "last_frame_at": row.get("last_frame_at"),
            "last_detection_at": row.get("last_detection_at"),
            "reconnect_count": row.get("reconnect_count", 0),
        })
    return jsonify({"cameras": out, "stats": database.get_stats()})


@app.route("/api/cameras", methods=["POST"])
def api_cameras_create():
    """
    Body JSON: { "name": str, "source_type": "webcam"|"phone"|"custom", "source_value": str }
    Starts a new camera worker and returns its assigned id.
    """
    if detector is None:
        return jsonify({"error": "Model still loading, try again shortly."}), 503

    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip() or "Camera"
    source_type = data.get("source_type", "")
    source_value = data.get("source_value", "")

    try:
        source = _build_source(source_type, source_value)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    camera_id = "cam_" + uuid.uuid4().hex[:8]
    _start_worker(camera_id, name, source)

    return jsonify({"id": camera_id, "name": name, "status": "starting"}), 201


@app.route("/api/cameras/<camera_id>/stop", methods=["POST"])
def api_cameras_stop(camera_id):
    with workers_lock:
        worker = workers.pop(camera_id, None)
    if worker is None:
        return jsonify({"error": "Camera not found"}), 404

    worker.stop()
    database.update_camera_status(camera_id, worker.name, "offline")
    return jsonify({"id": camera_id, "status": "stopped"})


@app.route("/api/status")
def api_status():
    # Kept for backward compatibility; same data as GET /api/cameras
    return api_cameras_list()


@app.route("/api/detections")
def api_detections():
    limit = 50
    return jsonify(database.get_recent_detections(limit=limit))


if __name__ == "__main__":
    start_system()
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=False, threaded=True)
