import sys
import os
import cv2
import json
import time

from flask import Flask, render_template, Response, send_from_directory, redirect, url_for

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.camera_service import CameraService
from alpr.pipeline import process_frame
from alpr.visualization import draw_detections

app = Flask(__name__)

camera_service = CameraService()

last_capture = {"image": None, "detections": []}
LOG_FILE = "logs/detections.json"

# ✅ FIXED: absolute capture directory
CAPTURE_DIR = os.path.join(os.path.dirname(__file__), "captures")


def load_detections():
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE) as f:
        return json.load(f)


# -----------------------------
# LIVE VIDEO STREAM
# -----------------------------
def generate_frames():
    while True:
        try:
            frame = camera_service.get_frame()

            detections_list = process_frame(frame)

            detections_list = [
                d for d in detections_list
                if d.get("text") and d.get("text").strip() != ""
            ]

            frame = draw_detections(frame, detections_list)

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue

            frame_bytes = buffer.tobytes()

            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' +
                frame_bytes +
                b'\r\n'
            )

        except Exception as e:
            print(f"Error generating frame: {e}")


@app.route("/")
def live_feed_page():
    return render_template("live_feed.html")


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# -----------------------------
# CAPTURE IMAGE
# -----------------------------
@app.route('/capture')
def capture_image():
    global last_capture

    frame = camera_service.get_frame()

    detected_plates = process_frame(frame)

    filtered_detections = [
        d for d in detected_plates
        if d.get("text") and d.get("text").strip() != ""
    ]

    frame_with_boxes = draw_detections(frame.copy(), filtered_detections)

    filename = f"capture_{int(time.time())}.jpg"
    os.makedirs(CAPTURE_DIR, exist_ok=True)

    filepath = os.path.join(CAPTURE_DIR, filename)

    cv2.imwrite(filepath, frame_with_boxes)

    last_capture["image"] = filename
    last_capture["detections"] = filtered_detections

    return redirect(url_for('captured_page'))

@app.route('/captured')
def captured_page():
    return render_template('captured.html', capture=last_capture)


# -----------------------------
# SERVE CAPTURED IMAGES (FIXED)
# -----------------------------
@app.route('/captures/<filename>')
def serve_image(filename):
    return send_from_directory(CAPTURE_DIR, filename)


# -----------------------------
# HISTORY
# -----------------------------
@app.route('/history')
def history_page():
    try:
        with open(LOG_FILE, "r") as f:
            logs = json.load(f)
    except:
        logs = []

    return render_template('history.html', logs=logs)


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5055, debug=True)