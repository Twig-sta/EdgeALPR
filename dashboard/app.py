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
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


# -----------------------------
# IMAGE CAPTURE
# -----------------------------
@app.route('/capture')
def capture_image():

    global last_capture

    frame = camera_service.get_frame()

    filename = f"capture_{int(time.time())}.jpg"

    capture_folder = os.path.join(os.path.dirname(__file__), "captures")
    os.makedirs(capture_folder, exist_ok=True)

    filepath = os.path.join(capture_folder, filename)

    cv2.imwrite(filepath, frame)

    detected_plates = process_frame(frame)

    last_capture["image"] = filename
    last_capture["detections"] = detected_plates

    return redirect(url_for('captured_page'))


@app.route('/captured')
def captured_page():
    return render_template('captured.html', capture=last_capture)


# -----------------------------
# SERVE CAPTURED IMAGES
# -----------------------------
@app.route('/captures/<filename>')
def serve_capture(filename):
    capture_folder = os.path.join(os.path.dirname(__file__), "captures")
    return send_from_directory(capture_folder, filename)


# -----------------------------
# RUN SERVER
# -----------------------------
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5055, debug=True)

