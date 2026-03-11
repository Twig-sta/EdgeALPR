from flask import Flask, render_template, Response
import cv2
import json
import os

from alpr.pipeline import process_frame
from alpr.visualization import draw_detections

app = Flask(__name__)

LOG_FILE = "logs/detections.json"

camera = cv2.VideoCapture(0)

def load_detections():
    if not os.path.exists(LOG_FILE):
        return []
    
    with open(LOG_FILE) as f:
        return json.load(f)
    
def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        detections = process_frame(frame)

        frame = draw_detections(frame, detections)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
@app.route("/")
def index():
    detections = load_detections()
    return render_template("index.html", detections=detections)

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5055, debug=True)