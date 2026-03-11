from flask import Flask, render_template
import json
import os

app = Flask(__name__)

LOG_FILE = "logs/detections.json"

def load_detections():
    if not os.path.exists(LOG_FILE):
        return []
    
    with open(LOG_FILE, "r") as f:
        return json.load(f)
    
@app.route("/")
def index():
    detections = load_detections()
    return render_template("index.html", detections=detections)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)