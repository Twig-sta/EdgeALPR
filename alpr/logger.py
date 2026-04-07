import json
import os
import cv2
from datetime import datetime

LOG_FILE = "logs/detections.json"


def log_detection(plate_text, image_path, status):

    entry = {
        "plate": plate_text,
        "image": image_path,
        "status": status,  
        "timestamp": datetime.now().isoformat()
    }

    try:
        with open(LOG_FILE, "r") as f:
            data = json.load(f)
    except:
        data = []

    data.append(entry)

    with open(LOG_FILE, "w") as f:
        json.dump(data, f, indent=2)