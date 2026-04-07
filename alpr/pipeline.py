# ** This file contains the main ALPR pipeline **#

import cv2
import pytesseract
import re
from alpr.detector import detect_plates
from alpr.utils import save_plate_image
from alpr.logger import log_detection 
from alpr.postprocess import load_authorized_plates, is_authorized

# Valid characters that appear on license plates: uppercase letters, numbers, and hyphens
VALID_PLATE_CHARACTERS = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-')

last_detected = set()

# Filter OCR output to only include valid license plate characters
def filter_plate_text(text):
    text = text.upper()

    text = re.sub(r'[^A-Z0-9]', '', text)

    if len(text) < 5:
        return ""

    return text

# This function processes a single video frame to detect license plates and extract text from them using OCR
def process_frame(frame):
    results = []

    authorized_plates = load_authorized_plates()
    plates = detect_plates(frame)

    for plate in plates:
        plate_img = plate['image']

        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray, config='--psm 8').strip()
        
        # Filter text
        text = filter_plate_text(text)

        image_path = None
        status = "unknown"

        if text:
            # ✅ Determine status
            if is_authorized(text, authorized_plates):
                status = "authorized"
            else:
                status = "unauthorized"

            # ✅ Prevent duplicate logging
            if text not in last_detected:
                last_detected.add(text)

                image_path = save_plate_image(plate_img)
                log_detection(text, image_path, status)

        results.append({
            'bbox': plate["bbox"],
            'text': text,
            'image': image_path,
            'status': status 
        })

    return results