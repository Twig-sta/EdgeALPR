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
    text = re.sub(r'[^A-Z0-9]', '', text.upper())

    # reject obvious junk
    if len(text) < 4 or len(text) > 8:
        return ""

    # must contain both letters AND numbers
    if not (re.search(r'[A-Z]', text) and re.search(r'[0-9]', text)):
        return ""

    return text

# This function processes a single video frame to detect license plates and extract text from them using OCR
def process_frame(frame):
    results = []

    authorized_plates = load_authorized_plates()
    plates = detect_plates(frame)

    plates = detect_plates(frame)

    def plate_score(p):
        x, y, w, h = p['bbox']
        area = w * h
        ratio = w / float(h)

    # ideal plate ratio ~4–5
        ratio_score = 1 - abs(ratio - 4.5) / 4.5

    # prefer wider plates
        width_score = w / 300.0

    # normalize area (avoid huge blobs winning)
        area_score = min(area / 20000.0, 1.0)

        return (ratio_score * 0.5) + (width_score * 0.3) + (area_score * 0.2)


    plates = sorted(plates, key=plate_score, reverse=True)

    if plates:
        plates = [plates[0]]

    for plate in plates:
        h, w = plate['image'].shape[:2]

        crop = plate['image'][int(h*0.35):int(h*0.75), 0:w]

        plate_img = crop

        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        
        # boost contrast
        gray = cv2.equalizeHist(gray)

        # reduce noise
        blur = cv2.GaussianBlur(gray, (5,5), 0)

        # threshold
        thresh = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
)

        text = pytesseract.image_to_string(
            thresh,
            config='--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        ).strip()    

        print("RAW OCR:", text)    
        
        # Filter text
        text = filter_plate_text(text)
        
        image_path = None
        status = "unknown"

        if text:
            if is_authorized(text, authorized_plates):
                status = "authorized"
            else:
                status = "unauthorized"

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