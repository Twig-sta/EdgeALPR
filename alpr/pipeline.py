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


def _plate_candidate_score(text):
    has_letter = any(char.isalpha() for char in text)
    has_digit = any(char.isdigit() for char in text)
    digit_count = sum(char.isdigit() for char in text)
    length_score = 3 if 6 <= len(text) <= 7 else 1
    mix_score = 2 if has_letter and has_digit else 0
    return length_score + mix_score + min(digit_count, 4) * 0.05 - abs(len(text) - 7) / 10


# Filter OCR output to only include valid license plate characters
def filter_plate_text(text):
    text = text.upper().replace(" ", "")
    possible = re.findall(r'[A-Z0-9]{5,8}', text)

    if not possible:
        compact = re.sub(r'[^A-Z0-9]', '', text)
        possible = [
            compact[index:index + 8]
            for index in range(max(1, len(compact) - 4))
            if 5 <= len(compact[index:index + 8]) <= 8
        ]

    if not possible:
        return ""

    return max(possible, key=_plate_candidate_score)


def _ocr_plate_image(plate_img):
    height, width = plate_img.shape[:2]
    crops = [
        plate_img,
        plate_img[int(height * 0.20):int(height * 0.88), int(width * 0.06):int(width * 0.91)],
        plate_img[int(height * 0.18):int(height * 0.88), int(width * 0.06):int(width * 0.91)],
        plate_img[int(height * 0.25):int(height * 0.86), int(width * 0.08):int(width * 0.90)],
    ]

    whitelist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    configs = [
        f"--psm 7 -c tessedit_char_whitelist={whitelist}",
        f"--psm 8 -c tessedit_char_whitelist={whitelist}",
        f"--psm 6 -c tessedit_char_whitelist={whitelist}",
    ]

    recognized = []
    for crop in crops:
        if crop.size == 0:
            continue

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

        variants = [
            gray,
            cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                5
            ),
        ]

        for image in variants:
            for config in configs:
                raw_text = pytesseract.image_to_string(image, config=config)
                text = filter_plate_text(raw_text)
                if text:
                    recognized.append(text)

    if not recognized:
        return ""

    return max(recognized, key=_plate_candidate_score)

def process_frame(frame):
    results = []

    authorized_plates = load_authorized_plates()
    plates = detect_plates(frame)

    for plate in plates:
        plate_img = plate['image']

        text = _ocr_plate_image(plate_img)

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

        if text:
            break

    return results
