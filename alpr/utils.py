#**Utility functions for ALPR preprocessing**#

# This module contains utility functions
import cv2 
import os
from importlib.resources import path
from datetime import datetime

# Directory to save captured license plate images
CAPTURE_DIR = "dashboard/captures/"

def save_plate_image(image):
    os.makedirs(CAPTURE_DIR, exist_ok=True)

    filename = f"capture_{int(datetime.now().timestamp())}.jpg"
    path = os.path.join(CAPTURE_DIR, filename)

    cv2.imwrite(path, image)

    print("Saved image at:", path)

    return path

# Preprocess the image for better license plate detection 
# This function converts the image to grayscale and applies a Gaussian blur to reduce noise
def preprocess_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    blur = cv2.GaussianBlur(gray, (5, 5), 0) 

    return blur

# Detect edges in the preprocessed image using Canny edge detection
# This function takes the blurred grayscale image and applies the Canny edge detection algorithm to find edges
def detect_edges(image):
    edges = cv2.Canny(image, 50, 150)

    edges = cv2.dilate(edges, None, iterations=2)
    edges = cv2.erode(edges, None, iterations=1)

    return edges

