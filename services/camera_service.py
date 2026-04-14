# services/camera_service.py
# This module defines the CameraService class, which is responsible for interfacing with the camera hardware to capture video frames. 
# It uses OpenCV to access the camera and retrieve frames in real-time. 

import cv2
class CameraService:
    # Initialize the camera service by opening a connection to the default camera (index 0).
    def __init__(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

        if not self.cap.isOpened():
            print("camera failed to open")
        else:
            print("camera opened successfully")

    # This method captures a single frame from the camera. It returns the captured frame if successful, or None if it fails to grab a frame.
    def get_frame(self):
        ret, frame = self.cap.read()

        if not ret:
            print("camera failed to grab frame")
            return None
        else:
            print("camera grabbed frame successfully")
        
        return frame
