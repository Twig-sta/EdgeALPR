from picamera2 import Picamera2
import cv2

class CameraService:
    def __init__(self):
        self.picam2 = Picamera2()

        self.picam2.configure(
            self.picam2.create_preview_configuration(
                main={"format": "RGB888", "size": (640, 480)}
            )
        )

        self.picam2.start()
        print("camera opened successfully (Picamera2)")

    def get_frame(self):
        frame = self.picam2.capture_array()

        if frame is None:
            print("camera failed to grab frame")
            return None

        # Convert RGB → BGR for OpenCV pipeline compatibility
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)