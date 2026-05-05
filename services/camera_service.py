import cv2
import os
import threading
import time

try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None


class CameraService:
    def __init__(self):
        self.cap = None
        self.picam2 = None
        self.frame_size = self._frame_size()
        self.lock = threading.Lock()
        self.latest_frame = None
        self.running = self._open_camera()

        if not self.running:
            print("camera failed to open")
        else:
            print("camera opened successfully")
            self.thread = threading.Thread(target=self._read_frames, daemon=True)
            self.thread.start()

    def _frame_size(self):
        width = int(os.environ.get("EDGEALPR_CAMERA_WIDTH", "640"))
        height = int(os.environ.get("EDGEALPR_CAMERA_HEIGHT", "480"))
        return width, height

    def _open_camera(self):
        if Picamera2 is not None:
            try:
                self.picam2 = Picamera2()
                self.picam2.configure(
                    self.picam2.create_preview_configuration(
                        main={"format": "RGB888", "size": self.frame_size}
                    )
                )
                self.picam2.start()
                return True
            except Exception as error:
                print(f"Picamera2 unavailable: {error}")
                self.picam2 = None

        avfoundation = getattr(cv2, "CAP_AVFOUNDATION", 0)
        self.cap = cv2.VideoCapture(0, avfoundation)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = cv2.VideoCapture(0)

        if self.cap.isOpened():
            width, height = self.frame_size
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        return self.cap.isOpened()

    def _read_frames(self):
        while self.running:
            if self.picam2 is not None:
                frame = self.picam2.capture_array()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                ret = True
            else:
                ret, frame = self.cap.read()

            if ret:
                with self.lock:
                    self.latest_frame = frame
            else:
                time.sleep(0.1)

    def get_frame(self):
        deadline = time.time() + 1.0
        while self.latest_frame is None and time.time() < deadline:
            time.sleep(0.03)

        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()

        print("camera failed to grab frame")
        return None

    def release(self):
        self.running = False
        if self.picam2 is not None:
            self.picam2.stop()
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
