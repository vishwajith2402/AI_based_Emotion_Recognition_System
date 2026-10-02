"""
Camera Stream Manager Module
AGB1303 - AI Problem Solving Techniques
Handles camera capture, frame acquisition, and fallbacks.
"""

import cv2
import numpy as np
import time


class CameraManager:
    """Manages OpenCV webcam connection and frame retrieval."""

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap = None
        self.is_open = False

    def open(self) -> bool:
        """Attempts to open the video capture device."""
        try:
            # On Windows, cv2.CAP_DSHOW provides fast startup
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                # Fallback to default backend
                self.cap = cv2.VideoCapture(self.camera_index)
            
            self.is_open = self.cap.isOpened()
            if self.is_open:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            return self.is_open
        except Exception as e:
            print(f"[Camera Error] Failed to open camera {self.camera_index}: {e}")
            self.is_open = False
            return False

    def read_frame(self) -> np.ndarray:
        """Reads a single frame from the camera."""
        if not self.is_open or self.cap is None:
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def close(self):
        """Safely releases the camera resource."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.is_open = False

    @staticmethod
    def generate_dummy_frame(text: str = "Webcam Inactive") -> np.ndarray:
        """Generates a styled placeholder frame when webcam is unavailable."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (30, 30, 35)  # Dark slate background
        
        cv2.putText(
            frame,
            text,
            (160, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (180, 180, 180),
            2,
            cv2.LINE_AA
        )
        return frame
