"""
Facial Preprocessing & Face Tracking Module
AGB1303 - AI Problem Solving Techniques
Handles Face Detection, Landmark Tracking (Eyes & Mouth), Normalization, and HUD Drawing.
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from src.config import FACE_IMAGE_SIZE, EMOTION_CLASSES, EMOTION_COLORS


class FacePreprocessor:
    """Detects faces, tracks landmarks (eyes/mouth), extracts ROIs, and formats for CNN."""

    def __init__(self, cascade_path: str = None):
        # Locate Haar Cascades
        if cascade_path and Path(cascade_path).exists():
            self.face_cascade = cv2.CascadeClassifier(str(cascade_path))
        else:
            default_xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.face_cascade = cv2.CascadeClassifier(default_xml)

        eye_xml = cv2.data.haarcascades + "haarcascade_eye.xml"
        smile_xml = cv2.data.haarcascades + "haarcascade_smile.xml"

        self.eye_cascade = cv2.CascadeClassifier(eye_xml)
        self.smile_cascade = cv2.CascadeClassifier(smile_xml)

        # Smooth tracking cache: {face_id: (x, y, w, h)}
        self.tracking_history = {}
        self.next_face_id = 1

    def detect_faces(self, frame_bgr: np.ndarray, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
        """Detects bounding boxes of faces in a BGR frame: [(x, y, w, h), ...]"""
        if frame_bgr is None or frame_bgr.size == 0:
            return []
        
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.face_cascade.empty():
            h, w = gray.shape
            return [(0, 0, w, h)]
            
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size
        )
        return faces

    def detect_landmarks(self, frame_bgr: np.ndarray, bbox: tuple) -> dict:
        """
        Detects internal facial landmarks within the face bounding box:
        - Eyes (left, right)
        - Mouth / Smile region
        """
        x, y, w, h = bbox
        face_gray = cv2.cvtColor(frame_bgr[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        
        landmarks = {"eyes": [], "mouth": None}

        # Detect eyes in upper 60% of face
        upper_face = face_gray[0:int(h * 0.6), :]
        if not self.eye_cascade.empty():
            eyes = self.eye_cascade.detectMultiScale(upper_face, scaleFactor=1.1, minNeighbors=4, minSize=(15, 15))
            for (ex, ey, ew, eh) in eyes[:2]:
                landmarks["eyes"].append((x + ex + ew // 2, y + ey + eh // 2))

        # Detect smile/mouth in lower 50% of face
        lower_face = face_gray[int(h * 0.5):, :]
        if not self.smile_cascade.empty():
            smiles = self.smile_cascade.detectMultiScale(lower_face, scaleFactor=1.7, minNeighbors=20, minSize=(25, 15))
            if len(smiles) > 0:
                sx, sy, sw, sh = smiles[0]
                landmarks["mouth"] = (x + sx + sw // 2, y + int(h * 0.5) + sy + sh // 2)

        return landmarks

    def preprocess_face(self, face_bgr_or_gray: np.ndarray, target_size=FACE_IMAGE_SIZE, enable_clahe: bool = True) -> np.ndarray:
        """
        Converts face crop to normalized grayscale 48x48 float32 array in [0, 1].
        Applies CLAHE (Contrast-Limited Adaptive Histogram Equalization) for robust
        invariance across South Asian / Indian skin tones (Fitzpatrick Types III-VI)
        and varying ambient lighting conditions.
        """
        if face_bgr_or_gray is None or face_bgr_or_gray.size == 0:
            return np.zeros(target_size, dtype=np.float32)

        if len(face_bgr_or_gray.shape) == 3:
            gray = cv2.cvtColor(face_bgr_or_gray, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_bgr_or_gray

        if enable_clahe:
            # CLAHE provides balanced contrast across darker and lighter facial regions without clipping highlights
            clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
            gray_eq = clahe.apply(gray)
        else:
            gray_eq = cv2.equalizeHist(gray)

        resized = cv2.resize(gray_eq, target_size, interpolation=cv2.INTER_AREA)
        normalized = resized.astype(np.float32) / 255.0
        return normalized

    def to_tensor(self, normalized_face: np.ndarray) -> torch.Tensor:
        """Converts (48, 48) normalized array into PyTorch batch tensor: (1, 1, 48, 48)."""
        tensor = torch.from_numpy(normalized_face).unsqueeze(0).unsqueeze(0)
        return tensor.float()

    def process_frame(self, frame_bgr: np.ndarray):
        """Processes frame, returning face tensors, bboxes, and internal landmarks."""
        faces = self.detect_faces(frame_bgr)
        processed_tensors = []
        valid_bboxes = []
        landmarks_list = []

        for (x, y, w, h) in faces:
            pad_w = int(w * 0.1)
            pad_h = int(h * 0.1)
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(frame_bgr.shape[1], x + w + pad_w)
            y2 = min(frame_bgr.shape[0], y + h + pad_h)

            face_roi = frame_bgr[y1:y2, x1:x2]
            if face_roi.size > 0:
                normalized = self.preprocess_face(face_roi)
                tensor = self.to_tensor(normalized)
                processed_tensors.append(tensor)
                valid_bboxes.append((x, y, w, h))
                landmarks_list.append(self.detect_landmarks(frame_bgr, (x, y, w, h)))

        return processed_tensors, valid_bboxes, landmarks_list

    @staticmethod
    def draw_futuristic_hud(
        frame: np.ndarray,
        bbox: tuple,
        emotion: str,
        confidence: float,
        face_id: int = 1,
        landmarks: dict = None,
        tracking_active: bool = True
    ):
        """Draws modern, futuristic HUD with corner brackets, glowing indicators, and landmarks."""
        x, y, w, h = bbox
        
        # Color: Emerald green if tracking active, amber if uncertain, red if error
        if tracking_active:
            glow_color = (0, 230, 118)   # Neon Emerald
        else:
            glow_color = (0, 215, 255)   # Amber/Cyan

        # 1. Corner brackets (futuristic targeting reticle)
        line_len = max(15, int(w * 0.18))
        thickness = 2

        # Top-Left
        cv2.line(frame, (x, y), (x + line_len, y), glow_color, thickness)
        cv2.line(frame, (x, y), (x, y + line_len), glow_color, thickness)
        # Top-Right
        cv2.line(frame, (x + w, y), (x + w - line_len, y), glow_color, thickness)
        cv2.line(frame, (x + w, y), (x + w, y + line_len), glow_color, thickness)
        # Bottom-Left
        cv2.line(frame, (x, y + h), (x + line_len, y + h), glow_color, thickness)
        cv2.line(frame, (x, y + h), (x, y + h - line_len), glow_color, thickness)
        # Bottom-Right
        cv2.line(frame, (x + w, y + h), (x + w - line_len, y + h), glow_color, thickness)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - line_len), glow_color, thickness)

        # Subtle thin boundary
        cv2.rectangle(frame, (x, y), (x + w, y + h), (glow_color[0]//3, glow_color[1]//3, glow_color[2]//3), 1)

        # 2. Draw Eye and Mouth Landmarks
        if landmarks:
            for (ex, ey) in landmarks.get("eyes", []):
                cv2.circle(frame, (ex, ey), 3, (0, 255, 255), -1)
                cv2.circle(frame, (ex, ey), 6, (0, 255, 255), 1)
            mouth_pt = landmarks.get("mouth")
            if mouth_pt:
                cv2.circle(frame, mouth_pt, 3, (255, 100, 255), -1)

        # 3. HUD Tags & Badges
        id_tag = f"ID #{face_id:02d}"
        pred_tag = f"{emotion.upper()} {confidence*100:.1f}%"

        # Upper Tag: ID & Tracking
        cv2.rectangle(frame, (x, max(0, y - 24)), (x + 85, y), (15, 23, 42), -1)
        cv2.putText(frame, id_tag, (x + 6, max(16, y - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

        # Lower Tag: Emotion & Confidence
        cv2.rectangle(frame, (x, y + h), (x + w, y + h + 24), (15, 23, 42), -1)
        cv2.putText(frame, pred_tag, (x + 8, y + h + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.50, glow_color, 1, cv2.LINE_AA)

        return frame

    @staticmethod
    def draw_clean_box(
        frame: np.ndarray,
        bbox: tuple,
        emotion: str,
        confidence: float,
        style_info: dict = None
    ):
        """Draws a clean, modern, medium-basic bounding box with emotion badge and morphological style tag."""
        x, y, w, h = bbox
        color_map = {
            "Angry": (68, 68, 239),     # Red
            "Happy": (16, 185, 129),    # Emerald
            "Neutral": (160, 160, 160), # Slate/Gray
            "Sad": (246, 130, 59),      # Blue
            "Surprise": (11, 158, 245)  # Amber
        }
        bgr = color_map.get(emotion, (16, 185, 129))
        
        # Clean bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), bgr, 2)
        
        # Emotion Label Banner on top
        label = f"{emotion} ({confidence * 100:.1f}%)"
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        tag_y = max(y - 6, text_h + 8)
        cv2.rectangle(frame, (x, tag_y - text_h - 6), (x + text_w + 12, tag_y + 4), bgr, -1)
        cv2.putText(frame, label, (x + 6, tag_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        # Morphological Style Bottom Tag (if provided)
        if style_info and isinstance(style_info, dict):
            m_s = style_info.get("mouth_style", "Normal").replace("Curved ", "").replace(" (Smiley)", "").replace(" (Frown)", "")
            e_s = style_info.get("eye_style", "Normal").replace(" (Drooping)", "")
            f_s = "Smooth" if "Smooth" in style_info.get("forehead_style", "") else ("Shrunk" if "Shrunk" in style_info.get("forehead_style", "") else "Raised")
            style_tag = f"Mouth: {m_s} | Eyes: {e_s} | Brow: {f_s}"

            (st_w, st_h), _ = cv2.getTextSize(style_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            bot_y = y + h + st_h + 8
            if bot_y < frame.shape[0]:
                cv2.rectangle(frame, (x, y + h), (x + st_w + 10, bot_y), (15, 23, 42), -1)
                cv2.putText(frame, style_tag, (x + 5, bot_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (226, 232, 240), 1, cv2.LINE_AA)

        return frame

    @staticmethod
    def draw_prediction(frame: np.ndarray, bbox: tuple, emotion: str, confidence: float, style_info: dict = None):
        """Draws clean medium-basic prediction box."""
        return FacePreprocessor.draw_clean_box(frame, bbox, emotion, confidence, style_info=style_info)

