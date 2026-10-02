"""
Facial Style & Micro-Expression Morphological Analyzer
======================================================
AGB1303 - AI Problem Solving Techniques (Batch 6)
Human Emotion Recognition Using Facial Expressions and Speech Modulation

Performs morphological computer-vision feature analysis on three facial zones:
1. Mouth Morphology & Curvature:
   - Curved Up (Smiley)
   - Curved Down (Frown)
   - Full Open / Gape
   - Tense / Compressed
   - Neutral Horizontal
2. Eye Aperture & Eyelid State:
   - Partially Closed / Drooping
   - Wide Open
   - Normal Open
3. Forehead & Glabellar Brow Furrows:
   - Shrunk / Wrinkles (Furrowed Corrugator)
   - Raised Lines (Frontalis Activation)
   - Plain Smooth Forehead

Outputs human-interpretable morphological reasoning:
- Mouth curved down and eyes closed partial -> Sad
- Some shrink on forehead, wide open of eye and full open/tense mouth -> Angry
- Mouth curved up like smiley face, normal eye and normal plain forehead without changes -> Happy
- Wide open eyes, open O-mouth, and raised forehead lines -> Surprise
- Mouth neutral horizontal, normal eye, and plain smooth forehead -> Neutral
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple
from src.config import EMOTION_CLASSES


class FacialStyleAnalyzer:
    """
    Morphological feature extraction and rule-based micro-expression style analyzer.
    """

    def __init__(self, cascades_dir: Optional[str] = None):
        # Load OpenCV Haar cascades
        default_dir = cv2.data.haarcascades
        self.eye_cascade = cv2.CascadeClassifier(default_dir + "haarcascade_eye.xml")
        self.smile_cascade = cv2.CascadeClassifier(default_dir + "haarcascade_smile.xml")

    def _prepare_face_image(self, face_img: np.ndarray, target_size: Tuple[int, int] = (160, 160)) -> np.ndarray:
        """Standardizes face crop to 160x160 grayscale matrix."""
        if face_img is None or face_img.size == 0:
            return np.zeros(target_size, dtype=np.uint8)

        if len(face_img.shape) == 3:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_img.copy()

        resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_CUBIC)
        return resized

    def _extract_mouth_morphology(self, face: np.ndarray, glab_furrow: float) -> Dict[str, Any]:
        """
        Analyzes the lower facial region (mouth and lips):
        - Detects mouth contour curvature (upwards/smile, downwards/frown, horizontal/neutral)
        - Detects vertical gape / mouth opening ratio
        - Checks smile cascade presence
        """
        h, w = face.shape
        # 1. Smile Cascade in lower half of face
        lower_face = face[int(h * 0.55):int(h * 0.95), int(w * 0.15):int(w * 0.85)]
        has_smile = False
        if not self.smile_cascade.empty() and lower_face.size > 0:
            smiles = self.smile_cascade.detectMultiScale(
                lower_face, scaleFactor=1.3, minNeighbors=18, minSize=(20, 12)
            )
            has_smile = len(smiles) > 0

        # 2. Oral Cavity Contour & Aspect Ratio (detects gaping / full open mouth or frown droop)
        mouth_roi = face[int(h * 0.62):int(h * 0.92), int(w * 0.20):int(w * 0.80)]
        mh, mw = mouth_roi.shape
        thresh = cv2.adaptiveThreshold(
            mouth_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4
        )
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        center_x = mw / 2.0
        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if area > 18:
                x, y, w_box, h_box = cv2.boundingRect(c)
                # Distance from horizontal center
                dist = abs((x + w_box / 2.0) - center_x)
                # Penalize contours glued to outer border
                if x > 2 and (x + w_box) < (mw - 2):
                    candidates.append((dist, area, c, (x, y, w_box, h_box)))
                else:
                    candidates.append((dist + 20.0, area, c, (x, y, w_box, h_box)))

        aspect = 0.35
        curve = 0.0
        w_c, h_c = 50, 18
        if candidates:
            # Pick contour closest to mouth center with significant area
            candidates.sort(key=lambda item: (item[0], -item[1]))
            best = candidates[0]
            c = best[2]
            pts = c.reshape(-1, 2)
            pt_l = pts[np.argmin(pts[:, 0])]
            pt_r = pts[np.argmax(pts[:, 0])]
            pt_top = pts[np.argmin(pts[:, 1])]
            pt_bot = pts[np.argmax(pts[:, 1])]
            w_c = max(1, pt_r[0] - pt_l[0])
            h_c = max(1, pt_bot[1] - pt_top[1])
            aspect = float(h_c / w_c)
            # Upward curve: corners higher physically (lower y) than mid
            corners_y = (float(pt_l[1]) + float(pt_r[1])) / 2.0
            mid_y = (float(pt_top[1]) + float(pt_bot[1])) / 2.0
            curve = float(mid_y - corners_y)

        # 3. Classify Mouth Style
        mouth_y_pos = best[3][1] if candidates else int(mh * 0.3)
        if aspect >= 0.70:
            mouth_style = "Full Open"
        elif has_smile or curve > 3.0:
            mouth_style = "Curved Up (Smiley)"
        elif aspect < 0.32 and glab_furrow > 28.0 and mouth_y_pos < int(mh * 0.4):
            mouth_style = "Tense / Compressed"
        elif glab_furrow < 25.0 and not has_smile and aspect < 0.55:
            mouth_style = "Neutral Horizontal"
        elif curve < -0.5 or (mouth_y_pos >= int(mh * 0.40) and not has_smile and glab_furrow > 30.0):
            mouth_style = "Curved Down (Frown)"
        elif abs(curve) <= 3.0:
            mouth_style = "Neutral Horizontal"
        else:
            mouth_style = "Curved Down (Frown)"

        return {
            "mouth_style": mouth_style,
            "has_smile": has_smile,
            "aspect_ratio": aspect,
            "curvature": curve,
            "contour_width": w_c,
            "contour_height": h_c
        }

    def _extract_eye_morphology(self, face: np.ndarray, glab_furrow: float, mouth_aspect: float) -> Dict[str, Any]:
        """
        Analyzes the ocular region:
        - Detects eye aperture / opening ratio
        - Checks for squinted/drooping or wide open staring states
        """
        h, w = face.shape
        upper_eyes = face[int(h * 0.22):int(h * 0.52), int(w * 0.12):int(w * 0.88)]
        eyes_cnt = 0
        if not self.eye_cascade.empty() and upper_eyes.size > 0:
            eyes = self.eye_cascade.detectMultiScale(
                upper_eyes, scaleFactor=1.1, minNeighbors=3, minSize=(14, 14)
            )
            eyes_cnt = len(eyes)

        # Eye ROI aperture calculation (left & right eye regions)
        left_eye = face[int(h * 0.24):int(h * 0.46), int(w * 0.16):int(w * 0.46)]
        right_eye = face[int(h * 0.24):int(h * 0.46), int(w * 0.54):int(w * 0.84)]

        def get_aperture(roi):
            if roi.size == 0:
                return 0.30
            thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            return float(np.mean(thresh > 0))

        aperture = (get_aperture(left_eye) + get_aperture(right_eye)) / 2.0

        # Classify Eye Style:
        # Drooping / partially closed: 0 eyes detected or (aperture < 0.28 and glab_furrow > 38.0)
        # Wide open: mouth full open OR (glab_furrow > 28.0 and eyes_cnt >= 1 and aperture > 0.40)
        # Normal open: standard relaxation
        if (glab_furrow > 38.0 and mouth_aspect < 0.50 and not (mouth_aspect >= 0.70)) or (eyes_cnt == 0 and aperture < 0.32):
            eye_style = "Partially Closed (Drooping)"
        elif mouth_aspect >= 0.70 or (glab_furrow > 28.0 and aperture > 0.48):
            eye_style = "Wide Open"
        else:
            eye_style = "Normal Open"

        return {
            "eye_style": eye_style,
            "eyes_detected": eyes_cnt,
            "aperture_ratio": aperture
        }

    def _extract_forehead_morphology(self, face: np.ndarray) -> Dict[str, Any]:
        """
        Analyzes the upper forehead and glabellar brow region:
        - Detects skin wrinkles / horizontal worry lines (Frontalis muscle activation)
        - Detects corrugator glabellar furrow / vertical shrink (Angry frown)
        - Detects plain smooth forehead without changes (Happy / Neutral)
        """
        h, w = face.shape

        # 1. Forehead horizontal worry lines
        fhead = face[int(h * 0.06):int(h * 0.28), int(w * 0.22):int(w * 0.78)]
        edges = cv2.Canny(fhead, 35, 110)
        edge_density = float(np.mean(edges > 0))
        sob_y = cv2.Sobel(fhead, cv2.CV_64F, 0, 1, ksize=3)
        horiz_energy = float(np.mean(np.abs(sob_y)))

        # 2. Glabellar Brow Furrow (between eyebrows)
        glab = face[int(h * 0.22):int(h * 0.36), int(w * 0.36):int(w * 0.64)]
        sob_x = cv2.Sobel(glab, cv2.CV_64F, 1, 0, ksize=3)
        glab_furrow = float(np.mean(np.abs(sob_x)))

        # Classify Forehead Style
        if glab_furrow >= 30.0:
            forehead_style = "Shrunk / Wrinkles (Furrowed)"
        elif horiz_energy >= 25.0 or edge_density >= 0.06:
            forehead_style = "Raised Lines"
        else:
            forehead_style = "Plain Smooth Forehead"

        return {
            "forehead_style": forehead_style,
            "glabella_furrow": glab_furrow,
            "horiz_energy": horiz_energy,
            "edge_density": edge_density
        }

    def analyze_face_style(
        self,
        face_bgr_or_gray: np.ndarray,
        full_frame_bgr: Optional[np.ndarray] = None,
        bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end morphological facial style analysis:
        - Extracts mouth, eye, and forehead morphology
        - Matches human-interpretable micro-expression rules
        - Calculates calibrated probability distribution across 5 emotions
        """
        # Crop from full frame if bbox provided
        if full_frame_bgr is not None and bbox is not None:
            x, y, w, h = bbox
            h_f, w_f = full_frame_bgr.shape[:2]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w_f, x + w), min(h_f, y + h)
            face_crop = full_frame_bgr[y1:y2, x1:x2]
        else:
            face_crop = face_bgr_or_gray

        face = self._prepare_face_image(face_crop)

        # 1. Forehead first (to inform ocular & mouth tension)
        forehead_info = self._extract_forehead_morphology(face)
        glab_furrow = forehead_info["glabella_furrow"]

        # 2. Mouth
        mouth_info = self._extract_mouth_morphology(face, glab_furrow)
        mouth_aspect = mouth_info["aspect_ratio"]

        # 3. Eyes
        eye_info = self._extract_eye_morphology(face, glab_furrow, mouth_aspect)

        m_style = mouth_info["mouth_style"]
        e_style = eye_info["eye_style"]
        f_style = forehead_info["forehead_style"]

        # ── Morphological Rule Scoring ─────────────────────────────────────────
        scores = {emo: 0.10 for emo in EMOTION_CLASSES}

        # 1. HAPPY: Mouth curved up (smiley) + normal eyes + plain smooth forehead
        if m_style == "Curved Up (Smiley)":
            scores["Happy"] += 0.70
        if e_style == "Normal Open":
            scores["Happy"] += 0.20
        if f_style == "Plain Smooth Forehead":
            scores["Happy"] += 0.30

        # 2. SAD: Mouth curved down (frown) + eyes partially closed (drooping)
        if m_style == "Curved Down (Frown)":
            scores["Sad"] += 0.70
        if e_style == "Partially Closed (Drooping)":
            scores["Sad"] += 0.50

        # 3. ANGRY: Forehead shrink/wrinkles + eyes wide/glaring + full open or tense mouth
        if f_style == "Shrunk / Wrinkles (Furrowed)":
            scores["Angry"] += 0.60
        if m_style in ["Tense / Compressed", "Full Open"]:
            scores["Angry"] += 0.45
        if e_style == "Wide Open" and f_style == "Shrunk / Wrinkles (Furrowed)":
            scores["Angry"] += 0.60

        # 4. SURPRISE: Wide open eyes + full open O-mouth + raised forehead lines
        if m_style == "Full Open":
            scores["Surprise"] += 0.65
        if f_style == "Raised Lines":
            scores["Surprise"] += 0.45
        if e_style == "Wide Open":
            scores["Surprise"] += 0.35

        # 5. NEUTRAL: Neutral horizontal mouth + normal eyes + plain smooth forehead
        # ONLY award Neutral when the mouth is genuinely neutral horizontal without expressivity
        if m_style == "Neutral Horizontal" and f_style != "Shrunk / Wrinkles (Furrowed)":
            scores["Neutral"] += 0.50
            if e_style == "Normal Open":
                scores["Neutral"] += 0.25
            if f_style == "Plain Smooth Forehead":
                scores["Neutral"] += 0.25

        # Softmax calibration with temperature
        temp = 0.45
        logits = np.array([scores[emo] for emo in EMOTION_CLASSES], dtype=np.float64)
        exp_logits = np.exp((logits - np.max(logits)) / temp)
        probs_arr = exp_logits / np.sum(exp_logits)
        probabilities = {emo: float(probs_arr[i]) for i, emo in enumerate(EMOTION_CLASSES)}

        # Predicted emotion & confidence with expressive salience calibration
        predicted_emotion = max(probabilities, key=probabilities.get)
        if predicted_emotion == "Neutral":
            non_neutral = {e: p for e, p in probabilities.items() if e != "Neutral"}
            best_nn = max(non_neutral, key=non_neutral.get)
            if non_neutral[best_nn] >= 0.22 and (probabilities["Neutral"] - non_neutral[best_nn]) < 0.15:
                predicted_emotion = best_nn
        confidence = probabilities[predicted_emotion]

        # Human-Interpretable Description
        if predicted_emotion == "Happy":
            style_description = "Mouth curved up like smiley face, normal eyes, and plain smooth forehead without changes."
        elif predicted_emotion == "Sad":
            style_description = "Mouth curved down (frown) and eyes partially closed / drooping."
        elif predicted_emotion == "Angry":
            style_description = "Shrunk furrow wrinkles on forehead, intense wide eyes, and tense/open mouth."
        elif predicted_emotion == "Surprise":
            style_description = "Wide open eyes, gaping open O-mouth, and raised forehead lines."
        else:
            style_description = "Neutral horizontal mouth, normal eyes, and plain smooth forehead without changes."

        return {
            "mouth_style": m_style,
            "eye_style": e_style,
            "forehead_style": f_style,
            "style_description": style_description,
            "predicted_emotion": predicted_emotion,
            "confidence": confidence,
            "probabilities": probabilities,
            "metrics": {
                "mouth_curvature": float(mouth_info["curvature"]),
                "mouth_aspect_ratio": float(mouth_info["aspect_ratio"]),
                "smile_detected": bool(mouth_info["has_smile"]),
                "eyes_detected": int(eye_info["eyes_detected"]),
                "eye_aperture_ratio": float(eye_info["aperture_ratio"]),
                "glabella_furrow": float(forehead_info["glabella_furrow"]),
                "forehead_edge_density": float(forehead_info["edge_density"]),
                "horiz_energy": float(forehead_info["horiz_energy"])
            }
        }
