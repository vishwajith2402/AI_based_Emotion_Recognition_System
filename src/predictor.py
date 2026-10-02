"""
Unified Multimodal Emotion Recognition Engine
AGB1303 - AI Problem Solving Techniques
Orchestrates facial analysis, acoustic speech analysis, speech-to-text semantic analysis,
and Tri-Modal decision fusion (Vision + Acoustics + Text).
"""

import numpy as np
import torch
from typing import Dict, Any, Optional
from src.face_preprocessing import FacePreprocessor
from src.speech_preprocessing import SpeechPreprocessor
from src.face_model import FacialEmotionModel
from src.speech_model import SpeechEmotionModel
from src.text_emotion import TextEmotionAnalyzer
from src.fusion import MultimodalFusion
from src.facial_style_analyzer import FacialStyleAnalyzer
from src.config import (
    EMOTION_CLASSES,
    DEFAULT_FACE_WEIGHT,
    DEFAULT_SPEECH_WEIGHT,
    DEFAULT_TEXT_WEIGHT
)


class EmotionPredictor:
    """Master engine coordinating face detection, speech audio, STT text semantics, and fusion."""

    def __init__(self, device: str = None, enable_smoothing: bool = True, smoothing_alpha: float = 0.65):
        self.face_preprocessor = FacePreprocessor()
        self.speech_preprocessor = SpeechPreprocessor()
        self.text_analyzer = TextEmotionAnalyzer()
        self.style_analyzer = FacialStyleAnalyzer()

        self.face_model = FacialEmotionModel(device=device)
        self.speech_model = SpeechEmotionModel(device=device)
        self.fusion = MultimodalFusion(
            face_weight=DEFAULT_FACE_WEIGHT,
            speech_weight=DEFAULT_SPEECH_WEIGHT,
            text_weight=DEFAULT_TEXT_WEIGHT
        )

        # Temporal EMA Smoothing (filters momentary frame jitters and noise)
        self.enable_smoothing = enable_smoothing
        self.smoothing_alpha = smoothing_alpha  # Current weight vs past memory (1 - alpha)
        self.smoothed_face_probs = None
        self.smoothed_speech_probs = None
        self.smoothed_text_probs = None

    def reset_smoothing(self):
        """Clears smoothing history for a fresh session."""
        self.smoothed_face_probs = None
        self.smoothed_speech_probs = None
        self.smoothed_text_probs = None

    def _apply_ema(self, new_probs: Dict[str, float], past_probs: Optional[Dict[str, float]]) -> Dict[str, float]:
        """Applies Exponential Moving Average smoothing over probability dictionary."""
        if past_probs is None or not self.enable_smoothing:
            return new_probs.copy()

        alpha = self.smoothing_alpha
        smoothed = {}
        for emo in EMOTION_CLASSES:
            smoothed[emo] = float(alpha * new_probs.get(emo, 0.0) + (1.0 - alpha) * past_probs.get(emo, 0.0))

        total = sum(smoothed.values())
        if total > 0:
            smoothed = {k: v / total for k, v in smoothed.items()}
        return smoothed

    def predict_image(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Analyzes a single image or video frame for facial emotion,
        combining deep CNN spatial features with morphological facial style analysis.
        """
        if image_bgr is None:
            return {
                "face_detected": False,
                "top_emotion": None,
                "confidence": 0.0,
                "probabilities": None,
                "facial_style": None
            }

        tensors, bboxes, landmarks_list = self.face_preprocessor.process_frame(image_bgr)
        if not tensors:
            # Fallback: whole-image crop
            normalized = self.face_preprocessor.preprocess_face(image_bgr)
            tensor = self.face_preprocessor.to_tensor(normalized)
            raw_emotion, raw_conf, raw_probs = self.face_model.predict(tensor)
            h, w = image_bgr.shape[:2]

            # Morphological Facial Style Analysis
            style_info = self.style_analyzer.analyze_face_style(image_bgr)
            style_probs = style_info.get("probabilities", raw_probs)

            # Ensemble blend: 82% Deep CNN + 18% Morphological Style Rules
            blended = {}
            for emo in EMOTION_CLASSES:
                blended[emo] = 0.82 * raw_probs.get(emo, 0.0) + 0.18 * style_probs.get(emo, 0.0)
            total = sum(blended.values())
            if total > 0:
                blended = {k: v / total for k, v in blended.items()}

            probs = self._apply_ema(blended, self.smoothed_face_probs)
            self.smoothed_face_probs = probs
            top_emotion = max(probs, key=probs.get)
            if top_emotion == "Neutral":
                non_neutral = {e: p for e, p in probs.items() if e != "Neutral"}
                best_nn = max(non_neutral, key=non_neutral.get)
                if non_neutral[best_nn] >= 0.22 and (probs["Neutral"] - non_neutral[best_nn]) < 0.15:
                    top_emotion = best_nn
            conf = probs[top_emotion]

            return {
                "face_detected": False,
                "top_emotion": top_emotion,
                "confidence": conf,
                "probabilities": probs,
                "bboxes": [(0, 0, w, h)],
                "landmarks": [],
                "facial_style": style_info,
                "style_description": style_info.get("style_description", "")
            }

        # Analyze primary (largest) face
        primary_tensor = tensors[0]
        primary_bbox = bboxes[0]
        raw_emotion, raw_conf, raw_probs = self.face_model.predict(primary_tensor)

        # Morphological Facial Style Analysis on face ROI
        x, y, w, h = primary_bbox
        face_crop = image_bgr[max(0, y):min(image_bgr.shape[0], y+h), max(0, x):min(image_bgr.shape[1], x+w)]
        style_info = self.style_analyzer.analyze_face_style(face_crop, full_frame_bgr=image_bgr, bbox=primary_bbox)
        style_probs = style_info.get("probabilities", raw_probs)

        # Ensemble blend: 82% Deep CNN + 18% Morphological Style Rules
        blended = {}
        for emo in EMOTION_CLASSES:
            blended[emo] = 0.82 * raw_probs.get(emo, 0.0) + 0.18 * style_probs.get(emo, 0.0)
        total = sum(blended.values())
        if total > 0:
            blended = {k: v / total for k, v in blended.items()}

        probs = self._apply_ema(blended, self.smoothed_face_probs)
        self.smoothed_face_probs = probs
        top_emotion = max(probs, key=probs.get)
        if top_emotion == "Neutral":
            non_neutral = {e: p for e, p in probs.items() if e != "Neutral"}
            best_nn = max(non_neutral, key=non_neutral.get)
            if non_neutral[best_nn] >= 0.22 and (probs["Neutral"] - non_neutral[best_nn]) < 0.15:
                top_emotion = best_nn
        conf = probs[top_emotion]

        return {
            "face_detected": True,
            "top_emotion": top_emotion,
            "confidence": conf,
            "probabilities": probs,
            "bboxes": bboxes,
            "landmarks": landmarks_list,
            "facial_style": style_info,
            "style_description": style_info.get("style_description", "")
        }

    def predict_audio(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """
        Analyzes an audio clip (numpy float32 array or loaded via path) for speech emotion with optional EMA temporal smoothing.
        """
        if audio_data is None:
            return {"speech_detected": False, "top_emotion": None, "confidence": 0.0, "probabilities": None}

        tensor, silent = self.speech_preprocessor.process_audio(audio_data)
        raw_emotion, raw_conf, raw_probs = self.speech_model.predict(tensor)

        if not silent:
            probs = self._apply_ema(raw_probs, self.smoothed_speech_probs)
            self.smoothed_speech_probs = probs
            top_emotion = max(probs, key=probs.get)
            if top_emotion == "Neutral":
                non_neutral = {e: p for e, p in probs.items() if e != "Neutral"}
                best_nn = max(non_neutral, key=non_neutral.get)
                if non_neutral[best_nn] >= 0.22 and (probs["Neutral"] - non_neutral[best_nn]) < 0.15:
                    top_emotion = best_nn
            conf = probs[top_emotion]
        else:
            probs = raw_probs
            top_emotion = raw_emotion
            conf = raw_conf

        return {
            "speech_detected": not silent,
            "top_emotion": top_emotion,
            "confidence": conf,
            "probabilities": probs,
            "is_silent": silent
        }

    def predict_audio_file(self, file_path: str) -> Dict[str, Any]:
        """Loads an audio file and executes speech emotion inference."""
        audio = self.speech_preprocessor.load_audio_file(file_path)
        return self.predict_audio(audio)

    def predict_text(self, text: str) -> Dict[str, Any]:
        """
        Analyzes transcribed text using VADER and Affective Lexicon density.
        """
        result = self.text_analyzer.predict_emotion_from_text(text)
        if result["has_text"]:
            probs = self._apply_ema(result["probabilities"], self.smoothed_text_probs)
            self.smoothed_text_probs = probs
            top_emo = max(probs, key=probs.get)
            result["probabilities"] = probs
            result["top_emotion"] = top_emo
            result["confidence"] = probs[top_emo]
        return result

    def transcribe_and_predict_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Transcribes audio buffer via STT and performs linguistic sentiment analysis.
        """
        text = self.text_analyzer.transcribe_audio(audio_data, sample_rate=sample_rate)
        return self.predict_text(text)

    def predict_multimodal(
        self,
        image_bgr: Optional[np.ndarray] = None,
        audio_data: Optional[np.ndarray] = None,
        text: Optional[str] = None,
        face_weight: float = DEFAULT_FACE_WEIGHT,
        speech_weight: float = DEFAULT_SPEECH_WEIGHT,
        text_weight: float = DEFAULT_TEXT_WEIGHT
    ) -> Dict[str, Any]:
        """
        Full end-to-end Tri-Modal prediction combining face, acoustic speech, and text semantics.
        """
        self.fusion.set_weights(face_weight, speech_weight, text_weight)

        face_res = self.predict_image(image_bgr) if image_bgr is not None else None
        speech_res = self.predict_audio(audio_data) if audio_data is not None else None

        # Text analysis: either provided directly or transcribed from audio
        text_res = None
        if text is not None and len(text.strip()) > 0:
            text_res = self.predict_text(text)
        elif audio_data is not None:
            # Only transcribe if audio is not silent
            if speech_res and speech_res.get("speech_detected", False):
                text_res = self.transcribe_and_predict_audio(audio_data)

        face_detected = face_res["face_detected"] if face_res else False
        face_probs = face_res["probabilities"] if face_res else None

        speech_detected = speech_res["speech_detected"] if speech_res else False
        speech_probs = speech_res["probabilities"] if speech_res else None

        text_detected = text_res["has_text"] if text_res else False
        text_probs = text_res["probabilities"] if text_res else None

        fusion_res = self.fusion.fuse(
            face_probs=face_probs,
            speech_probs=speech_probs,
            text_probs=text_probs,
            face_detected=face_detected,
            speech_detected=speech_detected,
            text_detected=text_detected
        )

        return {
            "facial": face_res,
            "speech": speech_res,
            "text": text_res,
            "fusion": fusion_res
        }
