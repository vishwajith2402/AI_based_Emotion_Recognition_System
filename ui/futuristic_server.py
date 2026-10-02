"""
Futuristic AI Dashboard Server
AGB1303 - AI Problem Solving Techniques
Batch 6: High-Performance FastAPI Backend for Multimodal Human Emotion Recognition.
"""

import sys
import os
import time
import base64
import io
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import soundfile as sf
import psutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from src.config import (
    EMOTION_CLASSES, EMOTION_COLORS, EMOTION_ICONS,
    FACIAL_TEST_DIR, SPEECH_TEST_DIR,
    DEFAULT_FACE_WEIGHT, DEFAULT_SPEECH_WEIGHT
)
from src.predictor import EmotionPredictor
from src.voice_assistant import assistant
from src.audio_recorder import AudioRecorder

app = FastAPI(title="Multimodal Emotion Recognition API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AI Master Predictor
predictor = EmotionPredictor()
recorder = AudioRecorder()
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)


class TTSRequest(BaseModel):
    text: str
    voice_id: Optional[str] = None
    rate: Optional[int] = 175
    volume: Optional[float] = 1.0


class FusionRequest(BaseModel):
    face_emotion: Optional[str] = None
    face_probs: Optional[Dict[str, float]] = None
    speech_emotion: Optional[str] = None
    speech_probs: Optional[Dict[str, float]] = None
    w_face: float = 0.5
    w_speech: float = 0.5


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serves the Futuristic AI Dashboard HTML."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Dashboard UI building in progress...</h1>")


@app.get("/api/system/status")
async def get_system_status():
    """Returns real-time system metrics, hardware states, and model statuses."""
    cpu_usage = psutil.cpu_percent()
    ram_usage = psutil.virtual_memory().percent
    ram_gb = round(psutil.virtual_memory().used / (1024 ** 3), 2)

    # Audio input devices
    audio_devices = AudioRecorder.list_devices()
    has_mic = len(audio_devices) > 0

    return {
        "status": "ready",
        "cpu_percent": cpu_usage,
        "ram_percent": ram_usage,
        "ram_used_gb": ram_gb,
        "facial_model_loaded": predictor.face_model.is_loaded,
        "speech_model_loaded": predictor.speech_model.is_loaded,
        "camera_available": True,
        "microphone_available": has_mic,
        "audio_devices": audio_devices,
        "tts_voices": assistant.available_voices,
        "emotion_classes": EMOTION_CLASSES,
        "emotion_colors": EMOTION_COLORS,
        "emotion_icons": EMOTION_ICONS
    }


@app.post("/api/predict/face")
async def predict_face(
    image_base64: Optional[str] = Form(None),
    sample_emotion: Optional[str] = Form(None)
):
    """Executes facial emotion analysis on camera frame or sample."""
    t0 = time.perf_counter()
    img_bgr = None

    if sample_emotion:
        sample_folder = FACIAL_TEST_DIR / sample_emotion.lower()
        files = list(sample_folder.glob("*.png")) + list(sample_folder.glob("*.jpg"))
        if files:
            img_bgr = cv2.imread(str(files[0]))

    elif image_base64:
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]
        img_bytes = base64.b64decode(image_base64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return {"face_detected": False, "error": "No valid image provided"}

    pred = predictor.predict_image(img_bgr)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "face_detected": pred["face_detected"],
        "top_emotion": pred["top_emotion"],
        "confidence": pred["confidence"],
        "probabilities": pred["probabilities"],
        "bboxes": pred.get("bboxes", []),
        "landmarks": pred.get("landmarks", []),
        "latency_ms": round(latency_ms, 2)
    }


@app.post("/api/predict/speech")
async def predict_speech(
    audio_base64: Optional[str] = Form(None),
    sample_emotion: Optional[str] = Form(None)
):
    """Executes speech emotion analysis on recorded audio or sample."""
    t0 = time.perf_counter()
    audio_data = None

    if sample_emotion:
        sample_folder = SPEECH_TEST_DIR / sample_emotion.lower()
        files = list(sample_folder.glob("*.wav")) + list(sample_folder.glob("*.mp3"))
        if files:
            audio_data = predictor.speech_preprocessor.load_audio_file(str(files[0]))

    elif audio_base64:
        if "," in audio_base64:
            audio_base64 = audio_base64.split(",")[1]
        raw_bytes = base64.b64decode(audio_base64)
        with io.BytesIO(raw_bytes) as bio:
            audio_data, sr = sf.read(bio)
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)

    if audio_data is None:
        return {"speech_detected": False, "error": "No valid audio provided"}

    pred = predictor.predict_audio(audio_data)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "speech_detected": pred["speech_detected"],
        "top_emotion": pred["top_emotion"],
        "confidence": pred["confidence"],
        "probabilities": pred["probabilities"],
        "is_silent": pred["is_silent"],
        "latency_ms": round(latency_ms, 2)
    }


@app.post("/api/predict/multimodal")
async def predict_multimodal(req: FusionRequest):
    """Computes decision-level multimodal fusion, conflict detection, and failovers."""
    t0 = time.perf_counter()
    predictor.fusion.set_weights(req.w_face, req.w_speech)

    fused_res = predictor.fusion.fuse(
        face_probs=req.face_probs,
        speech_probs=req.speech_probs,
        face_detected=(req.face_probs is not None),
        speech_detected=(req.speech_probs is not None)
    )

    latency_ms = (time.perf_counter() - t0) * 1000.0
    fused_res["latency_ms"] = round(latency_ms, 2)
    return fused_res


@app.post("/api/tts/speak")
async def tts_speak(req: TTSRequest):
    """Triggers offline non-blocking speech output via pyttsx3."""
    if req.rate:
        assistant.set_rate(req.rate)
    if req.volume:
        assistant.set_volume(req.volume)
    if req.voice_id:
        assistant.set_voice(req.voice_id)

    assistant.speak(req.text)
    return {"status": "enqueued", "text": req.text}


@app.get("/api/samples")
async def get_samples_catalog():
    """Returns catalog of sample emotions available for live testing."""
    return {
        "emotions": EMOTION_CLASSES,
        "sample_counts": {
            e: len(list((FACIAL_TEST_DIR / e.lower()).glob("*.png")))
            for e in EMOTION_CLASSES
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
