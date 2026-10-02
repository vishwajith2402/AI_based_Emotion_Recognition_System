"""
Project Configuration Module
AGB1303 - AI Problem Solving Techniques
Human Emotion Recognition Using Facial Expressions and Speech Modulation
"""

import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Ensure directories exist
for p in [MODELS_DIR, REPORTS_DIR, DATA_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Standardized Emotion Classes (Aligned across Facial & Speech branches)
EMOTION_CLASSES = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]
NUM_CLASSES = len(EMOTION_CLASSES)

CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(EMOTION_CLASSES)}
IDX_TO_CLASS = {idx: cls for idx, cls in enumerate(EMOTION_CLASSES)}

# Visual Presentation Colors & Icons
EMOTION_COLORS = {
    "Angry": "#EF4444",     # Red
    "Happy": "#10B981",     # Emerald Green
    "Neutral": "#6B7280",   # Gray
    "Sad": "#3B82F6",       # Blue
    "Surprise": "#F59E0B"   # Amber/Gold
}

EMOTION_ICONS = {
    "Angry": "😠",
    "Happy": "😊",
    "Neutral": "😐",
    "Sad": "😢",
    "Surprise": "😲"
}

# Facial Pipeline Settings
FACE_IMAGE_SIZE = (48, 48)   # (Height, Width)
FACE_CHANNELS = 1            # Grayscale for standardized FER
FACE_MODEL_PATH = MODELS_DIR / "facial_cnn.pth"

# Speech Pipeline Settings
AUDIO_SAMPLE_RATE = 16000    # 16 kHz standard
AUDIO_DURATION = 3.0         # 3 seconds fixed buffer window
N_MFCC = 40                  # 40 Mel-frequency cepstral coefficients
N_FFT = 2048
HOP_LENGTH = 512
MAX_AUDIO_FRAMES = 100       # Time frames (3.0 * 16000 / 512 ~ 94, padded to 100)
SPEECH_MODEL_PATH = MODELS_DIR / "speech_bilstm.pth"

# Multimodal & Tri-Modal Fusion Settings
DEFAULT_FACE_WEIGHT = 0.40       # Visual channel (Facial CNN)
DEFAULT_SPEECH_WEIGHT = 0.35     # Acoustic channel (Speech BiLSTM)
DEFAULT_TEXT_WEIGHT = 0.25       # Semantic channel (Transcribed Linguistic Sentiment)
DUAL_FACE_WEIGHT = 0.55          # Fallback when text is inactive
DUAL_SPEECH_WEIGHT = 0.45        # Fallback when text is inactive
CONFLICT_THRESHOLD = 0.40        # Threshold for modality disagreement alert
DEFAULT_ASSESSMENT_DURATION = 20.0  # Standard assessment window duration (seconds)

# Dataset Paths
FACIAL_DATA_DIR = DATA_DIR / "facial"
SPEECH_DATA_DIR = DATA_DIR / "speech"

FACIAL_TRAIN_DIR = FACIAL_DATA_DIR / "train"
FACIAL_VAL_DIR = FACIAL_DATA_DIR / "val"
FACIAL_TEST_DIR = FACIAL_DATA_DIR / "test"

SPEECH_TRAIN_DIR = SPEECH_DATA_DIR / "train"
SPEECH_VAL_DIR = SPEECH_DATA_DIR / "val"
SPEECH_TEST_DIR = SPEECH_DATA_DIR / "test"

# Standard Canonical Benchmark Test Samples (Representative verified exemplars)
CANONICAL_FACIAL_TEST_SAMPLES = {
    "Angry": "fer_Angry_04243.png",
    "Happy": "fer_Happy_07642.png",
    "Neutral": "fer_Neutral_05282.png",
    "Sad": "fer_Sad_05198.png",
    "Surprise": "fer_Surprise_03431.png"
}

CANONICAL_SPEECH_TEST_SAMPLES = {
    "Angry": "tess_Angry_00340.wav",
    "Happy": "happy_test_0005.wav",
    "Neutral": "ravdess_Neutral_00246.wav",
    "Sad": "tess_Sad_00342.wav",
    "Surprise": "ravdess_Surprise_00166.wav"
}
