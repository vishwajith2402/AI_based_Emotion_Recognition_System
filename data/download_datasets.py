"""
Public Benchmark Dataset Downloader & Importer
AGB1303 - AI Problem Solving Techniques
Provides instructions and automated utilities to download and integrate
full-scale benchmark datasets: FER-2013 (Facial) and RAVDESS (Speech).
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import shutil
import urllib.request
import zipfile
from src.config import FACIAL_DATA_DIR, SPEECH_DATA_DIR, EMOTION_CLASSES


BENCHMARK_INFO = {
    "FER-2013": {
        "modality": "Facial Expressions",
        "samples": "35,887 grayscale 48x48 images",
        "classes": "Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral",
        "description": "Standard benchmark created by Pierre-Luc Carrier and Aaron Courville.",
        "url": "https://www.kaggle.com/datasets/msambare/fer2013"
    },
    "RAVDESS": {
        "modality": "Speech Modulation & Audio",
        "samples": "1,440 speech audio files from 24 professional actors (12 female, 12 male)",
        "classes": "Calm, Happy, Sad, Angry, Fearful, Surprise, Disgust, Neutral",
        "description": "Ryerson Audio-Visual Database of Emotional Speech and Song.",
        "url": "https://zenodo.org/record/1188976"
    }
}


def print_dataset_instructions():
    """Prints clear instructions on how to use full academic datasets."""
    print("=" * 70)
    print("ACADEMIC BENCHMARK DATASETS FOR MULTIMODAL EMOTION RECOGNITION")
    print("=" * 70)
    for name, info in BENCHMARK_INFO.items():
        print(f"\n[{name}] ({info['modality']})")
        print(f"  • Size & Content : {info['samples']}")
        print(f"  • Emotion Classes: {info['classes']}")
        print(f"  • Source URL     : {info['url']}")
        print(f"  • Description    : {info['description']}")
    
    print("\n" + "-" * 70)
    print("HOW TO IMPORT EXTERNAL DATA:")
    print("1. For Facial Images:")
    print("   Place images into: data/facial/<train|val|test>/<emotion_name>/")
    print("   Supported formats: .png, .jpg, .jpeg (48x48 grayscale or RGB)")
    print("\n2. For Speech Audios:")
    print("   Place audio clips into: data/speech/<train|val|test>/<emotion_name>/")
    print("   Supported formats: .wav, .mp3 (16kHz recommended)")
    print("=" * 70)


def map_ravdess_filename(filename: str) -> str:
    """
    Decodes RAVDESS filename format: 03-01-0X-...
    Modality (01=full-AV, 02=video-only, 03=audio-only)
    Vocal channel (01=speech, 02=song)
    Emotion (01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised)
    """
    parts = Path(filename).stem.split("-")
    if len(parts) >= 3:
        emotion_code = parts[2]
        mapping = {
            "01": "neutral",
            "02": "neutral",  # Calm mapped to neutral
            "03": "happy",
            "04": "sad",
            "05": "angry",
            "08": "surprise"
        }
        return mapping.get(emotion_code, None)
    return None


if __name__ == "__main__":
    print_dataset_instructions()
