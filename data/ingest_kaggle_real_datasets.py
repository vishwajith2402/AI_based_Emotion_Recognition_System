"""
Ingest Kaggle Real-World Facial and Voice Datasets
AGB1303 - AI Problem Solving Techniques

Ingests real-time external academic datasets from Kaggle:
1. Cohn-Kanade (CK+) Facial Action Units Dataset (shawon10/ckplus)
2. In-The-Wild Facial Emotion Recognition Dataset (ananthu017/emotion-detection-fer)
3. Toronto Emotional Speech Set (TESS) Audio Dataset (ejlok1/toronto-emotional-speech-set-tess)
"""

import os
import sys
from pathlib import Path
import random
import cv2
import numpy as np
import soundfile as sf
import librosa
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    FACIAL_TRAIN_DIR, FACIAL_VAL_DIR, FACIAL_TEST_DIR,
    SPEECH_TRAIN_DIR, SPEECH_VAL_DIR, SPEECH_TEST_DIR,
    EMOTION_CLASSES
)

CKPLUS_CACHE = Path.home() / ".cache" / "kagglehub" / "datasets" / "shawon10" / "ckplus" / "versions" / "1" / "CK+48"
FER_KAGGLE_CACHE = Path.home() / ".cache" / "kagglehub" / "datasets" / "ananthu017" / "emotion-detection-fer" / "versions" / "1"
TESS_CACHE = Path.home() / ".cache" / "kagglehub" / "datasets" / "ejlok1" / "toronto-emotional-speech-set-tess" / "versions" / "1"


def assign_split_path(idx: int, total: int, base_train: Path, base_val: Path, base_test: Path, emotion: str) -> Path:
    r = idx / total
    if r < 0.70:
        target_dir = base_train / emotion
    elif r < 0.85:
        target_dir = base_val / emotion
    else:
        target_dir = base_test / emotion
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def ingest_ckplus_facial():
    print("=" * 70)
    print("INGESTING COHN-KANADE (CK+) CLINICAL FACIAL EMOTION DATASET (KAGGLE)")
    print("=" * 70)
    if not CKPLUS_CACHE.exists():
        print(f"[Warning] CK+ cache not found at: {CKPLUS_CACHE}")
        return 0

    mapping = {
        "anger": "Angry",
        "happy": "Happy",
        "sadness": "Sad",
        "surprise": "Surprise",
        "contempt": "Neutral"
    }

    total_added = 0
    for ck_folder, emo_class in mapping.items():
        src_folder = CKPLUS_CACHE / ck_folder
        if not src_folder.exists():
            continue

        images = sorted(list(src_folder.glob("*.png")) + list(src_folder.glob("*.jpg")))
        random.seed(42)
        random.shuffle(images)
        n = len(images)

        for i, img_path in enumerate(images):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # Standardize to 48x48
            if img.shape != (48, 48):
                img = cv2.resize(img, (48, 48), interpolation=cv2.INTER_AREA)

            target_dir = assign_split_path(i, n, FACIAL_TRAIN_DIR, FACIAL_VAL_DIR, FACIAL_TEST_DIR, emo_class)
            out_name = f"kaggle_ck_{emo_class}_{i:04d}.png"
            cv2.imwrite(str(target_dir / out_name), img)
            total_added += 1

            # Augmentation (horizontal flip) for training split
            if (i / n) < 0.70:
                flip = cv2.flip(img, 1)
                flip_name = f"kaggle_ck_{emo_class}_{i:04d}_flip.png"
                cv2.imwrite(str(target_dir / flip_name), flip)
                total_added += 1

        print(f"  [CK+ Processed] {emo_class}: {n} images imported & augmented from '{ck_folder}'")

    print(f"[Success] Imported {total_added} CK+ facial expression samples.\n")
    return total_added


def ingest_kaggle_fer_real():
    print("=" * 70)
    print("INGESTING REAL-WORLD FACIAL EMOTION DETECTION DATASET (KAGGLE)")
    print("=" * 70)
    if not FER_KAGGLE_CACHE.exists():
        print(f"[Warning] Kaggle FER cache not found at: {FER_KAGGLE_CACHE}")
        return 0

    mapping = {
        "angry": "Angry",
        "happy": "Happy",
        "neutral": "Neutral",
        "sad": "Sad",
        "surprised": "Surprise"
    }

    total_added = 0
    # Process both train and test partitions from Kaggle FER
    for sub in ["train", "test"]:
        sub_path = FER_KAGGLE_CACHE / sub
        if not sub_path.exists():
            continue

        for kaggle_emo, emo_class in mapping.items():
            folder = sub_path / kaggle_emo
            if not folder.exists():
                continue

            files = sorted(list(folder.glob("*.png")) + list(folder.glob("*.jpg")))
            random.seed(42 + len(sub))
            random.shuffle(files)
            # Add up to 500 diverse samples per emotion to ensure rich real-world distribution
            selected = files[:500]
            n = len(selected)

            for i, fpath in enumerate(selected):
                img = cv2.imread(str(fpath), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                if img.shape != (48, 48):
                    img = cv2.resize(img, (48, 48), interpolation=cv2.INTER_AREA)

                target_dir = assign_split_path(i, n, FACIAL_TRAIN_DIR, FACIAL_VAL_DIR, FACIAL_TEST_DIR, emo_class)
                out_name = f"kaggle_real_{sub}_{emo_class}_{i:05d}.png"
                cv2.imwrite(str(target_dir / out_name), img)
                total_added += 1

            print(f"  [Kaggle FER {sub.title()}] {emo_class}: {n} images integrated from '{kaggle_emo}'")

    print(f"[Success] Imported {total_added} real-world Kaggle facial emotion samples.\n")
    return total_added


def ingest_kaggle_tess_speech():
    print("=" * 70)
    print("INGESTING TORONTO EMOTIONAL SPEECH SET (TESS) REAL AUDIO DATASET (KAGGLE)")
    print("=" * 70)
    if not TESS_CACHE.exists():
        print(f"[Warning] TESS cache not found at: {TESS_CACHE}")
        return 0

    mapping = {
        "angry": "Angry",
        "happy": "Happy",
        "neutral": "Neutral",
        "sad": "Sad",
        "ps": "Surprise",
        "surprise": "Surprise",
        "pleasant_surprise": "Surprise",
        "pleasant_surprised": "Surprise"
    }

    wav_files = list(TESS_CACHE.glob("**/*.wav"))
    print(f"[INFO] Discovered {len(wav_files)} total TESS WAV recordings in cache.")

    classified = {c: [] for c in EMOTION_CLASSES}
    for wf in wav_files:
        fn = wf.name.lower()
        parent = wf.parent.name.lower()
        matched = None
        for key, cls_name in mapping.items():
            if f"_{key}.wav" in fn or f"_{key}" in fn or key in parent:
                matched = cls_name
                break
        if matched:
            classified[matched].append(wf)

    total_added = 0
    for emo_class in EMOTION_CLASSES:
        flist = classified[emo_class]
        random.seed(42)
        random.shuffle(flist)
        # Ingest up to 250 diverse speech clips per emotion
        selected = flist[:250]
        n = len(selected)

        for i, src_wav in enumerate(selected):
            try:
                y, sr = sf.read(str(src_wav))
                if y.ndim > 1:
                    y = np.mean(y, axis=1)
                if sr != 16000:
                    y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=16000)

                # Standardize duration to 3 seconds with subtle edge trimming
                y, _ = librosa.effects.trim(y, top_db=25)
                target_len = 16000 * 3
                if len(y) > target_len:
                    y = y[:target_len]
                elif len(y) < target_len:
                    y = np.pad(y, (0, target_len - len(y)), mode='constant')

                target_dir = assign_split_path(i, n, SPEECH_TRAIN_DIR, SPEECH_VAL_DIR, SPEECH_TEST_DIR, emo_class)
                out_name = f"kaggle_tess_{emo_class}_{i:04d}.wav"
                sf.write(str(target_dir / out_name), y.astype(np.float32), 16000)
                total_added += 1
            except Exception as e:
                pass

        print(f"  [TESS Processed] {emo_class}: {len(selected)} audio files standardized at 16kHz mono")

    print(f"[Success] Imported {total_added} real spoken voice clips from TESS.\n")
    return total_added


if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("AGB1303 - REAL-TIME KAGGLE & BENCHMARK DATASET INGESTION SUITE")
    print("#" * 70 + "\n")

    ck_cnt = ingest_ckplus_facial()
    fer_cnt = ingest_kaggle_fer_real()
    tess_cnt = ingest_kaggle_tess_speech()

    print("=" * 70)
    print("INGESTION SUMMARY")
    print("=" * 70)
    print(f"  • CK+ Clinical Action Unit Faces : {ck_cnt} samples")
    print(f"  • Real-World In-The-Wild Faces    : {fer_cnt} samples")
    print(f"  • TESS Real Spoken Voice Clips   : {tess_cnt} samples")
    print(f"  • Total Real-World Datasets Added: {ck_cnt + fer_cnt + tess_cnt} samples")
    print("=" * 70 + "\n")
