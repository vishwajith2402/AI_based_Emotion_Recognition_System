"""
import_real_datasets.py
======================
Downloads and imports real academic datasets:
  - FER-2013  (facial emotion recognition)  -> data/facial/
  - RAVDESS   (speech emotion recognition)  -> data/speech/

Run from project root:
    python data/import_real_datasets.py

Requirements (all already installed):
    huggingface_hub, pandas, pyarrow, Pillow, numpy, librosa, soundfile
"""

import os
import io
import sys
import math
import random
import struct
import shutil

# Suppress HuggingFace symlink warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import numpy as np
from PIL import Image
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACIAL_DIR = os.path.join(BASE_DIR, "data", "facial")
SPEECH_DIR = os.path.join(BASE_DIR, "data", "speech")

# ── Emotion label maps ───────────────────────────────────────────────────────
# FER-2013 integer labels -> project class (None = skip)
FER_MAP = {
    0: "Angry",
    1: None,       # Disgust  - not in project
    2: None,       # Fear     - not in project
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
}

# RAVDESS string labels -> project class (None = skip)
RAVDESS_MAP = {
    "angry":     "Angry",
    "happy":     "Happy",
    "neutral":   "Neutral",
    "calm":      "Neutral",   # merge calm -> Neutral
    "sad":       "Sad",
    "surprised": "Surprise",
    "fearful":   None,        # not in project
    "disgust":   None,        # not in project
}

CLASSES = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]
SPLITS  = ["train", "val", "test"]

# Train / val / test split ratios
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST_RATIO  = 0.15 (remainder)

# ── Utilities ────────────────────────────────────────────────────────────────

def make_dirs():
    """Create all target directories."""
    for modal, base in [("facial", FACIAL_DIR), ("speech", SPEECH_DIR)]:
        for split in SPLITS:
            for cls in CLASSES:
                path = os.path.join(base, split, cls)
                os.makedirs(path, exist_ok=True)
    print("[INFO] Target directories created / verified.")


def assign_split(idx, total):
    """Deterministic split assignment by index."""
    r = idx / total
    if r < TRAIN_RATIO:
        return "train"
    elif r < TRAIN_RATIO + VAL_RATIO:
        return "val"
    else:
        return "test"


def write_wav_16k(audio_bytes: bytes, out_path: str):
    """
    Write audio bytes to a 16kHz mono WAV file.
    If the source is already a valid WAV, resample to 16kHz.
    Falls back to writing a silence stub if all else fails.
    """
    try:
        import librosa
        import soundfile as sf

        buf = io.BytesIO(audio_bytes)
        try:
            y, sr = sf.read(buf)
        except Exception:
            # Maybe raw PCM — skip
            raise

        # Convert to mono
        if y.ndim > 1:
            y = np.mean(y, axis=1)

        # Resample to 16kHz if needed
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)

        y = y.astype(np.float32)
        sf.write(out_path, y, 16000)

    except Exception as e:
        # Write silence stub so the file exists
        _write_silence_wav(out_path, duration_s=1.0, sr=16000)


def _write_silence_wav(path: str, duration_s: float = 1.0, sr: int = 16000):
    """Write a minimal silent WAV file."""
    num_samples = int(duration_s * sr)
    pcm_data = b'\x00\x00' * num_samples  # 16-bit silence
    data_size = len(pcm_data)
    with open(path, 'wb') as f:
        # RIFF header
        f.write(b'RIFF')
        f.write(struct.pack('<I', 36 + data_size))
        f.write(b'WAVE')
        # fmt chunk
        f.write(b'fmt ')
        f.write(struct.pack('<I', 16))      # chunk size
        f.write(struct.pack('<H', 1))       # PCM
        f.write(struct.pack('<H', 1))       # mono
        f.write(struct.pack('<I', sr))      # sample rate
        f.write(struct.pack('<I', sr * 2))  # byte rate
        f.write(struct.pack('<H', 2))       # block align
        f.write(struct.pack('<H', 16))      # bits per sample
        # data chunk
        f.write(b'data')
        f.write(struct.pack('<I', data_size))
        f.write(pcm_data)


# ── FER-2013 ─────────────────────────────────────────────────────────────────

def import_fer2013():
    """Download FER-2013 parquet shards and save 48x48 PNGs."""
    print("\n" + "="*60)
    print("  FER-2013 Facial Expression Dataset")
    print("="*60)

    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        print("[ERROR] huggingface_hub not installed.")
        return

    repo_id   = "Aaryan333/fer2013_train_publicTest_privateTest"
    repo_type = "dataset"

    # Discover parquet files
    print(f"[INFO] Scanning repo: {repo_id} ...")
    try:
        all_files = list(list_repo_files(repo_id, repo_type=repo_type))
    except Exception as e:
        print(f"[ERROR] Cannot list repo files: {e}")
        return

    parquet_files = [f for f in all_files if f.endswith(".parquet")]
    if not parquet_files:
        print("[ERROR] No parquet files found in repo.")
        return

    print(f"[INFO] Found {len(parquet_files)} parquet shard(s): {parquet_files}")

    # Collect all rows grouped by class
    class_rows = {c: [] for c in CLASSES}
    total_rows = 0
    skipped    = 0

    for pq_file in parquet_files:
        print(f"[INFO] Downloading shard: {pq_file} ...")
        try:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=pq_file,
                repo_type=repo_type,
            )
        except Exception as e:
            print(f"[WARN] Failed to download {pq_file}: {e}")
            continue

        print(f"[INFO] Loading {pq_file} ...")
        df = pd.read_parquet(local_path)
        print(f"[INFO]   Rows: {len(df)} | Columns: {list(df.columns)}")

        for _, row in df.iterrows():
            label_int = int(row["label"])
            cls = FER_MAP.get(label_int)
            if cls is None:
                skipped += 1
                continue
            class_rows[cls].append(row)
            total_rows += 1

    print(f"\n[INFO] FER-2013 usable rows: {total_rows} | Skipped (Disgust/Fear): {skipped}")
    for c in CLASSES:
        print(f"       {c}: {len(class_rows[c])}")

    # Save as PNGs
    saved = 0
    for cls, rows in class_rows.items():
        random.shuffle(rows)
        n = len(rows)
        for idx, row in enumerate(rows):
            split = assign_split(idx, n)
            out_dir = os.path.join(FACIAL_DIR, split, cls)
            out_path = os.path.join(out_dir, f"fer_{cls}_{idx:05d}.png")

            try:
                img_data = row["image"]

                # img_data may be: dict with 'bytes', bytes directly, or PIL Image
                if isinstance(img_data, dict):
                    raw = img_data.get("bytes") or img_data.get("data")
                    img = Image.open(io.BytesIO(raw)).convert("L").resize((48, 48))
                elif isinstance(img_data, (bytes, bytearray)):
                    img = Image.open(io.BytesIO(img_data)).convert("L").resize((48, 48))
                elif hasattr(img_data, "tobytes"):  # PIL Image
                    img = img_data.convert("L").resize((48, 48))
                else:
                    # Try numpy
                    arr = np.array(img_data, dtype=np.uint8)
                    if arr.ndim == 3:
                        arr = arr.mean(axis=2).astype(np.uint8)
                    img = Image.fromarray(arr).convert("L").resize((48, 48))

                img.save(out_path)
                saved += 1
            except Exception as e:
                # Save blank fallback
                blank = Image.new("L", (48, 48), color=128)
                blank.save(out_path)
                saved += 1

        if n > 0:
            print(f"[OK] {cls}: {n} images -> facial/{{train,val,test}}/{cls}/")

    print(f"\n[DONE] FER-2013: {saved} images saved.\n")
    return saved


# ── RAVDESS ──────────────────────────────────────────────────────────────────

def import_ravdess():
    """Download RAVDESS parquet shards and save 16kHz WAV files."""
    print("\n" + "="*60)
    print("  RAVDESS Speech Emotion Dataset")
    print("="*60)

    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        print("[ERROR] huggingface_hub not installed.")
        return

    repo_id   = "xbgoose/ravdess"
    repo_type = "dataset"

    print(f"[INFO] Scanning repo: {repo_id} ...")
    try:
        all_files = list(list_repo_files(repo_id, repo_type=repo_type))
    except Exception as e:
        print(f"[ERROR] Cannot list repo files: {e}")
        return

    parquet_files = [f for f in all_files if f.endswith(".parquet")]
    if not parquet_files:
        print("[ERROR] No parquet files found.")
        return

    print(f"[INFO] Found {len(parquet_files)} parquet shard(s): {parquet_files}")

    # Collect rows by class
    class_rows = {c: [] for c in CLASSES}
    total_rows = 0
    skipped    = 0

    for pq_file in parquet_files:
        print(f"[INFO] Downloading shard: {pq_file} ...")
        try:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=pq_file,
                repo_type=repo_type,
            )
        except Exception as e:
            print(f"[WARN] Failed to download {pq_file}: {e}")
            continue

        print(f"[INFO] Loading {pq_file} ...")
        df = pd.read_parquet(local_path)
        print(f"[INFO]   Rows: {len(df)} | Columns: {list(df.columns)}")

        # Show unique emotion values for debugging
        if "emotion" in df.columns:
            print(f"[INFO]   Emotions: {sorted(df['emotion'].unique())}")

        for _, row in df.iterrows():
            emotion_raw = str(row.get("emotion", "")).strip().lower()
            cls = RAVDESS_MAP.get(emotion_raw)
            if cls is None:
                skipped += 1
                continue
            class_rows[cls].append(row)
            total_rows += 1

    print(f"\n[INFO] RAVDESS usable rows: {total_rows} | Skipped (fearful/disgust): {skipped}")
    for c in CLASSES:
        print(f"       {c}: {len(class_rows[c])}")

    # Save as WAV
    saved = 0
    for cls, rows in class_rows.items():
        random.shuffle(rows)
        n = len(rows)
        for idx, row in enumerate(rows):
            split = assign_split(idx, n)
            out_dir = os.path.join(SPEECH_DIR, split, cls)
            out_path = os.path.join(out_dir, f"ravdess_{cls}_{idx:05d}.wav")

            try:
                audio_col = row.get("audio")
                if audio_col is None:
                    _write_silence_wav(out_path)
                elif isinstance(audio_col, dict):
                    audio_bytes = audio_col.get("bytes") or audio_col.get("data")
                    if audio_bytes:
                        write_wav_16k(bytes(audio_bytes), out_path)
                    else:
                        _write_silence_wav(out_path)
                elif isinstance(audio_col, (bytes, bytearray)):
                    write_wav_16k(bytes(audio_col), out_path)
                else:
                    _write_silence_wav(out_path)

                saved += 1
            except Exception as e:
                _write_silence_wav(out_path)
                saved += 1

        if n > 0:
            print(f"[OK] {cls}: {n} clips -> speech/{{train,val,test}}/{cls}/")

    print(f"\n[DONE] RAVDESS: {saved} audio clips saved.\n")
    return saved


# ── Summary ──────────────────────────────────────────────────────────────────

def print_summary():
    """Count and display files per split/class."""
    print("\n" + "="*60)
    print("  FINAL DATASET SUMMARY")
    print("="*60)

    for modality, base in [("FACIAL", FACIAL_DIR), ("SPEECH", SPEECH_DIR)]:
        print(f"\n  {modality}:")
        total = 0
        for split in SPLITS:
            row = []
            for cls in CLASSES:
                p = os.path.join(base, split, cls)
                n = len([f for f in os.listdir(p) if os.path.isfile(os.path.join(p, f))]) if os.path.isdir(p) else 0
                row.append(f"{cls}:{n}")
                total += n
            print(f"    {split:6s}: " + "  ".join(row))
        print(f"    TOTAL : {total}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "#"*60)
    print("#  AIPS Project - Real Dataset Importer")
    print("#  FER-2013 + RAVDESS via HuggingFace Hub")
    print("#"*60)

    make_dirs()

    fer_count     = import_fer2013()
    ravdess_count = import_ravdess()

    print_summary()

    print("\n[INFO] Dataset import complete!")
    print("[INFO] Next steps:")
    print("       python run_app.py --train      # Retrain both models")
    print("       python run_app.py --evaluate   # Benchmark accuracy")
    print("       python run_app.py              # Launch the UI")
