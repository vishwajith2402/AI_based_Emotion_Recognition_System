"""
Dataset Generator & Pre-populator Module
AGB1303 - AI Problem Solving Techniques
Generates structured Facial Image datasets and Speech Modulation Audio datasets
for Angry, Happy, Neutral, Sad, and Surprise across train, val, and test splits.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import cv2
import numpy as np
import soundfile as sf
from src.config import (
    FACIAL_TRAIN_DIR, FACIAL_VAL_DIR, FACIAL_TEST_DIR,
    SPEECH_TRAIN_DIR, SPEECH_VAL_DIR, SPEECH_TEST_DIR,
    EMOTION_CLASSES, AUDIO_SAMPLE_RATE, FACE_IMAGE_SIZE
)


def create_synthetic_facial_image(emotion: str, size=(48, 48), seed=None) -> np.ndarray:
    """
    Synthesizes a 48x48 facial crop with characteristic emotional expression geometry:
    - Eyes, eyebrows, mouth curves, nose, and facial contours.
    """
    if seed is not None:
        np.random.seed(seed)

    # Base head oval canvas
    img = np.zeros(size, dtype=np.uint8)
    center = (size[0] // 2, size[1] // 2)
    
    # Random skin tone intensity
    base_intensity = np.random.randint(170, 220)
    cv2.ellipse(img, center, (18, 22), 0, 0, 360, int(base_intensity), -1)

    # Eyes positions
    left_eye = (17, 19)
    right_eye = (31, 19)
    
    # Emotion-specific facial geometries
    if emotion == "Happy":
        # Smiling mouth: arc curving upward
        cv2.ellipse(img, (24, 31), (10, 6), 0, 10, 170, 40, 2)
        # Cheeks highlight
        cv2.circle(img, (14, 27), 2, int(base_intensity + 20), -1)
        cv2.circle(img, (34, 27), 2, int(base_intensity + 20), -1)
        # Eyes slightly squinting
        cv2.ellipse(img, left_eye, (3, 2), 0, 0, 180, 30, -1)
        cv2.ellipse(img, right_eye, (3, 2), 0, 0, 180, 30, -1)
        # Raised relaxed eyebrows
        cv2.line(img, (13, 14), (20, 13), 50, 1)
        cv2.line(img, (28, 13), (35, 14), 50, 1)

    elif emotion == "Sad":
        # Downturned mouth: arc curving downward
        cv2.ellipse(img, (24, 37), (9, 5), 0, 190, 350, 40, 2)
        # Drooping eyes
        cv2.circle(img, left_eye, 2, 40, -1)
        cv2.circle(img, right_eye, 2, 40, -1)
        # Eyebrows angled up towards center (distressed)
        cv2.line(img, (13, 16), (20, 13), 50, 1)
        cv2.line(img, (28, 13), (35, 16), 50, 1)

    elif emotion == "Angry":
        # Tense straight/grimacing mouth
        cv2.line(img, (17, 34), (31, 34), 30, 2)
        # Wide intense eyes
        cv2.circle(img, left_eye, 3, 30, -1)
        cv2.circle(img, right_eye, 3, 30, -1)
        # Furrowed V-angled eyebrows (lowered inward)
        cv2.line(img, (13, 13), (21, 17), 40, 2)
        cv2.line(img, (27, 17), (35, 13), 40, 2)

    elif emotion == "Surprise":
        # Wide open oval mouth 'O'
        cv2.ellipse(img, (24, 34), (5, 8), 0, 0, 360, 30, -1)
        # Wide circular eyes
        cv2.circle(img, left_eye, 4, 30, -1)
        cv2.circle(img, left_eye, 1, 240, -1)
        cv2.circle(img, right_eye, 4, 30, -1)
        cv2.circle(img, right_eye, 1, 240, -1)
        # High arched eyebrows
        cv2.ellipse(img, (17, 13), (4, 3), 0, 180, 360, 40, 1)
        cv2.ellipse(img, (31, 13), (4, 3), 0, 180, 360, 40, 1)

    else:  # Neutral
        # Straight relaxed mouth line
        cv2.line(img, (18, 33), (30, 33), 50, 1)
        # Normal eyes
        cv2.circle(img, left_eye, 2, 40, -1)
        cv2.circle(img, right_eye, 2, 40, -1)
        # Horizontal neutral eyebrows
        cv2.line(img, (14, 15), (20, 15), 50, 1)
        cv2.line(img, (28, 15), (34, 15), 50, 1)

    # Nose marker
    cv2.line(img, (24, 22), (24, 26), 70, 1)

    # Add realistic subtle noise & Gaussian blur for texture realism
    noise = np.random.normal(0, 4, size).astype(np.float32)
    img_float = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(img_float, (3, 3), 0.5)

    return blurred


def create_synthetic_speech_audio(emotion: str, duration=3.0, sample_rate=AUDIO_SAMPLE_RATE, seed=None) -> np.ndarray:
    """
    Synthesizes a 16kHz audio sample with emotional speech acoustic modulation:
    - Fundamental frequency (pitch), harmonics, prosody/intonation, and tempo envelopes.
    """
    if seed is not None:
        np.random.seed(seed)

    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples, endpoint=False)

    # Emotion pitch contours and modulation parameters
    if emotion == "Happy":
        f0_base = 270.0 + np.random.uniform(-15, 15)
        # Undulating, animated pitch variation
        pitch_contour = f0_base + 40.0 * np.sin(2 * np.pi * 1.8 * t)
        # Syllabic bursts (3 syllables per sec)
        syllables = np.clip(np.sin(2 * np.pi * 3.5 * t), 0, 1) ** 2
        harmonic_decay = [0.6, 0.35, 0.2, 0.1]

    elif emotion == "Sad":
        f0_base = 125.0 + np.random.uniform(-10, 10)
        # Falling downward intonation, slow cadence
        pitch_contour = f0_base - 25.0 * (t / duration)
        syllables = np.clip(np.sin(2 * np.pi * 1.2 * t), 0, 1) ** 1.5
        harmonic_decay = [0.8, 0.2, 0.05, 0.01]

    elif emotion == "Angry":
        f0_base = 310.0 + np.random.uniform(-20, 20)
        # Harsh fluctuating high pitch with jitter
        jitter = np.random.normal(0, 15, len(t))
        pitch_contour = f0_base + jitter
        syllables = np.clip(np.sin(2 * np.pi * 4.2 * t), 0, 1) ** 3
        harmonic_decay = [0.5, 0.4, 0.3, 0.25]  # Rich harsh upper harmonics

    elif emotion == "Surprise":
        f0_base = 200.0 + np.random.uniform(-10, 10)
        # Rapid upward pitch sweep
        pitch_contour = f0_base + 180.0 * (t / duration) ** 1.5
        syllables = np.clip(np.sin(2 * np.pi * 2.5 * t), 0, 1)
        harmonic_decay = [0.6, 0.3, 0.2, 0.1]

    else:  # Neutral
        f0_base = 160.0 + np.random.uniform(-8, 8)
        pitch_contour = f0_base + 5.0 * np.sin(2 * np.pi * 0.8 * t)
        syllables = np.clip(np.sin(2 * np.pi * 2.2 * t), 0, 1) ** 2
        harmonic_decay = [0.7, 0.25, 0.1, 0.03]

    # Synthesize multi-harmonic formant sound
    phase = 2 * np.pi * np.cumsum(pitch_contour) / sample_rate
    signal = np.zeros(num_samples, dtype=np.float32)
    for h_idx, amp in enumerate(harmonic_decay, 1):
        signal += amp * np.sin(h_idx * phase)

    # Modulate with speech syllabic envelope
    speech_signal = signal * syllables

    # Add gentle ambient breath/noise
    noise = np.random.normal(0, 0.02, num_samples)
    final_audio = speech_signal + noise

    # Normalize amplitude
    max_amp = np.max(np.abs(final_audio))
    if max_amp > 0:
        final_audio = final_audio / max_amp * 0.90

    return final_audio.astype(np.float32)


def generate_all_datasets(samples_per_class_train=40, samples_per_class_val=10, samples_per_class_test=10):
    """
    Generates and organizes complete training, validation, and testing sets
    for both facial images and speech audio clips.
    """
    splits = [
        ("train", FACIAL_TRAIN_DIR, SPEECH_TRAIN_DIR, samples_per_class_train),
        ("val", FACIAL_VAL_DIR, SPEECH_VAL_DIR, samples_per_class_val),
        ("test", FACIAL_TEST_DIR, SPEECH_TEST_DIR, samples_per_class_test),
    ]

    print("=" * 60)
    print("Building Curated Trainable Datasets for Multimodal Emotion Recognition")
    print("=" * 60)

    total_images = 0
    total_audios = 0

    for split_name, face_dir, speech_dir, count in splits:
        print(f"\nProcessing [{split_name.upper()}] split ({count} samples/class)...")
        
        for emotion in EMOTION_CLASSES:
            # Setup directories
            face_cls_dir = face_dir / emotion.lower()
            speech_cls_dir = speech_dir / emotion.lower()
            face_cls_dir.mkdir(parents=True, exist_ok=True)
            speech_cls_dir.mkdir(parents=True, exist_ok=True)

            # Generate facial images
            for i in range(count):
                seed = hash(f"{split_name}_{emotion}_{i}") % 1000000
                face_img = create_synthetic_facial_image(emotion, size=FACE_IMAGE_SIZE, seed=seed)
                face_path = face_cls_dir / f"{emotion.lower()}_{split_name}_{i:04d}.png"
                cv2.imwrite(str(face_path), face_img)
                total_images += 1

            # Generate speech audio files
            for i in range(count):
                seed = hash(f"speech_{split_name}_{emotion}_{i}") % 1000000
                audio_data = create_synthetic_speech_audio(emotion, duration=3.0, seed=seed)
                speech_path = speech_cls_dir / f"{emotion.lower()}_{split_name}_{i:04d}.wav"
                sf.write(str(speech_path), audio_data, AUDIO_SAMPLE_RATE, subtype="PCM_16")
                total_audios += 1

            print(f"  [+] {emotion:8s}: {count} face images, {count} audio clips")

    print("\n" + "=" * 60)
    print(f"Dataset successfully built!")
    print(f"Total Facial Images generated : {total_images}")
    print(f"Total Speech Audio generated  : {total_audios}")
    print(f"Facial Root Directory        : {FACIAL_TRAIN_DIR.parent}")
    print(f"Speech Root Directory        : {SPEECH_TRAIN_DIR.parent}")
    print("=" * 60)


if __name__ == "__main__":
    generate_all_datasets()
