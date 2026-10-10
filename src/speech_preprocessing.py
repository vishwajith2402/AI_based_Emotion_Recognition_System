"""
Speech Preprocessing Module
AGB1303 - AI Problem Solving Techniques
Handles Audio Ingestion, Silence Removal, Normalization, and MFCC Extraction.
"""

import numpy as np
import librosa
import soundfile as sf
import torch
from pathlib import Path
from src.config import AUDIO_SAMPLE_RATE, AUDIO_DURATION, N_MFCC, N_FFT, HOP_LENGTH, MAX_AUDIO_FRAMES


class SpeechPreprocessor:
    """Extracts standardized MFCC features from speech audio files or live microphone buffers."""

    def __init__(
        self,
        sample_rate: int = AUDIO_SAMPLE_RATE,
        duration: float = AUDIO_DURATION,
        n_mfcc: int = N_MFCC,
        max_frames: int = MAX_AUDIO_FRAMES
    ):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mfcc = n_mfcc
        self.max_frames = max_frames
        self.target_length = int(sample_rate * duration)

    def load_audio_file(self, file_path: str) -> np.ndarray:
        """Loads and resamples audio file to standardized sample rate."""
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
            return audio
        except Exception as e:
            # Fallback using soundfile
            data, sr = sf.read(file_path)
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            if sr != self.sample_rate:
                data = librosa.resample(data, orig_sr=sr, target_sr=self.sample_rate)
            return data.astype(np.float32)

    def normalize_and_trim(self, audio: np.ndarray, top_db: int = 25) -> np.ndarray:
        """Trims silence and normalizes audio amplitude."""
        if audio is None or len(audio) == 0:
            return np.zeros(self.target_length, dtype=np.float32)

        # Trim leading and trailing silence
        try:
            trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
            if len(trimmed) > int(self.sample_rate * 0.4):  # Keep trimmed if > 400ms
                audio = trimmed
        except Exception:
            pass

        # Adjust length to target_length
        if len(audio) < self.target_length:
            # Pad with silence or repeat
            pad_width = self.target_length - len(audio)
            audio = np.pad(audio, (0, pad_width), mode='constant')
        else:
            # Center crop or head crop
            audio = audio[:self.target_length]

        # Peak normalization
        max_val = np.max(np.abs(audio))
        if max_val > 1e-6:
            audio = audio / max_val

        return audio.astype(np.float32)

    def is_silent(self, audio: np.ndarray, energy_threshold: float = 0.005) -> bool:
        """Checks if the audio signal is below conversational energy threshold."""
        if audio is None or len(audio) == 0:
            return True
        rms = np.sqrt(np.mean(np.square(audio)))
        return float(rms) < energy_threshold

    def extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """
        Extracts MFCC features.
        Returns array of shape: (max_frames, n_mfcc) e.g., (100, 40).
        """
        normalized_audio = self.normalize_and_trim(audio)
        
        # Extract MFCC: shape (n_mfcc, time_steps)
        mfcc = librosa.feature.mfcc(
            y=normalized_audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )
        
        # Standardize features (mean=0, std=1 across time)
        mfcc_mean = np.mean(mfcc, axis=1, keepdims=True)
        mfcc_std = np.std(mfcc, axis=1, keepdims=True) + 1e-6
        mfcc_norm = (mfcc - mfcc_mean) / mfcc_std

        # Transpose to (time_steps, n_mfcc) for sequence modeling
        mfcc_t = mfcc_norm.T

        # Pad or truncate to max_frames
        if mfcc_t.shape[0] < self.max_frames:
            pad_len = self.max_frames - mfcc_t.shape[0]
            mfcc_t = np.pad(mfcc_t, ((0, pad_len), (0, 0)), mode='constant')
        else:
            mfcc_t = mfcc_t[:self.max_frames, :]

        return mfcc_t.astype(np.float32)

    def extract_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extracts comprehensive multi-feature acoustic representation based on
        Shivam Burnwal's Speech Emotion Recognition pipeline:
        - MFCC (40 coefficients)
        - Zero Crossing Rate (ZCR) (1 dimension)
        - Root Mean Square Energy (RMS) (1 dimension)
        - Mel Spectrogram (40 mel filterbanks)
        Total feature dimensionality: 82 features per time frame.
        Returns array of shape: (max_frames, 82) e.g., (100, 82).
        """
        normalized_audio = self.normalize_and_trim(audio)

        # 1. MFCC (40 dimensions)
        mfcc = librosa.feature.mfcc(
            y=normalized_audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )
        mfcc_m = np.mean(mfcc, axis=1, keepdims=True)
        mfcc_s = np.std(mfcc, axis=1, keepdims=True) + 1e-6
        mfcc_norm = (mfcc - mfcc_m) / mfcc_s

        # 2. Zero Crossing Rate (1 dimension)
        zcr = librosa.feature.zero_crossing_rate(y=normalized_audio, hop_length=HOP_LENGTH)
        zcr_m = np.mean(zcr, axis=1, keepdims=True)
        zcr_s = np.std(zcr, axis=1, keepdims=True) + 1e-6
        zcr_norm = (zcr - zcr_m) / zcr_s

        # 3. Root Mean Square Energy (1 dimension)
        rms = librosa.feature.rms(y=normalized_audio, hop_length=HOP_LENGTH)
        rms_m = np.mean(rms, axis=1, keepdims=True)
        rms_s = np.std(rms, axis=1, keepdims=True) + 1e-6
        rms_norm = (rms - rms_m) / rms_s

        # 4. Mel Spectrogram (40 dimensions)
        mel = librosa.feature.melspectrogram(
            y=normalized_audio,
            sr=self.sample_rate,
            n_mels=self.n_mfcc,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_m = np.mean(mel_db, axis=1, keepdims=True)
        mel_s = np.std(mel_db, axis=1, keepdims=True) + 1e-6
        mel_norm = (mel_db - mel_m) / mel_s

        # Concatenate along feature axis: (40 + 1 + 1 + 40 = 82, time_steps)
        combined = np.concatenate([mfcc_norm, zcr_norm, rms_norm, mel_norm], axis=0)

        # Transpose to (time_steps, 82)
        combined_t = combined.T

        # Pad or truncate to max_frames
        if combined_t.shape[0] < self.max_frames:
            pad_len = self.max_frames - combined_t.shape[0]
            combined_t = np.pad(combined_t, ((0, pad_len), (0, 0)), mode='constant')
        else:
            combined_t = combined_t[:self.max_frames, :]

        return combined_t.astype(np.float32)

    def to_tensor(self, features: np.ndarray) -> torch.Tensor:
        """Converts (time_steps, feature_dim) array into PyTorch batch tensor: (1, time_steps, feature_dim)."""
        tensor = torch.from_numpy(features).unsqueeze(0)  # (1, T, F)
        return tensor.float()

    def process_audio(self, audio_data: np.ndarray):
        """
        Full pipeline for an audio array.
        Returns:
            tensor: torch.Tensor of shape (1, 100, 82)
            silent: bool indicating if audio is below energy threshold
        """
        silent = self.is_silent(audio_data)
        features = self.extract_features(audio_data)
        tensor = self.to_tensor(features)
        return tensor, silent
