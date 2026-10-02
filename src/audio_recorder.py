"""
Audio Recorder Module
AGB1303 - AI Problem Solving Techniques
Handles microphone recording, buffer capture, and device checking via sounddevice.
"""

try:
    import sounddevice as sd
except (ImportError, OSError):
    sd = None

import numpy as np
import soundfile as sf
from pathlib import Path
from src.config import AUDIO_SAMPLE_RATE, AUDIO_DURATION


class AudioRecorder:
    """Records microphone audio using sounddevice."""

    def __init__(self, sample_rate: int = AUDIO_SAMPLE_RATE, duration: float = AUDIO_DURATION):
        self.sample_rate = sample_rate
        self.duration = duration

    @staticmethod
    def list_devices():
        """Returns list of available audio input devices."""
        if sd is None:
            return []
        try:
            devices = sd.query_devices()
            input_devices = [
                {"id": idx, "name": dev["name"], "channels": dev["max_input_channels"]}
                for idx, dev in enumerate(devices)
                if dev["max_input_channels"] > 0
            ]
            return input_devices
        except Exception as e:
            print(f"[Audio Error] Device query failed: {e}")
            return []

    def record_seconds(self, duration: float = None, device_id: int = None) -> np.ndarray:
        """
        Synchronously records audio for a fixed duration.
        Returns:
            audio: 1D numpy float32 array normalized to [-1.0, 1.0].
        """
        dur = duration or self.duration
        num_frames = int(dur * self.sample_rate)

        if sd is None:
            print("[Audio Warning] SoundDevice / PortAudio is not available in this environment.")
            return None

        try:
            recording = sd.rec(
                num_frames,
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                device=device_id
            )
            sd.wait()
            return recording.flatten()
        except Exception as e:
            print(f"[Audio Error] Recording failed: {e}")
            return None

    def save_wav(self, audio: np.ndarray, file_path: str):
        """Saves float32 audio array to a standard 16-bit PCM WAV file."""
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(file_path, audio, self.sample_rate, subtype="PCM_16")

    @staticmethod
    def synthesize_mock_speech(emotion: str = "Neutral", duration: float = 3.0, sample_rate: int = 16000) -> np.ndarray:
        """Generates representative acoustic audio signals for pipeline testing."""
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        
        # Base pitch (F0) and harmonic variations based on emotion acoustics
        pitch_map = {
            "Angry": 320.0,     # High pitch, high energy, jitter
            "Happy": 280.0,     # Elevated pitch, undulating inflection
            "Neutral": 180.0,   # Moderate steady pitch
            "Sad": 130.0,       # Low pitch, slow cadence
            "Surprise": 350.0   # Sharp rising inflection
        }
        f0 = pitch_map.get(emotion, 180.0)

        # Generate harmonic series to simulate vocal formants
        signal = (
            0.5 * np.sin(2 * np.pi * f0 * t) +
            0.25 * np.sin(2 * np.pi * (2 * f0) * t) +
            0.15 * np.sin(2 * np.pi * (3 * f0) * t)
        )

        # Modulate envelope
        if emotion == "Angry":
            noise = np.random.normal(0, 0.05, len(t))
            signal = signal + noise
            envelope = np.clip(1.2 * np.abs(np.sin(2 * np.pi * 3 * t)), 0.2, 1.0)
        elif emotion == "Happy":
            envelope = np.abs(np.sin(2 * np.pi * 1.5 * t))
        elif emotion == "Sad":
            envelope = np.exp(-1.5 * t) + 0.2
        elif emotion == "Surprise":
            envelope = np.clip(np.exp(t) / np.exp(duration), 0.1, 1.0)
        else:
            envelope = 0.8 * np.ones_like(t)

        audio = (signal * envelope).astype(np.float32)
        # Normalize
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        return audio


class ContinuousAudioStream:
    """
    Continuously streams microphone audio into a rolling window buffer (e.g. 3.0 seconds).
    Enables low-latency real-time speech emotion recognition.
    """

    def __init__(self, sample_rate: int = AUDIO_SAMPLE_RATE, duration: float = AUDIO_DURATION):
        import threading
        self.sample_rate = sample_rate
        self.buffer_size = int(duration * sample_rate)
        self.buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.lock = threading.Lock()
        self.stream = None
        self.is_running = False
        self.current_volume = 0.0
        self.is_speech_active = False
        self.silence_threshold = 0.008  # RMS energy threshold for speech activity

    def _audio_callback(self, indata, frames, time_info, status):
        """Called by sounddevice for each chunk of incoming mic audio."""
        if not self.is_running:
            return
        chunk = indata[:, 0]
        # Calculate instantaneous RMS energy
        rms = float(np.sqrt(np.mean(chunk**2)))
        self.current_volume = rms
        self.is_speech_active = (rms > self.silence_threshold)

        with self.lock:
            chunk_len = len(chunk)
            if chunk_len >= self.buffer_size:
                self.buffer[:] = chunk[-self.buffer_size:]
            else:
                self.buffer[:-chunk_len] = self.buffer[chunk_len:]
                self.buffer[-chunk_len:] = chunk

    def start(self, device_id: int = None) -> bool:
        """Starts the background audio stream."""
        if sd is None:
            print("[ContinuousAudioStream] SoundDevice / PortAudio is not available in this environment.")
            return False
        if self.is_running:
            return True
        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                callback=self._audio_callback,
                device=device_id,
                blocksize=1024
            )
            self.stream.start()
            self.is_running = True
            return True
        except Exception as e:
            print(f"[ContinuousAudioStream] Could not open microphone: {e}")
            self.is_running = False
            return False

    def stop(self):
        """Stops and cleans up the stream."""
        self.is_running = False
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    def get_audio_buffer(self) -> np.ndarray:
        """Returns a copy of the latest rolling 3-second audio buffer."""
        with self.lock:
            return self.buffer.copy()

    def get_volume(self) -> float:
        """Returns current mic RMS volume level [0.0 - 1.0]."""
        return self.current_volume

    def has_speech(self) -> bool:
        """Returns True if current audio energy exceeds silence threshold."""
        return self.is_speech_active

