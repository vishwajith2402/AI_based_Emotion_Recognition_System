"""
AI Voice Assistant (Voice Virtualizer) Module
AGB1303 - AI Problem Solving Techniques
Handles offline Text-to-Speech (TTS) using pyttsx3 with non-blocking threaded execution.
"""

import threading
import queue
import pyttsx3
from typing import List, Dict


class VoiceAssistant:
    """Pluggable TTS Voice Assistant with voice selection and volume/rate controls."""

    def __init__(self):
        self.speech_queue = queue.Queue()
        self.is_speaking = False
        self.muted = False
        self.rate = 175
        self.volume = 1.0
        self.voice_id = None
        self.available_voices = self._query_voices()

        # Set default voice
        if self.available_voices:
            self.voice_id = self.available_voices[0]["id"]

        # Worker thread for non-blocking speech
        self.worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.worker_thread.start()

    def _query_voices(self) -> List[Dict[str, str]]:
        """Queries native installed voices on Windows."""
        voices_list = []
        try:
            engine = pyttsx3.init()
            voices = engine.getProperty("voices")
            for v in voices:
                voices_list.append({
                    "id": v.id,
                    "name": v.name,
                    "languages": getattr(v, "languages", ["en"])
                })
            engine.stop()
        except Exception as e:
            print(f"[VoiceAssistant Error] Could not query voices: {e}")
        return voices_list

    def _speech_worker(self):
        """Dedicated thread to handle pyttsx3 COM loop without blocking main thread."""
        while True:
            text = self.speech_queue.get()
            if text is None:
                break

            if self.muted or not text.strip():
                self.speech_queue.task_done()
                continue

            try:
                self.is_speaking = True
                engine = pyttsx3.init()
                engine.setProperty("rate", self.rate)
                engine.setProperty("volume", self.volume)
                if self.voice_id:
                    try:
                        engine.setProperty("voice", self.voice_id)
                    except Exception:
                        pass

                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except Exception as e:
                print(f"[VoiceAssistant Error] Speech synthesis failed: {e}")
            finally:
                self.is_speaking = False
                self.speech_queue.task_done()

    def speak(self, text: str):
        """Enqueues text for non-blocking speech synthesis."""
        if not self.muted and text.strip():
            self.speech_queue.put(text)

    def set_rate(self, rate: int):
        """Sets speech speed in words per minute (e.g., 120-240)."""
        self.rate = max(80, min(300, rate))

    def set_volume(self, volume: float):
        """Sets speech volume in [0.0, 1.0]."""
        self.volume = max(0.0, min(1.0, volume))

    def set_voice(self, voice_id: str):
        """Selects voice by ID."""
        self.voice_id = voice_id

    def set_mute(self, mute: bool):
        """Mutes or unmutes speech output."""
        self.muted = mute


# Singleton instance
assistant = VoiceAssistant()
