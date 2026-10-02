"""
text_emotion.py
===============
Speech-to-Text (STT) and Semantic Linguistic Emotion Analysis Module.
Course: AGB1303 - AI Problem Solving Techniques (Batch 6)

Provides:
1. Speech-to-Text transcription from 16kHz audio buffers using speech_recognition.
2. Semantic emotion classification using NLTK VADER and Affective Lexicon density.
3. Probability distribution across: Angry, Happy, Neutral, Sad, Surprise.
4. Tri-Modal failover compatibility.
"""

import re
import numpy as np
import speech_recognition as sr
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from src.config import EMOTION_CLASSES, EMOTION_COLORS, EMOTION_ICONS


class TextEmotionAnalyzer:
    """
    Transcribes spoken audio into text and predicts emotional sentiment probabilities.
    """

    # Comprehensive Affective Emotion Lexicons
    EMOTION_LEXICON = {
        "Angry": {
            "angry", "mad", "furious", "rage", "hate", "hateful", "annoyed", "irritated",
            "frustrated", "unfair", "stupid", "idiot", "disgust", "awful", "terrible",
            "damn", "hell", "pissed", "unacceptable", "offensive", "horrible", "hostile",
            "bitter", "yell", "screaming", "ridiculous", "shut", "stop", "worst", "liar"
        },
        "Happy": {
            "happy", "glad", "joy", "joyful", "smile", "smiling", "great", "wonderful",
            "love", "loving", "awesome", "fantastic", "delighted", "yay", "haha", "laugh",
            "laughing", "beautiful", "enjoy", "enjoying", "best", "good", "proud", "thank",
            "thanks", "congrats", "celebration", "super", "perfect", "blessed", "exciting",
            "excited", "cheerful", "pleasure", "fun", "cool", "nice", "brilliant"
        },
        "Sad": {
            "sad", "crying", "cry", "tears", "sorrow", "depressed", "depression", "unhappy",
            "lonely", "grief", "pain", "painful", "hurt", "hurts", "miss", "sorry", "loss",
            "lost", "helpless", "hopeless", "down", "gloomy", "miserable", "heartbroken",
            "regret", "disappointed", "disappointment", "unfortunate", "tragic", "bad"
        },
        "Surprise": {
            "wow", "whoa", "what", "really", "shock", "shocked", "shocking", "omg",
            "unexpected", "unbelievable", "surprise", "surprised", "surprising", "stunned",
            "sudden", "amazing", "amazed", "impossible", "seriously", "gasp", "miracle",
            "incredible", "unreal", "startling", "astonished"
        },
        "Neutral": {
            "okay", "alright", "fine", "yes", "no", "think", "maybe", "know", "say",
            "today", "tomorrow", "work", "report", "system", "data", "computer", "normal",
            "check", "test", "proceed", "here", "there", "is", "are", "have", "please",
            "hello", "hi", "morning", "afternoon", "listening", "speaking", "standard"
        }
    }

    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True

        # Ensure VADER lexicon is loaded
        try:
            self.vader = SentimentIntensityAnalyzer()
        except Exception:
            try:
                nltk.download("vader_lexicon", quiet=True)
                self.vader = SentimentIntensityAnalyzer()
            except Exception as e:
                print(f"[Warning] Could not initialize VADER: {e}")
                self.vader = None

    def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribes numpy audio buffer into text.
        Returns empty string if speech is unintelligible, silent, or offline.
        """
        if audio_data is None or len(audio_data) == 0:
            return ""

        try:
            # Convert float32 in [-1.0, 1.0] -> int16 PCM
            if audio_data.dtype in (np.float32, np.float64):
                pcm_data = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
            elif audio_data.dtype == np.int16:
                pcm_data = audio_data
            else:
                pcm_data = audio_data.astype(np.int16)

            pcm_bytes = pcm_data.tobytes()
            sr_audio = sr.AudioData(pcm_bytes, sample_rate=sample_rate, sample_width=2)

            # Transcribe via Google Speech Recognition API (zero API key needed)
            text = self.recognizer.recognize_google(sr_audio, language="en-US")
            return text.strip()
        except sr.UnknownValueError:
            # Speech was unintelligible or silence
            return ""
        except sr.RequestError as e:
            # Network issue or rate limit
            return ""
        except Exception as e:
            return ""

    def predict_emotion_from_text(self, text: str) -> dict:
        """
        Analyzes the emotional polarity and semantic keyword density of text.
        Returns calibrated probability distribution across the 5 project classes.
        """
        if not text or len(text.strip()) == 0:
            # Uniform neutral baseline when no text
            uniform_prob = 1.0 / len(EMOTION_CLASSES)
            return {
                "text": "",
                "has_text": False,
                "top_emotion": "Neutral",
                "confidence": uniform_prob,
                "probabilities": {c: uniform_prob for c in EMOTION_CLASSES},
                "vader_scores": {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0},
                "detected_keywords": []
            }

        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        tokens = set(cleaned.split())

        # 1. Keyword Lexicon Activations
        keyword_hits = {c: [] for c in EMOTION_CLASSES}
        for emo, words in self.EMOTION_LEXICON.items():
            for tok in tokens:
                if tok in words:
                    keyword_hits[emo].append(tok)

        # 2. VADER Sentiment Scoring
        if self.vader:
            v_scores = self.vader.polarity_scores(text)
        else:
            v_scores = {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}

        compound = v_scores["compound"]
        pos = v_scores["pos"]
        neg = v_scores["neg"]
        neu = v_scores["neu"]

        # Base logits initialized equally to avoid neutral bias
        logits = {
            "Angry": 0.2,
            "Happy": 0.2,
            "Neutral": 0.2,
            "Sad": 0.2,
            "Surprise": 0.2
        }

        # 3. Modulate logits using VADER valence polarity
        if compound >= 0.25:
            # Positive valence
            logits["Happy"] += compound * 2.2
            logits["Surprise"] += compound * 0.8
            logits["Neutral"] += neu * 0.2
        elif compound <= -0.25:
            # Negative valence: differentiate between Angry and Sad
            abs_comp = abs(compound)
            # High intensity punctuation or tokens favor Angry over Sad
            has_exclamation = "!" in text or any(w in tokens for w in ["damn", "shut", "hate", "mad", "worst"])
            if has_exclamation:
                logits["Angry"] += abs_comp * 2.2
                logits["Sad"] += abs_comp * 0.9
            else:
                logits["Sad"] += abs_comp * 1.8
                logits["Angry"] += abs_comp * 1.2
            logits["Neutral"] += neu * 0.2
        else:
            # Neutral / mild polarity
            logits["Neutral"] += 1.5 + (neu * 0.8)

        # 4. Modulate logits using Affective Lexicon keyword matches
        for emo, hits in keyword_hits.items():
            if hits:
                logits[emo] += len(hits) * 1.5

        # 5. Softmax Normalization
        max_l = max(logits.values())
        exp_logits = {c: np.exp(logits[c] - max_l) for c in EMOTION_CLASSES}
        sum_exp = sum(exp_logits.values())
        probs = {c: float(exp_logits[c] / sum_exp) for c in EMOTION_CLASSES}

        top_emo = max(probs, key=probs.get)
        if top_emo == "Neutral":
            non_neutral = {e: p for e, p in probs.items() if e != "Neutral"}
            best_nn = max(non_neutral, key=non_neutral.get)
            if non_neutral[best_nn] >= 0.22 and (probs["Neutral"] - non_neutral[best_nn]) < 0.15:
                top_emo = best_nn
        confidence = probs[top_emo]

        all_detected_kw = []
        for emo, hits in keyword_hits.items():
            for h in hits:
                all_detected_kw.append(f"{h} ({emo})")

        return {
            "text": text,
            "has_text": True,
            "top_emotion": top_emo,
            "confidence": confidence,
            "probabilities": probs,
            "vader_scores": v_scores,
            "detected_keywords": all_detected_kw
        }

    def transcribe_and_predict(self, audio_data: np.ndarray, sample_rate: int = 16000) -> dict:
        """
        Combines STT transcription and semantic emotion classification in a single call.
        """
        text = self.transcribe_audio(audio_data, sample_rate=sample_rate)
        return self.predict_emotion_from_text(text)
