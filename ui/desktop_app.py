"""
Medium-Basic Real-Time Desktop UI
Human Emotion Recognition Using Facial Expressions and Speech Modulation
AGB1303 - AI Problem Solving Techniques (Batch 6)

Features:
  - Real-time simultaneous camera and microphone analysis
  - Speech-to-Text & Semantic Tri-Modal Fusion (Vision + Acoustics + Text)
  - Real-Time Emotion Timeline & Live Mood Trajectory Graph (Russell's Circumplex Affect Model)
  - Emotional Stability Index & Dominant Mood session analytics
  - Clean, professional, medium-basic layout (no distracting sci-fi HUD)
  - Real-time speech prosody stream with live volume meter
  - Real-time facial emotion recognition with clean bounding box
  - Weighted multimodal decision fusion with conflict & failover detection
  - 20-Second Simultaneous Tri-Modal Assessment with Markdown export
  - Optional offline text-to-speech voice assistant
"""

import sys
import os
import time
import threading
import collections
import tkinter as tk
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from PIL import Image, ImageTk
import customtkinter as ctk

from src.config import (
    EMOTION_CLASSES, EMOTION_COLORS, EMOTION_ICONS,
    DEFAULT_FACE_WEIGHT, DEFAULT_SPEECH_WEIGHT, DEFAULT_TEXT_WEIGHT,
    DEFAULT_ASSESSMENT_DURATION
)
from src.predictor import EmotionPredictor
from src.audio_recorder import ContinuousAudioStream
from src.voice_assistant import VoiceAssistant
from src.face_preprocessing import FacePreprocessor
from src.assessment import AssessmentSession

# Set clean medium-basic theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class EmotionTimelineTracker:
    """
    Maintains a rolling temporal history of multimodal emotion decisions and calculates
    Affective Valence (-1.0 to +1.0) and Emotional Stability.
    """

    VALENCE_MAP = {
        "Happy": 1.0,      # Positive High
        "Surprise": 0.4,   # Alert / Pleasant
        "Neutral": 0.0,    # Equilibrium
        "Sad": -0.6,       # Low Energy Negative
        "Angry": -0.9      # High Energy Negative
    }

    def __init__(self, max_points: int = 100):
        self.max_points = max_points
        self.history = collections.deque(maxlen=max_points)
        self.start_time = None
        self.session_seconds = 0.0
        self.emotion_counts = collections.Counter()
        self.total_samples = 0
        self.conflict_count = 0

    def reset(self):
        """Resets the history for a new session."""
        self.history.clear()
        self.start_time = time.time()
        self.session_seconds = 0.0
        self.emotion_counts.clear()
        self.total_samples = 0
        self.conflict_count = 0

    def add_point(self, dominant_emotion: str, confidence: float, probabilities: dict = None, is_conflict: bool = False):
        """Appends a new time point to the live emotion timeline."""
        if self.start_time is None:
            self.start_time = time.time()
        self.session_seconds = time.time() - self.start_time

        # Calculate instantaneous Affective Valence
        valence = 0.0
        if probabilities:
            for emo, p in probabilities.items():
                valence += p * self.VALENCE_MAP.get(emo, 0.0)
        else:
            valence = self.VALENCE_MAP.get(dominant_emotion, 0.0)

        valence = float(np.clip(valence, -1.0, 1.0))

        point = {
            "time": self.session_seconds,
            "emotion": dominant_emotion,
            "valence": valence,
            "confidence": float(confidence),
            "conflict": bool(is_conflict)
        }
        self.history.append(point)
        self.emotion_counts[dominant_emotion] += 1
        self.total_samples += 1
        if is_conflict:
            self.conflict_count += 1

    def get_stats(self) -> dict:
        """Computes live session analytics."""
        if not self.history or self.total_samples == 0:
            return {
                "session_time_str": "00:00",
                "dominant_emotion": "Neutral",
                "dominant_pct": 0,
                "stability_pct": 100,
                "stability_label": "Calm",
                "avg_valence": 0.0,
                "valence_label": "Equilibrium",
                "conflict_count": 0
            }

        mins = int(self.session_seconds // 60)
        secs = int(self.session_seconds % 60)
        time_str = f"{mins:02d}:{secs:02d}"

        # Dominant emotion: prioritize active non-neutral expressions over resting baseline
        non_neutral_counts = {e: count for e, count in self.emotion_counts.items() if e != "Neutral" and count > 0}
        if non_neutral_counts:
            most_common = max(non_neutral_counts, key=non_neutral_counts.get)
            count = non_neutral_counts[most_common]
            dominant_pct = int((count / max(1, self.total_samples)) * 100)
        elif self.emotion_counts:
            most_common, count = self.emotion_counts.most_common(1)[0]
            dominant_pct = int((count / max(1, self.total_samples)) * 100)
        else:
            most_common, dominant_pct = "Neutral", 0

        # Emotional Stability: inverse of rolling variance in valence
        valences = [p["valence"] for p in self.history]
        std_dev = float(np.std(valences)) if len(valences) > 1 else 0.0
        stability_score = int(np.clip((1.0 - (std_dev * 1.4)) * 100, 15, 100))

        if stability_score >= 85:
            stab_label = "Very Stable"
        elif stability_score >= 65:
            stab_label = "Steady"
        elif stability_score >= 45:
            stab_label = "Moderate"
        else:
            stab_label = "Dynamic"

        # Average Valence
        avg_val = float(np.mean(valences))
        if avg_val > 0.25:
            val_label = "Positive"
        elif avg_val < -0.25:
            val_label = "Negative"
        else:
            val_label = "Neutral"

        return {
            "session_time_str": time_str,
            "dominant_emotion": most_common,
            "dominant_pct": dominant_pct,
            "stability_pct": stability_score,
            "stability_label": stab_label,
            "avg_valence": avg_val,
            "valence_label": val_label,
            "conflict_count": self.conflict_count
        }

    def render_canvas(self, canvas: tk.Canvas):
        """Draws the live scrolling mood trajectory graph and timeline ribbon."""
        canvas.delete("all")
        width = canvas.winfo_width()
        height = canvas.winfo_height()
        if width <= 20 or height <= 20:
            return

        margin_left = 68
        margin_right = 20
        margin_top = 16
        margin_bottom = 26

        plot_w = width - margin_left - margin_right
        plot_h = height - margin_top - margin_bottom
        zero_y = margin_top + (plot_h / 2.0)

        # 1. Background Grid & Zones
        canvas.create_rectangle(margin_left, margin_top, width - margin_right, height - margin_bottom, fill="#0F172A", outline="#1E293B", width=1)
        canvas.create_line(margin_left, zero_y, width - margin_right, zero_y, fill="#334155", width=1, dash=(4, 4))

        quarter_h = plot_h / 4.0
        canvas.create_line(margin_left, zero_y - quarter_h, width - margin_right, zero_y - quarter_h, fill="#1E293B", width=1)
        canvas.create_line(margin_left, zero_y + quarter_h, width - margin_right, zero_y + quarter_h, fill="#1E293B", width=1)

        # Y-Axis Labels
        canvas.create_text(margin_left - 8, margin_top + 4, text="Positive (+1.0)", fill="#10B981", font=("Segoe UI", 8), anchor="e")
        canvas.create_text(margin_left - 8, zero_y, text="Neutral (0.0)", fill="#94A3B8", font=("Segoe UI", 8), anchor="e")
        canvas.create_text(margin_left - 8, height - margin_bottom - 4, text="Negative (-1.0)", fill="#EF4444", font=("Segoe UI", 8), anchor="e")

        # X-Axis Time Label
        canvas.create_text(width - margin_right, height - margin_bottom + 12, text="Time (Latest →)", fill="#64748B", font=("Segoe UI", 8), anchor="e")

        if not self.history:
            canvas.create_text(width / 2.0, height / 2.0, text="Awaiting multimodal input...", fill="#475569", font=("Segoe UI", 11))
            return

        pts = list(self.history)
        n = len(pts)
        if n < 2:
            return

        step_x = plot_w / max(1, self.max_points - 1)
        coords = []
        for i, pt in enumerate(pts):
            idx = (self.max_points - n) + i
            x = margin_left + (idx * step_x)
            val = pt["valence"]
            y = zero_y - (val * (plot_h / 2.0) * 0.90)
            coords.append((x, y))

        # 2. Draw Bottom Emotion Ribbon
        ribbon_y = height - margin_bottom + 4
        for i in range(len(coords) - 1):
            x1, _ = coords[i]
            x2, _ = coords[i + 1]
            emo = pts[i]["emotion"]
            bar_color = EMOTION_COLORS.get(emo, "#3B82F6")
            canvas.create_rectangle(x1, ribbon_y, x2 + 1, ribbon_y + 4, fill=bar_color, outline="")

        # 3. Draw Connecting Splines
        for i in range(len(coords) - 1):
            x1, y1 = coords[i]
            x2, y2 = coords[i + 1]
            val = (pts[i]["valence"] + pts[i+1]["valence"]) / 2.0

            if val > 0.20:
                line_color = "#10B981"
            elif val < -0.20:
                line_color = "#EF4444"
            else:
                line_color = "#60A5FA"

            canvas.create_line(x1, y1, x2, y2, fill=line_color, width=2)

        # 4. Draw Current Trajectory Head (Glowing Dot)
        last_x, last_y = coords[-1]
        last_emo = pts[-1]["emotion"]
        head_color = EMOTION_COLORS.get(last_emo, "#3B82F6")
        canvas.create_oval(last_x - 5, last_y - 5, last_x + 5, last_y + 5, fill=head_color, outline="#F8FAFC", width=2)


class RealtimeEmotionApp(ctk.CTk):
    """Clean, medium-basic desktop application with real-time camera, mic, STT, and emotion timeline."""

    def __init__(self):
        super().__init__()

        self.title("Human Emotion Recognition - Tri-Modal AI (Vision + Acoustics + Text)")
        self.geometry("1240x870")
        self.minsize(1080, 740)

        # AI Pipelines
        self.predictor = EmotionPredictor()
        self.audio_stream = ContinuousAudioStream()
        self.voice_assistant = VoiceAssistant()
        self.timeline_tracker = EmotionTimelineTracker(max_points=100)
        self.assessment_session = AssessmentSession(target_duration=DEFAULT_ASSESSMENT_DURATION)

        # State Variables
        self.is_running = False
        self.is_assessing = False
        self.mic_muted = False
        self.tts_enabled = False
        self.last_spoken_emotion = None
        self.last_speech_time = 0.0

        # Weights
        self.w_face = DEFAULT_FACE_WEIGHT
        self.w_speech = DEFAULT_SPEECH_WEIGHT
        self.w_text = DEFAULT_TEXT_WEIGHT

        # Data Shared Across Threads
        self.lock = threading.Lock()
        self.latest_frame = None
        self.face_res = None
        self.speech_res = None
        self.text_res = None
        self.fused_res = None
        self.current_volume = 0.0
        self.is_speech_active = False

        # Threads
        self.video_thread = None
        self.audio_thread = None
        self.text_thread = None
        self.cam_cap = None

        self.build_ui()

        # Handle window close
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # Start UI periodic tick loop (runs at ~25 FPS)
        self.after(40, self.ui_tick)

    def build_ui(self):
        """Constructs clean, medium-basic layout with live timeline graph."""
        # ── 1. Top Header ──────────────────────────────────────────────────────
        header = ctk.CTkFrame(self, fg_color="#1E293B", corner_radius=10, height=58)
        header.pack(fill="x", padx=16, pady=(10, 6))
        header.pack_propagate(False)

        left_header = ctk.CTkFrame(header, fg_color="transparent")
        left_header.pack(side="left", padx=16, pady=6)

        title = ctk.CTkLabel(
            left_header,
            text="🎭 Human Emotion Recognition System",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#F8FAFC"
        )
        title.pack(anchor="w")

        subtitle = ctk.CTkLabel(
            left_header,
            text="Speech-to-Text & Semantic Tri-Modal Fusion (Vision + Acoustics + Text) | AGB1303 Batch 6",
            font=ctk.CTkFont(size=11),
            text_color="#94A3B8"
        )
        subtitle.pack(anchor="w")

        # Right Header Badges
        right_header = ctk.CTkFrame(header, fg_color="transparent")
        right_header.pack(side="right", padx=16, pady=10)

        self.lbl_system_status = ctk.CTkLabel(
            right_header,
            text="● SYSTEM IDLE",
            fg_color="#334155",
            text_color="#CBD5E1",
            font=ctk.CTkFont(size=12, weight="bold"),
            corner_radius=6,
            padx=12,
            pady=4
        )
        self.lbl_system_status.pack(side="right")

        # ── 2. Middle Main Content Area (Two Columns) ──────────────────────────
        content = ctk.CTkFrame(self, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=16, pady=2)

        # ── Left Column: Live Inputs (Camera & Mic & STT) ─────────────────────
        left_col = ctk.CTkFrame(content, fg_color="#0F172A", corner_radius=12)
        left_col.pack(side="left", fill="both", expand=True, padx=(0, 6))

        cam_header_row = ctk.CTkFrame(left_col, fg_color="transparent")
        cam_header_row.pack(fill="x", padx=14, pady=(8, 4))

        cam_title = ctk.CTkLabel(
            cam_header_row,
            text="📹 Real-Time Video Stream",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#F1F5F9"
        )
        cam_title.pack(side="left")

        self.lbl_fps = ctk.CTkLabel(
            cam_header_row,
            text="30 FPS",
            font=ctk.CTkFont(size=11),
            text_color="#64748B"
        )
        self.lbl_fps.pack(side="right")

        # Video Canvas
        self.video_label = ctk.CTkLabel(
            left_col,
            text="Click 'Start Real-Time Analysis' below to activate Camera & Microphone",
            fg_color="#1E293B",
            text_color="#94A3B8",
            corner_radius=8,
            width=520,
            height=290
        )
        self.video_label.pack(padx=14, pady=2, fill="both", expand=True)

        # ── Facial Style & Micro-Expression Morphological Analysis Card ─────────
        self.style_card = ctk.CTkFrame(left_col, fg_color="#1E293B", corner_radius=8)
        self.style_card.pack(fill="x", padx=14, pady=3)

        s_head = ctk.CTkFrame(self.style_card, fg_color="transparent")
        s_head.pack(fill="x", padx=10, pady=(5, 2))

        ctk.CTkLabel(
            s_head,
            text="🎭 Facial Style & Micro-Expression",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#F59E0B"
        ).pack(side="left")

        self.lbl_style_diag_badge = ctk.CTkLabel(
            s_head,
            text="Neutral Style",
            fg_color="#334155",
            text_color="#CBD5E1",
            font=ctk.CTkFont(size=10, weight="bold"),
            corner_radius=4,
            padx=8,
            pady=1
        )
        self.lbl_style_diag_badge.pack(side="right")

        # Zone badges row: Mouth | Eyes | Forehead
        badges_row = ctk.CTkFrame(self.style_card, fg_color="transparent")
        badges_row.pack(fill="x", padx=8, pady=(2, 2))

        self.lbl_badge_mouth = ctk.CTkLabel(
            badges_row,
            text="👄 Mouth: Neutral",
            font=ctk.CTkFont(size=10),
            fg_color="#0F172A",
            corner_radius=5,
            text_color="#E2E8F0",
            padx=6,
            pady=2
        )
        self.lbl_badge_mouth.pack(side="left", padx=2, fill="x", expand=True)

        self.lbl_badge_eyes = ctk.CTkLabel(
            badges_row,
            text="👁️ Eyes: Normal",
            font=ctk.CTkFont(size=10),
            fg_color="#0F172A",
            corner_radius=5,
            text_color="#E2E8F0",
            padx=6,
            pady=2
        )
        self.lbl_badge_eyes.pack(side="left", padx=2, fill="x", expand=True)

        self.lbl_badge_forehead = ctk.CTkLabel(
            badges_row,
            text="🧠 Brow: Smooth",
            font=ctk.CTkFont(size=10),
            fg_color="#0F172A",
            corner_radius=5,
            text_color="#E2E8F0",
            padx=6,
            pady=2
        )
        self.lbl_badge_forehead.pack(side="left", padx=2, fill="x", expand=True)

        self.lbl_style_reasoning = ctk.CTkLabel(
            self.style_card,
            text="“Neutral horizontal mouth, normal eyes, and plain smooth forehead.”",
            font=ctk.CTkFont(size=10, slant="italic"),
            text_color="#94A3B8",
            wraplength=480,
            justify="left",
            anchor="w"
        )
        self.lbl_style_reasoning.pack(fill="x", padx=10, pady=(1, 5))

        # Microphone Volume Indicator Row
        mic_frame = ctk.CTkFrame(left_col, fg_color="#1E293B", corner_radius=8)
        mic_frame.pack(fill="x", padx=14, pady=4)

        self.lbl_mic_icon = ctk.CTkLabel(
            mic_frame,
            text="🎤 Mic Level:",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#CBD5E1"
        )
        self.lbl_mic_icon.pack(side="left", padx=(12, 6), pady=5)

        self.bar_mic_volume = ctk.CTkProgressBar(
            mic_frame,
            orientation="horizontal",
            height=9,
            progress_color="#10B981"
        )
        self.bar_mic_volume.set(0.0)
        self.bar_mic_volume.pack(side="left", fill="x", expand=True, padx=8, pady=5)

        self.lbl_mic_activity = ctk.CTkLabel(
            mic_frame,
            text="Idle / Muted",
            font=ctk.CTkFont(size=11),
            text_color="#94A3B8",
            width=90
        )
        self.lbl_mic_activity.pack(side="right", padx=(6, 12), pady=5)

        # ── Transcribed Speech & Semantics Banner ──────────────────────────────
        stt_frame = ctk.CTkFrame(left_col, fg_color="#1E293B", corner_radius=8)
        stt_frame.pack(fill="x", padx=14, pady=4)

        stt_head = ctk.CTkFrame(stt_frame, fg_color="transparent")
        stt_head.pack(fill="x", padx=10, pady=(6, 2))

        ctk.CTkLabel(
            stt_head,
            text="💬 Transcribed Speech & Semantics",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#38BDF8"
        ).pack(side="left")

        self.lbl_stt_sentiment_badge = ctk.CTkLabel(
            stt_head,
            text="Neutral",
            fg_color="#334155",
            text_color="#CBD5E1",
            font=ctk.CTkFont(size=10, weight="bold"),
            corner_radius=4,
            padx=8,
            pady=2
        )
        self.lbl_stt_sentiment_badge.pack(side="right")

        self.lbl_transcribed_text = ctk.CTkLabel(
            stt_frame,
            text="“Listening for spoken words...”",
            font=ctk.CTkFont(size=11, slant="italic"),
            text_color="#CBD5E1",
            wraplength=480,
            justify="left",
            anchor="w"
        )
        self.lbl_transcribed_text.pack(fill="x", padx=12, pady=(2, 6))

        # Controls Row
        ctrl_frame = ctk.CTkFrame(left_col, fg_color="transparent")
        ctrl_frame.pack(fill="x", padx=14, pady=4)

        self.btn_toggle_run = ctk.CTkButton(
            ctrl_frame,
            text="▶ Start Real-Time Analysis",
            command=self.toggle_live_analysis,
            fg_color="#2563EB",
            hover_color="#1D4ED8",
            font=ctk.CTkFont(size=12, weight="bold"),
            height=34
        )
        self.btn_toggle_run.pack(side="left", fill="x", expand=True, padx=(0, 6))

        self.btn_mute_mic = ctk.CTkButton(
            ctrl_frame,
            text="🎤 Mute Mic",
            command=self.toggle_mute_mic,
            fg_color="#334155",
            hover_color="#475569",
            width=95,
            height=34
        )
        self.btn_mute_mic.pack(side="left", padx=4)

        self.chk_tts = ctk.CTkCheckBox(
            ctrl_frame,
            text="Voice Output",
            command=self.toggle_tts,
            font=ctk.CTkFont(size=11)
        )
        self.chk_tts.pack(side="right", padx=(8, 0))

        # 20-Second Simultaneous Tri-Modal Assessment Row
        self.assess_frame = ctk.CTkFrame(left_col, fg_color="#1E293B", corner_radius=8)
        self.assess_frame.pack(fill="x", padx=14, pady=(2, 8))

        self.btn_20s_assess = ctk.CTkButton(
            self.assess_frame,
            text="⏱️ Start 20s Assessment",
            command=self.start_20s_assessment,
            fg_color="#7C3AED",
            hover_color="#6D28D9",
            font=ctk.CTkFont(size=11, weight="bold"),
            height=30
        )
        self.btn_20s_assess.pack(side="left", padx=8, pady=5)
        self.btn_30s_assess = self.btn_20s_assess  # Backward compatibility alias

        self.lbl_assess_timer = ctk.CTkLabel(
            self.assess_frame,
            text="Simultaneous 20s Vision, Voice & Text Diagnostic",
            font=ctk.CTkFont(size=11),
            text_color="#CBD5E1"
        )
        self.lbl_assess_timer.pack(side="left", padx=6, pady=5)

        self.bar_assess_progress = ctk.CTkProgressBar(
            self.assess_frame,
            orientation="horizontal",
            height=8,
            progress_color="#8B5CF6"
        )
        self.bar_assess_progress.set(0.0)
        self.bar_assess_progress.pack(side="left", fill="x", expand=True, padx=8, pady=5)

        self.btn_cancel_assess = ctk.CTkButton(
            self.assess_frame,
            text="Cancel",
            command=self.cancel_20s_assessment,
            fg_color="#334155",
            hover_color="#475569",
            width=55,
            height=26,
            font=ctk.CTkFont(size=10)
        )

        # ── Right Column: Real-Time Results & Tri-Modal Fusion ────────────────
        right_col = ctk.CTkFrame(content, fg_color="#0F172A", corner_radius=12, width=470)
        right_col.pack(side="right", fill="both", padx=(6, 0))
        right_col.pack_propagate(False)

        res_title = ctk.CTkLabel(
            right_col,
            text="⚡ Tri-Modal Emotion Analysis",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#F1F5F9"
        )
        res_title.pack(anchor="w", padx=16, pady=(8, 4))

        # 1. Primary Fused Emotion Result Card
        self.card_fused = ctk.CTkFrame(right_col, fg_color="#1E293B", corner_radius=10, border_width=2, border_color="#3B82F6")
        self.card_fused.pack(fill="x", padx=16, pady=3)

        self.lbl_card_title = ctk.CTkLabel(
            self.card_fused,
            text="FINAL COMBINED EMOTION (TRI-MODAL FUSION)",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#94A3B8"
        )
        self.lbl_card_title.pack(pady=(6, 1))

        self.lbl_fused_emotion = ctk.CTkLabel(
            self.card_fused,
            text="😐 Neutral",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color="#60A5FA"
        )
        self.lbl_fused_emotion.pack(pady=1)

        self.lbl_fused_conf = ctk.CTkLabel(
            self.card_fused,
            text="Combined Confidence: 0.0%",
            font=ctk.CTkFont(size=12),
            text_color="#CBD5E1"
        )
        self.lbl_fused_conf.pack(pady=1)

        # Modality Concordance / Conflict Status Pill
        self.lbl_conflict_status = ctk.CTkLabel(
            self.card_fused,
            text="Modality Status: Waiting for inputs",
            fg_color="#334155",
            text_color="#CBD5E1",
            font=ctk.CTkFont(size=10),
            corner_radius=6,
            padx=10,
            pady=3
        )
        self.lbl_conflict_status.pack(pady=(2, 6))

        # 2. Tri-Modal Unimodal Split Badges (3 Channels)
        unimodal_frame = ctk.CTkFrame(right_col, fg_color="transparent")
        unimodal_frame.pack(fill="x", padx=16, pady=4)

        # 2a. Facial Badge
        self.card_face = ctk.CTkFrame(unimodal_frame, fg_color="#1E293B", corner_radius=8)
        self.card_face.pack(side="left", fill="both", expand=True, padx=(0, 2))

        lbl_f_head = ctk.CTkLabel(self.card_face, text="📷 Vision", font=ctk.CTkFont(size=10, weight="bold"), text_color="#94A3B8")
        lbl_f_head.pack(pady=(4, 1))
        self.lbl_face_val = ctk.CTkLabel(self.card_face, text="Neutral (0%)", font=ctk.CTkFont(size=11, weight="bold"), text_color="#F1F5F9")
        self.lbl_face_val.pack(pady=(0, 4))

        # 2b. Speech Badge
        self.card_speech = ctk.CTkFrame(unimodal_frame, fg_color="#1E293B", corner_radius=8)
        self.card_speech.pack(side="left", fill="both", expand=True, padx=2)

        lbl_s_head = ctk.CTkLabel(self.card_speech, text="🎙️ Acoustics", font=ctk.CTkFont(size=10, weight="bold"), text_color="#94A3B8")
        lbl_s_head.pack(pady=(4, 1))
        self.lbl_speech_val = ctk.CTkLabel(self.card_speech, text="Neutral (0%)", font=ctk.CTkFont(size=11, weight="bold"), text_color="#F1F5F9")
        self.lbl_speech_val.pack(pady=(0, 4))

        # 2c. Text Semantics Badge
        self.card_text = ctk.CTkFrame(unimodal_frame, fg_color="#1E293B", corner_radius=8)
        self.card_text.pack(side="left", fill="both", expand=True, padx=(2, 0))

        lbl_t_head = ctk.CTkLabel(self.card_text, text="💬 Semantics", font=ctk.CTkFont(size=10, weight="bold"), text_color="#94A3B8")
        lbl_t_head.pack(pady=(4, 1))
        self.lbl_text_val = ctk.CTkLabel(self.card_text, text="Neutral (0%)", font=ctk.CTkFont(size=11, weight="bold"), text_color="#F1F5F9")
        self.lbl_text_val.pack(pady=(0, 4))

        # 3. Class Probabilities Progress Bars
        bars_card = ctk.CTkFrame(right_col, fg_color="#1E293B", corner_radius=10)
        bars_card.pack(fill="both", expand=True, padx=16, pady=3)

        bars_title = ctk.CTkLabel(
            bars_card,
            text="Emotion Probability Breakdown",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#CBD5E1"
        )
        bars_title.pack(anchor="w", padx=14, pady=(6, 2))

        self.prob_bars = {}
        for emotion in EMOTION_CLASSES:
            row = ctk.CTkFrame(bars_card, fg_color="transparent")
            row.pack(fill="x", padx=14, pady=2)

            icon = EMOTION_ICONS.get(emotion, "")
            lbl = ctk.CTkLabel(
                row,
                text=f"{icon} {emotion:<8}",
                width=80,
                anchor="w",
                font=ctk.CTkFont(size=10)
            )
            lbl.pack(side="left")

            bar = ctk.CTkProgressBar(
                row,
                orientation="horizontal",
                height=9,
                progress_color=EMOTION_COLORS.get(emotion, "#3B82F6")
            )
            bar.set(0.2)
            bar.pack(side="left", fill="x", expand=True, padx=6)

            val_lbl = ctk.CTkLabel(row, text="20%", width=36, font=ctk.CTkFont(size=10))
            val_lbl.pack(side="right")

            self.prob_bars[emotion] = (bar, val_lbl)

        # 4. Fusion Weights Information Card
        slider_card = ctk.CTkFrame(right_col, fg_color="#1E293B", corner_radius=10)
        slider_card.pack(fill="x", padx=16, pady=(3, 6))

        self.lbl_slider = ctk.CTkLabel(
            slider_card,
            text="Tri-Modal Weights: 40% Vision | 35% Acoustics | 25% Semantics",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#94A3B8"
        )
        self.lbl_slider.pack(pady=(4, 1))

        self.slider_weight = ctk.CTkSlider(
            slider_card,
            from_=0.1,
            to=0.9,
            number_of_steps=16,
            command=self.on_slider_changed
        )
        self.slider_weight.set(0.4)
        self.slider_weight.pack(fill="x", padx=14, pady=(1, 6))

        # ── 3. Bottom Panel: Real-Time Emotion Timeline & Live Mood Graph ─────
        timeline_panel = ctk.CTkFrame(self, fg_color="#0F172A", corner_radius=12, height=160)
        timeline_panel.pack(fill="x", padx=16, pady=(4, 10))
        timeline_panel.pack_propagate(False)

        # Timeline Header Row with Live Analytics Badges
        t_header = ctk.CTkFrame(timeline_panel, fg_color="transparent")
        t_header.pack(fill="x", padx=14, pady=(6, 2))

        t_title = ctk.CTkLabel(
            t_header,
            text="📈 Real-Time Emotion Timeline & Live Mood Trajectory",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#F1F5F9"
        )
        t_title.pack(side="left")

        # Stats Badges on Right
        stats_frame = ctk.CTkFrame(t_header, fg_color="transparent")
        stats_frame.pack(side="right")

        self.lbl_stat_time = ctk.CTkLabel(
            stats_frame,
            text="⏱️ Session: 00:00",
            font=ctk.CTkFont(size=11),
            text_color="#94A3B8",
            padx=6
        )
        self.lbl_stat_time.pack(side="left")

        self.lbl_stat_dominant = ctk.CTkLabel(
            stats_frame,
            text="🎭 Dominant: Neutral",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#60A5FA",
            padx=6
        )
        self.lbl_stat_dominant.pack(side="left")

        self.lbl_stat_stability = ctk.CTkLabel(
            stats_frame,
            text="⚖️ Stability: 100% (Calm)",
            font=ctk.CTkFont(size=11),
            text_color="#10B981",
            padx=6
        )
        self.lbl_stat_stability.pack(side="left")

        self.lbl_stat_valence = ctk.CTkLabel(
            stats_frame,
            text="📊 Valence: 0.00",
            font=ctk.CTkFont(size=11),
            text_color="#CBD5E1",
            padx=6
        )
        self.lbl_stat_valence.pack(side="left")

        # Interactive Canvas for Scrolling Graph
        self.canvas_timeline = tk.Canvas(
            timeline_panel,
            bg="#0B132B",
            highlightthickness=0,
            height=100
        )
        self.canvas_timeline.pack(fill="both", expand=True, padx=14, pady=(2, 6))

    # ── Control Callbacks ─────────────────────────────────────────────────────

    def on_slider_changed(self, val):
        self.w_face = float(val)
        rem = 1.0 - self.w_face
        self.w_speech = rem * 0.58
        self.w_text = rem * 0.42

        face_pct = int(self.w_face * 100)
        speech_pct = int(self.w_speech * 100)
        text_pct = int(self.w_text * 100)

        self.lbl_slider.configure(
            text=f"Tri-Modal Weights: {face_pct}% Vision | {speech_pct}% Acoustics | {text_pct}% Semantics"
        )
        self.predictor.fusion.set_weights(self.w_face, self.w_speech, self.w_text)

    def toggle_mute_mic(self):
        self.mic_muted = not self.mic_muted
        if self.mic_muted:
            self.btn_mute_mic.configure(text="🔇 Unmute Mic", fg_color="#DC2626")
        else:
            self.btn_mute_mic.configure(text="🎤 Mute Mic", fg_color="#334155")

    def toggle_tts(self):
        self.tts_enabled = bool(self.chk_tts.get())

    def toggle_live_analysis(self):
        if not self.is_running:
            self.start_live_analysis()
        else:
            self.stop_live_analysis()

    def start_live_analysis(self):
        self.timeline_tracker.reset()

        # Open Camera
        self.cam_cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cam_cap.isOpened():
            self.cam_cap = cv2.VideoCapture(0)

        if not self.cam_cap.isOpened():
            self.video_label.configure(text="⚠️ Camera device could not be opened.\nChecking microphone only...", image="")
            self.cam_cap = None

        # Start Microphone Stream
        mic_ok = self.audio_stream.start()
        if not mic_ok:
            print("[Warning] Microphone stream could not be started.")

        self.is_running = True
        self.btn_toggle_run.configure(text="⏹ Stop Analysis", fg_color="#DC2626", hover_color="#B91C1C")
        self.lbl_system_status.configure(text="● LIVE ANALYZING", fg_color="#15803D", text_color="#DCFCE7")

        # Start Workers
        if self.cam_cap is not None:
            self.video_thread = threading.Thread(target=self._video_worker, daemon=True)
            self.video_thread.start()

        self.audio_thread = threading.Thread(target=self._audio_worker, daemon=True)
        self.audio_thread.start()

        self.text_thread = threading.Thread(target=self._text_worker, daemon=True)
        self.text_thread.start()

    def stop_live_analysis(self):
        self.is_running = False
        if self.cam_cap is not None:
            self.cam_cap.release()
            self.cam_cap = None

        self.audio_stream.stop()

        self.btn_toggle_run.configure(text="▶ Start Real-Time Analysis", fg_color="#2563EB", hover_color="#1D4ED8")
        self.lbl_system_status.configure(text="○ SYSTEM IDLE", fg_color="#334155", text_color="#CBD5E1")
        self.video_label.configure(text="Live Analysis Stopped\nClick 'Start Real-Time Analysis' to resume", image="")
        self.bar_mic_volume.set(0.0)
        self.lbl_mic_activity.configure(text="Idle / Muted")

        if self.is_assessing:
            self.cancel_20s_assessment()

    # ── 20-Second Simultaneous Tri-Modal Assessment ───────────────────────────

    def start_20s_assessment(self):
        """Initiates the 20-second simultaneous tri-modal evaluation."""
        if not self.is_running:
            self.start_live_analysis()

        self.timeline_tracker.reset()
        self.assessment_session.start()
        self.is_assessing = True

        self.btn_20s_assess.configure(state="disabled")
        self.btn_cancel_assess.pack(side="right", padx=(4, 8), pady=5)
        self.lbl_assess_timer.configure(text="⏱️ Assessment: 20.0s remaining...", text_color="#A78BFA")
        self.lbl_system_status.configure(text="⏱️ 20s ASSESSMENT IN PROGRESS", fg_color="#6D28D9", text_color="#EDE9FE")
        self.bar_assess_progress.set(0.0)

    # Alias for backward compatibility
    start_30s_assessment = start_20s_assessment

    def cancel_20s_assessment(self):
        """Cancels an ongoing assessment."""
        self.is_assessing = False
        self.assessment_session.stop()
        self.btn_20s_assess.configure(state="normal")
        self.btn_cancel_assess.pack_forget()
        self.lbl_assess_timer.configure(text="Assessment Cancelled", text_color="#94A3B8")
        self.bar_assess_progress.set(0.0)
        if self.is_running:
            self.lbl_system_status.configure(text="● LIVE ANALYZING", fg_color="#15803D", text_color="#DCFCE7")

    # Alias for backward compatibility
    cancel_30s_assessment = cancel_20s_assessment

    def show_assessment_report_modal(self, report: dict):
        """Displays the comprehensive Final Result of Analysis modal dialog."""
        modal = ctk.CTkToplevel(self)
        modal.title(f"20-Second Tri-Modal Assessment Report - {report.get('session_id', 'ASSESS')}")
        modal.geometry("780x740")
        modal.minsize(700, 620)
        modal.configure(fg_color="#0F172A")
        modal.transient(self)
        modal.grab_set()

        # Center modal on parent
        modal.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() - 780) // 2
        y = self.winfo_y() + (self.winfo_height() - 740) // 2
        modal.geometry(f"780x740+{max(0, x)}+{max(0, y)}")

        scroll = ctk.CTkScrollableFrame(modal, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=20, pady=16)

        # Header Title
        title_row = ctk.CTkFrame(scroll, fg_color="transparent")
        title_row.pack(fill="x", pady=(0, 10))

        lbl_head = ctk.CTkLabel(
            title_row,
            text="📋 20-Second Tri-Modal Emotion Assessment",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#F8FAFC"
        )
        lbl_head.pack(side="left")

        lbl_sess = ctk.CTkLabel(
            title_row,
            text=f"Session: {report.get('session_id', '')} | {report.get('duration_seconds', 20)}s",
            font=ctk.CTkFont(size=11),
            text_color="#94A3B8"
        )
        lbl_sess.pack(side="right")

        # 1. Executive Summary Hero Card
        dom_emo = report.get("dominant_emotion", "Neutral")
        dom_color = EMOTION_COLORS.get(dom_emo, "#3B82F6")
        dom_icon = EMOTION_ICONS.get(dom_emo, "🎭")
        dom_pct = report.get("dominant_percentage", 0.0)
        dom_conf = report.get("dominant_confidence", 0.0)

        hero_card = ctk.CTkFrame(scroll, fg_color="#1E293B", corner_radius=12, border_width=2, border_color=dom_color)
        hero_card.pack(fill="x", pady=8)

        hero_top = ctk.CTkFrame(hero_card, fg_color="transparent")
        hero_top.pack(fill="x", padx=20, pady=(14, 6))

        ctk.CTkLabel(
            hero_top,
            text="DOMINANT RECOGNIZED EMOTION (TRI-MODAL)",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#94A3B8"
        ).pack(anchor="w")

        ctk.CTkLabel(
            hero_top,
            text=f"{dom_icon} {dom_emo}",
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color=dom_color
        ).pack(anchor="w", pady=(2, 4))

        resting_pct = report.get("resting_neutral_pct", 0.0)
        if dom_emo != "Neutral" and resting_pct > 0:
            hero_desc = f"Primary Recognized Expression: {dom_emo} ({dom_pct}% active display, {resting_pct}% resting baseline) with {dom_conf}% average model confidence."
        else:
            hero_desc = f"Detected across {dom_pct}% of the 20-second session with {dom_conf}% average model confidence."

        ctk.CTkLabel(
            hero_top,
            text=hero_desc,
            font=ctk.CTkFont(size=12),
            text_color="#E2E8F0"
        ).pack(anchor="w")

        # Badges row: Affective Valence & Stability
        badges_row = ctk.CTkFrame(hero_card, fg_color="transparent")
        badges_row.pack(fill="x", padx=20, pady=(4, 14))

        val_val = report.get("affective_valence", 0.0)
        val_sign = "+" if val_val >= 0 else ""
        val_box = ctk.CTkFrame(badges_row, fg_color="#0F172A", corner_radius=8)
        val_box.pack(side="left", padx=(0, 8), fill="x", expand=True)
        ctk.CTkLabel(val_box, text=f"📊 Affective Valence: {val_sign}{val_val:.2f}", font=ctk.CTkFont(size=12, weight="bold"), text_color="#38BDF8").pack(padx=10, pady=(6, 1))
        ctk.CTkLabel(val_box, text=report.get("valence_label", "Neutral"), font=ctk.CTkFont(size=10), text_color="#94A3B8").pack(padx=10, pady=(0, 6))

        stab_score = report.get("stability_score", 100)
        stab_box = ctk.CTkFrame(badges_row, fg_color="#0F172A", corner_radius=8)
        stab_box.pack(side="right", padx=(8, 0), fill="x", expand=True)
        ctk.CTkLabel(stab_box, text=f"⚖️ Stability Score: {stab_score}%", font=ctk.CTkFont(size=12, weight="bold"), text_color="#10B981").pack(padx=10, pady=(6, 1))
        ctk.CTkLabel(stab_box, text=report.get("stability_label", "Calm"), font=ctk.CTkFont(size=10), text_color="#94A3B8").pack(padx=10, pady=(0, 6))

        # 2. Tri-Modal Modality Performance Comparison (3 Columns)
        mod_frame = ctk.CTkFrame(scroll, fg_color="transparent")
        mod_frame.pack(fill="x", pady=6)

        # 2a. Vision (CNN)
        f_sum = report.get("face_summary", {})
        f_card = ctk.CTkFrame(mod_frame, fg_color="#1E293B", corner_radius=10)
        f_card.pack(side="left", fill="both", expand=True, padx=(0, 4))

        ctk.CTkLabel(f_card, text="📷 Vision (CNN)", font=ctk.CTkFont(size=11, weight="bold"), text_color="#F1F5F9").pack(anchor="w", padx=12, pady=(10, 4))
        ctk.CTkLabel(f_card, text=f"• Dominant: {f_sum.get('dominant_emotion', 'Neutral')} ({f_sum.get('dominant_pct', 0)}%)", font=ctk.CTkFont(size=10), text_color="#CBD5E1").pack(anchor="w", padx=12, pady=1)
        ctk.CTkLabel(f_card, text=f"• Tracking: {f_sum.get('engagement_pct', 0)}%", font=ctk.CTkFont(size=10), text_color="#CBD5E1").pack(anchor="w", padx=12, pady=1)
        ctk.CTkLabel(f_card, text=f"• Conf: {f_sum.get('mean_confidence', 0)}%", font=ctk.CTkFont(size=10), text_color="#CBD5E1").pack(anchor="w", padx=12, pady=(1, 10))

        # 2b. Acoustics (BiLSTM)
        s_sum = report.get("speech_summary", {})
        s_card = ctk.CTkFrame(mod_frame, fg_color="#1E293B", corner_radius=10)
        s_card.pack(side="left", fill="both", expand=True, padx=4)

        ctk.CTkLabel(s_card, text="🎙️ Acoustics (BiLSTM)", font=ctk.CTkFont(size=11, weight="bold"), text_color="#F1F5F9").pack(anchor="w", padx=12, pady=(10, 4))
        ctk.CTkLabel(s_card, text=f"• Dominant: {s_sum.get('dominant_emotion', 'Neutral')} ({s_sum.get('dominant_pct', 0)}%)", font=ctk.CTkFont(size=10), text_color="#CBD5E1").pack(anchor="w", padx=12, pady=1)
        ctk.CTkLabel(s_card, text=f"• Active Time: {s_sum.get('speaking_time_pct', 0)}%", font=ctk.CTkFont(size=10), text_color="#CBD5E1").pack(anchor="w", padx=12, pady=1)
        ctk.CTkLabel(s_card, text=f"• Conf: {s_sum.get('mean_confidence', 0)}%", font=ctk.CTkFont(size=10), text_color="#CBD5E1").pack(anchor="w", padx=12, pady=(1, 10))

        # 2c. Semantics (STT + VADER)
        t_sum = report.get("text_summary", {})
        t_card = ctk.CTkFrame(mod_frame, fg_color="#1E293B", corner_radius=10)
        t_card.pack(side="left", fill="both", expand=True, padx=(4, 0))

        ctk.CTkLabel(t_card, text="💬 Semantics (NLP)", font=ctk.CTkFont(size=11, weight="bold"), text_color="#F1F5F9").pack(anchor="w", padx=12, pady=(10, 4))
        ctk.CTkLabel(t_card, text=f"• Dominant: {t_sum.get('dominant_emotion', 'Neutral')} ({t_sum.get('dominant_pct', 0)}%)", font=ctk.CTkFont(size=10), text_color="#CBD5E1").pack(anchor="w", padx=12, pady=1)
        ctk.CTkLabel(t_card, text=f"• Phrases: {t_sum.get('utterances_count', 0)} captured", font=ctk.CTkFont(size=10), text_color="#CBD5E1").pack(anchor="w", padx=12, pady=1)
        ctk.CTkLabel(t_card, text=f"• Conf: {t_sum.get('mean_confidence', 0)}%", font=ctk.CTkFont(size=10), text_color="#CBD5E1").pack(anchor="w", padx=12, pady=(1, 10))

        # Congruence Banner
        cong = report.get("multimodal_congruence", {})
        cong_banner = ctk.CTkFrame(scroll, fg_color="#1E293B", corner_radius=8)
        cong_banner.pack(fill="x", pady=6)
        ctk.CTkLabel(
            cong_banner,
            text=f"🤝 Tri-Modal Concordance: {cong.get('concordance_rate_pct', 100)}% cross-channel alignment | Divergence/Conflict: {cong.get('conflict_rate_pct', 0)}%",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#10B981" if cong.get('conflict_rate_pct', 0) < 15 else "#F59E0B"
        ).pack(padx=14, pady=8)

        # 2d. Facial Morphological Style Summary Card
        style_summary = report.get("facial_style_summary", {})
        if style_summary:
            style_card = ctk.CTkFrame(scroll, fg_color="#1E293B", corner_radius=10)
            style_card.pack(fill="x", pady=6)

            st_head = ctk.CTkFrame(style_card, fg_color="transparent")
            st_head.pack(fill="x", padx=16, pady=(10, 4))
            ctk.CTkLabel(
                st_head,
                text="🎭 Facial Style & Micro-Expression Morphological Analysis",
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color="#F59E0B"
            ).pack(side="left")

            st_emo = style_summary.get("dominant_style_emotion", "Neutral")
            ctk.CTkLabel(
                st_head,
                text=f"{EMOTION_ICONS.get(st_emo, '🎭')} {st_emo}",
                font=ctk.CTkFont(size=11, weight="bold"),
                fg_color=EMOTION_COLORS.get(st_emo, "#334155"),
                text_color="#FFFFFF",
                corner_radius=4,
                padx=8,
                pady=2
            ).pack(side="right")

            # Metrics row
            st_metrics = ctk.CTkFrame(style_card, fg_color="transparent")
            st_metrics.pack(fill="x", padx=16, pady=2)

            ctk.CTkLabel(st_metrics, text=f"• Mouth: {style_summary.get('dominant_mouth_style', 'Neutral Horizontal')}", font=ctk.CTkFont(size=11), text_color="#CBD5E1").pack(side="left", padx=(0, 14))
            ctk.CTkLabel(st_metrics, text=f"• Eyes: {style_summary.get('dominant_eye_style', 'Normal Open')}", font=ctk.CTkFont(size=11), text_color="#CBD5E1").pack(side="left", padx=14)
            ctk.CTkLabel(st_metrics, text=f"• Brow: {style_summary.get('dominant_forehead_style', 'Plain Smooth Forehead')}", font=ctk.CTkFont(size=11), text_color="#CBD5E1").pack(side="left", padx=14)

            # Morphological Reasoning text
            ctk.CTkLabel(
                style_card,
                text=f"“{style_summary.get('style_description', '')}”",
                font=ctk.CTkFont(size=11, slant="italic"),
                text_color="#94A3B8",
                wraplength=660,
                justify="left"
            ).pack(anchor="w", padx=16, pady=(4, 10))

        # 3. Transcribed Utterances Section (if any)
        utterances = t_sum.get("utterances", [])
        if utterances:
            utt_card = ctk.CTkFrame(scroll, fg_color="#1E293B", corner_radius=10)
            utt_card.pack(fill="x", pady=6)

            ctk.CTkLabel(
                utt_card,
                text="💬 Transcribed Spoken Phrases & Linguistic Sentiment",
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color="#F1F5F9"
            ).pack(anchor="w", padx=16, pady=(10, 6))

            for u in utterances[:6]:  # Show top 6
                u_row = ctk.CTkFrame(utt_card, fg_color="transparent")
                u_row.pack(fill="x", padx=16, pady=2)
                ctk.CTkLabel(u_row, text=f"[{u['time']}s]", font=ctk.CTkFont(size=10), text_color="#94A3B8", width=42).pack(side="left")
                ctk.CTkLabel(u_row, text=f"“{u['text']}”", font=ctk.CTkFont(size=10, slant="italic"), text_color="#E2E8F0").pack(side="left", padx=6)
                ctk.CTkLabel(u_row, text=f"{EMOTION_ICONS.get(u['emotion'], '')} {u['emotion']}", font=ctk.CTkFont(size=10, weight="bold"), text_color=EMOTION_COLORS.get(u['emotion'], "#3B82F6")).pack(side="right")

            ctk.CTkLabel(utt_card, text="", height=4).pack()

        # 4. Complete Emotion Breakdown Bars
        dist_card = ctk.CTkFrame(scroll, fg_color="#1E293B", corner_radius=10)
        dist_card.pack(fill="x", pady=6)

        ctk.CTkLabel(
            dist_card,
            text="📊 20-Second Emotion Distribution Breakdown",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#F1F5F9"
        ).pack(anchor="w", padx=16, pady=(10, 6))

        for emo, d in report.get("emotion_distribution", {}).items():
            row = ctk.CTkFrame(dist_card, fg_color="transparent")
            row.pack(fill="x", padx=16, pady=3)
            ctk.CTkLabel(row, text=f"{d['icon']} {emo:<9}", width=90, anchor="w", font=ctk.CTkFont(size=11)).pack(side="left")
            p_bar = ctk.CTkProgressBar(row, orientation="horizontal", height=9, progress_color=d["color"])
            p_bar.set(d["percentage"] / 100.0)
            p_bar.pack(side="left", fill="x", expand=True, padx=8)
            ctk.CTkLabel(row, text=f"{d['percentage']}%", width=45, font=ctk.CTkFont(size=11, weight="bold")).pack(side="right")

        ctk.CTkLabel(dist_card, text="", height=4).pack()

        # 5. Behavioral & Diagnostic Interpretation
        diag_card = ctk.CTkFrame(scroll, fg_color="#1E293B", corner_radius=10)
        diag_card.pack(fill="x", pady=6)

        ctk.CTkLabel(
            diag_card,
            text="🧠 Affective Computing Diagnostic Summary",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#CBD5E1"
        ).pack(anchor="w", padx=16, pady=(10, 4))

        diag_txt = report.get("diagnostic_summary", "Assessment concluded normally.")
        ctk.CTkLabel(
            diag_card,
            text=diag_txt,
            font=ctk.CTkFont(size=11),
            text_color="#94A3B8",
            wraplength=660,
            justify="left"
        ).pack(anchor="w", padx=16, pady=(2, 12))

        # Bottom Buttons
        btn_bar = ctk.CTkFrame(scroll, fg_color="transparent")
        btn_bar.pack(fill="x", pady=(10, 4))

        lbl_saved = ctk.CTkLabel(btn_bar, text="", font=ctk.CTkFont(size=11), text_color="#10B981")
        lbl_saved.pack(side="left", padx=4)

        def do_export():
            out_dir = os.path.join(PROJECT_ROOT, "reports")
            os.makedirs(out_dir, exist_ok=True)
            sess_id = report.get("session_id", f"ASSESS-{int(time.time())}")
            out_file = os.path.join(out_dir, f"{sess_id}_report.md")
            self.assessment_session.export_markdown_report(out_file)
            lbl_saved.configure(text=f"✅ Saved to: reports/{os.path.basename(out_file)}")

        btn_export = ctk.CTkButton(
            btn_bar,
            text="💾 Export Markdown Report",
            command=do_export,
            fg_color="#059669",
            hover_color="#047857",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        btn_export.pack(side="right", padx=6)

        def do_rerun():
            modal.destroy()
            self.start_20s_assessment()

        btn_rerun = ctk.CTkButton(
            btn_bar,
            text="🔄 Run New 20s Assessment",
            command=do_rerun,
            fg_color="#7C3AED",
            hover_color="#6D28D9",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        btn_rerun.pack(side="right", padx=6)

        btn_close = ctk.CTkButton(
            btn_bar,
            text="✕ Close",
            command=modal.destroy,
            fg_color="#334155",
            hover_color="#475569",
            width=80,
            font=ctk.CTkFont(size=12)
        )
        btn_close.pack(side="right", padx=6)

    # ── Background Worker Threads ─────────────────────────────────────────────

    def _video_worker(self):
        """Continuously captures camera frames, runs facial CNN, and draws clean box."""
        while self.is_running and self.cam_cap is not None:
            ret, frame = self.cam_cap.read()
            if not ret or frame is None:
                time.sleep(0.03)
                continue

            frame_bgr = cv2.flip(frame, 1)  # Natural mirror view
            pred = self.predictor.predict_image(frame_bgr)

            # Draw clean box with morphological style tag
            if pred.get("face_detected") and pred.get("bboxes"):
                bbox = pred["bboxes"][0]
                top_emo = pred.get("top_emotion", "Neutral")
                conf = pred.get("confidence", 0.0)
                style_info = pred.get("facial_style")
                FacePreprocessor.draw_clean_box(frame_bgr, bbox, top_emo, conf, style_info=style_info)

            with self.lock:
                self.latest_frame = frame_bgr
                self.face_res = pred

            time.sleep(0.03)  # ~30 FPS

    def _audio_worker(self):
        """Continuously pulls audio buffer from stream and runs speech BiLSTM."""
        while self.is_running:
            vol = self.audio_stream.get_volume()
            has_speech = self.audio_stream.has_speech()

            with self.lock:
                self.current_volume = vol
                self.is_speech_active = has_speech

            if not self.mic_muted and has_speech:
                audio_buffer = self.audio_stream.get_audio_buffer()
                if audio_buffer is not None and len(audio_buffer) > 0:
                    pred = self.predictor.predict_audio(audio_buffer)
                    with self.lock:
                        self.speech_res = pred
            else:
                with self.lock:
                    if self.mic_muted:
                        self.speech_res = None

            time.sleep(0.25)  # Speech inference every 250ms

    def _text_worker(self):
        """Asynchronously transcribes audio and analyzes semantic sentiment without stalling audio stream."""
        while self.is_running:
            if not self.mic_muted and self.is_speech_active:
                audio_buffer = self.audio_stream.get_audio_buffer()
                if audio_buffer is not None and len(audio_buffer) >= 16000:  # At least 1.0s
                    try:
                        text_pred = self.predictor.transcribe_and_predict_audio(audio_buffer)
                        if text_pred and text_pred.get("has_text"):
                            with self.lock:
                                self.text_res = text_pred
                    except Exception:
                        pass
            time.sleep(1.2)  # Check speech-to-text every 1.2s

    # ── Main Thread GUI Periodic Tick ──────────────────────────────────────────

    def ui_tick(self):
        """Called every 40ms (~25 FPS) on main Tkinter thread to update all UI elements."""
        if self.is_running:
            with self.lock:
                frame = self.latest_frame.copy() if self.latest_frame is not None else None
                face_p = self.face_res.copy() if self.face_res is not None else None
                speech_p = self.speech_res.copy() if self.speech_res is not None else None
                text_p = self.text_res.copy() if self.text_res is not None else None
                vol = self.current_volume
                speech_active = self.is_speech_active

            # 1. Update Video Frame
            if frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(frame_rgb)
                img_tk = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(520, 290))
                self.video_label.configure(image=img_tk, text="")

            # 1b. Update Facial Morphological Style Card
            f_style = face_p.get("facial_style") if face_p else None
            if f_style and face_p.get("face_detected"):
                m_s = f_style.get("mouth_style", "Neutral Horizontal")
                e_s = f_style.get("eye_style", "Normal Open")
                f_s = f_style.get("forehead_style", "Plain Smooth Forehead")
                s_emo = f_style.get("predicted_emotion", "Neutral")
                s_conf = f_style.get("confidence", 0.0)
                s_desc = f_style.get("style_description", "")

                m_short = m_s.replace(" (Smiley)", "").replace(" (Frown)", "").replace("Curved ", "")
                e_short = e_s.replace(" (Drooping)", "")
                f_short = "Smooth" if "Smooth" in f_s else ("Shrunk" if "Shrunk" in f_s else "Raised")

                self.lbl_badge_mouth.configure(text=f"👄 Mouth: {m_short}")
                self.lbl_badge_eyes.configure(text=f"👁️ Eyes: {e_short}")
                self.lbl_badge_forehead.configure(text=f"🧠 Brow: {f_short}")
                self.lbl_style_diag_badge.configure(
                    text=f"{EMOTION_ICONS.get(s_emo, '🎭')} {s_emo} ({int(s_conf*100)}%)",
                    fg_color=EMOTION_COLORS.get(s_emo, "#334155"),
                    text_color="#FFFFFF"
                )
                self.lbl_style_reasoning.configure(text=f"“{s_desc}”", text_color="#E2E8F0")
            else:
                self.lbl_badge_mouth.configure(text="👄 Mouth: —")
                self.lbl_badge_eyes.configure(text="👁️ Eyes: —")
                self.lbl_badge_forehead.configure(text="🧠 Brow: —")
                self.lbl_style_diag_badge.configure(text="Awaiting Face", fg_color="#334155", text_color="#CBD5E1")
                self.lbl_style_reasoning.configure(text="“Awaiting face detection for micro-expression analysis...”", text_color="#64748B")

            # 2. Update Microphone Volume Meter
            clamped_vol = min(1.0, vol * 6.0)
            self.bar_mic_volume.set(clamped_vol)
            if self.mic_muted:
                self.lbl_mic_activity.configure(text="Muted", text_color="#EF4444")
            elif speech_active:
                self.lbl_mic_activity.configure(text=f"Speaking ({int(clamped_vol*100)}%)", text_color="#10B981")
            else:
                self.lbl_mic_activity.configure(text="Silent", text_color="#94A3B8")

            # 3. Update Transcribed Speech Banner
            if text_p and text_p.get("has_text"):
                t_words = text_p.get("text", "")
                t_emo = text_p.get("top_emotion", "Neutral")
                t_conf = text_p.get("confidence", 0.0)
                t_icon = EMOTION_ICONS.get(t_emo, "💬")
                self.lbl_transcribed_text.configure(text=f"“{t_words}”", text_color="#F1F5F9")
                self.lbl_stt_sentiment_badge.configure(
                    text=f"{t_icon} {t_emo} ({int(t_conf*100)}%)",
                    fg_color=EMOTION_COLORS.get(t_emo, "#334155"),
                    text_color="#FFFFFF"
                )
            elif speech_active:
                self.lbl_transcribed_text.configure(text="“Transcribing spoken phrase...”", text_color="#94A3B8")
            else:
                self.lbl_transcribed_text.configure(text="“Awaiting vocal input...”", text_color="#64748B")

            # 4. Compute Tri-Modal Multimodal Fusion
            f_probs = face_p.get("probabilities") if face_p else None
            s_probs = speech_p.get("probabilities") if (speech_p and not self.mic_muted and speech_active) else None
            t_probs = text_p.get("probabilities") if (text_p and text_p.get("has_text")) else None

            f_detected = bool(face_p and face_p.get("face_detected"))
            s_detected = bool(s_probs is not None)
            t_detected = bool(t_probs is not None)

            fused = self.predictor.fusion.fuse(
                face_probs=f_probs,
                speech_probs=s_probs,
                text_probs=t_probs,
                face_detected=f_detected,
                speech_detected=s_detected,
                text_detected=t_detected
            )

            # 5. Update Fused Card
            final_emo = fused["final_emotion"]
            conf = fused["confidence"]
            status = fused["status"]
            conflict_detected = fused.get("conflict_detected", False)
            color = EMOTION_COLORS.get(final_emo, "#3B82F6")
            icon = EMOTION_ICONS.get(final_emo, "🎭")

            self.lbl_fused_emotion.configure(text=f"{icon} {final_emo}", text_color=color)
            self.lbl_fused_conf.configure(text=f"Combined Confidence: {conf*100:.1f}%")
            self.card_fused.configure(border_color=color)

            # Status pill styling
            if conflict_detected:
                reason = fused.get("conflict_reason", "Divergence detected")
                self.lbl_conflict_status.configure(
                    text=f"⚠️ {reason}",
                    fg_color="#7F1D1D",
                    text_color="#FCA5A5"
                )
            elif len(fused.get("active_modalities", [])) >= 2:
                self.lbl_conflict_status.configure(
                    text=f"✅ Concordant ({status})",
                    fg_color="#14532D",
                    text_color="#86EFAC"
                )
            else:
                self.lbl_conflict_status.configure(
                    text=f"Modality Mode: {status}",
                    fg_color="#334155",
                    text_color="#CBD5E1"
                )

            # 6. Update Individual Badges (3 Badges)
            if face_p and f_detected:
                f_top = face_p.get("top_emotion", "Neutral")
                f_c = face_p.get("confidence", 0.0)
                self.lbl_face_val.configure(text=f"{f_top} ({f_c*100:.1f}%)", text_color=EMOTION_COLORS.get(f_top, "#F1F5F9"))
            else:
                self.lbl_face_val.configure(text="No Face", text_color="#94A3B8")

            if speech_p and s_detected:
                s_top = speech_p.get("top_emotion", "Neutral")
                s_c = speech_p.get("confidence", 0.0)
                self.lbl_speech_val.configure(text=f"{s_top} ({s_c*100:.1f}%)", text_color=EMOTION_COLORS.get(s_top, "#F1F5F9"))
            elif self.mic_muted:
                self.lbl_speech_val.configure(text="Mic Muted", text_color="#EF4444")
            else:
                self.lbl_speech_val.configure(text="Silent", text_color="#94A3B8")

            if text_p and t_detected:
                t_top = text_p.get("top_emotion", "Neutral")
                t_c = text_p.get("confidence", 0.0)
                self.lbl_text_val.configure(text=f"{t_top} ({t_c*100:.1f}%)", text_color=EMOTION_COLORS.get(t_top, "#F1F5F9"))
            else:
                self.lbl_text_val.configure(text="No Speech Text", text_color="#94A3B8")

            # 7. Update Probability Progress Bars
            fused_probs = fused.get("fused_probabilities", {})
            for emotion, (bar, val_lbl) in self.prob_bars.items():
                p = fused_probs.get(emotion, 0.0)
                bar.set(p)
                val_lbl.configure(text=f"{int(p * 100)}%")

            # 8. Add Data Point to Emotion Timeline Tracker
            self.timeline_tracker.add_point(
                dominant_emotion=final_emo,
                confidence=conf,
                probabilities=fused_probs,
                is_conflict=conflict_detected
            )

            # 9. Update Timeline Analytics Header Badges
            stats = self.timeline_tracker.get_stats()
            self.lbl_stat_time.configure(text=f"⏱️ Session: {stats['session_time_str']}")
            dom_emo = stats["dominant_emotion"]
            dom_icon = EMOTION_ICONS.get(dom_emo, "")
            self.lbl_stat_dominant.configure(
                text=f"🎭 Dominant: {dom_icon} {dom_emo} ({stats['dominant_pct']}%)",
                text_color=EMOTION_COLORS.get(dom_emo, "#60A5FA")
            )
            self.lbl_stat_stability.configure(
                text=f"⚖️ Stability: {stats['stability_pct']}% ({stats['stability_label']})"
            )
            val_val = stats["avg_valence"]
            val_sign = "+" if val_val >= 0 else ""
            self.lbl_stat_valence.configure(
                text=f"📊 Valence: {val_sign}{val_val:.2f} ({stats['valence_label']})"
            )

            # 10. Render Live Canvas
            self.timeline_tracker.render_canvas(self.canvas_timeline)

            # 11. Optional TTS Announcement
            now = time.time()
            if self.tts_enabled and conf > 0.60 and final_emo != self.last_spoken_emotion and (now - self.last_speech_time) > 4.0:
                self.last_spoken_emotion = final_emo
                self.last_speech_time = now
                self.voice_assistant.speak(f"Detected {final_emo}")

            # 12. 20-Second Simultaneous Tri-Modal Assessment Tracking
            if self.is_assessing:
                self.assessment_session.record_tick(
                    face_pred=face_p,
                    speech_pred=speech_p,
                    fused_pred=fused,
                    current_volume=vol,
                    is_speaking=speech_active,
                    text_pred=text_p
                )
                rem = self.assessment_session.remaining_seconds()
                prog = self.assessment_session.progress()
                self.lbl_assess_timer.configure(text=f"⏱️ Assessment: {rem:.1f}s left ({int(prog*100)}%)")
                self.bar_assess_progress.set(prog)

                if self.assessment_session.is_finished():
                    self.is_assessing = False
                    self.btn_20s_assess.configure(state="normal")
                    self.btn_cancel_assess.pack_forget()
                    self.lbl_assess_timer.configure(text="✅ Assessment Completed!", text_color="#10B981")
                    self.bar_assess_progress.set(1.0)
                    self.lbl_system_status.configure(text="● ASSESSMENT COMPLETE", fg_color="#15803D", text_color="#DCFCE7")

                    report = self.assessment_session.generate_final_report()

                    # Optional TTS announcement of final result
                    if self.tts_enabled:
                        dom = report.get("dominant_emotion", "Neutral")
                        conf_pct = int(report.get("dominant_confidence", 0))
                        self.voice_assistant.speak(f"20-second assessment complete. Dominant emotion is {dom} with {conf_pct} percent confidence.")

                    # Open Final Result Report Modal Dialog
                    self.after(300, lambda: self.show_assessment_report_modal(report))

        # Re-queue next tick
        self.after(40, self.ui_tick)

    def on_close(self):
        """Cleanup upon closing application."""
        self.is_running = False
        if self.cam_cap is not None:
            self.cam_cap.release()
        self.audio_stream.stop()
        self.destroy()


def launch_desktop():
    """Entry point to launch the clean medium-basic desktop application."""
    app = RealtimeEmotionApp()
    app.mainloop()


if __name__ == "__main__":
    launch_desktop()
