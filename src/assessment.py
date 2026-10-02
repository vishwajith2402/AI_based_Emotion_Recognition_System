"""
assessment.py
=============
20-Second Simultaneous Tri-Modal Emotion Assessment Engine.
Course: AGB1303 - AI Problem Solving Techniques (Batch 6)

Collects and aggregates continuous real-time data streams across:
- Facial Expression Recognition (CNN)
- Speech Modulation Analysis (MFCC + BiLSTM)
- Speech-to-Text & Semantic Sentiment (STT + VADER + Lexicon)

Calculates:
- 20-Second Dominant Overall Emotion & Combined Confidence
- Affective Valence Trajectory (Russell's Circumplex Affect Model)
- Emotional Stability Index (Variance of Valence over time)
- Tri-Modal Congruence & Cross-Modality Divergence Ratio
- Speaking Time Ratio, Face Engagement Ratio & Linguistic Density
- Professional Behavioral & Affective Computing Diagnostic Insights
"""

import time
import json
import collections
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any, List

from src.config import (
    EMOTION_CLASSES, EMOTION_COLORS, EMOTION_ICONS,
    DEFAULT_ASSESSMENT_DURATION
)


class AssessmentSession:
    """
    Orchestrates a 20-second simultaneous tri-modal emotion assessment session.
    """

    VALENCE_MAP = {
        "Happy": 1.0,
        "Surprise": 0.4,
        "Neutral": 0.0,
        "Sad": -0.6,
        "Angry": -0.9
    }

    def __init__(self, target_duration: float = DEFAULT_ASSESSMENT_DURATION):
        self.target_duration = target_duration
        self.start_time = None
        self.end_time = None
        self.is_active = False

        # Recorded Time-Series Data
        self.samples: List[Dict[str, Any]] = []
        self.transcribed_utterances: List[Dict[str, Any]] = []

    def start(self):
        """Starts a fresh 20-second assessment session."""
        self.start_time = time.time()
        self.end_time = None
        self.is_active = True
        self.samples.clear()
        self.transcribed_utterances.clear()

    def stop(self):
        """Manually stops the session."""
        if self.is_active:
            self.end_time = time.time()
            self.is_active = False

    def elapsed_seconds(self) -> float:
        if not self.start_time:
            return 0.0
        if self.is_active:
            return time.time() - self.start_time
        return (self.end_time or time.time()) - self.start_time

    def remaining_seconds(self) -> float:
        return max(0.0, self.target_duration - self.elapsed_seconds())

    def progress(self) -> float:
        """Returns session progress between 0.0 and 1.0."""
        return min(1.0, self.elapsed_seconds() / max(0.1, self.target_duration))

    def is_finished(self) -> bool:
        if not self.is_active:
            return False
        if self.elapsed_seconds() >= self.target_duration:
            self.stop()
            return True
        return False

    def record_tick(
        self,
        face_pred: Optional[dict],
        speech_pred: Optional[dict],
        fused_pred: Optional[dict],
        current_volume: float,
        is_speaking: bool,
        text_pred: Optional[dict] = None
    ):
        """
        Records a synchronized tri-modal snapshot during the 20-second window.
        """
        if not self.is_active:
            return

        now = self.elapsed_seconds()

        # Instantaneous Affective Valence
        valence = 0.0
        if fused_pred and "fused_probabilities" in fused_pred:
            for emo, p in fused_pred["fused_probabilities"].items():
                valence += p * self.VALENCE_MAP.get(emo, 0.0)
        elif fused_pred:
            valence = self.VALENCE_MAP.get(fused_pred.get("final_emotion", "Neutral"), 0.0)
        valence = float(np.clip(valence, -1.0, 1.0))

        # 1. Face Data
        face_valid = bool(face_pred and face_pred.get("face_detected"))
        face_emo = face_pred.get("top_emotion", "Neutral") if face_valid else None
        face_conf = float(face_pred.get("confidence", 0.0)) if face_valid else 0.0

        # 2. Speech Data
        speech_valid = bool(speech_pred and is_speaking)
        speech_emo = speech_pred.get("top_emotion", "Neutral") if speech_valid else None
        speech_conf = float(speech_pred.get("confidence", 0.0)) if speech_valid else 0.0

        # 3. Text Semantics Data
        text_valid = bool(text_pred and text_pred.get("has_text"))
        text_emo = text_pred.get("top_emotion", "Neutral") if text_valid else None
        text_conf = float(text_pred.get("confidence", 0.0)) if text_valid else 0.0
        raw_text = text_pred.get("text", "") if text_valid else ""

        if text_valid and raw_text:
            # Avoid consecutive duplicate records
            if not self.transcribed_utterances or self.transcribed_utterances[-1]["text"] != raw_text:
                self.transcribed_utterances.append({
                    "time": round(now, 1),
                    "text": raw_text,
                    "emotion": text_emo,
                    "confidence": text_conf,
                    "keywords": text_pred.get("detected_keywords", [])
                })

        # 4. Fused Data
        fused_emo = fused_pred.get("final_emotion", "Neutral") if fused_pred else "Neutral"
        fused_conf = float(fused_pred.get("confidence", 0.0)) if fused_pred else 0.0
        is_conflict = bool(fused_pred.get("conflict_detected", False)) if fused_pred else False

        # Morphological Facial Style
        facial_style = face_pred.get("facial_style") if face_pred else None

        entry = {
            "time": now,
            "face_valid": face_valid,
            "face_emotion": face_emo,
            "face_confidence": face_conf,
            "facial_style": facial_style,
            "speech_valid": speech_valid,
            "speech_emotion": speech_emo,
            "speech_confidence": speech_conf,
            "text_valid": text_valid,
            "text_emotion": text_emo,
            "text_confidence": text_conf,
            "text_raw": raw_text,
            "fused_emotion": fused_emo,
            "fused_confidence": fused_conf,
            "fused_probabilities": fused_pred.get("fused_probabilities", {}) if fused_pred else {},
            "valence": valence,
            "volume": float(current_volume),
            "is_speaking": bool(is_speaking),
            "is_conflict": is_conflict
        }

        self.samples.append(entry)

    def generate_final_report(self) -> dict:
        """
        Compiles the complete 20-second analytical report across all 3 modalities.
        """
        if not self.samples:
            return {
                "status": "NO_DATA",
                "message": "No data points were collected during the assessment window."
            }

        total_ticks = len(self.samples)
        duration = self.elapsed_seconds()

        # 1. Emotion Frequencies & Expressive Dominant Emotion Resolution
        fused_counts = collections.Counter(s["fused_emotion"] for s in self.samples)

        # Priority to non-neutral active emotional expressions
        expressive_counts = collections.Counter({
            e: count for e, count in fused_counts.items()
            if e != "Neutral" and count > 0
        })

        resting_neutral_ticks = fused_counts.get("Neutral", 0)
        resting_neutral_pct = round((resting_neutral_ticks / total_ticks) * 100.0, 1)

        if expressive_counts:
            # Pick the most prevalent non-neutral emotion displayed during the session
            dominant_emo, exp_count = expressive_counts.most_common(1)[0]
            dominant_pct = round((exp_count / total_ticks) * 100.0, 1)
            # Confidence of occurrences of this dominant expressive emotion
            dom_confidences = [s["fused_confidence"] for s in self.samples if s["fused_emotion"] == dominant_emo]
            mean_dom_conf = float(np.mean(dom_confidences)) if dom_confidences else 0.0
        else:
            # If all ticks predicted Neutral, inspect continuous probabilities across non-neutral classes
            mean_nn_probs = {
                c: float(np.mean([s.get("fused_probabilities", {}).get(c, 0.0) for s in self.samples]))
                for c in EMOTION_CLASSES if c != "Neutral"
            }
            if mean_nn_probs and max(mean_nn_probs.values()) >= 0.12:
                dominant_emo = max(mean_nn_probs, key=mean_nn_probs.get)
                dominant_pct = round(mean_nn_probs[dominant_emo] * 100.0, 1)
                dom_confidences = [s.get("fused_probabilities", {}).get(dominant_emo, 0.0) for s in self.samples]
                mean_dom_conf = float(np.mean(dom_confidences)) if dom_confidences else 0.50
            else:
                dominant_emo, dom_count = fused_counts.most_common(1)[0]
                dominant_pct = (dom_count / total_ticks) * 100.0
                dom_confidences = [s["fused_confidence"] for s in self.samples if s["fused_emotion"] == dominant_emo]
                mean_dom_conf = float(np.mean(dom_confidences)) if dom_confidences else 0.0

        # Full Distribution across 5 classes
        distribution = {}
        for cls in EMOTION_CLASSES:
            count = fused_counts.get(cls, 0)
            distribution[cls] = {
                "count": count,
                "percentage": round((count / total_ticks) * 100.0, 1),
                "icon": EMOTION_ICONS.get(cls, ""),
                "color": EMOTION_COLORS.get(cls, "#3B82F6")
            }

        # 2. Valence & Mood Trajectory
        valences = [s["valence"] for s in self.samples]
        mean_valence = float(np.mean(valences))
        min_valence = float(np.min(valences))
        max_valence = float(np.max(valences))
        val_std = float(np.std(valences)) if len(valences) > 1 else 0.0

        if mean_valence > 0.25:
            val_label = "Positive Affect (Engaged/Optimistic)"
            val_sentiment = "Positive"
        elif mean_valence < -0.25:
            val_label = "Negative Affect (Stress/Displeasure/Tension)"
            val_sentiment = "Negative"
        else:
            val_label = "Equilibrium / Neutral Affect (Composed/Focused)"
            val_sentiment = "Neutral"

        # Emotional Stability Score (0 to 100)
        stability_score = int(np.clip((1.0 - (val_std * 1.35)) * 100.0, 10, 100))
        if stability_score >= 85:
            stability_label = "Highly Stable (Consistent Emotional Expression)"
        elif stability_score >= 65:
            stability_label = "Steady (Controlled Emotional State)"
        elif stability_score >= 45:
            stability_label = "Moderate (Responsive to Stimuli)"
        else:
            stability_label = "Dynamic (Frequent Emotional Shifts)"

        # 3. Facial Modality Performance
        face_valid_ticks = sum(1 for s in self.samples if s["face_valid"])
        face_engagement_pct = round((face_valid_ticks / total_ticks) * 100.0, 1)

        face_emos = [s["face_emotion"] for s in self.samples if s["face_valid"]]
        if face_emos:
            face_counter = collections.Counter(face_emos)
            nn_face = {e: c for e, c in face_counter.items() if e != "Neutral" and c > 0}
            if nn_face:
                dom_face_emo = max(nn_face, key=nn_face.get)
                f_count = nn_face[dom_face_emo]
            else:
                dom_face_emo, f_count = face_counter.most_common(1)[0]
            face_dom_pct = round((f_count / len(face_emos)) * 100.0, 1)
            face_confs = [s["face_confidence"] for s in self.samples if s["face_valid"] and s["face_emotion"] == dom_face_emo]
            mean_face_conf = float(np.mean(face_confs)) if face_confs else 0.0
        else:
            dom_face_emo = "No Face Detected"
            face_dom_pct = 0.0
            mean_face_conf = 0.0

        # 3b. Morphological Facial Style Performance
        styles = [s["facial_style"] for s in self.samples if s.get("facial_style")]
        if styles:
            m_styles = [st["mouth_style"] for st in styles if "mouth_style" in st]
            e_styles = [st["eye_style"] for st in styles if "eye_style" in st]
            f_styles = [st["forehead_style"] for st in styles if "forehead_style" in st]
            s_emos = [st["predicted_emotion"] for st in styles if "predicted_emotion" in st]

            nn_m_styles = [m for m in m_styles if m != "Neutral Horizontal"]
            dom_mouth = collections.Counter(nn_m_styles).most_common(1)[0][0] if nn_m_styles else (collections.Counter(m_styles).most_common(1)[0][0] if m_styles else "Neutral Horizontal")

            nn_f_styles = [f for f in f_styles if f != "Plain Smooth Forehead"]
            dom_fhead = collections.Counter(nn_f_styles).most_common(1)[0][0] if nn_f_styles else (collections.Counter(f_styles).most_common(1)[0][0] if f_styles else "Plain Smooth Forehead")

            dom_eye = collections.Counter(e_styles).most_common(1)[0][0] if e_styles else "Normal Open"

            nn_s_emos = [e for e in s_emos if e != "Neutral"]
            dom_s_emo = collections.Counter(nn_s_emos).most_common(1)[0][0] if nn_s_emos else (collections.Counter(s_emos).most_common(1)[0][0] if s_emos else "Neutral")
            style_desc = styles[-1].get("style_description", "")
        else:
            dom_mouth = "Neutral Horizontal"
            dom_eye = "Normal Open"
            dom_fhead = "Plain Smooth Forehead"
            dom_s_emo = "Neutral"
            style_desc = "Neutral horizontal mouth, normal eyes, and plain smooth forehead without changes."

        facial_style_summary = {
            "dominant_style_emotion": dom_s_emo,
            "dominant_mouth_style": dom_mouth,
            "dominant_eye_style": dom_eye,
            "dominant_forehead_style": dom_fhead,
            "style_description": style_desc
        }

        # 4. Speech Modality Performance
        speaking_ticks = sum(1 for s in self.samples if s["is_speaking"])
        speaking_time_pct = round((speaking_ticks / total_ticks) * 100.0, 1)

        speech_emos = [s["speech_emotion"] for s in self.samples if s["speech_valid"]]
        if speech_emos:
            speech_counter = collections.Counter(speech_emos)
            nn_speech = {e: c for e, c in speech_counter.items() if e != "Neutral" and c > 0}
            if nn_speech:
                dom_speech_emo = max(nn_speech, key=nn_speech.get)
                s_count = nn_speech[dom_speech_emo]
            else:
                dom_speech_emo, s_count = speech_counter.most_common(1)[0]
            speech_dom_pct = round((s_count / len(speech_emos)) * 100.0, 1)
            speech_confs = [s["speech_confidence"] for s in self.samples if s["speech_valid"] and s["speech_emotion"] == dom_speech_emo]
            mean_speech_conf = float(np.mean(speech_confs)) if speech_confs else 0.0
        else:
            dom_speech_emo = "Silent / Inactive"
            speech_dom_pct = 0.0
            mean_speech_conf = 0.0

        volumes = [s["volume"] for s in self.samples]
        mean_volume = float(np.mean(volumes)) if volumes else 0.0

        # 5. Semantic / Text Modality Performance
        text_valid_ticks = sum(1 for s in self.samples if s["text_valid"])
        text_coverage_pct = round((text_valid_ticks / total_ticks) * 100.0, 1)

        text_emos = [s["text_emotion"] for s in self.samples if s["text_valid"]]
        if text_emos:
            text_counter = collections.Counter(text_emos)
            nn_text = {e: c for e, c in text_counter.items() if e != "Neutral" and c > 0}
            if nn_text:
                dom_text_emo = max(nn_text, key=nn_text.get)
                t_count = nn_text[dom_text_emo]
            else:
                dom_text_emo, t_count = text_counter.most_common(1)[0]
            text_dom_pct = round((t_count / len(text_emos)) * 100.0, 1)
            text_confs = [s["text_confidence"] for s in self.samples if s["text_valid"] and s["text_emotion"] == dom_text_emo]
            mean_text_conf = float(np.mean(text_confs)) if text_confs else 0.0
        else:
            dom_text_emo = "No Text Detected"
            text_dom_pct = 0.0
            mean_text_conf = 0.0

        # 6. Tri-Modal Congruence & Cross-Modality Divergence
        multi_active_ticks = sum(
            1 for s in self.samples
            if sum([s["face_valid"], s["speech_valid"], s["text_valid"]]) >= 2
        )
        conflict_ticks = sum(1 for s in self.samples if s["is_conflict"])

        if multi_active_ticks > 0:
            concordance_pct = round(((multi_active_ticks - conflict_ticks) / multi_active_ticks) * 100.0, 1)
            concordance_pct = max(0.0, concordance_pct)
            conflict_pct = round((conflict_ticks / multi_active_ticks) * 100.0, 1)
        else:
            concordance_pct = 100.0
            conflict_pct = 0.0

        # 7. Formulate Diagnostic Behavioral & Semantic Insight
        diagnostic_insights = []

        if dominant_emo == "Happy":
            if concordance_pct >= 80:
                diagnostic_insights.append("Subject demonstrated high tri-modal congruence across facial smile expression, vocal prosody, and positive linguistic vocabulary.")
            else:
                diagnostic_insights.append("Subject showed positive visual/vocal cues with subtle semantic or prosodic discrepancies.")
        elif dominant_emo == "Neutral":
            diagnostic_insights.append("Subject maintained high emotional composure, cognitive stability, and neutral linguistic articulation throughout the 20-second window.")
        elif dominant_emo == "Sad":
            diagnostic_insights.append("Subject exhibited subdued affective valence reflected in low vocal energy, melancholy lexicon choices, and downward facial markers.")
        elif dominant_emo == "Angry":
            diagnostic_insights.append("Subject exhibited high-intensity negative arousal with sharp acoustic bursts and critical linguistic keyword density.")
        elif dominant_emo == "Surprise":
            diagnostic_insights.append("Subject displayed dynamic facial eyebrow elevation and sudden acoustic pitch transitions.")

        if len(self.transcribed_utterances) > 0:
            diagnostic_insights.append(f"Speech-to-text successfully captured {len(self.transcribed_utterances)} distinct spoken phrases.")
        else:
            diagnostic_insights.append("No intelligible speech phrases recognized; system gracefully operated on vision and acoustic energy.")

        if conflict_pct > 25:
            diagnostic_insights.append(f"Noteworthy cross-modal discrepancy observed ({conflict_pct}% conflict rate), indicating potential verbal sarcasm, affective masking, or irony.")

        report = {
            "status": "SUCCESS",
            "session_id": f"ASSESS-{int(time.time())}",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": round(duration, 1),
            "total_frames_analyzed": total_ticks,

            # Executive Summary
            "dominant_emotion": dominant_emo,
            "dominant_percentage": round(dominant_pct, 1),
            "resting_neutral_pct": resting_neutral_pct,
            "dominant_confidence": round(mean_dom_conf * 100.0 if mean_dom_conf <= 1.0 else mean_dom_conf, 1),
            "dominant_icon": EMOTION_ICONS.get(dominant_emo, "🎭"),
            "dominant_color": EMOTION_COLORS.get(dominant_emo, "#3B82F6"),

            # Valence & Stability
            "affective_valence": round(mean_valence, 2),
            "valence_min": round(min_valence, 2),
            "valence_max": round(max_valence, 2),
            "valence_label": val_label,
            "valence_sentiment": val_sentiment,
            "stability_score": stability_score,
            "stability_label": stability_label,

            # Modality Performance Breakdown
            "face_summary": {
                "engagement_pct": face_engagement_pct,
                "dominant_emotion": dom_face_emo,
                "dominant_pct": face_dom_pct,
                "mean_confidence": round(mean_face_conf * 100.0, 1)
            },
            "speech_summary": {
                "speaking_time_pct": speaking_time_pct,
                "dominant_emotion": dom_speech_emo,
                "dominant_pct": speech_dom_pct,
                "mean_confidence": round(mean_speech_conf * 100.0, 1),
                "mean_volume_rms": round(mean_volume, 3)
            },
            "text_summary": {
                "coverage_pct": text_coverage_pct,
                "dominant_emotion": dom_text_emo,
                "dominant_pct": text_dom_pct,
                "mean_confidence": round(mean_text_conf * 100.0, 1),
                "utterances_count": len(self.transcribed_utterances),
                "utterances": self.transcribed_utterances
            },
            "multimodal_congruence": {
                "concordance_rate_pct": concordance_pct,
                "conflict_rate_pct": conflict_pct,
                "conflict_ticks": conflict_ticks,
                "multi_active_ticks": multi_active_ticks
            },

            # Morphological Facial Style Summary
            "facial_style_summary": facial_style_summary,

            # Class Distribution
            "emotion_distribution": distribution,

            # Diagnostic Interpretation
            "diagnostic_summary": " ".join(diagnostic_insights)
        }

        return report

    def export_markdown_report(self, filepath: str) -> str:
        """
        Exports the 20-second assessment results to a clean Markdown file.
        """
        report = self.generate_final_report()
        if report.get("status") != "SUCCESS":
            return ""

        f_style = report.get("facial_style_summary", {})

        md_content = f"""# 20-Second Tri-Modal Emotion Assessment Report 📋

**Session ID**: `{report['session_id']}`  
**Timestamp**: {report['timestamp']}  
**Evaluation Duration**: {report['duration_seconds']} seconds ({report['total_frames_analyzed']} tri-modal snapshots)  
**System Architecture**: Tri-Modal Deep Learning & NLP + Facial Morphological Style (Facial CNN + Speech BiLSTM + STT Semantic VADER + Morphological Micro-Expression Engine)

---

## 🏆 Executive Summary

- **Dominant Recognized Emotion**: **{report['dominant_icon']} {report['dominant_emotion']}**
- **Session Dominance**: **{report['dominant_percentage']}%** of evaluation time
- **Mean Model Confidence**: **{report['dominant_confidence']}%**
- **Affective Valence ($V$)**: **{report['affective_valence']:+.2f}** ({report['valence_label']})
- **Emotional Stability**: **{report['stability_score']}%** ({report['stability_label']})

---

## 🎭 Facial Morphological Style & Micro-Expression Reasoning

- **Dominant Morphological Expression**: **{f_style.get('dominant_style_emotion', 'Neutral')}**
- **Mouth Curvature & Opening**: `{f_style.get('dominant_mouth_style', 'Neutral Horizontal')}`
- **Eye Aperture & Eyelid State**: `{f_style.get('dominant_eye_style', 'Normal Open')}`
- **Forehead & Brow Furrows**: `{f_style.get('dominant_forehead_style', 'Plain Smooth Forehead')}`
- **Morphological Diagnostic Reason**: *"{f_style.get('style_description', 'Plain smooth forehead, normal eyes, and horizontal mouth.')}"*

---

## 🔬 Modality-by-Modality Deep Dive (Tri-Modal)

| Metric | Facial Expression (CNN) | Speech Prosody (BiLSTM) | Transcribed Semantics (STT + VADER) | Tri-Modal Fused System |
| :--- | :---: | :---: | :---: | :---: |
| **Dominant Category** | {report['face_summary']['dominant_emotion']} | {report['speech_summary']['dominant_emotion']} | {report['text_summary']['dominant_emotion']} | **{report['dominant_emotion']}** |
| **Modality Confidence** | {report['face_summary']['mean_confidence']}% | {report['speech_summary']['mean_confidence']}% | {report['text_summary']['mean_confidence']}% | **{report['dominant_confidence']}%** |
| **Presence / Coverage** | {report['face_summary']['engagement_pct']}% tracking rate | {report['speech_summary']['speaking_time_pct']}% speaking time | {report['text_summary']['utterances_count']} phrases captured | **100% active stream** |
| **Modality Concordance** | — | — | — | **{report['multimodal_congruence']['concordance_rate_pct']}% aligned** |
| **Cross-Modality Conflict** | — | — | — | **{report['multimodal_congruence']['conflict_rate_pct']}% divergence** |

---

## 💬 Transcribed Spoken Utterances & Semantic Lexicon
"""
        if self.transcribed_utterances:
            md_content += "| Time | Transcribed Text | Semantic Emotion | Confidence | Keywords |\n"
            md_content += "| :--- | :--- | :---: | :---: | :--- |\n"
            for u in self.transcribed_utterances:
                kw_str = ", ".join(u["keywords"]) if u["keywords"] else "—"
                md_content += f"| `{u['time']}s` | \"{u['text']}\" | **{u['emotion']}** | {round(u['confidence']*100, 1)}% | {kw_str} |\n"
        else:
            md_content += "\n*No spoken utterances transcribed during this evaluation.*  \n"

        md_content += f"""
---

## 📊 Complete 20-Second Emotion Distribution

| Emotion Class | Snapshot Count | Percentage of Session | Trajectory Status |
| :--- | :---: | :---: | :--- |
"""
        for cls, d in report["emotion_distribution"].items():
            bar_len = int(d['percentage'] / 5)
            bar_str = "█" * bar_len + "░" * (20 - bar_len)
            md_content += f"| **{d['icon']} {cls}** | {d['count']} | {d['percentage']}% | `{bar_str}` |\n"

        md_content += f"""
---

## 🧠 Diagnostic Behavioral Interpretation

> {report['diagnostic_summary']}

---
*Report generated automatically by AIPS Tri-Modal Emotion Recognition System (AGB1303 Batch 6)*
"""

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(md_content)

        return filepath
