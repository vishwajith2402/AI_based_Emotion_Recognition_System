"""
Multimodal Feature Fusion Module
AGB1303 - AI Problem Solving Techniques
Implements Decision-Level Weighted Fusion, Modality Conflict Analysis,
and Dynamic Tri-Modal Fallbacks (Vision + Acoustics + Text).
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from src.config import (
    EMOTION_CLASSES,
    DEFAULT_FACE_WEIGHT,
    DEFAULT_SPEECH_WEIGHT,
    DEFAULT_TEXT_WEIGHT,
    DUAL_FACE_WEIGHT,
    DUAL_SPEECH_WEIGHT,
    CONFLICT_THRESHOLD
)


class MultimodalFusion:
    """
    Combines visual (facial), acoustic (speech), and linguistic (text) predictions
    with dynamic weighting, tri-modal conflict detection, and graceful fallbacks.
    """

    def __init__(
        self,
        face_weight: float = DEFAULT_FACE_WEIGHT,
        speech_weight: float = DEFAULT_SPEECH_WEIGHT,
        text_weight: float = DEFAULT_TEXT_WEIGHT
    ):
        self.face_weight = face_weight
        self.speech_weight = speech_weight
        self.text_weight = text_weight
        self._normalize_base_weights()

    def _normalize_base_weights(self):
        total = self.face_weight + self.speech_weight + self.text_weight
        if total > 0:
            self.face_weight /= total
            self.speech_weight /= total
            self.text_weight /= total
        else:
            self.face_weight = DEFAULT_FACE_WEIGHT
            self.speech_weight = DEFAULT_SPEECH_WEIGHT
            self.text_weight = DEFAULT_TEXT_WEIGHT

    def set_weights(self, face_weight: float, speech_weight: float, text_weight: Optional[float] = None):
        """Allows dynamic adjustment of fusion modality weights."""
        self.face_weight = max(0.0, face_weight)
        self.speech_weight = max(0.0, speech_weight)
        if text_weight is not None:
            self.text_weight = max(0.0, text_weight)
        self._normalize_base_weights()

    def compute_pair_conflict(
        self,
        p1: Dict[str, float],
        p2: Dict[str, float],
        name1: str,
        name2: str
    ) -> Tuple[bool, float, str]:
        """Calculates conflict severity between any two modality probability distributions."""
        v1 = np.array([p1.get(c, 0.0) for c in EMOTION_CLASSES])
        v2 = np.array([p2.get(c, 0.0) for c in EMOTION_CLASSES])

        top1 = EMOTION_CLASSES[int(np.argmax(v1))]
        top2 = EMOTION_CLASSES[int(np.argmax(v2))]

        # Total variation distance: 0.5 * sum(|P - Q|) in [0, 1]
        tv_dist = float(0.5 * np.sum(np.abs(v1 - v2)))
        is_conflict = (top1 != top2) and (tv_dist > CONFLICT_THRESHOLD)

        if is_conflict:
            msg = f"{name1} expresses '{top1}' but {name2} conveys '{top2}' (discrepancy: {tv_dist:.2f})"
        else:
            msg = f"{name1} and {name2} aligned"

        return is_conflict, tv_dist, msg

    def compute_tri_conflict(
        self,
        active_modalities: Dict[str, Dict[str, float]]
    ) -> Tuple[bool, float, str]:
        """
        Evaluates cross-modal conflict across all currently active modalities.
        Can detect verbal irony / sarcasm (e.g. happy text vs angry speech/face).
        """
        keys = list(active_modalities.keys())
        if len(keys) < 2:
            return False, 0.0, "Single or zero modality active; no conflict possible."

        conflicts = []
        max_dist = 0.0

        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                m1, m2 = keys[i], keys[j]
                is_conf, dist, reason = self.compute_pair_conflict(
                    active_modalities[m1], active_modalities[m2], m1, m2
                )
                if dist > max_dist:
                    max_dist = dist
                if is_conf:
                    conflicts.append(reason)

        if conflicts:
            combined_reason = " | ".join(conflicts)
            return True, max_dist, f"Divergence detected: {combined_reason}"
        else:
            return False, max_dist, "Active modalities aligned within normal concordant variance."

    def fuse(
        self,
        face_probs: Optional[Dict[str, float]],
        speech_probs: Optional[Dict[str, float]],
        text_probs: Optional[Dict[str, float]] = None,
        face_detected: bool = True,
        speech_detected: bool = True,
        text_detected: bool = False
    ) -> Dict:
        """
        Performs robust Tri-Modal decision fusion (Vision + Acoustics + Text).
        Dynamically re-normalizes weights across any combination of active channels.
        Handles missing modalities gracefully with automatic fallbacks.
        """
        active_modalities = {}
        active_weights = {}

        if face_detected and face_probs is not None:
            active_modalities["Face (Vision)"] = face_probs
            active_weights["Face"] = self.face_weight

        if speech_detected and speech_probs is not None:
            active_modalities["Speech (Acoustics)"] = speech_probs
            active_weights["Speech"] = self.speech_weight

        if text_detected and text_probs is not None:
            active_modalities["Text (Semantics)"] = text_probs
            active_weights["Text"] = self.text_weight

        # Case A: Tri-Modal or Multi-Modal Active
        if len(active_modalities) >= 2:
            # Re-normalize weights among active modalities
            w_sum = sum(active_weights.values())
            if w_sum > 0:
                norm_weights = {k: v / w_sum for k, v in active_weights.items()}
            else:
                norm_weights = {k: 1.0 / len(active_weights) for k in active_weights}

            fused_probs = {c: 0.0 for c in EMOTION_CLASSES}

            if "Face" in norm_weights:
                w = norm_weights["Face"]
                for c in EMOTION_CLASSES:
                    fused_probs[c] += w * face_probs[c]

            if "Speech" in norm_weights:
                w = norm_weights["Speech"]
                for c in EMOTION_CLASSES:
                    fused_probs[c] += w * speech_probs[c]

            if "Text" in norm_weights:
                w = norm_weights["Text"]
                for c in EMOTION_CLASSES:
                    fused_probs[c] += w * text_probs[c]

            # Ensure sum = 1.0
            total_p = sum(fused_probs.values())
            if total_p > 0:
                fused_probs = {k: float(v / total_p) for k, v in fused_probs.items()}

            best_emotion = max(fused_probs, key=fused_probs.get)
            # Expressive Salience Calibration: prevent neutral resting baseline from dominating active expressions
            if best_emotion == "Neutral":
                non_neutral = {e: p for e, p in fused_probs.items() if e != "Neutral"}
                best_nn = max(non_neutral, key=non_neutral.get)
                if non_neutral[best_nn] >= 0.22 and (fused_probs["Neutral"] - non_neutral[best_nn]) < 0.15:
                    best_emotion = best_nn
            confidence = fused_probs[best_emotion]

            is_conflict, conflict_score, reason = self.compute_tri_conflict(active_modalities)

            if len(active_modalities) == 3:
                mode_str = "Tri-Modal Fused (Vision + Acoustics + Text)"
            else:
                mode_str = f"Dual-Modal Fused ({' + '.join(active_modalities.keys())})"

            status = mode_str if not is_conflict else f"{mode_str} [Conflict: {reason}]"

            return {
                "final_emotion": best_emotion,
                "confidence": confidence,
                "fused_probabilities": fused_probs,
                "status": status,
                "conflict_detected": is_conflict,
                "conflict_score": conflict_score,
                "conflict_reason": reason,
                "weights_used": norm_weights,
                "active_modalities": list(active_modalities.keys())
            }

        # Case B: Exactly 1 Modality Active (Unimodal Fallbacks)
        elif len(active_modalities) == 1:
            mod_name = list(active_modalities.keys())[0]
            raw_probs = list(active_modalities.values())[0]
            best_emotion = max(raw_probs, key=raw_probs.get)
            if best_emotion == "Neutral":
                non_neutral = {e: p for e, p in raw_probs.items() if e != "Neutral"}
                best_nn = max(non_neutral, key=non_neutral.get)
                if non_neutral[best_nn] >= 0.22 and (raw_probs["Neutral"] - non_neutral[best_nn]) < 0.15:
                    best_emotion = best_nn
            confidence = raw_probs[best_emotion]

            return {
                "final_emotion": best_emotion,
                "confidence": confidence,
                "fused_probabilities": raw_probs.copy(),
                "status": f"{mod_name}-Only Fallback",
                "conflict_detected": False,
                "conflict_score": 0.0,
                "conflict_reason": f"Only {mod_name} active",
                "weights_used": {mod_name.split()[0]: 1.0},
                "active_modalities": [mod_name]
            }

        # Case C: Zero Active Modalities (Idle baseline)
        else:
            uniform_prob = 1.0 / len(EMOTION_CLASSES)
            uniform_dict = {c: uniform_prob for c in EMOTION_CLASSES}
            return {
                "final_emotion": "Neutral",
                "confidence": uniform_prob,
                "fused_probabilities": uniform_dict,
                "status": "No Active Modality (Idle)",
                "conflict_detected": False,
                "conflict_score": 0.0,
                "conflict_reason": "No input signals detected",
                "weights_used": {},
                "active_modalities": []
            }
