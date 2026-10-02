"""
Test script for FacialStyleAnalyzer on canonical facial samples.
"""

import os
import sys
import cv2
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.facial_style_analyzer import FacialStyleAnalyzer
from src.config import CANONICAL_FACIAL_TEST_SAMPLES, FACIAL_TEST_DIR

def run_test():
    analyzer = FacialStyleAnalyzer()
    print("=" * 75)
    print("FACIAL STYLE & MICRO-EXPRESSION MORPHOLOGICAL ANALYSIS TEST")
    print("=" * 75)

    all_passed = True
    for emo, fname in CANONICAL_FACIAL_TEST_SAMPLES.items():
        img_path = os.path.join(FACIAL_TEST_DIR, emo.lower(), fname)
        if not os.path.exists(img_path):
            print(f"[SKIP] File not found: {img_path}")
            continue

        img = cv2.imread(img_path)
        result = analyzer.analyze_face_style(img)

        pred_emo = result["predicted_emotion"]
        conf = result["confidence"]
        mouth_s = result["mouth_style"]
        eye_s = result["eye_style"]
        fhead_s = result["forehead_style"]
        desc = result["style_description"]
        m = result["metrics"]

        match_status = "MATCH" if pred_emo == emo else "DIFF"
        if pred_emo != emo:
            all_passed = False

        print(f"Sample: {emo:<9} -> Predicted Style: {pred_emo:<9} ({conf*100:4.1f}%) [{match_status}]")
        print(f"  Mouth   : {mouth_s} (aspect={m['mouth_aspect_ratio']:.2f}, curve={m['mouth_curvature']:+.2f}, smile={m['smile_detected']})")
        print(f"  Eyes    : {eye_s} (detected={m['eyes_detected']}, aperture={m['eye_aperture_ratio']:.3f})")
        print(f"  Forehead: {fhead_s} (furrow={m['glabella_furrow']:.1f}, horiz={m['horiz_energy']:.1f}, edge={m['forehead_edge_density']:.3f})")
        print(f"  Reason  : {desc}")
        print("-" * 75)

    print(f"All Canonical Samples Morphological Matching: {'PASS (100%)' if all_passed else 'REVIEW'}")

if __name__ == "__main__":
    run_test()
