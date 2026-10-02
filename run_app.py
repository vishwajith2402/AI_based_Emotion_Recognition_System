"""
Master Application Launcher
AGB1303 - AI Problem Solving Techniques
Batch 6: Human Emotion Recognition Using Facial Expressions and Speech Modulation
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="Launch Human Emotion Recognition (Facial Expressions + Speech Modulation)"
    )
    parser.add_argument(
        "--ui",
        choices=["desktop", "web", "futuristic"],
        default="desktop",
        help="Interface type: 'desktop' (Clean Medium-Basic Real-Time Mic & Camera UI, Default), 'web' (Streamlit Studio), or 'futuristic' (Cyberpunk Glassmorphism)"
    )
    parser.add_argument(
        "--test-all",
        action="store_true",
        help="Run comprehensive test suite across all expressions, speech modulations, conflicts, and fallbacks"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run training pipelines for both Facial CNN and Speech BiLSTM"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run comparative benchmark evaluation across unimodal and multimodal pipelines"
    )
    parser.add_argument(
        "--assess-20s", "--assess",
        dest="assess_20s",
        action="store_true",
        help="Run a 20-second simultaneous facial, vocal and semantic assessment test"
    )
    parser.add_argument(
        "--assess-30s",
        dest="assess_20s",
        action="store_true",
        help="Legacy alias for 20-second assessment test"
    )
    parser.add_argument(
        "--rebuild-data",
        action="store_true",
        help="Rebuild curated facial images and speech audio datasets"
    )

    args = parser.parse_args()

    # Action 0: 20-Second Assessment Test
    if args.assess_20s:
        print("[Launcher] Starting 20-Second Simultaneous Tri-Modal Assessment Test...")
        from src.assessment import AssessmentSession
        from src.predictor import EmotionPredictor
        import time, glob

        predictor = EmotionPredictor()
        session = AssessmentSession(target_duration=3.0)
        session.start()
        print("[Assessment] Collecting simultaneous visual, acoustic, and semantic test vectors...")
        test_images = glob.glob(str(PROJECT_ROOT / "data" / "facial" / "test" / "*" / "*.png"))[:10]
        test_audios = glob.glob(str(PROJECT_ROOT / "data" / "speech" / "test" / "*" / "*.wav"))[:10]
        sample_phrases = [
            "I am so joyful and happy with this amazing result!",
            "Everything is proceeding as expected according to schedule.",
            "This makes me so proud and thrilled!",
            "Standard operating procedure is being verified.",
            "I am genuinely delighted with how smoothly everything works."
        ]
        import cv2
        import soundfile as sf
        for idx, (img_p, aud_p) in enumerate(zip(test_images, test_audios)):
            cv_img = cv2.imread(img_p)
            f_res = predictor.predict_image(cv_img)
            y, sr = sf.read(aud_p)
            s_res = predictor.predict_audio(y)
            phrase = sample_phrases[idx % len(sample_phrases)]
            t_res = predictor.predict_text(phrase)
            fused = predictor.fusion.fuse(
                face_probs=f_res.get("probabilities"),
                speech_probs=s_res.get("probabilities"),
                text_probs=t_res.get("probabilities"),
                face_detected=f_res.get("face_detected", True),
                speech_detected=True,
                text_detected=True
            )
            session.record_tick(f_res, s_res, fused, current_volume=0.25, is_speaking=True, text_pred=t_res)
            time.sleep(0.05)

        rep = session.generate_final_report()
        print(f"\n[Assessment Complete]")
        print(f"  Dominant Emotion : {rep['dominant_emotion']} ({rep['dominant_percentage']}%)")
        print(f"  Mean Confidence  : {rep['dominant_confidence']}%")
        print(f"  Valence Score    : {rep['affective_valence']:+.2f} ({rep['valence_label']})")
        print(f"  Stability Index  : {rep['stability_score']}% ({rep['stability_label']})")
        print(f"  Concordance Rate : {rep['multimodal_congruence']['concordance_rate_pct']}%")
        print(f"  Phrases Captured : {rep['text_summary']['utterances_count']}")
        print(f"  Diagnostic Note  : {rep['diagnostic_summary']}")
        out_f = str(PROJECT_ROOT / "reports" / "assessment_test_report.md")
        session.export_markdown_report(out_f)
        print(f"[Export] Saved report to {out_f}\n")
        return

    # Action 1: Run Full Test Suite
    if args.test_all:
        print("[Launcher] Starting comprehensive test suite for all expressions and modulations...")
        from test_all_modalities import run_full_suite
        run_full_suite()
        return

    # Action 2: Rebuild Data
    if args.rebuild_data:
        print("[Launcher] Rebuilding datasets...")
        from data.dataset_builder import generate_all_datasets
        generate_all_datasets()
        return

    # Action 3: Train Models
    if args.train:
        print("[Launcher] Starting Facial CNN Training...")
        from train_facial import train_facial_model
        train_facial_model(epochs=6, batch_size=128)

        print("\n[Launcher] Starting Speech BiLSTM Training...")
        from train_speech import train_speech_model
        train_speech_model(epochs=15, batch_size=32)
        return

    # Action 4: Evaluate
    if args.evaluate:
        print("[Launcher] Running Benchmark Evaluation...")
        from evaluate import evaluate_all
        evaluate_all()
        return

    # Action 5: Launch Selected User Interface
    if args.ui == "futuristic":
        print("[Launcher] Launching Futuristic Dark-Glassmorphism AI Desktop Dashboard...")
        from ui.futuristic_desktop import launch_futuristic_app
        launch_futuristic_app()

    elif args.ui == "web":
        print("[Launcher] Starting Streamlit Web Dashboard...")
        web_app_script = PROJECT_ROOT / "ui" / "web_app.py"
        cmd = [sys.executable, "-m", "streamlit", "run", str(web_app_script)]
        subprocess.run(cmd)

    elif args.ui == "desktop":
        print("[Launcher] Starting Windows Desktop Interface...")
        from ui.desktop_app import launch_desktop
        launch_desktop()


if __name__ == "__main__":
    main()
