import sys
from pathlib import Path

PROJECT_ROOT = Path("D:/Practice Projects/AIPS - AI in Problem Solving Techniques/AIPS Prototype")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predictor import EmotionPredictor

pred = EmotionPredictor()

print("\n--- Verifying All 5 Long Audio Presets ---")
for emo in ['happy', 'sad', 'surprise', 'angry', 'neutral']:
    p = PROJECT_ROOT / "demo_samples" / "speech" / f"{emo}.wav"
    pred.reset_smoothing()
    res = pred.predict_audio_file(str(p))
    top = res["top_emotion"]
    conf = res["confidence"]
    print(f"{emo.upper()}: Top={top} ({conf*100:.1f}%) | Probs={res['probabilities']}")
