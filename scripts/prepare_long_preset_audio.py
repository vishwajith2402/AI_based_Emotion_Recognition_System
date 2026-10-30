import os
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa

PROJECT_ROOT = Path("D:/Practice Projects/AIPS - AI in Problem Solving Techniques/AIPS Prototype")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

hf_dir = Path.home() / ".cache" / "huggingface" / "hub" / "datasets--humyn-labs--Indian-Emotional-Speech-Corpus" / "snapshots" / "7f5dbd0d03eab228990e6fa43afdcac99506aab8" / "Indian Corpus"

# Load source audios (all resampled to 16kHz mono)
print("Loading high-quality speech source recordings...")
y_east, _ = librosa.load(str(hf_dir / "Old_Male_East.wav"), sr=16000)
y_north, _ = librosa.load(str(hf_dir / "Old_Male_North.wav"), sr=16000)

# Segments definition:
# 1. Happy (Old_Male_East 0.5s to 9.0s -> 8.5s)
# "Last Monday was perfect! I got the job I had been dreaming of! I screamed, hugged my parents, and we even had cake for breakfast."
happy_clip = y_east[int(0.5 * 16000):int(9.0 * 16000)]

# 2. Sad (Old_Male_East 9.5s to 17.5s -> 8.0s)
# "Two days later, my mom called... our dog Bruno had passed away in his sleep. I didn't even get to say goodbye."
sad_clip = y_east[int(9.5 * 16000):int(17.5 * 16000)]

# 3. Surprise (Old_Male_East 17.5s to 26.5s -> 9.0s)
# "And just when I thought the week couldn't get any crazier, I got a surprise call! I was selected for a global fellowship I never even applied for!"
surprise_clip = y_east[int(17.5 * 16000):int(26.5 * 16000)]

# 4. Angry (Old_Male_East 27.0s to 35.5s -> 8.5s)
# "Earlier that day, my Wi-Fi cut out again during an important meeting... third time this week! It's beyond frustrating!"
angry_clip = y_east[int(27.0 * 16000):int(35.5 * 16000)]

# 5. Neutral (Old_Male_North 36.0s to 44.0s -> 8.0s)
# "What a week. Honestly, I'm still processing it all, trying to reflect calmly on everything that happened."
neutral_clip = y_north[int(36.0 * 16000):int(44.0 * 16000)]

clips = {
    "happy": happy_clip,
    "sad": sad_clip,
    "surprise": surprise_clip,
    "angry": angry_clip,
    "neutral": neutral_clip,
}

# Normalize each to peak amplitude 0.95
for name, clip in clips.items():
    peak = np.max(np.abs(clip))
    if peak > 0:
        clips[name] = (clip / peak) * 0.95

# Output directories
out_dirs = [
    PROJECT_ROOT / "demo_samples" / "speech",
    PROJECT_ROOT / "ui" / "static" / "demo_samples" / "speech"
]

for od in out_dirs:
    od.mkdir(parents=True, exist_ok=True)
    for name, clip in clips.items():
        out_path = od / f"{name}.wav"
        sf.write(str(out_path), clips[name], 16000)
        dur = len(clips[name]) / 16000
        print(f"Saved {name}.wav to {od} ({dur:.2f}s, peak={np.max(np.abs(clips[name])):.2f})")

print("All 5 audio presets generated successfully!")
