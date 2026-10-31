# 🎭 EmotiSync Presentation Test Media Pack
**AGB1303 - AI Problem Solving Techniques (TCPR) | Batch 6**

This directory contains pre-verified, high-confidence media files specifically curated for live demonstration in the **Media Upload Studio Mode** (Real Media Mode) of EmotiSync.

---

## 📁 Directory Structure & Verified Files

### 1. `1_FACIAL_IMAGES/` (Verified Visual Expressions)
Use these in the **Step 1: Visual Modality (Video / Image)** upload zone.
- `1_Happy_Smile.png` — **96.5% Happy** (Clear Duchenne zygomatic corner elevation & cheek squinch)
- `2_Sad_Grief.png` — **81.6% Sad** (Depressed lip commissures AU15 & grief slant AU1)
- `3_Angry_Furious.png` — **94.4% Angry** (Intense corrugator supercilii glabella furrow AU4)
- `4_Surprise_Astonished.png` — **98.2% Surprise** (Marked palpebral fissure expansion & open jaw AU26)
- `5_Neutral_Calm.png` — **93.3% Neutral** (Balanced resting horizontal mouth & smooth brow)

### 2. `2_SPEECH_AUDIO/` (Natural Spoken Utterances - 8-9 seconds)
Use these in the **Step 2: Vocal Modality (Speech Audio)** upload zone.
- `1_Happy_Speech.wav` — **95.7% Happy** (*"Last Monday was perfect! I got the job I had been dreaming of..."*)
- `2_Sad_Speech.wav` — **88.2% Sad** (*"Two days later, my mom called... our dog Bruno had passed away..."*)
- `3_Angry_Speech.wav` — **97.8% Angry** (*"Earlier that day, my Wi-Fi cut out again during an important meeting..."*)
- `4_Surprise_Speech.wav` — **93.9% Surprise** (*"And just when I thought the week couldn't get any crazier, I got a surprise call!..."*)
- `5_Neutral_Speech.wav` — **96.9% Neutral** (*"What a week. Honestly, I'm still processing it all, trying to reflect calmly..."*)

### 3. `3_BENCHMARK_VOCAL_UTTERANCES/` (TESS & RAVDESS Reference Audio)
- `1_TESS_Angry_Vocal.wav` — **99.9% Angry**
- `2_RAVDESS_Happy_Vocal.wav` — **99.8% Happy**
- `3_TESS_Sad_Vocal.wav` — **99.5% Sad**
- `4_RAVDESS_Surprise_Vocal.wav` — **99.1% Surprise**
- `5_RAVDESS_Neutral_Vocal.wav` — **92.1% Neutral**

### 4. `4_PRESENTATION_DEMO_SCENARIOS/` (Pre-Paired Multi-Modal Demos)
Each folder has `face_input.png` and `voice_input.wav` ready to demonstrate:
- **`Scenario_A_Concordant_Happy/`**: Demonstrates 100% harmonious Happy fusion (>94% certainty).
- **`Scenario_B_Concordant_Sad/`**: Demonstrates subtle sorrow & melancholic prosody fusion.
- **`Scenario_C_Concordant_Angry/`**: Demonstrates high-arousal agitated fusion.
- **`Scenario_D_Concordant_Surprise/`**: Demonstrates sudden awe / startle fusion.
- **`Scenario_E_Sarcasm_HappyFace_AngryVoice/`**: ⚡ **Key Demo!** Upload `face_input.png` (Smiling Face) and `voice_input.wav` (Angry Voice). EmotiSync triggers a real-time **Cross-Modal Conflict Alert (Verbal Sarcasm / Hostility)**!
- **`Scenario_F_Emotional_Masking_HappyFace_SadVoice/`**: ⚡ **Key Demo!** Upload `face_input.png` (Smiling Face) and `voice_input.wav` (Sad Weeping Voice). EmotiSync detects **Emotional Masking / Suppressed Grief**!

---

## 🚀 How to Run During Presentation
1. Open EmotiSync and switch to **MEDIA UPLOAD STUDIO** tab.
2. In **Step 1 (Visual Modality)**, click "Browse" or drag & drop any image from `1_FACIAL_IMAGES/`.
3. In **Step 2 (Vocal Modality)**, click "Browse" or drag & drop any audio from `2_SPEECH_AUDIO/`.
4. Click **"⚡ RUN MODEL (MULTIMODAL ANALYSIS)"**.
5. The pop-up result window will immediately display the high-confidence facial analysis, vocal analysis, Action Units breakdown, and final fused verdict!
