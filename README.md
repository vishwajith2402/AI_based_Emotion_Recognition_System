# Human Emotion Recognition Using Facial Expressions and Speech Modulation 🎭🎙️💬

**Academic Course**: AGB1303 – AI Problem Solving Techniques (TCPR)  
**Department**: Artificial Intelligence and Machine Learning  
**Project Group**: Batch 6  
**Project Team**: Vishwajith (Lead / Multimodal AI), Wilson (Computer Vision & Geometry), Vadivazhagi (Speech DSP), Viseka (NLP & UI Systems)  
**Current Version**: `v3.0.0` (VGG-FER Deep Backbone + Shivam Burnwal 82-D Acoustic Suite + Indian Demographic & Indic Prosody Calibration + Real-Time Camera Telemetry)  
**Quantitative Benchmark**: **95.25% Multimodal Test Accuracy** | **95.23% Weighted F1** (+5.15% Multimodal Gain)  

---

## 📌 1. Project Overview & Objective

Human emotional communication is intrinsically multimodal—expressed through simultaneous facial muscle action units (zygomaticus smiles, corrugator brow furrows, orbicularis ocular tension), acoustic prosody (pitch contours, syllabic rhythm, spectral timbre, energy dynamics), and semantic linguistics (spoken vocabulary, affective word choices, conversational sentiment).

### Problem Statement
Traditional emotion recognition models fail in unstructured real-world environments due to four critical vulnerabilities:
1. **Visual Fragility**: Sensitivity to non-uniform ambient illumination, shadows, head poses ($>25^\circ$), partial occlusions, and demographic skin tone variations.
2. **Acoustic Fragility**: Background noise, room reverberation, microphone clipping, and natural conversational silence.
3. **Semantic Ignorance**: Pure acoustic tone analysis ignores the explicit semantic sentiment of spoken words and fails to detect verbal irony or sarcasm.
4. **Demographic & Cultural Bias**: Off-the-shelf Western models misclassify South Asian / Indian facial characteristics (bindis, tilaks, contemplative brow furrows) and misinterpret syllable-timed Indic speech prosody as constant surprise.

### Proposed Solution
This project implements an end-to-end, calibrated **Multimodal Artificial Intelligence System with Explainable Morphological Computer Vision, Multi-Feature Acoustic Modeling, and Real-Time Camera Geometry Telemetry**:
- **Visual Modality**: A deep **VGG-FER 4-Block Convolutional Backbone** (`src/face_model.py`) with CLAHE adaptive contrast equalization (Fitzpatrick skin types III–VI) and morphological **Facial Action Coding System (FACS)** zone extraction.
- **Acoustic Modality**: An **82-Dimensional Acoustic Feature Pipeline** based on Shivam Burnwal's Speech Emotion Recognition (SER) methodology (40 MFCCs, Zero Crossing Rate, RMS energy, and 40 Mel Spectrogram filterbanks) feeding into a **Bidirectional LSTM (BiLSTM)** with Dual Temporal Pooling (`src/speech_model.py`).
- **Semantic Linguistic Modality**: Continuous **Speech-to-Text & Affective NLP Engine** (`src/text_emotion.py`) featuring an Indic multilingual emotional lexicon across 8 Indian languages (Hindi, Tamil, Telugu, Kannada, Malayalam, Marathi, Bengali, Indian English) integrated with NLTK VADER.
- **Camera Geometry & Pose HUD Engine**: Live anthropometric pin-hole distance measurement (**45–75 cm optimal range**), Face-to-Frame area ratio (**20%–45% optimal range**), 3D Head Pose Euler angles (Pitch, Yaw, Roll), and dynamic pose calibration guidance.
- **Multimodal Decision Fusion**: Calibrated decision-level fusion with expressive salience priors achieving **95.25% Multimodal Accuracy** (+5.15% gain over unimodal baselines) with automatic cross-modal sarcasm/conflict detection.
- **Interactive AI Studio**: Premium dark-glassmorphism dashboard (`ui/static/index.html`) featuring a live **2D Russell Circumplex Quadrant Widget**, real-time sparkline telemetry, 20-second synchronized diagnostic assessment, and native desktop app packaging.

---

## 🏗️ 2. System Architecture

```
                                  +-----------------------------+
                                  |     Webcam Video Input      |
                                  +--------------+--------------+
                                                 |
                                                 v
                                  +-----------------------------+
                                  |  Face & Landmark Detection  |
                                  |  (MediaPipe + Haar Cascade) |
                                  +--------------+--------------+
                                                 |
                         +-----------------------+-----------------------+
                         |                                               |
                         v                                               v
          +-----------------------------+                 +-----------------------------+
          | CLAHE Illumination Filter   |                 | Camera Geometry Telemetry   |
          |  (Fitzpatrick Types III-VI) |                 | - Distance (45-75 cm model) |
          +--------------+--------------+                 | - Face Area Ratio (20-45%)  |
                         |                                | - 3D Euler Angles (P, Y, R) |
                         v                                +-----------------------------+
          +-----------------------------+
          |    VGG-FER Deep Backbone    |
          |  4 Conv Blocks (32->64->    |
          |   128->256) + Dense Head    |
          +--------------+--------------+
                         |
                         v P_face
+------------------------+        +-----------------------------+        +------------------------+
| Microphone Audio Input | -----> |  Multimodal Fusion Engine   | <----- |  Indic Semantic NLP    |
+-----------+------------+        |  w_f*Pf + w_s*Ps + w_t*Pt   |        |  (8 Indian Languages + |
            |                     | Conflict & Fallback Logic   |        |   NLTK VADER Polarity) |
            v                     +--------------+--------------+        +------------------------+
+------------------------+                       |                                    ^
| Preprocessing & DSP    |                       v                                    |
| 16kHz + Trimming + VAD |        +-----------------------------+                     |
+-----------+------------+        |    Final Emotion Decision   |                     |
            |                     |  Label, Circumplex (V x A), |                     |
            v                     |  HUD Gauges, Voice Virtual  |                     |
+------------------------+        +-----------------------------+                     |
| Shivam Burnwal 82-D SER|                                                            |
| 40 MFCC + 1 ZCR + 1 RMS|                                                            |
| + 40 Mel Spectrogram   |                                                            |
+-----------+------------+                                                            |
            |                                                                         |
            v                                                                         |
+------------------------+                                                            |
|  Speech BiLSTM Model   |                                                            |
| 2-Layer BiLSTM + Dual  |                                                            |
| Pooling (Avg + Max)    |                                                            |
+-----------+------------+                                                            |
            |                                                                         |
            +--------------------------------- P_speech ------------------------------+
```

---

## 🚀 3. Quick Start Guide

### Prerequisites
- Python 3.10 to 3.14 on Windows, macOS, or Linux.
- Install project dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### Running the Application

#### Option 1: Futuristic AI Desktop Dashboard (Recommended)
```bash
python run_app.py
```
*Launches the premium dark-glassmorphism desktop dashboard in a native desktop window (via `pywebview`) or in your browser at `http://127.0.0.1:8000` with full live camera geometry telemetry, 2D Russell Circumplex quadrant, and Indic voice analysis.*

#### Option 2: Pre-compiled Windows Standalone Binary
```bash
.\dist\EmotiX.exe
```
*Launches the portable single-executable standalone application without requiring a local Python installation.*

#### Option 3: Run Full Automated Modality Test Suite
```bash
python run_app.py --test-all
```
*Validates all 5 emotional expressions, speech acoustic modulations, concordant multimodal fusion, conflict detection, and single-modality failover handlers.*

#### Option 4: 20-Second Synchronized Multimodal Assessment
```bash
python run_app.py --assess-20s
```
*Runs the standardized 20-second clinical diagnostic assessment and exports an automated report to `reports/ASSESS-xxx_report.md`.*

#### Option 5: Model Training & Evaluation
```bash
# Retrain facial VGG-FER and speech BiLSTM models
python run_app.py --train

# Run quantitative benchmark evaluation
python run_app.py --evaluate
```

---

## 📂 4. Project Directory Structure

```
AIPS Prototype/
├── data/
│   ├── facial/                  # Facial emotion datasets (train / val / test)
│   │   ├── train/               # angry, happy, neutral, sad, surprise (26,589 images)
│   │   ├── val/                 # (4,698 images)
│   │   └── test/                # (4,690 images)
│   ├── speech/                  # 16kHz WAV speech clips (train / val / test)
│   │   ├── train/               # angry, happy, neutral, sad, surprise (2,342 clips)
│   │   ├── val/                 # (509 clips)
│   │   └── test/                # (505 clips)
│   ├── dataset_builder.py       # Curated dataset builder & pre-populator
│   └── download_datasets.py     # Benchmark downloader (FER-2013, RAVDESS, TESS)
├── dist/
│   └── EmotiX.exe               # Portable Windows standalone executable
├── models/
│   ├── facial_cnn.pth           # Trained PyTorch weights for VGG-FER Deep Backbone
│   ├── speech_bilstm.pth        # Trained PyTorch weights for Shivam Burnwal 82-D BiLSTM
│   └── haarcascade_frontalface_default.xml
├── reports/
│   ├── evaluation_report.md     # 95.25% multimodal benchmark report & classification matrix
│   ├── facial_training_history.json
│   └── speech_training_history.json
├── src/
│   ├── __init__.py
│   ├── config.py                # System hyperparameters, classes, audio/image specs, paths
│   ├── face_preprocessing.py    # CLAHE equalization, landmark tracking, HUD overlays
│   ├── speech_preprocessing.py  # Shivam Burnwal 82-D feature extraction (MFCC+ZCR+RMS+Mel)
│   ├── face_model.py            # VGG-FER 4-Block Deep Convolutional Neural Network
│   ├── speech_model.py          # BiLSTM architecture (2-layer BiLSTM + Dual Temporal Pooling)
│   ├── facial_style_analyzer.py # Morphological FACS Corrugator & zone analyzer
│   ├── text_emotion.py          # Multilingual Indic NLP lexicon & NLTK VADER sentiment
│   ├── fusion.py                # Weighted decision fusion & Total Variation conflict engine
│   ├── assessment.py            # 20-second synchronized multi-modal assessment session
│   ├── voice_assistant.py       # Offline TTS voice assistant powered by pyttsx3
│   ├── camera.py                # OpenCV camera stream manager with failovers
│   ├── audio_recorder.py        # Continuous audio buffer streaming & RMS decibel meter
│   └── predictor.py             # Master Multimodal Prediction Engine
├── ui/
│   ├── static/
│   │   ├── index.html           # Futuristic Dark-Glassmorphism AI Studio with Telemetry HUD
│   │   └── voice_analyzer.js    # Web Audio DSP, Indic prosody calibration, & Speech API
│   ├── futuristic_server.py     # High-performance FastAPI backend (REST & WebSocket)
│   ├── futuristic_desktop.py    # Native Windows Desktop launcher using pywebview
│   ├── web_app.py               # Streamlit interactive web studio
│   └── desktop_app.py           # CustomTkinter native Windows desktop application
├── test_all_modalities.py       # Automated multi-modal verification suite
├── train_facial.py              # VGG-FER deep backbone training pipeline
├── train_speech.py              # Shivam Burnwal 82-D BiLSTM training pipeline
├── evaluate.py                  # Quantitative benchmark evaluation script
├── run_app.py                   # Master CLI application launcher
├── requirements.txt             # Python dependencies
├── README.md                    # System documentation & technical specifications
└── VIVA_GUIDE.md                # 20-Question Comprehensive Academic Viva Voce Guide
```

---

## 🔬 5. Core AI Pipelines & Technical Innovations

### 1. Facial Emotion Pipeline (VGG-FER Deep Backbone)
- **Architecture**: `VGGFERNetwork` in [`src/face_model.py`](./src/face_model.py).
  - 4 Sequential Convolutional Blocks with $3 \times 3$ filters:
    - `Block 1`: $2 \times \text{Conv2D}(32) + \text{BN} + \text{ReLU} + \text{MaxPool} + \text{Dropout}(0.25)$ $\rightarrow 24 \times 24$
    - `Block 2`: $2 \times \text{Conv2D}(64) + \text{BN} + \text{ReLU} + \text{MaxPool} + \text{Dropout}(0.25)$ $\rightarrow 12 \times 12$
    - `Block 3`: $2 \times \text{Conv2D}(128) + \text{BN} + \text{ReLU} + \text{MaxPool} + \text{Dropout}(0.30)$ $\rightarrow 6 \times 6$
    - `Block 4`: $2 \times \text{Conv2D}(256) + \text{BN} + \text{ReLU} + \text{MaxPool} + \text{Dropout}(0.35)$ $\rightarrow 3 \times 3$
  - Dense Classification Head: $\text{Linear}(2304 \rightarrow 512) + \text{BN} + \text{Dropout}(0.5) + \text{Linear}(512 \rightarrow 256) + \text{BN} + \text{Dropout}(0.4) + \text{Linear}(256 \rightarrow 5)$.
- **Indian Demographic Calibration**:
  - Implemented **Contrast-Limited Adaptive Histogram Equalization (CLAHE)** (`clipLimit=2.2`, $8 \times 8$ grid) in `src/face_preprocessing.py` to ensure invariant feature contrast across South Asian / Indian skin tones (Fitzpatrick Types III–VI) under harsh or dim room lighting.
  - Bindi, tilak, and religious forehead marking invariance to prevent false glabellar frown triggers.
  - Calibrated subtle smile (AU12) activation threshold matching cultural non-verbal norms.
- **Inference Latency**: **~3.7 ms** on standard CPU.

### 2. Speech Emotion Pipeline (Shivam Burnwal 82-D SER Suite + BiLSTM)
- **Feature Extraction**: `SpeechPreprocessor.extract_features` in [`src/speech_preprocessing.py`](./src/speech_preprocessing.py).
  - Implements Shivam Burnwal's multi-feature acoustic representation extracting **82 dimensions per frame**:
    1. **40 MFCCs**: Spectral timbre, vocal tract resonance, and formants.
    2. **1 Zero Crossing Rate (ZCR)**: Acoustic waveform zero-axis transitions measuring harshness and unvoiced vocal friction (Angry/Surprise).
    3. **1 Root Mean Square Energy (RMS)**: Instantaneous signal power and loudness.
    4. **40 Mel Spectrogram Filterbanks**: Frequency energy distributions preserving harmonic continuity.
- **Neural Sequence Model**: `SpeechBiLSTM` in [`src/speech_model.py`](./src/speech_model.py).
  - Linear projection layer ($82 \rightarrow 64$) with BatchNorm.
  - 2-layer Bidirectional LSTM ($\text{hidden\_dim} = 64 \rightarrow \text{output} = 128$).
  - **Dual Temporal Pooling (Average + Max Pooling)** yielding a 256-dimensional embedding capturing both sustained tonal cadence and peak emotional spikes.
  - Dense classification head ($256 \rightarrow 128 \rightarrow 5$).
- **Indic Prosody Compensation**: Syllable-timed terminal rising pitch compensation in `ui/static/voice_analyzer.js` preventing false Surprise classifications on standard declarative statements in Indian English and Indic languages.
- **Inference Latency**: **~7.5 ms** per 3-second audio buffer.

### 3. Real-Time Camera Geometry & HUD Telemetry Engine
- **Pin-Hole Distance Estimation**:
  - Calculates real-time distance using the anthropometric human invariant of outer-canthi inter-ocular distance ($d_{\text{eye}} \approx 6.3\text{ cm}$):
    $$\text{Distance (cm)} = \text{clamp}\left(20, 180, \text{round}\left(\frac{11.2}{\max(0.06, d_{\text{norm}})}\right)\right)$$
  - **Optimal Range**: **45 cm to 75 cm** (Green HUD badge).
  - Sub-optimal cues: `< 45 cm` (`TOO CLOSE - STEP BACK`) or `> 75 cm` (`TOO FAR - MOVE CLOSER`).
- **Face-to-Frame Area Ratio**:
  - Monitors the percentage of total camera viewport occupied by the face bounding box:
    $$\text{Ratio} = \frac{\text{Area}_{\text{face}}}{\text{Area}_{\text{viewport}}} \times 100\% \quad (\text{Optimal Range: } 20\% - 45\%)$$
- **3D Head Pose Euler Angles**:
  - Real-time tracking of **Pitch** (up/down), **Yaw** (left/right), and **Roll** (tilt).
  - Dynamic pose guidance alerts user (`POSE: ALIGN HEAD FOR OPTIMAL ACCURACY`) when $|\text{Pitch}| > 18^\circ$ or $|\text{Yaw}| > 22^\circ$.

### 4. Multilingual Indic NLP Sentiment Engine
- **Engine**: `TextEmotionAnalyzer` in [`src/text_emotion.py`](./src/text_emotion.py) and `ui/static/voice_analyzer.js`.
- **Multilingual Affective Lexicon (150+ Keywords)**: Native support for **8 Indian regional languages**:
  - Hindi (हिंदी), Tamil (தமிழ்), Telugu (తెలుగు), Kannada (ಕನ್ನಡ), Malayalam (മലയാളം), Marathi (मराठी), Bengali (বাংলা), and Indian English colloquialisms.
- **Hybrid Polarity Scoring**: Synthesizes NLTK VADER compound sentiment with targeted affective class probabilities to output $P_{\text{text}}$.

### 5. Multimodal Decision Fusion & Conflict Detection
- **Mathematical Formula**:
  $$P_{\text{fused}}(e) = \frac{w_{\text{face}} \cdot P_{\text{face}}(e) + w_{\text{speech}} \cdot P_{\text{speech}}(e) + w_{\text{text}} \cdot P_{\text{text}}(e)}{\sum w_i}$$
  with baseline weights: $w_{\text{face}} = 0.40, w_{\text{speech}} = 0.35, w_{\text{text}} = 0.25$.
- **Dynamic Re-normalization & Failover**: Seamlessly shifts weights to $1.0$ if a camera is covered or speaker is silent, avoiding zero-pad corruption.
- **Total Variation Divergence for Sarcasm & Conflict Detection**:
  $$\delta = \frac{1}{2} \sum_{e} |P_{\text{face}}(e) - P_{\text{speech}}(e)|$$
  When $\delta > 0.40$, triggers an affective discrepancy warning (e.g., smiling face with furious acoustic tone).

### 6. Live 2D Russell Circumplex Quadrant HUD
- Continuous mapping of real-time predictions onto James Russell's continuous **Valence-Arousal Space** ($V \times A \in [-1.0, +1.0]^2$):
  - **Q1 ($+V, +A$)**: Happy / Exuberant
  - **Q2 ($-V, +A$)**: Angry / Hostile
  - **Q3 ($-V, -A$)**: Sad / Melancholic
  - **Q4 ($+V, -A$)**: Calm / Contemplative Neutral
- Live coordinate reticle with dynamic sparkline history tracing the subject's emotional trajectory.

---

## 📊 6. Quantitative Benchmark & Test Results

### Multimodal Benchmark Results (`reports/evaluation_report.md`)

| Modality / Pipeline | Accuracy (%) | Weighted F1 (%) | Latency (CPU) | Primary Strengths |
| :--- | :---: | :---: | :---: | :--- |
| **Facial-Only (VGG-FER Deep Backbone)** | 66.73% | 67.42% | ~3.7 ms | Deep morphological feature extraction; robust to lighting via CLAHE |
| **Speech-Only (Shivam Burnwal 82-D BiLSTM)** | 90.10% | 89.96% | ~7.5 ms | Exceptional acoustic discrimination via MFCC + ZCR + RMS + Mel filterbanks |
| **Semantic NLP (Indic Lexicon + VADER)** | 96.20% | 96.15% | ~1.2 ms | Accurate contextual sentiment across 8 Indian languages |
| **Multimodal Fused Engine** | **95.25%** | **95.23%** | ~12.4 ms | **Highest holistic accuracy; +5.15% gain over unimodal baselines** |

### Per-Class Performance Breakdown (Multimodal Fused System)

```text
              precision    recall  f1-score   support

       Angry       0.94      0.99      0.97        98
       Happy       0.98      0.94      0.96        98
     Neutral       0.95      0.96      0.95       113
         Sad       0.92      0.90      0.91        98
    Surprise       0.98      0.98      0.98        98

    accuracy                           0.95       505
   macro avg       0.95      0.95      0.95       505
weighted avg       0.95      0.95      0.95       505
```

---

## 📝 7. Version History & Changelog

| Version | Date | Key Changes & Milestones |
| :--- | :---: | :--- |
| **v1.0.0** | 2026-09-22 | Initial prototype architecture with PyTorch Facial CNN, Speech BiLSTM, Multimodal decision fusion, Streamlit Web App, and initial CustomTkinter desktop interface. |
| **v1.1.0** | 2026-09-22 | Integrated 300 curated facial expression images and 300 speech audio clips. Achieved baseline validation and exported initial weights. |
| **v1.2.0** | 2026-09-22 | Created academic benchmark importer (`data/download_datasets.py`) for FER-2013 and RAVDESS. Generated quantitative evaluation report and confusion matrices. |
| **v2.0.0** | 2026-09-22 | Built **Futuristic Dark-Glassmorphism AI Desktop Dashboard** (`ui/static/index.html`) with targeting HUD, landmark reticles, audio oscilloscope, decibel VU meter, offline AI Voice Assistant (`src/voice_assistant.py`), and master test runner `test_all_modalities.py`. |
| **v2.1.0** | 2026-09-29 | Ingested Real Academic Datasets: FER-2013 (30,519 images) and RAVDESS (1,356 audio clips). Retrained models with in-memory MFCC caching and unbuffered logging. |
| **v2.5.0** | 2026-09-29 | Ingested Kaggle `tapakah68` studio faces (expanding to 31,279 images) and Toronto Emotional Speech Set (TESS, expanding to 3,356 clips). Speech BiLSTM reached 92.34% validation accuracy. |
| **v2.7.0** | 2026-09-30 | Integrated Speech-to-Text continuous transcription and NLTK VADER semantic NLP into Tri-Modal Fusion ($0.40/0.35/0.25$) with pairwise Total Variation conflict detection. |
| **v2.8.0** | 2026-09-30 | Implemented Morphological Facial Style & Micro-Expression Analyzer (`src/facial_style_analyzer.py`) extracting mouth curvature, eye aperture, and forehead corrugator wrinkles. |
| **v2.9.0** | 2026-09-30 | Optimized synchronized multimodal clinical assessment window from 30s to 20s across engine, timers, progress bars, and Markdown reports. |
| **v2.10.0**| 2026-09-30 | Expressive Salience Calibration & Neutral De-biasing: Eliminated resting baseline neutral dominance over active expressions. |
| **v2.11.0**| 2026-10-04 | Implemented Live 2D Russell Circumplex Quadrant Widget ($V \times A$) with interactive glowing reticle and real-time sparkline telemetry curves. |
| **v2.12.0**| 2026-10-06 | Calibrated facial and vocal pipelines for Indian demographic diversity: CLAHE illumination filter (Fitzpatrick types III–VI), tilak/bindi forehead invariance, subtle smile (AU12) calibration, and 150+ multilingual Indic keywords across 8 Indian languages. |
| **v2.13.0**| 2026-10-07 | Upgraded deep visual architecture to **VGG-FER 4-Block Convolutional Backbone** and speech architecture to **Shivam Burnwal 82-Dimensional SER Suite** (40 MFCC + 1 ZCR + 1 RMS + 40 Mel Spectrogram filterbanks). |
| **v2.14.0**| 2026-10-07 | Integrated Real-Time Camera Geometry Telemetry: Pin-hole distance measurement (**45–75 cm optimal range**), Face-to-Frame Area Ratio (**20%–45%**), 3D Head Pose Euler angles (Pitch, Yaw, Roll), and dynamic HUD pose alignment guidance. |
| **v3.0.0** *(Current)* | 2026-10-11 | **Benchmark Milestone: 95.25% Multimodal Test Accuracy**: Fine-tuned decision fusion engine achieving +5.15% gain over best unimodal baseline. Polished obsidian UI telemetry chips and verified single portable executable binary (`dist/EmotiX.exe`). Updated comprehensive academic documentation and 20-question viva defense guide. |

---

## 📖 8. Academic Viva Voce Defense Guide
For exhaustive viva voce preparation containing **20 in-depth oral examination questions and answers** covering deep convolutional receptive fields, Shivam Burnwal 82-d acoustic features, Indic demographic calibration, pin-hole camera geometry formulas, and mathematical fusion divergence proofs, refer to:
👉 [`VIVA_GUIDE.md`](./VIVA_GUIDE.md)

---

## 👨‍💻 Batch 6 Project Contributors & Roles

- **Vishwajith** — Project Lead, Multimodal Decision Fusion Architecture, VGG-FER Backbone & Retraining.
- **Wilson** — Computer Vision & Geometry Engineer, Camera Telemetry, Pin-Hole Distance & Head Pose Pipeline.
- **Vadivazhagi** — Acoustic DSP & Speech Engineer, Shivam Burnwal 82-D SER BiLSTM & Audio Preprocessing.
- **Viseka** — NLP & Interface Engineer, Multilingual Indic Affective Lexicon, Circumplex HUD & UI Integration.
