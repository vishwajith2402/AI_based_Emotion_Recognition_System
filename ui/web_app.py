"""
Streamlit Multimodal Emotion Recognition Dashboard
AGB1303 - AI Problem Solving Techniques
Batch 6: Human Emotion Recognition Using Facial Expressions and Speech Modulation
"""

import sys
from pathlib import Path

# Setup project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import io
import time
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.config import (
    EMOTION_CLASSES, EMOTION_COLORS, EMOTION_ICONS,
    FACE_MODEL_PATH, SPEECH_MODEL_PATH, FACIAL_TEST_DIR, SPEECH_TEST_DIR
)
from src.face_preprocessing import FacePreprocessor
from src.speech_preprocessing import SpeechPreprocessor
from src.face_model import FacialEmotionModel
from src.speech_model import SpeechEmotionModel
from src.fusion import MultimodalFusion
from src.audio_recorder import AudioRecorder


# Page Configuration
st.set_page_config(
    page_title="Multimodal Emotion Recognition",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4F46E5, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #9CA3AF;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #1F2937;
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid #374151;
        margin-bottom: 1rem;
    }
    .badge-fused {
        display: inline-block;
        padding: 0.5rem 1.2rem;
        font-size: 1.6rem;
        font-weight: 700;
        border-radius: 8px;
        color: white;
    }
    .conflict-box {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid #EF4444;
        border-radius: 8px;
        padding: 0.8rem;
        color: #FCA5A5;
        font-weight: 500;
        margin-top: 0.8rem;
    }
    .concordant-box {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid #10B981;
        border-radius: 8px;
        padding: 0.8rem;
        color: #6EE7B7;
        font-weight: 500;
        margin-top: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Caches loaded neural network pipelines."""
    face_model = FacialEmotionModel(str(FACE_MODEL_PATH))
    speech_model = SpeechEmotionModel(str(SPEECH_MODEL_PATH))
    face_prep = FacePreprocessor()
    speech_prep = SpeechPreprocessor()
    fusion_engine = MultimodalFusion()
    return face_model, speech_model, face_prep, speech_prep, fusion_engine


face_model, speech_model, face_prep, speech_prep, fusion_engine = load_models()

# Sidebar
st.sidebar.title("🎛️ Control Panel")
st.sidebar.markdown("**Course**: AGB1303 – AI Problem Solving")
st.sidebar.markdown("**Project**: Human Emotion Recognition")
st.sidebar.markdown("---")

st.sidebar.subheader("⚖️ Modality Weights")
w_face = st.sidebar.slider("Facial Weight (w_face)", 0.0, 1.0, 0.5, 0.05)
w_speech = 1.0 - w_face
st.sidebar.markdown(f"**Speech Weight (w_speech)**: `{w_speech:.2f}`")
fusion_engine.set_weights(w_face, w_speech)

st.sidebar.markdown("---")
st.sidebar.subheader("Hardware & Model Status")
face_status = "✅ Loaded" if face_model.is_loaded else "⚠️ Default"
speech_status = "✅ Loaded" if speech_model.is_loaded else "⚠️ Default"
st.sidebar.markdown(f"- Facial CNN: **{face_status}**")
st.sidebar.markdown(f"- Speech BiLSTM: **{speech_status}**")


# Header
st.markdown('<div class="main-header">Human Emotion Recognition System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multimodal AI Architecture combining Facial Expressions (CNN) & Speech Modulation (MFCC + BiLSTM)</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Multimodal Live Inference",
    "📊 Modality Visualizer & Analytics",
    "⚙️ Model Training & Evaluation",
    "📖 Viva & System Architecture Guide"
])


# ==========================================
# TAB 1: LIVE INFERENCE
# ==========================================
with tab1:
    col_left, col_right = st.columns(2)

    # ----------------- FACIAL BRANCH -----------------
    with col_left:
        st.subheader("👁️ 1. Facial Expression Branch")
        face_input_type = st.radio("Select Facial Input Source:", ["Upload Face Photo", "Take Camera Photo", "Use Sample Dataset Face"], horizontal=True)

        face_image_bgr = None
        if face_input_type == "Upload Face Photo":
            uploaded_file = st.file_uploader("Upload Image (JPG, PNG)", type=["jpg", "jpeg", "png"], key="face_uploader")
            if uploaded_file is not None:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                face_image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        elif face_input_type == "Take Camera Photo":
            camera_img = st.camera_input("Capture Camera Frame", key="face_cam")
            if camera_img is not None:
                file_bytes = np.asarray(bytearray(camera_img.read()), dtype=np.uint8)
                face_image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        else:
            selected_emotion = st.selectbox("Pick Emotion Sample:", EMOTION_CLASSES, index=1, key="face_sample_emotion")
            sample_folder = FACIAL_TEST_DIR / selected_emotion.lower()
            sample_files = list(sample_folder.glob("*.png"))
            demo_fallback = PROJECT_ROOT / "demo_samples" / "facial" / f"{selected_emotion.lower()}.png"
            if sample_files:
                face_image_bgr = cv2.imread(str(sample_files[0]))
            elif demo_fallback.exists():
                face_image_bgr = cv2.imread(str(demo_fallback))
            else:
                st.warning(f"No sample image found for '{selected_emotion}'. Please upload an image.")

        # Process Facial Image
        face_probs = None
        face_top_emotion = None
        face_confidence = 0.0

        if face_image_bgr is not None:
            display_img = face_image_bgr.copy()
            faces = face_prep.detect_faces(display_img)

            if len(faces) > 0:
                x, y, w, h = faces[0]
                face_roi = display_img[y:y+h, x:x+w]
                normalized = face_prep.preprocess_face(face_roi)
            else:
                normalized = face_prep.preprocess_face(display_img)

            f_tensor = face_prep.to_tensor(normalized)
            face_top_emotion, face_confidence, face_probs = face_model.predict(f_tensor)

            # Draw detection
            if len(faces) > 0:
                FacePreprocessor.draw_prediction(display_img, faces[0], face_top_emotion, face_confidence)

            st.image(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB), caption=f"Facial Prediction: {face_top_emotion} ({face_confidence*100:.1f}%)", use_container_width=True)

            # Facial Probabilities Bar Chart
            df_face = pd.DataFrame({
                "Emotion": EMOTION_CLASSES,
                "Probability": [face_probs[c] * 100 for c in EMOTION_CLASSES]
            })
            st.bar_chart(df_face.set_index("Emotion"), height=180)
        else:
            st.info("Provide a facial image to activate the visual modality.")

    # ----------------- SPEECH BRANCH -----------------
    with col_right:
        st.subheader("🎙️ 2. Speech Modulation Branch")
        speech_input_type = st.radio("Select Audio Input Source:", ["Upload Audio File", "Synthesize Voice Inflection", "Use Sample Dataset Audio"], horizontal=True)

        audio_data = None
        if speech_input_type == "Upload Audio File":
            uploaded_audio = st.file_uploader("Upload Audio (WAV, MP3)", type=["wav", "mp3"], key="audio_uploader")
            if uploaded_audio is not None:
                audio_data = speech_prep.load_audio_file(uploaded_audio)
                st.audio(uploaded_audio)

        elif speech_input_type == "Synthesize Voice Inflection":
            synth_emotion = st.selectbox("Acoustic Emotion Profile:", EMOTION_CLASSES, index=1, key="synth_speech_emotion")
            if st.button("Generate & Test Voice Modulations", key="synth_btn"):
                audio_data = AudioRecorder.synthesize_mock_speech(synth_emotion, duration=3.0)
                # Convert for st.audio
                import soundfile as sf
                wav_io = io.BytesIO()
                sf.write(wav_io, audio_data, 16000, format='WAV', subtype='PCM_16')
                st.audio(wav_io.getvalue(), format="audio/wav")

        else:
            selected_audio_emotion = st.selectbox("Pick Dataset Audio Emotion:", EMOTION_CLASSES, index=1, key="speech_sample_emotion")
            audio_folder = SPEECH_TEST_DIR / selected_audio_emotion.lower()
            audio_files = list(audio_folder.glob("*.wav"))
            demo_audio_fallback = PROJECT_ROOT / "demo_samples" / "speech" / f"{selected_audio_emotion.lower()}.wav"
            if audio_files:
                audio_data = speech_prep.load_audio_file(str(audio_files[0]))
                with open(audio_files[0], "rb") as af:
                    st.audio(af.read(), format="audio/wav")
            elif demo_audio_fallback.exists():
                audio_data = speech_prep.load_audio_file(str(demo_audio_fallback))
                with open(demo_audio_fallback, "rb") as af:
                    st.audio(af.read(), format="audio/wav")
            else:
                st.warning(f"No sample audio found for '{selected_audio_emotion}'. Please upload or synthesize audio.")

        # Process Speech Audio
        speech_probs = None
        speech_top_emotion = None
        speech_confidence = 0.0

        if audio_data is not None:
            s_tensor, is_silent = speech_prep.process_audio(audio_data)
            speech_top_emotion, speech_confidence, speech_probs = speech_model.predict(s_tensor)

            # Waveform Plot
            fig, ax = plt.subplots(figsize=(6, 1.8))
            ax.plot(audio_data[::10], color="#06B6D4", linewidth=0.8)
            ax.set_title(f"Speech Audio Waveform (MFCC Analyzed) | Silence: {is_silent}", fontsize=9, color="#E5E7EB")
            ax.axis('off')
            fig.patch.set_facecolor('#111827')
            st.pyplot(fig)

            # Speech Probabilities Bar Chart
            df_speech = pd.DataFrame({
                "Emotion": EMOTION_CLASSES,
                "Probability": [speech_probs[c] * 100 for c in EMOTION_CLASSES]
            })
            st.bar_chart(df_speech.set_index("Emotion"), height=180)
        else:
            st.info("Provide an audio input to activate the acoustic modality.")

    # ----------------- MULTIMODAL FUSION CARD -----------------
    st.markdown("---")
    st.subheader("⚡ 3. Multimodal Feature Fusion Result")

    fusion_res = fusion_engine.fuse(
        face_probs=face_probs,
        speech_probs=speech_probs,
        face_detected=(face_probs is not None),
        speech_detected=(speech_probs is not None)
    )

    final_emotion = fusion_res["final_emotion"]
    confidence = fusion_res["confidence"]
    status = fusion_res["status"]
    is_conflict = fusion_res["conflict_detected"]
    color = EMOTION_COLORS.get(final_emotion, "#10B981")
    icon = EMOTION_ICONS.get(final_emotion, "🎭")

    m_col1, m_col2, m_col3 = st.columns([1.5, 1, 1])

    with m_col1:
        st.markdown(f"""
        <div style="background: #111827; border: 2px solid {color}; border-radius: 12px; padding: 1.5rem; text-align: center;">
            <div style="font-size: 1rem; color: #9CA3AF; text-transform: uppercase; letter-spacing: 1px;">Combined Multimodal Prediction</div>
            <div style="font-size: 2.8rem; font-weight: 800; color: {color}; margin: 0.5rem 0;">
                {icon} {final_emotion}
            </div>
            <div style="font-size: 1.2rem; color: #F3F4F6;">
                Confidence: <b>{confidence * 100:.1f}%</b>
            </div>
            <div style="font-size: 0.85rem; color: #6B7280; margin-top: 0.5rem;">
                Weights: Visual ({w_face*100:.0f}%) | Acoustic ({w_speech*100:.0f}%)
            </div>
        </div>
        """, unsafe_allow_html=True)

    with m_col2:
        st.markdown(f"**Modality Status**: `{status}`")
        if is_conflict:
            st.markdown(f"""
            <div class="conflict-box">
                ⚠️ <b>Modality Conflict Alert!</b><br>
                {fusion_res["conflict_reason"]}<br>
                <small>Divergence Index: {fusion_res["conflict_score"]:.2f}</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="concordant-box">
                ✅ <b>Modality Alignment Strong</b><br>
                Facial cues and vocal prosody are in harmonic agreement.
            </div>
            """, unsafe_allow_html=True)

    with m_col3:
        st.markdown("**Comparison Breakdown**:")
        st.markdown(f"- Facial Top: **{face_top_emotion or 'N/A'}** ({face_confidence*100:.1f}%)")
        st.markdown(f"- Speech Top: **{speech_top_emotion or 'N/A'}** ({speech_confidence*100:.1f}%)")
        st.markdown(f"- Fused Top: **{final_emotion}** ({confidence*100:.1f}%)")


# ==========================================
# TAB 2: ANALYTICS & RADAR
# ==========================================
with tab2:
    st.subheader("📊 Comparative Modality Distribution")
    
    # Combined Multi-bar DataFrame
    df_combined = pd.DataFrame({
        "Emotion": EMOTION_CLASSES,
        "Facial (CNN) %": [face_probs[c] * 100 if face_probs else 0 for c in EMOTION_CLASSES],
        "Speech (BiLSTM) %": [speech_probs[c] * 100 if speech_probs else 0 for c in EMOTION_CLASSES],
        "Fused %": [fusion_res["fused_probabilities"][c] * 100 for c in EMOTION_CLASSES]
    })
    
    st.dataframe(df_combined, use_container_width=True)
    st.bar_chart(df_combined.set_index("Emotion"))


# ==========================================
# TAB 3: TRAINING & EVALUATION STUDIO
# ==========================================
with tab3:
    st.subheader("⚙️ Neural Network Training & Evaluation Studio")
    
    t_col1, t_col2 = st.columns(2)
    with t_col1:
        st.markdown("### Retrain Facial CNN")
        st.caption("Trains 3-block Deep CNN on 48x48 normalized facial expressions.")
        epochs_face = st.slider("Facial Epochs", 2, 20, 5, key="ep_face")
        if st.button("🚀 Train Facial CNN", key="btn_train_face"):
            with st.spinner("Training Facial CNN..."):
                from train_facial import train_facial_model
                train_facial_model(epochs=epochs_face)
                st.success("Facial CNN retraining complete! Weights reloaded.")

    with t_col2:
        st.markdown("### Retrain Speech BiLSTM")
        st.caption("Trains 2-layer Bidirectional LSTM on 40-coefficient MFCC time-series.")
        epochs_speech = st.slider("Speech Epochs", 2, 20, 5, key="ep_speech")
        if st.button("🚀 Train Speech BiLSTM", key="btn_train_speech"):
            with st.spinner("Training Speech BiLSTM..."):
                from train_speech import train_speech_model
                train_speech_model(epochs=epochs_speech)
                st.success("Speech BiLSTM retraining complete! Weights reloaded.")

    st.markdown("---")
    st.markdown("### 🏆 Run Benchmark Evaluation")
    if st.button("Run Comprehensive Unimodal vs Multimodal Evaluation"):
        with st.spinner("Evaluating models on test splits..."):
            from evaluate import evaluate_all
            evaluate_all()
            report_file = PROJECT_ROOT / "reports" / "evaluation_report.md"
            if report_file.exists():
                st.markdown(report_file.read_text(encoding="utf-8"))


# ==========================================
# TAB 4: VIVA & ARCHITECTURE GUIDE
# ==========================================
with tab4:
    st.subheader("📖 Project Overview & Viva Defense Guide")
    st.markdown("""
    ### 🎯 Project Title
    **Human Emotion Recognition Using Facial Expressions and Speech Modulation**  
    *Course: AGB1303 – AI Problem Solving Techniques (Department of AI & ML)*

    ---

    ### 🏗️ Complete System Workflow
    1. **Data Acquisition**: Captures visual frames via OpenCV VideoCapture and audio signals via SoundDevice.
    2. **Facial Processing (CNN)**:
       - Haar Cascade / DNN face localization.
       - Cropping and histogram equalization to 48×48 normalized grayscale.
       - 3 Convolutional blocks with Batch Normalization and Dropout for spatial feature extraction.
    3. **Speech Processing (MFCC + BiLSTM)**:
       - 16,000 Hz resampling, silence trimming, and peak normalization.
       - Extraction of 40 Mel-Frequency Cepstral Coefficients (MFCCs).
       - 2-layer Bidirectional LSTM capturing forward and backward temporal prosody and intonation.
    4. **Multimodal Feature Fusion**:
       - Weighted decision-level probability fusion:
         $$P_{\\text{fused}}(e) = w_{\\text{face}} P_{\\text{face}}(e) + w_{\\text{speech}} P_{\\text{speech}}(e)$$
       - Total Variation Distance conflict detection.
       - Dynamic single-modality fallback if face is occluded or audio is silent.
    """)
