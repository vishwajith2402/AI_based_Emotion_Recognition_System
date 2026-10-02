# Viva Voce & Project Review Defense Guide 🎓
**Course**: AGB1303 – AI Problem Solving Techniques (TCPR)  
**Department**: Artificial Intelligence and Machine Learning  
**Project**: Human Emotion Recognition Using Facial Expressions and Speech Modulation (Batch 6)

---

## 🌟 Quick Project Pitch (1-Minute Elevator Summary)
> *"Respected guide and examiners, our project is **Human Emotion Recognition Using Facial Expressions and Speech Modulation**. While existing systems rely solely on a single modality—such as facial image analysis or speech audio—they struggle in noisy real-world environments with poor lighting, head turns, or background sounds. We have engineered an end-to-end multimodal deep learning system that combines visual cues using a **Deep Convolutional Neural Network (CNN)** and acoustic vocal patterns using **MFCC feature extraction with a Bidirectional LSTM (BiLSTM)**. These two independent information streams are synthesized through a **Multimodal Feature Fusion Engine** with dynamic confidence weighting, single-modality failover, and conflict detection. Our prototype includes a real-time web dashboard and Windows desktop app capable of live inference and interactive model retraining."*

---

## ❓ 15 Most Common Viva Questions & Model Answers

### Q1: What is Multimodal Emotion Recognition, and why is it superior to unimodal systems?
**Answer**:
Unimodal systems process either visual or acoustic data in isolation. They face key real-world failure modes:
1. **Visual failure**: Occlusion (hand on face, mask), dim lighting, extreme head poses, or low-resolution webcams.
2. **Acoustic failure**: Background babble, microphone distortion, ambient noise, or natural human silence.
By combining both modalities, **multimodal recognition leverages complementary information**. When one modality is compromised or ambiguous, the other provides strong compensatory evidence, yielding higher overall accuracy and operational reliability.

---

### Q2: Why did you choose a Convolutional Neural Network (CNN) for facial emotion recognition?
**Answer**:
CNNs are the gold standard for computer vision because of:
1. **Local receptive fields**: They detect local visual patterns like eye corners, eyebrow angles, and mouth curves.
2. **Weight sharing & translation invariance**: A smile is recognized whether the mouth is centered or shifted in the frame.
3. **Hierarchical feature extraction**: Lower layers learn edges and textures, intermediate layers capture facial landmarks (lips, nose, eyelids), and higher dense layers learn abstract emotional geometry.

Our architecture features 3 convolutional blocks with Batch Normalization (to stabilize gradient propagation) and Dropout (to prevent overfitting).

---

### Q3: What is MFCC and why is it used for speech emotion recognition?
**Answer**:
**MFCC** stands for **Mel-Frequency Cepstral Coefficients**.
1. Human auditory perception is non-linear—we perceive pitch differences much more acutely at lower frequencies (below 1 kHz) than at higher frequencies.
2. The **Mel scale** mimics the cochlea's frequency response.
3. MFCC extraction computes:
   - Fourier Transform (FFT) of short-time windowed audio frames.
   - Mel-filterbank triangular bandpass filtering.
   - Logarithm of filter energies.
   - Discrete Cosine Transform (DCT) to decorrelate filter outputs.
4. We extract **40 MFCC coefficients** per frame, which capture vocal timbre, formant resonance, and vocal tract shape without being sensitive to language vocabulary.

---

### Q4: Why use a Bidirectional LSTM (BiLSTM) instead of standard LSTM or MLP for speech?
**Answer**:
Speech is an inherently sequential, temporal signal where emotional expression unfolds over time (prosody, cadence, rising or falling intonation):
1. **Standard LSTMs** only process speech forward in time ($t_0 \rightarrow t_n$), missing future context.
2. **BiLSTM** processes the MFCC sequence simultaneously in **both directions** (forward $t_0 \rightarrow t_n$ and backward $t_n \rightarrow t_0$). This allows the network to evaluate how an emotional exclamation begins and decays relative to its full sentence prosody.
3. We follow the BiLSTM with **Dual Temporal Pooling (Average + Max Pooling)** across time steps to capture both sustained vocal tone and peak acoustic energy bursts.

---

### Q5: What are the different types of Multimodal Fusion, and which one did you use?
**Answer**:
There are three fundamental fusion strategies:
1. **Early / Feature-Level Fusion**: Concatenating raw feature vectors (e.g. CNN embedding + MFCC embedding) into a single large vector before classification. *Challenge*: Requires synchronized multimodal dataset pairs and risks high dimensionality.
2. **Late / Decision-Level Fusion**: Independently predicting probability vectors $P_{\text{face}}$ and $P_{\text{speech}}$ and mathematically combining them. *Advantage*: Highly modular, allows independent unimodal training, and trivially supports single-modality fallback if one input drops out.
3. **Hybrid Fusion**: Combines intermediate cross-attention with decision weights.

In our prototype, we implement **Decision-Level Weighted Fusion**:
$$P_{\text{fused}}(e) = w_{\text{face}} \cdot P_{\text{face}}(e) + w_{\text{speech}} \cdot P_{\text{speech}}(e)$$
where weights can be statically balanced ($0.5/0.5$) or dynamically modulated based on modality signal quality.

---

### Q6: What happens if a person is smiling (happy face) but yelling in anger (angry voice)? How does your system handle conflict?
**Answer**:
This is termed a **Modality Conflict**.
1. Our `MultimodalFusion` module calculates the **Total Variation Divergence** between the facial probability distribution and speech probability distribution:
   $$\delta = \frac{1}{2} \sum_{e} |P_{\text{face}}(e) - P_{\text{speech}}(e)|$$
2. If $\delta > 0.40$ and the top predicted classes differ, the system raises a **Modality Conflict Alert** on the UI, displaying:
   `"Divergence detected: Face expresses 'Happy' but voice conveys 'Angry'"`.
3. This informs human supervisors of sarcasm, masked emotion, or environmental ambiguity rather than silently presenting an uncalibrated compromise.

---

### Q7: What happens if the user is silent or turns their head away from the camera?
**Answer**:
The system implements **Graceful Fallback Handling**:
- **Silence**: The `SpeechPreprocessor` checks Root-Mean-Square (RMS) acoustic energy. If energy falls below conversational threshold ($< 0.005$), the audio is marked silent and the system operates in **Facial-Only Fallback** ($w_{\text{face}}=1.0$).
- **Face Occluded**: If OpenCV Haar Cascade/DNN finds no face bounding boxes, the system operates in **Speech-Only Fallback** ($w_{\text{speech}}=1.0$).
- **Neither Available**: The system gracefully reports an Idle/Neutral baseline without crashing.

---

### Q8: What emotion categories are recognized by the system?
**Answer**:
We standardized on 5 core universal emotional states:
1. **Angry**: High energy, tense vocal harmonics, lowered/furrowed eyebrows.
2. **Happy**: Upward mouth arc, smiling cheek elevation, melodic pitch inflection.
3. **Neutral**: Rested facial geometry, flat monotone speech prosody.
4. **Sad**: Downturned mouth, drooping eyelids, falling pitch contour, low acoustic energy.
5. **Surprise**: Wide circular eyes, high arched eyebrows, steep upward acoustic pitch sweep.

---

### Q9: Which datasets are supported for training?
**Answer**:
1. **Curated Project Dataset**: A balanced, pre-packaged dataset of 300 facial images and 300 16kHz audio clips stored in `data/facial/` and `data/speech/` across `train/`, `val/`, and `test/` splits.
2. **Academic Benchmarks Supported**:
   - **FER-2013**: 35,887 facial expression grayscale images.
   - **RAVDESS**: Ryerson Audio-Visual Database of Emotional Speech and Song (1,440 recordings from 24 actors).
   - We provide `data/download_datasets.py` with filename parsers to map benchmark labels to our 5 unified classes.

---

### Q10: What deep learning framework and libraries did you use?
**Answer**:
- **Deep Learning**: PyTorch (`torch`, `nn.Module`, `optim.Adam`, `CrossEntropyLoss`).
- **Computer Vision**: OpenCV (`cv2`) for Haar Cascade face detection, bounding box overlay, and video capture.
- **Audio Processing**: Librosa and SoundFile for 16kHz resampling, silence trimming, and 40 MFCC extraction; SoundDevice for microphone recording.
- **User Interface**: Streamlit for the web application and CustomTkinter for the native Windows desktop app.

---

### Q11: How do you prevent overfitting in the neural networks?
**Answer**:
1. **Data Augmentation**: Random horizontal flipping and brightness scaling on facial training images.
2. **Dropout Regularization**: 25% dropout after convolutional and pooling blocks, 50% dropout in dense layers, and 35% dropout in the BiLSTM head.
3. **Batch Normalization**: Placed after convolutional layers and dense layers to prevent internal covariate shift.
4. **Weight Decay ($L_2$ Regularization)**: $1 \times 10^{-4}$ in Adam optimizer.
5. **Validation Checkpointing**: Saving the model state dictionary only when validation accuracy improves.

---

### Q12: How are audio files normalized before MFCC extraction?
**Answer**:
1. Resampled to 16,000 Hz.
2. Silence trimmed using `librosa.effects.trim` with a 25 dB threshold.
3. Padded or cropped to a fixed duration of 3.0 seconds (yielding uniform 100-frame sequence length).
4. Peak amplitude normalized to prevent microphone distance from altering feature magnitudes.
5. MFCCs are z-score standardized (zero mean, unit variance) across time frames.

---

### Q13: Can this system be deployed in real-time?
**Answer**:
Yes.
- Facial CNN inference takes under **15 milliseconds** per frame on CPU.
- 3-second speech MFCC extraction and BiLSTM inference takes under **40 milliseconds**.
- The system easily sustains 30+ FPS video analysis with concurrent audio chunk processing.

---

### Q14: What are the primary limitations of the current prototype?
**Answer**:
1. **Inference vs Internal State**: The system detects physical *expressions* of emotion, which do not always correlate 1:1 with an individual's genuine internal psychological state.
2. **Microphone Variability**: Low-quality condenser microphones can distort higher audio harmonics.
3. **Extreme Poses**: Haar Cascades fail when the face is turned greater than $45^\circ$ from the camera (can be improved with MediaPipe Face Mesh).

---

### Q15: What are the future enhancements (Future Scope)?
**Answer**:
1. **Attention-based Cross-Modal Fusion**: Utilizing Transformer Cross-Attention (e.g. Perceiver or Multimodal Transformer) to model fine-grained temporal alignment between specific phonemes and lip movements.
2. **3D Facial Landmarks / MediaPipe**: Tracking 468 3D facial landmarks for robust 3D head pose invariance.
3. **Micro-Expression Detection**: Incorporating optical flow to detect fleeting subconscious micro-expressions lasting $< 200\text{ ms}$.
