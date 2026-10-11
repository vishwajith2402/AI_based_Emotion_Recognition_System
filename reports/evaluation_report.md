# Multimodal Emotion Recognition Evaluation Report

**Course**: AGB1303 – AI Problem Solving Techniques  
**Project**: Human Emotion Recognition Using Facial Expressions and Speech Modulation (Batch 6)

## 1. Quantitative Benchmark Results

| Modality / Model | Accuracy (%) | Weighted F1 (%) | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: |
| **Facial Expression (CNN)** | 65.31% | 65.89% | Standard | Standard |
| **Speech Modulation (BiLSTM)** | 93.75% | 93.72% | Standard | Standard |
| **Multimodal Fused System** | **95.94%** | **95.92%** | High | High |

> **Multimodal Gain**: Fusion demonstrates **+2.19%** accuracy difference over single-modality baselines.

## 2. Classification Report (Fused System)

```text
              precision    recall  f1-score   support

       Angry       0.96      0.97      0.96       125
       Happy       0.94      0.96      0.95       125
     Neutral       0.97      0.96      0.97       140
         Sad       0.95      0.91      0.93       125
    Surprise       0.97      0.99      0.98       125

    accuracy                           0.96       640
   macro avg       0.96      0.96      0.96       640
weighted avg       0.96      0.96      0.96       640

```
