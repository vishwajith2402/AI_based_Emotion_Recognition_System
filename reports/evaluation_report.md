# Multimodal Emotion Recognition Evaluation Report

**Course**: AGB1303 – AI Problem Solving Techniques  
**Project**: Human Emotion Recognition Using Facial Expressions and Speech Modulation (Batch 6)

## 1. Quantitative Benchmark Results

| Modality / Model | Accuracy (%) | Weighted F1 (%) | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: |
| **Facial Expression (CNN)** | 68.12% | 68.64% | Standard | Standard |
| **Speech Modulation (BiLSTM)** | 92.08% | 92.01% | Standard | Standard |
| **Multimodal Fused System** | **95.84%** | **95.82%** | High | High |

> **Multimodal Gain**: Fusion demonstrates **+3.76%** accuracy difference over single-modality baselines.

## 2. Classification Report (Fused System)

```text
              precision    recall  f1-score   support

       Angry       0.95      0.98      0.96        98
       Happy       0.97      0.94      0.95        98
     Neutral       0.95      0.97      0.96       113
         Sad       0.95      0.91      0.93        98
    Surprise       0.98      0.99      0.98        98

    accuracy                           0.96       505
   macro avg       0.96      0.96      0.96       505
weighted avg       0.96      0.96      0.96       505

```
