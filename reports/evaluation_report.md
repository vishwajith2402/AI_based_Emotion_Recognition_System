# Multimodal Emotion Recognition Evaluation Report

**Course**: AGB1303 – AI Problem Solving Techniques  
**Project**: Human Emotion Recognition Using Facial Expressions and Speech Modulation (Batch 6)

## 1. Quantitative Benchmark Results

| Modality / Model | Accuracy (%) | Weighted F1 (%) | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: |
| **Facial Expression (CNN)** | 66.73% | 67.42% | Standard | Standard |
| **Speech Modulation (BiLSTM)** | 90.10% | 89.96% | Standard | Standard |
| **Multimodal Fused System** | **95.25%** | **95.23%** | High | High |

> **Multimodal Gain**: Fusion demonstrates **+5.15%** accuracy difference over single-modality baselines.

## 2. Classification Report (Fused System)

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
