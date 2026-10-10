# Multimodal Emotion Recognition Evaluation Report

**Course**: AGB1303 – AI Problem Solving Techniques  
**Project**: Human Emotion Recognition Using Facial Expressions and Speech Modulation (Batch 6)

## 1. Quantitative Benchmark Results

| Modality / Model | Accuracy (%) | Weighted F1 (%) | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: |
| **Facial Expression (CNN)** | 66.73% | 67.42% | Standard | Standard |
| **Speech Modulation (BiLSTM)** | 90.10% | 89.96% | Standard | Standard |
| **Multimodal Fused System** | **93.07%** | **93.02%** | High | High |

> **Multimodal Gain**: Fusion demonstrates **+2.97%** accuracy difference over single-modality baselines.

## 2. Classification Report (Fused System)

```text
              precision    recall  f1-score   support

       Angry       0.89      1.00      0.94        98
       Happy       0.94      0.94      0.94        98
     Neutral       0.99      0.82      0.90       113
         Sad       0.87      0.93      0.90        98
    Surprise       0.98      0.98      0.98        98

    accuracy                           0.93       505
   macro avg       0.93      0.93      0.93       505
weighted avg       0.93      0.93      0.93       505

```
