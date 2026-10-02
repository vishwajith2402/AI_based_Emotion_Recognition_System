# Multimodal Emotion Recognition Evaluation Report

**Course**: AGB1303 – AI Problem Solving Techniques  
**Project**: Human Emotion Recognition Using Facial Expressions and Speech Modulation (Batch 6)

## 1. Quantitative Benchmark Results

| Modality / Model | Accuracy (%) | Weighted F1 (%) | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: |
| **Facial Expression (CNN)** | 69.70% | 69.99% | Standard | Standard |
| **Speech Modulation (BiLSTM)** | 91.09% | 90.99% | Standard | Standard |
| **Multimodal Fused System** | **94.46%** | **94.45%** | High | High |

> **Multimodal Gain**: Fusion demonstrates **+3.37%** accuracy difference over single-modality baselines.

## 2. Classification Report (Fused System)

```text
              precision    recall  f1-score   support

       Angry       0.92      0.98      0.95        98
       Happy       0.90      0.94      0.92        98
     Neutral       0.96      0.96      0.96       113
         Sad       0.98      0.89      0.93        98
    Surprise       0.97      0.95      0.96        98

    accuracy                           0.94       505
   macro avg       0.95      0.94      0.94       505
weighted avg       0.95      0.94      0.94       505

```
