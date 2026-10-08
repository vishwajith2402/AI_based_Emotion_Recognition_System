# Multimodal Emotion Recognition Evaluation Report

**Course**: AGB1303 – AI Problem Solving Techniques  
**Project**: Human Emotion Recognition Using Facial Expressions and Speech Modulation (Batch 6)

## 1. Quantitative Benchmark Results

| Modality / Model | Accuracy (%) | Weighted F1 (%) | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: |
| **Facial Expression (CNN)** | 70.30% | 70.14% | Standard | Standard |
| **Speech Modulation (BiLSTM)** | 92.48% | 92.41% | Standard | Standard |
| **Multimodal Fused System** | **94.06%** | **94.03%** | High | High |

> **Multimodal Gain**: Fusion demonstrates **+1.58%** accuracy difference over single-modality baselines.

## 2. Classification Report (Fused System)

```text
              precision    recall  f1-score   support

       Angry       0.92      0.99      0.96        98
       Happy       0.91      0.96      0.94        98
     Neutral       0.95      0.92      0.94       113
         Sad       0.93      0.88      0.91        98
    Surprise       0.98      0.96      0.97        98

    accuracy                           0.94       505
   macro avg       0.94      0.94      0.94       505
weighted avg       0.94      0.94      0.94       505

```
