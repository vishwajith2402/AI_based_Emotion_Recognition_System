# Multimodal Emotion Recognition Evaluation Report

**Course**: AGB1303 – AI Problem Solving Techniques  
**Project**: Human Emotion Recognition Using Facial Expressions and Speech Modulation (Batch 6)

## 1. Quantitative Benchmark Results

| Modality / Model | Accuracy (%) | Weighted F1 (%) | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: |
| **Facial Expression (CNN)** | 67.61% | 68.07% | Standard | Standard |
| **Speech Modulation (BiLSTM)** | 95.65% | 95.64% | Standard | Standard |
| **Multimodal Fused System** | **97.72%** | **97.71%** | High | High |

> **Multimodal Gain**: Fusion demonstrates **+2.07%** accuracy difference over single-modality baselines.

## 2. Classification Report (Fused System)

```text
              precision    recall  f1-score   support

       Angry       0.98      0.98      0.98       181
       Happy       0.97      0.98      0.97       181
     Neutral       0.97      0.99      0.98       196
         Sad       0.98      0.94      0.96       181
    Surprise       0.98      0.99      0.99       181

    accuracy                           0.98       920
   macro avg       0.98      0.98      0.98       920
weighted avg       0.98      0.98      0.98       920

```
