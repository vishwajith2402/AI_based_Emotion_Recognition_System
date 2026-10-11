# Multimodal Emotion Recognition Evaluation Report

**Course**: AGB1303 – AI Problem Solving Techniques  
**Project**: Human Emotion Recognition Using Facial Expressions and Speech Modulation (Batch 6)

## 1. Quantitative Benchmark Results

| Modality / Model | Accuracy (%) | Weighted F1 (%) | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: |
| **Facial Expression (CNN)** | 66.81% | 67.22% | Standard | Standard |
| **Speech Modulation (BiLSTM)** | 93.91% | 93.87% | Standard | Standard |
| **Multimodal Fused System** | **96.23%** | **96.21%** | High | High |

> **Multimodal Gain**: Fusion demonstrates **+2.32%** accuracy difference over single-modality baselines.

## 2. Classification Report (Fused System)

```text
              precision    recall  f1-score   support

       Angry       0.95      0.96      0.96       135
       Happy       0.97      0.96      0.97       135
     Neutral       0.95      0.98      0.97       150
         Sad       0.96      0.91      0.94       135
    Surprise       0.98      0.99      0.99       135

    accuracy                           0.96       690
   macro avg       0.96      0.96      0.96       690
weighted avg       0.96      0.96      0.96       690

```
