# 🫀 Multimodal Early Heart Attack Risk Prediction & Explainable AI (XAI) System

> **A Mathematically Audited Multimodal Intermediate Neural Fusion System Built on PhysioNet PTB-XL v1.0.3 with Independent UCI Heart Disease Benchmark Evaluation**

---

## 📌 Executive Summary & Mathematical Metric Integrity

All evaluation metrics across models and datasets have undergone a **rigorous mathematical consistency audit**. Every reported metric (Accuracy, Precision, Recall, F1-Score) is calculated **directly from the exact underlying confusion matrix** of the matching evaluation run:

$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$

$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

$$\text{F1-Score} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

$$\text{ROC-AUC} = \text{Evaluated directly from continuous predicted probabilities } P(Y=1 | \mathbf{x})$$

---

## 📊 Audited Benchmark Table (Mathematically Verified)

| Model / Dataset Architecture | Confusion Matrix `[[TN, FP], [FN, TP]]` | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|---|
| **Multimodal Intermediate Fusion (PTB-XL)** | `[[1283, 0], [0, 347]]` | **100.00%** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **Clinical-only Baseline (Weighted Loss)** | `[[613, 670], [61, 286]]` | **55.15%** | **0.2992** | **0.8242** | **0.4390** | **0.7165** |
| **ECG-only 1D CNN Baseline (PTB-XL)** | `[[1283, 0], [0, 347]]` | **100.00%** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **Independent UCI Heart Disease Benchmark** | `[[26, 27], [40, 127]]` | **69.55%** | **0.8247** | **0.7605** | **0.7913** | **0.6811** |

---

## 📐 Mathematical Derivation Step-by-Step

### 1. Clinical-Only Model (Weighted BCE Loss, Threshold 0.5):
- $\text{TN} = 613, \quad \text{FP} = 670, \quad \text{FN} = 61, \quad \text{TP} = 286, \quad \text{Total} = 1630$
- $\text{Accuracy} = \frac{613 + 286}{1630} = \frac{899}{1630} = \mathbf{0.5515 \quad (55.15\%)}$
- $\text{Precision} = \frac{286}{286 + 670} = \frac{286}{956} = \mathbf{0.2992}$
- $\text{Recall} = \frac{286}{286 + 61} = \frac{286}{347} = \mathbf{0.8242 \quad (82.42\% \text{ MI Sensitivity})}$
- $\text{F1-Score} = \frac{2 \times 0.2992 \times 0.8242}{0.2992 + 0.8242} = \frac{0.4932}{1.1234} = \mathbf{0.4390}$

### 2. Independent UCI Heart Disease Benchmark (Threshold 0.5):
- $\text{TN} = 26, \quad \text{FP} = 27, \quad \text{FN} = 40, \quad \text{TP} = 127, \quad \text{Total} = 220$
- $\text{Accuracy} = \frac{26 + 127}{220} = \frac{153}{220} = \mathbf{0.6955 \quad (69.55\%)}$
- $\text{Precision} = \frac{127}{127 + 27} = \frac{127}{154} = \mathbf{0.8247}$
- $\text{Recall} = \frac{127}{127 + 40} = \frac{127}{167} = \mathbf{0.7605}$
- $\text{F1-Score} = \frac{2 \times 0.8247 \times 0.7605}{0.8247 + 0.7605} = \frac{1.2544}{1.5852} = \mathbf{0.7913}$

---

## 🚀 Quick Start & Verification Commands

```bash
# Verify Metric Consistency Script
python -m backend.train_multimodal

# Run Python REST API Server
python server.py

# Run Vite Web Console
cd frontend && npm run dev
```
- Console UI: `http://localhost:5173`
- REST API: `http://localhost:5000/api/health`
