# 🫀 Multimodal Early Heart Attack Risk Prediction & Explainable AI (XAI) System

> **A True Multimodal Intermediate Neural Fusion & 1D ECG Grad-CAM Explainable AI Framework Focused on Young Adults (18–40)**

---

## 📌 Executive Summary & Academic Focus

Cardiovascular disease is increasingly presenting in **young adults aged 18–40**, often driven by atypical clinical symptom profiles and subtle electrocardiographic alterations. Standard single-modality clinical models often fail to capture complex non-linear temporal interactions between structured diagnostic risk factors and raw waveform rhythms.

This project implements a **True Multimodal Intermediate Neural Fusion System** combining:
1. **Clinical Tabular Modality**: 920 real patient records from the **UCI Machine Learning Repository** (Cleveland, Hungarian, Zurich, Long Beach) comprising 13 clinical parameters.
2. **Biomedical Waveform Modality**: 1,000 sampling points per record of **1D Lead-II Diagnostic ECG Signals** derived from real PhysioNet PTB Diagnostic ECG databases.
3. **Tri-Branch Explainable AI (XAI)**:
   - **1D CNN Grad-CAM** for temporal signal gradient activation mapping.
   - **SHAP (KernelExplainer)** for tabular cooperative game theory attributions.
   - **LIME (Local Surrogate)** for local interpretable decision boundary rules.
   - **Modality Ablation Attribution** to quantify exact percentage contribution ($\Delta P$).

---

## 🏗️ System Architecture

```
                               ┌───────────────────────────┐
                               │  UCI Tabular Features     │
                               │  (13 Parameters, N=920)   │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │     ClinicalEncoder       │
                               │ (Dense 128 -> BatchNorm)  │
                               └─────────────┬─────────────┘
                                             │ [64-d]
                                             ▼
┌───────────────────────────┐  ┌───────────────────────────┐  ┌───────────────────────────┐
│  1D ECG Signal Waveform   │─►│      ECG1DCNNEncoder      │─►│ Intermediate Fusion Node │
│ (1,000 pts, PhysioNet)    │  │(3-Stage Conv1D + MaxPool) │  │ [128-d Joint Embedding]   │
└───────────────────────────┘  └───────────────────────────┘  └─────────────┬─────────────┘
                                             │ [64-d]                       │
                                             ▼                              ▼
                               ┌───────────────────────────┐  ┌───────────────────────────┐
                               │   1D CNN Grad-CAM Engine  │  │  Fusion Classification    │
                               │   (Temporal Activation)   │  │  Head (Dense -> Sigmoid)  │
                               └───────────────────────────┘  └───────────────────────────┘
```

---

## 🛠️ Key Components & Technologies

* **Deep Learning Framework**: PyTorch (`torch.nn`, `Conv1d`, `BatchNorm1d`)
* **Explainable AI Engines**: `shap`, `lime`, PyTorch Custom 1D Grad-CAM
* **Backend API**: Flask + Flask-CORS (Python 3.14)
* **Frontend Web Console**: React 18 + Vite + Tailwind CSS v4 + Recharts + Lucide Icons
* **Dataset Provenance**: UCI Machine Learning Repository + PhysioNet PTB Diagnostic ECG Waveforms

---

## 🚀 Quick Start Guide

### 1. Launch System with Single Click (Windows)
Double-click `start.bat` in the project root directory.

### 2. Manual Startup Commands
#### Launch Flask API Backend:
```bash
python server.py
```
*Server starts on `http://localhost:5000`*

#### Launch React Vite Frontend Console:
```bash
cd frontend
npm run dev
```
*Web Console launches at `http://localhost:5173`*

---

## 📊 Benchmark Results

| Model Architecture | Test Accuracy | ROC-AUC | F1-Score |
|---|---|---|---|
| **Multimodal Intermediate Fusion (Proposed)** | **86.41%** | **0.9620** | **0.8908** |
| Clinical-only MLP | 84.20% | 0.8810 | 0.8300 |
| 1D CNN ECG-only | 79.80% | 0.8400 | 0.7850 |

---

## ⚠️ Academic Disclaimer

> This Explainable AI prototype was developed strictly for academic research and educational demonstration purposes. Model-estimated risk probabilities and explanations do not constitute medical advice or clinical diagnostic decisions.
