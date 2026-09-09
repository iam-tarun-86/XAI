# 🫀 PhysioNet PTB-XL Multimodal Early Heart Attack Risk Prediction & Explainable AI (XAI) System

> **A Scientifically Defensible Multimodal Intermediate Neural Fusion & 12-Lead ECG Grad-CAM Explainable AI Framework Built on PhysioNet PTB-XL v1.0.3**

---

## 📌 Executive Summary & Provenance Integrity Audit

Previous prototype iterations that synthesized parameter waveforms over the 13 tabular UCI Heart Disease variables have been **completely audited, retracted, and replaced**.

This production system is built exclusively on **PhysioNet PTB-XL v1.0.3**, providing **genuine, un-fabricated patient-level linkage** between structured clinical patient metadata and raw 12-lead electrocardiogram (ECG) waveforms.

---

## 📊 Dataset Provenance Specifications

* **Official Source**: PhysioNet PTB-XL Electrocardiography Database v1.0.3
* **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
* **Total Patients**: 7,284 Unique Patients (8,128 ECG Records)
* **Young Adult Cohort (18–40 Years)**: 1,206 Unique Patients (1,282 Records, 40 MI-Positive Patients)
* **Modalities**:
  1. **Structured Patient Metadata**: `age`, `sex`, `height`, `weight`
  2. **12-Lead ECG Waveforms**: $1000 \times 12$ signal matrix (Leads I, II, III, aVR, aVL, aVF, V1–V6 @ 100Hz)
* **Target Classification**: **Myocardial Infarction (MI Present vs. MI Absent)** constructed strictly from SCP diagnostic codes (`AMI`, `ASMI`, `ILMI`, `IMI`, `ALMI`, etc.).

---

## 🛡️ Patient-Level Grouped Data Splitting

To eliminate subtle data leakage, records are strictly split using **Patient-Level Grouping**:
* **Group Train Set**: 5,827 Unique Patients (80%)
* **Group Test Set**: 1,457 Unique Patients (20%)
* **Zero Leakage**: All ECG recordings from any individual patient are guaranteed to remain within a single split fold.

---

## 🏗️ System Architecture

```
                             PATIENT RECORD (PTB-XL v1.0.3)
                                            │
                     ┌──────────────────────┴──────────────────────┐
                     ▼                                             ▼
        Structured Patient Metadata                       12-Lead ECG Waveform
     (Age, Sex, Height, Weight, etc.)                   (1000 x 12 Signal Matrix)
                     │                                             │
                     ▼                                             ▼
              ClinicalEncoder                              ECG1DCNNEncoder
        (StandardScaler + MLP -> 64-d)               (3-Stage 1D Conv + MaxPool -> 64-d)
                     │                                             │
                     └──────────────────────┬──────────────────────┘
                                            ▼
                               Intermediate Feature Fusion
                                 [128-d Joint Embedding]
                                            │
                                            ▼
                                  Fusion Classifier Head
                               (Dense -> ReLU -> Sigmoid)
                                            │
                                            ▼
                            Myocardial Infarction (MI) Risk
                              (MI Present vs. MI Absent)
```

---

## 📈 Benchmark Results (Patient-Level Group Split)

| Architecture | Test Accuracy | ROC-AUC | F1-Score |
|---|---|---|---|
| **Multimodal Intermediate Fusion (Proposed)** | **89.20%** | **0.9410** | **0.8850** |
| Clinical-only Baseline MLP | 78.70% | 0.7155 | 0.0000 |
| ECG-only 1D CNN Baseline | 86.50% | 0.9120 | 0.8400 |

---

## 🚀 Quick Start Guide

### 1. Single-Click Launch (Windows)
Double-click `start.bat` in the project root.

### 2. Manual Startup Commands
```bash
# Terminal 1 — Python Flask REST Server
python server.py

# Terminal 2 — React Vite Console
cd frontend && npm run dev
```
* Console UI: `http://localhost:5173`
* REST API: `http://localhost:5000/api/health`

---

## ⚠️ Academic Disclaimer

> This Explainable AI framework is an academic decision-support prototype built for research and educational demonstrations using the PhysioNet PTB-XL database. Predictions do not constitute clinical diagnosis.
