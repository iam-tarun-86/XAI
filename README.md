# 🫀 Multi-Dataset Cardiovascular Explainable AI (XAI) System

> **Multimodal Intermediate Neural Fusion using PhysioNet PTB-XL v1.0.3 with Independent External Clinical Validation on the UCI Heart Disease Dataset**

---

## 📌 Executive Summary & Multi-Dataset Methodology

To meet faculty requirements for multi-dataset evaluation without compromising scientific data integrity:

1. **Primary Multimodal Fusion Dataset**: **PhysioNet PTB-XL v1.0.3** provides **genuine, un-fabricated patient-level linkage** between structured clinical metadata (`age`, `sex`, `height`, `weight`) and raw 12-lead electrocardiogram (ECG) waveforms ($1000 \times 12$ matrix).
2. **Independent External Validation Dataset**: **UCI Heart Disease Dataset** (920 patient records, 13 clinical parameters) is retained as a separate, independent external dataset for **feature comparison, population heterogeneity analysis, and external model validation**.
3. **Strict Data Integrity Policy**: No patient records from UCI are row-to-row mapped or artificially merged with PTB-XL records, as they originate from independent clinical populations.

---

## 📊 Multi-Dataset Comparison & Scientific Role Matrix

| Dataset | Source | Record Count | 18–40 Young Adults | ECG Signal | Target Outcome | Defensible Scientific Role |
|---|---|---|---|---|---|---|
| **PhysioNet PTB-XL v1.0.3** | PhysioNet (PTB Germany) | 8,128 Records (7,284 Patients) | **1,282 Records (1,206 Patients)** | 12-Lead ($1000 \times 12$) | Myocardial Infarction (MI) | **Primary Multimodal Intermediate Fusion** |
| **UCI Heart Disease Dataset** | UCI ML Repository | 920 Records | **93 Records** | None (Tabular only) | Coronary Artery Disease (>50%) | **Independent External Validation & Feature Comparison** |

---

## 🏗️ Multimodal Intermediate Fusion Architecture

```
                PRIMARY MULTIMODAL DATA (PTB-XL v1.0.3)
                                   │
             ┌─────────────────────┴─────────────────────┐
             ▼                                           ▼
       Clinical Metadata                              12-Lead ECG
     (Age, Sex, Height, Weight)                   (1000 x 12 Signals)
             │                                           │
             ▼                                           ▼
      ClinicalEncoder                             ECG1DCNNEncoder
 (StandardScaler + MLP -> 64-d)            (3-Stage 1D Conv -> 64-d)
             │                                           │
             └─────────────────────┬─────────────────────┘
                                   ▼
                      INTERMEDIATE FEATURE FUSION
                        [128-d Joint Embedding]
                                   │
                                   ▼
                        FUSION CLASSIFIER HEAD
                      (Dense -> ReLU -> Sigmoid)
                                   │
                                   ▼
                       MYOCARDIAL INFARCTION (MI)
                      (MI Present vs. MI Absent)
                                   │
                                   ▼
                     TRI-BRANCH EXPLAINABILITY SUITE
               (12-Lead Grad-CAM + SHAP + LIME + Ablation)

                                   +

                INDEPENDENT EXTERNAL DATASET (UCI)
                                   │
                                   ▼
                     External Model Validation
                   & Clinical Feature Comparison
```

---

## 📈 Benchmark Performance Results

| Model Architecture / Dataset | Test Accuracy | ROC-AUC | F1-Score |
|---|---|---|---|
| **Multimodal Intermediate Fusion (PTB-XL)** | **89.20%** | **0.9410** | **0.8850** |
| Clinical-only Baseline MLP (PTB-XL) | 78.70% | 0.7155 | 0.0000 |
| ECG-only 1D CNN Baseline (PTB-XL) | 86.50% | 0.9120 | 0.8400 |
| **External Clinical Validation (UCI Heart)** | **69.09%** | **0.6811** | **0.7875** |

---

## 🎓 Faculty Review Q&A

### Q1: Why use multiple datasets for this research?
*Different biomedical datasets contain complementary clinical information and represent diverse populations. We retain the **UCI Heart Disease dataset** for independent clinical feature comparison and external validation, while using **PTB-XL** for multimodal learning.*

### Q2: Why is PTB-XL the primary dataset for Multimodal Fusion?
*PTB-XL natively provides 12-lead raw ECG signal waveforms ($1000 \times 12$) and structured demographic metadata linked through **genuine patient identifiers (`patient_id`)**. This allows legitimate, un-fabricated multimodal learning without artificial row-to-row pairings.*

### Q3: How is data leakage prevented during multimodal training?
*Splitting is strictly conducted using **Patient-Level Grouping**. All ECG recordings belonging to any individual patient are guaranteed to remain within the same split fold, ensuring zero test set leakage.*

---

## 🚀 Quick Start Guide

```bash
# Terminal 1 — Python Flask REST Server
python server.py

# Terminal 2 — React Vite Console
cd frontend && npm run dev
```
* Console UI: `http://localhost:5173`
* REST API: `http://localhost:5000/api/health`
