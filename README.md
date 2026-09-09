# 🫀 Early Heart Attack Risk Prediction for Young Adults Using Explainable AI

Academic lab project for the **Explainable Artificial Intelligence (23AM501)** course, CSE (AI & ML), Sri Krishna College of Technology.

---

## 📐 System Architecture Framework

![Explainable AI Architecture Framework](./xai_framework_diagram.png)

---

## 🌟 Executive Abstract & Motivation

Early prediction and risk stratification of cardiovascular disease in young adults (aged 18–40) is an increasingly vital clinical research problem. Traditional risk scoring models frequently underestimate cardiovascular vulnerabilities in younger populations due to non-typical presentation and lifestyle interaction effects.

This system combines machine learning classifiers (**Random Forest**) and deep learning representations (**PyTorch Tabular Neural Network**) trained on the real **UCI Heart Disease Dataset** (920 patient records) with a three-perspective Explainable AI suite:
1. **SHAP (SHapley Additive exPlanations)** — Cooperative game-theoretic feature attribution.
2. **LIME (Local Interpretable Model-agnostic Explanations)** — Local linear surrogate decision rules.
3. **Tabular Grad-CAM (Gradient-weighted Activation Mapping)** — Intermediate neural network activation gradients.

---

## ⚠️ Academic & Medical Disclaimer

> **This system is an academic Explainable AI research prototype and decision-support demonstration. Predictions are statistical model estimates and should NOT be used for direct clinical diagnosis or patient treatment.**

---

## 🔑 Key System Capabilities

* 🎯 **18–40 Young Adult Risk Cohort**: First-class support and dedicated analysis mode for evaluating risk patterns in patients aged 18 to 40.
* 🧪 **Real UCI Dataset Integration**: Trained on 920 real clinical records combining Cleveland Clinic, Hungarian Institute, University Hospital Zurich, and V.A. Medical Center Long Beach.
* 📊 **Three-Way XAI Perspective**: Instant multi-view feature attributions (SHAP Waterfall + LIME Local Linear Surrogate + Tabular Grad-CAM Heatmap) for any patient.
* 🔬 **Live Risk Prediction & Explanation**: Enter custom clinical parameters to generate real-time risk probabilities and instant XAI breakdowns.
* ⚡ **Model Evaluation & Error Analysis**: Comprehensive metrics (Accuracy, Precision, Recall, F1, ROC-AUC), Confusion Matrix, and clinical error trade-off explanations (False Positives vs. False Negatives).

---

## 🚀 Getting Started & How to Run

### One-Click Launch (Windows)

Double-click `start.bat` in File Explorer or run in Command Prompt / PowerShell:
```cmd
start.bat
```

The script automatically verifies python/npm environments, launches both backend and frontend, and cleans up all processes when closed.

Once launched, open your web browser:
* 🌐 **React Clinical AI Console**: `http://localhost:5173`
* ⚡ **Flask REST API Engine**: `http://localhost:5000`

---

## 📁 Repository Structure

```
├── dataset/
│   ├── heart_dataset.csv     # Real UCI Heart Disease tabular dataset (920 patient records)
│   └── README.md             # Dataset documentation, clinical units, and cohort stats
├── frontend/                 # React 19 + Vite + Tailwind CSS v4 Clinical Console
│   ├── src/App.jsx           # 9-Tab Clinical Research Dashboard with 3-Way XAI Panel
│   └── src/index.css         # Glassmorphism styling and Tailwind CSS v4 directives
├── server.py                 # Flask REST API (Random Forest, SHAP, LIME, PyTorch Grad-CAM)
├── Heart_Attack_XAI_Analysis.ipynb # 16-Phase Academic Research Notebook
├── xai_framework_diagram.png # High-resolution architecture framework diagram
├── xai_framework_diagram.pdf # PDF format of the system architecture diagram
├── start.bat                 # Windows process lifecycle manager script
└── requirements.txt          # Python dependencies
```

---

## 📊 XAI Methodologies Comparison

| Attribute | SHAP (Shapley Additive exPlanations) | LIME (Local Surrogates) | Tabular Grad-CAM |
|---|---|---|---|
| **Theoretical Foundation** | Cooperative Game Theory (Shapley values) | Local Surrogate Linear Models | Backpropagated Neural Gradients |
| **Explanation Scope** | Global + Local | Local Neighborhood | Intermediate Neural Representations |
| **Consistency** | Deterministic & Axiomatic | Stochastic (perturbation-based) | Gradient-exact to Neural Target |
| **Axiomatic Guarantees** | Efficiency, Symmetry, Additivity | Heuristic-based | Gradient chain rule attribution |

---

## 🛠️ Tech Stack

* **ML & XAI Engine**: Python 3.12+, `scikit-learn`, `shap`, `lime`, `torch` (PyTorch), `pandas`, `numpy`, `matplotlib`, `plotly`
* **Backend API**: Flask, Flask-CORS
* **Frontend Web App**: React 19, Vite, Tailwind CSS v4, Recharts, Lucide Icons

---

## 🎓 Academic Course Context

Developed for the **Explainable Artificial Intelligence (23AM501)** course to demonstrate transparent, interpretable, and auditable machine learning model decisions in high-stakes healthcare risk assessment.
