# SHAP vs LIME: Explainable AI Lab

Academic lab project for the **Explainable Artificial Intelligence (23AM501)** course, CSE (AI & ML), Sri Krishna College of Technology.

## Overview

This project explores two of the most widely used frameworks for model interpretability — **SHAP (SHapley Additive exPlanations)** and **LIME (Local Interpretable Model-agnostic Explanations)** — by training a Random Forest classifier on the Wine dataset and analyzing its predictions through both global and local explanation techniques.

## What's covered

- **Model training** — Random Forest classifier trained on the Wine dataset
- **Global interpretability (SHAP)** — SHAP summary plots and feature importance rankings across the entire dataset
- **Local interpretability (SHAP)** — Waterfall plots and interactive force plots explaining individual predictions
- **Feature interactions** — SHAP dependence analysis to understand how features interact with each other
- **Local interpretability (LIME)** — Surrogate linear models for explaining single predictions
- **Comparative analysis** — Measuring agreement/disagreement between SHAP and LIME explanations on the same instances
- **Case study** — Deep-dive into a specific high-uncertainty prediction, examining which features pulled the model's decision in opposite directions

## Key takeaway

| | SHAP | LIME |
|---|---|---|
| Basis | Game theory (Shapley values) | Local surrogate linear models |
| Consistency | Deterministic, highly consistent | Stochastic, can vary between runs |
| Speed | Slower (esp. KernelSHAP) | Faster, samples locally |
| Scope | Global + local explanations | Primarily local |
| Guarantees | Mathematically fair attribution | Heuristic, no formal guarantees |

**Use SHAP** when you need theoretically sound, consistent feature attribution across a model. **Use LIME** when you need a fast, intuitive explanation for one specific prediction, especially on high-dimensional data.

## Tech stack

Python, scikit-learn, SHAP, LIME, pandas, matplotlib

## Course context

Built as a hands-on lab to understand explainable AI techniques — a growing requirement in real-world ML systems where model decisions need to be interpretable and auditable, not just accurate.
