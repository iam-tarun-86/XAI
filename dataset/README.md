# 🫀 UCI Heart Disease Dataset (Young Adults 18–40 Focus)

This directory contains the real, publicly available **UCI Heart Disease Dataset** fetched from the **UCI Machine Learning Repository** (combining Cleveland Clinic, Hungarian Institute of Cardiology, University Hospital Zurich Switzerland, and V.A. Medical Center Long Beach).

---

## 📁 Files in this Directory

* **`heart_dataset.csv`**: The processed tabular dataset (920 real patient records, 13 clinical parameters + target variable).
* **`README.md`**: Dataset documentation, feature definitions, and cohort breakdown.

---

## 📊 Real Patient Statistics & Cohort Breakdown

* **Source**: UCI Machine Learning Repository (Four Combined International Clinical Databases)
* **Total Real Records**: 920 patients
* **Age Range**: 28.0 to 77.0 years
* **Young Adult Cohort (18–40 years)**: 93 real patient records (61 Low Risk, 32 Elevated Risk)
* **Older Adult Cohort (>40 years)**: 827 real patient records
* **Target Class Distribution**: 509 Elevated Risk (55.3%), 411 Low Risk (44.7%)

---

## 🧪 Clinical Feature Definitions

| Feature Name | UCI Code | Data Type | Units / Encoding | Clinical Description |
|---|---|---|---|---|
| **age** | `age` | Numerical | Years (28–77) | Patient age. Focused on early risk detection in young adults (18–40). |
| **sex** | `sex` | Categorical | 0: Female, 1: Male | Biological sex of patient. |
| **chest_pain_type** | `cp` | Categorical | 0: Typical, 1: Atypical, 2: Non-anginal, 3: Asymptomatic | Subtype of chest pain reported. |
| **resting_bp** | `trestbps` | Numerical | mmHg (80–200) | Resting blood pressure at hospital admission. |
| **cholesterol** | `chol` | Numerical | mg/dL (85–603) | Serum cholesterol measurement. |
| **fasting_bs** | `fbs` | Binary | 0: No, 1: Yes (>120 mg/dL) | Fasting blood sugar elevation (>120 mg/dL). |
| **resting_ecg** | `restecg` | Categorical | 0: Normal, 1: ST-T Wave Abnormality, 2: LV Hypertrophy | Resting electrocardiographic results. |
| **max_heart_rate** | `thalach` | Numerical | bpm (60–202) | Maximum heart rate achieved during stress test. |
| **exercise_angina** | `exang` | Binary | 0: No, 1: Yes | Exercise-induced angina pectoris. |
| **st_depression** | `oldpeak` | Numerical | mm (-2.6–6.2) | ST depression induced by exercise relative to rest. |
| **st_slope** | `slope` | Categorical | 1: Upsloping, 2: Flat, 3: Downsloping | Slope of peak exercise ST segment. |
| **num_major_vessels** | `ca` | Numerical | 0 to 3 vessels | Number of major vessels colored by fluoroscopy. |
| **thalassemia** | `thal` | Categorical | 3: Normal, 6: Fixed Defect, 7: Reversable Defect | Thalassemia blood condition status. |

---

## 🎯 Target Definition

* **`0`**: **Low Model-Estimated Risk** (Absence of coronary artery disease, <50% diameter narrowing)
* **`1`**: **Elevated Model-Estimated Risk** (Presence of coronary artery disease, >50% diameter narrowing)

---

## ⚠️ Academic & Medical Disclaimer

> This dataset and associated models represent an academic Explainable AI prototype developed for research and educational demonstrations. It is not intended for clinical diagnosis or direct patient management.
