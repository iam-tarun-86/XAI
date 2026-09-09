# 🍇 Wine Recognition Dataset Overview

This directory contains the dataset used for training the Random Forest Classifier and generating Explainable AI (SHAP & LIME) attributions.

---

## 📁 Files in this Directory

* **`wine_dataset.csv`**: The complete tabular dataset exported as CSV (178 samples, 13 features + target variables).
* **`README.md`**: Dataset documentation and feature descriptions.

---

## 📊 Dataset Metadata

* **Total Samples**: 178 wine instances
* **Total Features**: 13 continuous chemical properties
* **Classes**: 3 wine cultivars (`class_0`, `class_1`, `class_2`)
* **Problem Framing**: Binary classification (`Wine Type 0` vs. `Other Wine Types`) for SHAP & LIME local model explanations.

---

## 🧪 Chemical Feature Descriptions

| Feature Name | Description | Units / Scale |
|---|---|---|
| **alcohol** | Alcohol content in the wine sample | Percentage (%) |
| **malic_acid** | Amount of Malic Acid | g/L |
| **ash** | Quantity of inorganic ash content | g/L |
| **alcalinity_of_ash** | Alcalinity level of the ash | mEq/L |
| **magnesium** | Magnesium concentration | mg/L |
| **total_phenols** | Total polyphenols present | Absorbance / Concentration |
| **flavanoids** | Flavanoid polyphenols | Absorbance / Concentration |
| **nonflavanoid_phenols** | Non-flavanoid polyphenols | Concentration |
| **proanthocyanins** | Proanthocyanins level | Concentration |
| **color_intensity** | Visual color intensity level | Color intensity index |
| **hue** | Hue ratio of the wine color | Ratio |
| **od280/od315_of_diluted_wines** | Protein content measure (OD280/OD315 ratio) | Ratio |
| **proline** | Proline amino acid concentration | mg/L |

---

## 🎯 Target Labels

* **`0` (`class_0`)**: Wine Type 0 (59 samples)
* **`1` (`class_1`)**: Wine Type 1 (71 samples)
* **`2` (`class_2`)**: Wine Type 2 (48 samples)

In the XAI model dashboard:
* **Target = 0**: `Wine Type 0`
* **Target = 1**: `Other Wine Types` (`class_1` + `class_2`)
