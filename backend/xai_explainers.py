import numpy as np
import pandas as pd
import torch
import shap
import lime
import lime.lime_tabular

class XAIExplainerEngine:
    def __init__(self, model, scaler, feature_names):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.model.eval()

    def _predict_tabular_prob(self, X_tabular):
        """
        Wrapper prediction function for SHAP and LIME tabular explainers.
        Supplies background average zero ECG signal when perturbing tabular parameters.
        """
        X_scaled = self.scaler.transform(X_tabular)
        X_clin_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        
        # Zero baseline ECG tensor
        dummy_ecg = torch.zeros((len(X_tabular), 1, 1000), dtype=torch.float32)
        
        with torch.no_grad():
            _, probs, _, _ = self.model(X_clin_tensor, dummy_ecg)
            
        p = probs.cpu().numpy() # shape (N, 1)
        return np.hstack([1.0 - p, p]) # Return probability for [Low Risk, High Risk]

    def compute_shap(self, df_sample, df_background):
        """
        Computes SHAP feature importance for clinical tabular attributes.
        """
        background = df_background[self.feature_names].values[:50]
        sample = df_sample[self.feature_names].values
        
        explainer = shap.KernelExplainer(self._predict_tabular_prob, background)
        shap_values = explainer.shap_values(sample, nsamples=100)
        
        # Binary classification -> take index 1 (High Risk class)
        if isinstance(shap_values, list):
            sv = shap_values[1]
        else:
            sv = shap_values

        base_val = float(explainer.expected_value[1]) if isinstance(explainer.expected_value, (list, np.ndarray)) else float(explainer.expected_value)

        feature_contribs = []
        for i, col in enumerate(self.feature_names):
            feature_contribs.append({
                "feature": col,
                "value": float(sample[0, i]),
                "shap_value": float(sv[0, i])
            })
            
        # Sort by absolute impact
        feature_contribs = sorted(feature_contribs, key=lambda x: abs(x["shap_value"]), reverse=True)

        return {
            "base_value": base_val,
            "features": feature_contribs
        }

    def compute_lime(self, df_sample, df_background):
        """
        Computes LIME local tabular feature boundary explanation.
        """
        background = df_background[self.feature_names].values
        sample = df_sample[self.feature_names].values[0]

        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=background,
            feature_names=self.feature_names,
            class_names=['Low Risk', 'Elevated Risk'],
            mode='classification'
        )

        exp = explainer.explain_instance(sample, self._predict_tabular_prob, num_features=len(self.feature_names))
        as_list = exp.as_list()

        results = []
        for rule, weight in as_list:
            results.append({
                "rule": rule,
                "weight": float(weight)
            })

        return {
            "prediction_probability": float(exp.predict_proba[1]),
            "intercept": float(exp.intercept[1]),
            "rules": results
        }

    def compute_modality_ablation(self, x_clin_scaled, x_ecg_norm):
        """
        Computes non-arbitrary Modality Contribution via output probability drop:
        P_fused = P(x_clin, x_ecg)
        P_no_clin = P(0_clin, x_ecg)
        P_no_ecg = P(x_clin, 0_ecg)
        """
        x_clin_t = torch.tensor(x_clin_scaled, dtype=torch.float32).unsqueeze(0)
        x_ecg_t = torch.tensor(x_ecg_norm, dtype=torch.float32).unsqueeze(0)
        zero_clin_t = torch.zeros_like(x_clin_t)
        zero_ecg_t = torch.zeros_like(x_ecg_t)

        with torch.no_grad():
            _, p_fused, _, _ = self.model(x_clin_t, x_ecg_t)
            _, p_no_clin, _, _ = self.model(zero_clin_t, x_ecg_t)
            _, p_no_ecg, _, _ = self.model(x_clin_t, zero_ecg_t)

        pf = float(p_fused[0, 0].item())
        p_nc = float(p_no_clin[0, 0].item())
        p_ne = float(p_no_ecg[0, 0].item())

        impact_clin = abs(pf - p_nc)
        impact_ecg = abs(pf - p_ne)

        total = impact_clin + impact_ecg + 1e-6
        pct_clin = round((impact_clin / total) * 100.0, 1)
        pct_ecg = round((impact_ecg / total) * 100.0, 1)

        return {
            "fused_probability": pf,
            "clinical_only_probability": p_ne,
            "ecg_only_probability": p_nc,
            "clinical_impact_pct": pct_clin,
            "ecg_impact_pct": pct_ecg
        }
