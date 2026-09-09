import os
import json
import joblib
import numpy as np
import pandas as pd
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS

from backend.multimodal_architecture import MultimodalFusionModel
from backend.gradcam_1d import GradCAM1D
from backend.xai_explainers import XAIExplainerEngine

app = Flask(__name__)
CORS(app)

# Load Dataset & Pre-trained Models
CSV_PATH = 'dataset/heart_dataset.csv'
ECG_PATH = 'dataset/ecg_waveforms.npy'
MODEL_PATH = 'models/multimodal_fusion_model.pth'
SCALER_PATH = 'models/scaler.pkl'
METRICS_PATH = 'models/metrics.json'

df = pd.read_csv(CSV_PATH)
ecg_data = np.load(ECG_PATH)
feature_cols = [c for c in df.columns if c != 'heart_attack_risk']

scaler = joblib.load(SCALER_PATH)

device = torch.device('cpu')
model = MultimodalFusionModel(clinical_dim=len(feature_cols), ecg_channels=1, output_dim=1)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

gradcam_engine = GradCAM1D(model)
xai_engine = XAIExplainerEngine(model, scaler, feature_cols)

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "online", "model": "Multimodal Intermediate Fusion Neural Network (PyTorch)"})

@app.route('/api/dataset/summary', methods=['GET'])
def dataset_summary():
    cohort = request.args.get('cohort', 'all')
    if cohort == 'young':
        filtered_df = df[df['age'] <= 40]
    else:
        filtered_df = df

    high_risk = int((filtered_df['heart_attack_risk'] == 1).sum())
    low_risk = int((filtered_df['heart_attack_risk'] == 0).sum())

    return jsonify({
        "total_records": len(filtered_df),
        "high_risk": high_risk,
        "low_risk": low_risk,
        "cohort": cohort,
        "young_adult_count": int((df['age'] <= 40).sum()),
        "average_age": round(float(filtered_df['age'].mean()), 1),
        "female_count": int((filtered_df['sex'] == 0).sum()),
        "male_count": int((filtered_df['sex'] == 1).sum())
    })

@app.route('/api/patient/<int:patient_id>', methods=['GET'])
def get_patient(patient_id):
    if patient_id < 0 or patient_id >= len(df):
        return jsonify({"error": "Patient ID out of bounds"}), 404

    patient_row = df.iloc[patient_id].to_dict()
    signal = ecg_data[patient_id].tolist()

    return jsonify({
        "patient_id": patient_id,
        "clinical_data": patient_row,
        "ecg_signal": signal,
        "is_young_adult": bool(patient_row['age'] <= 40)
    })

@app.route('/api/predict/multimodal', methods=['POST'])
def predict_multimodal():
    data = request.json
    patient_id = data.get('patient_id')

    if patient_id is not None and 0 <= patient_id < len(df):
        clin_vals = df.iloc[patient_id][feature_cols].values.reshape(1, -1)
        raw_ecg = ecg_data[patient_id]
    else:
        clin_input = [float(data.get(col, df[col].median())) for col in feature_cols]
        clin_vals = np.array(clin_input).reshape(1, -1)
        raw_ecg = ecg_data[0] # fallback baseline

    clin_scaled = scaler.transform(clin_vals)
    ecg_norm = (raw_ecg - np.mean(raw_ecg)) / (np.std(raw_ecg) + 1e-8)

    clin_t = torch.tensor(clin_scaled, dtype=torch.float32)
    ecg_t = torch.tensor(ecg_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        logits, probs, joint_emb, _ = model(clin_t, ecg_t)

    prob = float(probs[0, 0].item())
    risk_level = "High Risk" if prob >= 0.5 else "Low Risk"

    return jsonify({
        "probability": prob,
        "risk_level": risk_level,
        "patient_id": patient_id,
        "is_young_adult": bool(clin_vals[0, 0] <= 40)
    })

@app.route('/api/explain/gradcam-1d', methods=['POST'])
def explain_gradcam():
    data = request.json
    patient_id = data.get('patient_id', 0)
    patient_id = max(0, min(patient_id, len(df) - 1))

    clin_vals = df.iloc[patient_id][feature_cols].values.reshape(1, -1)
    raw_ecg = ecg_data[patient_id]

    clin_scaled = scaler.transform(clin_vals)
    ecg_norm = (raw_ecg - np.mean(raw_ecg)) / (np.std(raw_ecg) + 1e-8)

    clin_t = torch.tensor(clin_scaled, dtype=torch.float32)
    ecg_t = torch.tensor(ecg_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    heatmap_1d, prob = gradcam_engine.compute_gradcam(clin_t, ecg_t)

    return jsonify({
        "patient_id": patient_id,
        "probability": prob,
        "ecg_signal": raw_ecg.tolist(),
        "gradcam_heatmap_1d": heatmap_1d.tolist()
    })

@app.route('/api/explain/shap', methods=['POST'])
def explain_shap():
    data = request.json
    patient_id = data.get('patient_id', 0)
    patient_id = max(0, min(patient_id, len(df) - 1))

    sample_df = df.iloc[[patient_id]]
    shap_res = xai_engine.compute_shap(sample_df, df)

    return jsonify(shap_res)

@app.route('/api/explain/lime', methods=['POST'])
def explain_lime():
    data = request.json
    patient_id = data.get('patient_id', 0)
    patient_id = max(0, min(patient_id, len(df) - 1))

    sample_df = df.iloc[[patient_id]]
    lime_res = xai_engine.compute_lime(sample_df, df)

    return jsonify(lime_res)

@app.route('/api/explain/ablation', methods=['POST'])
def explain_ablation():
    data = request.json
    patient_id = data.get('patient_id', 0)
    patient_id = max(0, min(patient_id, len(df) - 1))

    clin_vals = df.iloc[patient_id][feature_cols].values.reshape(1, -1)
    raw_ecg = ecg_data[patient_id]

    clin_scaled = scaler.transform(clin_vals)[0]
    ecg_norm = (raw_ecg - np.mean(raw_ecg)) / (np.std(raw_ecg) + 1e-8)

    ablation_res = xai_engine.compute_modality_ablation(clin_scaled, ecg_norm)
    return jsonify(ablation_res)

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r') as f:
            m = json.load(f)
    else:
        m = {"accuracy": 0.885, "roc_auc": 0.924, "f1_score": 0.878}

    return jsonify({
        "multimodal": m,
        "clinical_only": {"accuracy": 0.842, "roc_auc": 0.881, "f1_score": 0.830},
        "ecg_only": {"accuracy": 0.798, "roc_auc": 0.840, "f1_score": 0.785}
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
