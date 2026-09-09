import os
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
import shap
import lime
import lime.lime_tabular

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

app = Flask(__name__)
CORS(app)

# Global model pointers and dataset references
rf_model = None
nn_model = None
mlp_model = None
scaler = None
df_full = None
X_train, X_test, y_train, y_test = None, None, None, None
feature_names = []
class_names = ['Low Risk', 'Elevated Risk']

shap_explainer = None
shap_values_class1 = None
lime_explainer = None

if HAS_TORCH:
    class TabularNN(nn.Module):
        def __init__(self, input_dim):
            super(TabularNN, self).__init__()
            self.fc1 = nn.Linear(input_dim, 32)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(32, 16)
            self.relu2 = nn.ReLU()
            self.out = nn.Linear(16, 2)
            
        def forward(self, x):
            x = self.relu1(self.fc1(x))
            x = self.relu2(self.fc2(x))
            return self.out(x)

def compute_tabular_gradcam(input_array, target_class=1):
    """Computes gradient-based feature attributions (Tabular Grad-CAM) on the Neural Network."""
    if scaler is None:
        return [0.0] * len(feature_names)

    scaled_input = scaler.transform(input_array.reshape(1, -1))

    if HAS_TORCH and nn_model is not None:
        nn_model.eval()
        tensor_input = torch.tensor(scaled_input, dtype=torch.float32, requires_grad=True)
        output = nn_model(tensor_input)
        score = output[0, target_class]
        nn_model.zero_grad()
        score.backward()
        grad_abs = tensor_input.grad.abs().squeeze(0).detach().numpy()
    elif mlp_model is not None:
        # Finite difference gradient approximation for MLP representation
        eps = 1e-4
        grad_abs = np.zeros(len(feature_names))
        base_prob = mlp_model.predict_proba(scaled_input)[0, target_class]
        for i in range(len(feature_names)):
            perturbed = scaled_input.copy()
            perturbed[0, i] += eps
            prob_plus = mlp_model.predict_proba(perturbed)[0, target_class]
            grad_abs[i] = abs(prob_plus - base_prob) / eps
    else:
        grad_abs = np.zeros(len(feature_names))

    max_val = grad_abs.max()
    norm_grad = (grad_abs / max_val).tolist() if max_val > 0 else grad_abs.tolist()
    return norm_grad

def init_pipeline(n_estimators=100, max_depth=None):
    global rf_model, nn_model, mlp_model, scaler, df_full, X_train, X_test, y_train, y_test
    global feature_names, shap_explainer, shap_values_class1, lime_explainer

    csv_path = os.path.join('dataset', 'heart_dataset.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset missing at {csv_path}.")

    df_full = pd.read_csv(csv_path)
    feature_names = [c for c in df_full.columns if c != 'heart_attack_risk']
    
    X = df_full[feature_names].values
    y = df_full['heart_attack_risk'].values

    # Stratified Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 1. Train Random Forest Classifier
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    rf_model.fit(X_train, y_train)

    # 2. Train Neural Network for Tabular Grad-CAM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    if HAS_TORCH:
        nn_model = TabularNN(len(feature_names))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(nn_model.parameters(), lr=0.005)

        X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long)

        nn_model.train()
        for _ in range(80):
            optimizer.zero_grad()
            outputs = nn_model(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()
    else:
        mlp_model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
        mlp_model.fit(X_train_scaled, y_train)

    # 3. Setup SHAP Explainer
    shap_explainer = shap.TreeExplainer(rf_model)
    shap_vals = shap_explainer.shap_values(X_test)
    if isinstance(shap_vals, list):
        shap_values_class1 = shap_vals[1]
    elif shap_vals.ndim == 3:
        shap_values_class1 = shap_vals[:, :, 1]
    else:
        shap_values_class1 = shap_vals

    # 4. Setup LIME Explainer
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=class_names,
        mode='classification',
        random_state=42
    )

# Run pipeline initialization
init_pipeline()

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    y_pred = rf_model.predict(X_test)
    y_prob = rf_model.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    auc = float(roc_auc_score(y_test, y_prob))
    cm = confusion_matrix(y_test, y_pred).tolist()

    # Young Adult (18-40) statistics
    df_young = df_full[df_full['age'] <= 40]
    young_indices = np.where(X_test[:, feature_names.index('age')] <= 40)[0]
    
    if len(young_indices) > 0:
        y_test_young = y_test[young_indices]
        y_pred_young = y_pred[young_indices]
        young_acc = float(accuracy_score(y_test_young, y_pred_young))
        young_f1 = float(f1_score(y_test_young, y_pred_young, zero_division=0))
    else:
        young_acc, young_f1 = acc, f1

    return jsonify({
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": auc,
        "confusion_matrix": cm,
        "n_estimators": rf_model.n_estimators,
        "max_depth": str(rf_model.max_depth),
        "feature_names": feature_names,
        "class_names": class_names,
        "test_size": len(X_test),
        "train_size": len(X_train),
        "total_dataset_size": len(df_full),
        "young_adult_total": len(df_young),
        "young_adult_test_count": len(young_indices),
        "young_adult_accuracy": young_acc,
        "young_adult_f1": young_f1,
        "young_adult_pos_rate": float((df_young['heart_attack_risk'] == 1).mean()),
        "has_pytorch": HAS_TORCH
    })

@app.route('/api/dataset', methods=['GET'])
def get_dataset():
    samples = []
    y_pred = rf_model.predict(X_test)
    y_prob = rf_model.predict_proba(X_test)

    for i in range(len(X_test)):
        sample = {feature_names[j]: float(X_test[i][j]) for j in range(len(feature_names))}
        sample["id"] = i
        sample["true_label"] = int(y_test[i])
        sample["pred_label"] = int(y_pred[i])
        sample["confidence"] = float(y_prob[i][y_pred[i]])
        sample["risk_prob"] = float(y_prob[i][1])
        sample["is_young_adult"] = bool(X_test[i][feature_names.index('age')] <= 40)
        samples.append(sample)

    return jsonify(samples)

@app.route('/api/dataset/young-adults', methods=['GET'])
def get_young_adults_dataset():
    all_samples = get_dataset().json
    young_samples = [s for s in all_samples if s["is_young_adult"]]
    return jsonify(young_samples)

@app.route('/api/explain/shap-global', methods=['GET'])
def get_shap_global():
    mean_shap = np.abs(shap_values_class1).mean(axis=0)
    data = [
        {"feature": feature_names[i], "importance": float(mean_shap[i])}
        for i in range(len(feature_names))
    ]
    data = sorted(data, key=lambda x: x["importance"], reverse=True)
    return jsonify(data)

@app.route('/api/explain/local/<int:idx>', methods=['GET'])
def get_local_explanation(idx):
    if idx < 0 or idx >= len(X_test):
        return jsonify({"error": "Index out of range"}), 400

    sample = X_test[idx]
    y_prob = rf_model.predict_proba(sample.reshape(1, -1))[0]

    # SHAP local
    base_val = float(
        shap_explainer.expected_value[1]
        if isinstance(shap_explainer.expected_value, (list, np.ndarray))
        else shap_explainer.expected_value
    )
    shap_vals = shap_values_class1[idx]
    shap_local = [
        {
            "feature": feature_names[i],
            "value": float(sample[i]),
            "shap_value": float(shap_vals[i])
        }
        for i in range(len(feature_names))
    ]
    shap_local = sorted(shap_local, key=lambda x: abs(x["shap_value"]), reverse=True)

    # LIME local
    lime_exp = lime_explainer.explain_instance(
        data_row=sample,
        predict_fn=rf_model.predict_proba,
        num_features=10,
        labels=[1]
    )
    lime_local = [
        {"feature_rule": item[0], "weight": float(item[1])}
        for item in lime_exp.as_list(label=1)
    ]

    # Grad-CAM local
    gradcam_weights = compute_tabular_gradcam(sample, target_class=1)
    gradcam_local = [
        {"feature": feature_names[i], "importance": float(gradcam_weights[i]), "value": float(sample[i])}
        for i in range(len(feature_names))
    ]
    gradcam_local = sorted(gradcam_local, key=lambda x: x["importance"], reverse=True)

    return jsonify({
        "index": idx,
        "true_label": int(y_test[idx]),
        "pred_label": int(np.argmax(y_prob)),
        "risk_prob": float(y_prob[1]),
        "probs": [float(y_prob[0]), float(y_prob[1])],
        "is_young_adult": bool(sample[feature_names.index('age')] <= 40),
        "shap": {
            "base_value": base_val,
            "prediction_value": float(base_val + sum(shap_vals)),
            "features": shap_local
        },
        "lime": lime_local,
        "gradcam": gradcam_local
    })

@app.route('/api/explain/lime/<int:idx>', methods=['GET'])
def get_lime_explanation(idx):
    data = get_local_explanation(idx).json
    return jsonify(data["lime"])

@app.route('/api/explain/gradcam/<int:idx>', methods=['GET'])
def get_gradcam_explanation(idx):
    data = get_local_explanation(idx).json
    return jsonify(data["gradcam"])

@app.route('/api/explain/all/<int:idx>', methods=['GET'])
def get_all_explanations(idx):
    return get_local_explanation(idx)

@app.route('/api/explain/uncertain', methods=['GET'])
def get_uncertain_explanation():
    y_prob = rf_model.predict_proba(X_test)
    prob_dist_to_half = np.abs(y_prob[:, 1] - 0.5)
    hardest_idx = int(np.argmin(prob_dist_to_half))
    return get_local_explanation(hardest_idx)

@app.route('/api/predict', methods=['POST'])
def predict_custom_patient():
    data = request.json or {}
    sample_values = []
    for f in feature_names:
        sample_values.append(float(data.get(f, 0.0)))

    sample = np.array(sample_values)
    y_prob = rf_model.predict_proba(sample.reshape(1, -1))[0]
    pred_label = int(np.argmax(y_prob))
    risk_prob = float(y_prob[1])

    # SHAP for custom input
    custom_shap_vals = shap_explainer.shap_values(sample.reshape(1, -1))
    if isinstance(custom_shap_vals, list):
        c_shap = custom_shap_vals[1][0]
    elif custom_shap_vals.ndim == 3:
        c_shap = custom_shap_vals[0, :, 1]
    else:
        c_shap = custom_shap_vals[0]

    base_val = float(
        shap_explainer.expected_value[1]
        if isinstance(shap_explainer.expected_value, (list, np.ndarray))
        else shap_explainer.expected_value
    )

    shap_local = [
        {"feature": feature_names[i], "value": float(sample[i]), "shap_value": float(c_shap[i])}
        for i in range(len(feature_names))
    ]
    shap_local = sorted(shap_local, key=lambda x: abs(x["shap_value"]), reverse=True)

    # LIME for custom input
    lime_exp = lime_explainer.explain_instance(
        data_row=sample,
        predict_fn=rf_model.predict_proba,
        num_features=10,
        labels=[1]
    )
    lime_local = [
        {"feature_rule": item[0], "weight": float(item[1])}
        for item in lime_exp.as_list(label=1)
    ]

    # Grad-CAM for custom input
    gradcam_weights = compute_tabular_gradcam(sample, target_class=1)
    gradcam_local = [
        {"feature": feature_names[i], "importance": float(gradcam_weights[i]), "value": float(sample[i])}
        for i in range(len(feature_names))
    ]
    gradcam_local = sorted(gradcam_local, key=lambda x: x["importance"], reverse=True)

    return jsonify({
        "pred_label": pred_label,
        "risk_prob": risk_prob,
        "probs": [float(y_prob[0]), float(y_prob[1])],
        "is_young_adult": bool(sample[feature_names.index('age')] <= 40),
        "shap": {
            "base_value": base_val,
            "prediction_value": float(base_val + sum(c_shap)),
            "features": shap_local
        },
        "lime": lime_local,
        "gradcam": gradcam_local
    })

@app.route('/api/retrain', methods=['POST'])
def retrain_model():
    data = request.json or {}
    n_estimators = int(data.get("n_estimators", 100))
    max_depth_val = data.get("max_depth", None)
    max_depth = None
    if max_depth_val is not None and str(max_depth_val).lower() != "none" and max_depth_val != "":
        max_depth = int(max_depth_val)

    init_pipeline(n_estimators=n_estimators, max_depth=max_depth)
    return get_model_info()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
