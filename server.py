from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import shap
import lime
import lime.lime_tabular

app = Flask(__name__)
# Enable CORS for frontend integration
CORS(app)

# Global variables for model, data, and explainers
model = None
X_train, X_test, y_train, y_test = None, None, None, None
feature_names = []
class_names = ['Wine Type 0', 'Other Wine Types']
lime_explainer = None
shap_explainer = None
shap_values_class1 = None

def init_dataset_and_model(n_estimators=100, max_depth=None):
    global model, X_train, X_test, y_train, y_test, feature_names, lime_explainer, shap_explainer, shap_values_class1
    
    # Load dataset
    wine = load_wine()
    df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
    feature_names = list(wine.feature_names)
    df['target'] = (wine.target > 0).astype(int)
    
    X = df[feature_names].values
    y = df['target'].values
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Initialize Random Forest Model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Setup Explainers
    shap_explainer = shap.TreeExplainer(model)
    shap_vals = shap_explainer.shap_values(X_test)
    
    # Handle SHAP dimensionality across different versions
    if isinstance(shap_vals, list):
        shap_values_class1 = shap_vals[1]
    elif shap_vals.ndim == 3:
        shap_values_class1 = shap_vals[:, :, 1]
    else:
        shap_values_class1 = shap_vals
        
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=class_names,
        mode='classification',
        random_state=42
    )

# Initial load
init_dataset_and_model()

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    return jsonify({
        "accuracy": acc,
        "f1_score": f1,
        "confusion_matrix": cm,
        "n_estimators": model.n_estimators,
        "max_depth": str(model.max_depth),
        "feature_names": feature_names,
        "class_names": class_names,
        "test_size": len(X_test),
        "train_size": len(X_train)
    })

@app.route('/api/dataset', methods=['GET'])
def get_dataset():
    # Return samples of test set for explorer
    samples = []
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    for i in range(len(X_test)):
        sample = {feature_names[j]: float(X_test[i][j]) for j in range(len(feature_names))}
        sample["id"] = i
        sample["true_label"] = int(y_test[i])
        sample["pred_label"] = int(y_pred[i])
        sample["confidence"] = float(y_prob[i][y_pred[i]])
        samples.append(sample)
        
    return jsonify(samples)

@app.route('/api/retrain', methods=['POST'])
def retrain_model():
    data = request.json or {}
    n_estimators = int(data.get("n_estimators", 100))
    max_depth_val = data.get("max_depth", None)
    
    max_depth = None
    if max_depth_val is not None and str(max_depth_val).lower() != "none" and max_depth_val != "":
        max_depth = int(max_depth_val)
        
    init_dataset_and_model(n_estimators=n_estimators, max_depth=max_depth)
    return get_model_info()

@app.route('/api/explain/shap-global', methods=['GET'])
def get_shap_global():
    # Return global feature importances based on SHAP values
    mean_shap = np.abs(shap_values_class1).mean(axis=0)
    data = [
        {"feature": feature_names[i], "importance": float(mean_shap[i])}
        for i in range(len(feature_names))
    ]
    # Sort by importance descending
    data = sorted(data, key=lambda x: x["importance"], reverse=True)
    return jsonify(data)

@app.route('/api/explain/local/<int:idx>', methods=['GET'])
def get_local_explanation(idx):
    if idx < 0 or idx >= len(X_test):
        return jsonify({"error": "Index out of range"}), 400
        
    sample = X_test[idx]
    y_prob = model.predict_proba(sample.reshape(1, -1))[0]
    
    # 1. SHAP local info (waterfall structure)
    # base_value is expected value of Class 1
    base_val = float(shap_explainer.expected_value[1] if isinstance(shap_explainer.expected_value, (list, np.ndarray)) else shap_explainer.expected_value)
    shap_vals = shap_values_class1[idx]
    
    shap_local = []
    for i in range(len(feature_names)):
        shap_local.append({
            "feature": feature_names[i],
            "value": float(sample[i]),
            "shap_value": float(shap_vals[i])
        })
    # Sort features by absolute contribution descending
    shap_local = sorted(shap_local, key=lambda x: abs(x["shap_value"]), reverse=True)
    
    # 2. LIME local info
    lime_exp = lime_explainer.explain_instance(
        data_row=sample,
        predict_fn=model.predict_proba,
        num_features=10,
        labels=[1]
    )
    # LIME weights for class 1
    lime_list = lime_exp.as_list(label=1)
    lime_local = [
        {"feature_rule": item[0], "weight": float(item[1])}
        for item in lime_list
    ]
    
    return jsonify({
        "index": idx,
        "true_label": int(y_test[idx]),
        "pred_label": int(np.argmax(y_prob)),
        "probs": [float(y_prob[0]), float(y_prob[1])],
        "shap": {
            "base_value": base_val,
            "prediction_value": float(base_val + sum(shap_vals)),
            "features": shap_local
        },
        "lime": lime_local
    })

@app.route('/api/explain/uncertain', methods=['GET'])
def get_uncertain_explanation():
    # Find sample closest to 0.5 probability of Class 1
    y_prob = model.predict_proba(X_test)
    prob_dist_to_half = np.abs(y_prob[:, 1] - 0.5)
    hardest_idx = int(np.argmin(prob_dist_to_half))
    
    return get_local_explanation(hardest_idx)

if __name__ == '__main__':
    # Run Flask server on port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
