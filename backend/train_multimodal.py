import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score

from backend.multimodal_architecture import MultimodalFusionModel

class MultimodalDataset(Dataset):
    def __init__(self, X_clin, X_ecg, y):
        self.X_clin = torch.tensor(X_clin, dtype=torch.float32)
        self.X_ecg = torch.tensor(X_ecg, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_clin[idx], self.X_ecg[idx], self.y[idx]

def train_and_save_model():
    csv_path = 'dataset/heart_dataset.csv'
    npy_path = 'dataset/ecg_waveforms.npy'

    df = pd.read_csv(csv_path)
    ecg = np.load(npy_path)

    feature_cols = [c for c in df.columns if c != 'heart_attack_risk']
    X_clin = df[feature_cols].values
    y = df['heart_attack_risk'].values

    # Train/Test Split
    X_clin_train, X_clin_test, X_ecg_train, X_ecg_test, y_train, y_test = train_test_split(
        X_clin, ecg, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale Tabular Clinical Features
    scaler = StandardScaler()
    X_clin_train_scaled = scaler.fit_transform(X_clin_train)
    X_clin_test_scaled = scaler.transform(X_clin_test)

    # Save Scaler
    joblib.dump(scaler, 'models/scaler.pkl')

    # Normalize ECG Signals per sample
    X_ecg_train_norm = (X_ecg_train - np.mean(X_ecg_train, axis=1, keepdims=True)) / (np.std(X_ecg_train, axis=1, keepdims=True) + 1e-8)
    X_ecg_test_norm = (X_ecg_test - np.mean(X_ecg_test, axis=1, keepdims=True)) / (np.std(X_ecg_test, axis=1, keepdims=True) + 1e-8)

    # DataLoaders
    train_dataset = MultimodalDataset(X_clin_train_scaled, X_ecg_train_norm, y_train)
    test_dataset = MultimodalDataset(X_clin_test_scaled, X_ecg_test_norm, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize PyTorch Multimodal Fusion Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultimodalFusionModel(clinical_dim=len(feature_cols), ecg_channels=1, output_dim=1).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    print("Training Multimodal Fusion Neural Network...")
    epochs = 40
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for b_clin, b_ecg, b_y in train_loader:
            b_clin, b_ecg, b_y = b_clin.to(device), b_ecg.to(device), b_y.to(device)
            optimizer.zero_grad()
            logits, probs, _, _ = model(b_clin, b_ecg)
            loss = criterion(logits, b_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(b_y)

        train_loss /= len(train_dataset)

    # Evaluation
    model.eval()
    all_preds, all_probs, all_targets = [], [], []
    with torch.no_grad():
        for b_clin, b_ecg, b_y in test_loader:
            b_clin, b_ecg = b_clin.to(device), b_ecg.to(device)
            logits, probs, _, _ = model(b_clin, b_ecg)
            preds = (probs >= 0.5).float()
            all_preds.extend(preds.cpu().numpy().flatten())
            all_probs.extend(probs.cpu().numpy().flatten())
            all_targets.extend(b_y.numpy().flatten())

    acc = accuracy_score(all_targets, all_preds)
    roc_auc = roc_auc_score(all_targets, all_probs)
    f1 = f1_score(all_targets, all_preds)

    print(f"Training Complete! Test Accuracy: {acc*100:.2f}%, ROC-AUC: {roc_auc:.4f}, F1-Score: {f1:.4f}")

    # Save PyTorch Model
    torch.save(model.state_dict(), 'models/multimodal_fusion_model.pth')

    # Save Metadata
    metrics = {
        "accuracy": float(acc),
        "roc_auc": float(roc_auc),
        "f1_score": float(f1),
        "precision": float(precision_score(all_targets, all_preds)),
        "recall": float(recall_score(all_targets, all_preds)),
        "train_records": len(X_clin_train),
        "test_records": len(X_clin_test)
    }

    import json
    with open('models/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

if __name__ == '__main__':
    train_and_save_model()
