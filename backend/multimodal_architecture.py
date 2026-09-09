import torch
import torch.nn as nn
import torch.nn.functional as F

class ClinicalEncoder(nn.Module):
    """
    MLP Encoder for Structured Patient Metadata (Age, Sex, Height, Weight).
    Input dim = 4 -> Dense(64) -> BatchNorm -> ReLU -> Dropout -> Dense(64).
    """
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=64):
        super(ClinicalEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        h = self.relu(self.bn1(self.fc1(x)))
        h = self.dropout(h)
        out = self.relu(self.bn2(self.fc2(h)))
        return out

class ECG1DCNNEncoder(nn.Module):
    """
    Multi-Lead (12-Channel) 1D Convolutional Neural Network for ECG Signals (T=1000).
    Input shape: (Batch, 12, 1000)
    Stage 1: Conv1D(12 -> 32, k=15, s=2) -> BatchNorm -> Pool
    Stage 2: Conv1D(32 -> 64, k=9, s=1) -> BatchNorm -> Pool
    Stage 3: Conv1D(64 -> 64, k=5, s=1) -> BatchNorm (Target for 1D Grad-CAM) -> AdaptiveAvgPool
    """
    def __init__(self, in_channels=12, output_dim=64):
        super(ECG1DCNNEncoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=15, stride=2, padding=7)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=9, stride=1, padding=4)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm1d(64)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (B, 12, 1000)
        c1 = self.relu(self.bn1(self.conv1(x)))
        p1 = self.pool1(c1)
        
        c2 = self.relu(self.bn2(self.conv2(p1)))
        p2 = self.pool2(c2)
        
        c3 = self.relu(self.bn3(self.conv3(p2)))
        
        g = self.global_pool(c3).squeeze(-1) # (B, 64)
        out = self.relu(self.fc(g)) # (B, 64)
        return out, c3

class ClinicalOnlyModel(nn.Module):
    """Baseline Clinical-Only MLP Model."""
    def __init__(self, input_dim=4, output_dim=1):
        super(ClinicalOnlyModel, self).__init__()
        self.encoder = ClinicalEncoder(input_dim=input_dim, output_dim=64)
        self.head = nn.Linear(64, output_dim)

    def forward(self, x_clin):
        emb = self.encoder(x_clin)
        logits = self.head(emb)
        probs = torch.sigmoid(logits)
        return logits, probs, emb

class ECGOnlyModel(nn.Module):
    """Baseline ECG-Only 1D CNN Model."""
    def __init__(self, in_channels=12, output_dim=1):
        super(ECGOnlyModel, self).__init__()
        self.encoder = ECG1DCNNEncoder(in_channels=in_channels, output_dim=64)
        self.head = nn.Linear(64, output_dim)

    def forward(self, x_ecg):
        emb, c3 = self.encoder(x_ecg)
        logits = self.head(emb)
        probs = torch.sigmoid(logits)
        return logits, probs, emb, c3

class MultimodalFusionModel(nn.Module):
    """
    Feature-Level Intermediate Multimodal Neural Fusion Architecture.
    Concatenates Clinical Embedding (64-d) + ECG Embedding (64-d) -> Joint Embedding (128-d)
    followed by a non-linear Fusion Classifier Head.
    """
    def __init__(self, clinical_dim=4, ecg_channels=12, output_dim=1):
        super(MultimodalFusionModel, self).__init__()
        self.clinical_encoder = ClinicalEncoder(input_dim=clinical_dim, output_dim=64)
        self.ecg_encoder = ECG1DCNNEncoder(in_channels=ecg_channels, output_dim=64)
        
        self.fusion_fc1 = nn.Linear(64 + 64, 64)
        self.bn_fusion = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fusion_out = nn.Linear(64, output_dim)

    def forward(self, x_clin, x_ecg):
        h_clin = self.clinical_encoder(x_clin)
        h_ecg, c3_feature_maps = self.ecg_encoder(x_ecg)
        
        joint_emb = torch.cat([h_clin, h_ecg], dim=1) # (B, 128)
        
        f = self.relu(self.bn_fusion(self.fusion_fc1(joint_emb)))
        f = self.dropout(f)
        logits = self.fusion_out(f)
        probs = torch.sigmoid(logits)
        
        return logits, probs, joint_emb, c3_feature_maps
