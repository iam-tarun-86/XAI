import torch
import torch.nn as nn
import torch.nn.functional as F

class ClinicalEncoder(nn.Module):
    """
    MLP Encoder for 13 Tabular Clinical Features.
    Maps input vector (13) -> Dense(128) -> ReLU -> BatchNorm -> Dropout -> Dense(64).
    """
    def __init__(self, input_dim=13, hidden_dim=128, output_dim=64):
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
    1D Convolutional Neural Network Encoder for Time-Series 1D ECG Signals (T=1000).
    Features 3 conv stages with 1D spatial convolutions, BatchNorm1d, MaxPool1d,
    and Global Average Pooling.
    """
    def __init__(self, in_channels=1, output_dim=64):
        super(ECG1DCNNEncoder, self).__init__()
        # Conv Stage 1
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=15, stride=2, padding=7)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Conv Stage 2
        self.conv2 = nn.Conv1d(16, 32, kernel_size=9, stride=1, padding=4)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Conv Stage 3 (Target for 1D Grad-CAM)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm1d(64)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (B, 1, T) or (B, T)
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        c1 = self.relu(self.bn1(self.conv1(x)))
        p1 = self.pool1(c1)
        
        c2 = self.relu(self.bn2(self.conv2(p1)))
        p2 = self.pool2(c2)
        
        c3 = self.relu(self.bn3(self.conv3(p2)))
        
        g = self.global_pool(c3).squeeze(-1) # (B, 64)
        out = self.relu(self.fc(g)) # (B, output_dim)
        return out, c3 # Return feature embedding & 1D activation maps for Grad-CAM

class MultimodalFusionModel(nn.Module):
    """
    Feature-Level Multimodal Intermediate Fusion Architecture.
    Concatenates Clinical Embedding (64-d) + ECG Signal Embedding (64-d) -> Joint Embedding (128-d)
    followed by a non-linear Fusion Classification Head.
    """
    def __init__(self, clinical_dim=13, ecg_channels=1, output_dim=1):
        super(MultimodalFusionModel, self).__init__()
        self.clinical_encoder = ClinicalEncoder(input_dim=clinical_dim, output_dim=64)
        self.ecg_encoder = ECG1DCNNEncoder(in_channels=ecg_channels, output_dim=64)
        
        # Joint Fusion Classification Head
        self.fusion_fc1 = nn.Linear(64 + 64, 64)
        self.bn_fusion = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fusion_out = nn.Linear(64, output_dim)

    def forward(self, x_clin, x_ecg):
        h_clin = self.clinical_encoder(x_clin)
        h_ecg, c3_feature_maps = self.ecg_encoder(x_ecg)
        
        # Intermediate/Feature-Level Concatenation
        joint_emb = torch.cat([h_clin, h_ecg], dim=1) # (B, 128)
        
        f = self.relu(self.bn_fusion(self.fusion_fc1(joint_emb)))
        f = self.dropout(f)
        logits = self.fusion_out(f)
        probs = torch.sigmoid(logits)
        
        return logits, probs, joint_emb, c3_feature_maps
