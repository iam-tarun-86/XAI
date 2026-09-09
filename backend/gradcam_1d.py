import torch
import torch.nn.functional as F
import numpy as np

class GradCAM1D:
    """
    1D CNN Class Activation Mapping (Grad-CAM) for Time-Series ECG Waveforms.
    Calculates exact temporal gradient activation map:
    alpha_k = (1/T_map) * sum_{t} (partial y^c / partial A_t^k)
    L_{Grad-CAM}^{1D}(t) = ReLU( sum_{k} alpha_k * A_t^k )
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        
        # Hook into conv3 layer of ECG1DCNNEncoder
        target_layer = self.model.ecg_encoder.conv3
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def compute_gradcam(self, x_clin_tensor, x_ecg_tensor):
        """
        Computes 1D Grad-CAM heatmap normalized to [0, 1] aligned with original T=1000 ECG points.
        """
        self.model.zero_grad()
        
        # Enable gradients for backward pass
        logits, probs, joint_emb, c3_maps = self.model(x_clin_tensor, x_ecg_tensor)
        
        # Target logit score
        score = logits[0, 0]
        score.backward()

        # Gradients: (1, C, T_map), Activations: (1, C, T_map)
        grads = self.gradients[0]     # (64, T_map)
        acts = self.activations[0]     # (64, T_map)

        # Alpha weights = Mean gradient per channel across temporal dimension
        weights = torch.mean(grads, dim=1, keepdim=True) # (64, 1)

        # Weighted sum of feature maps
        cam_1d = torch.sum(weights * acts, dim=0) # (T_map,)
        cam_1d = F.relu(cam_1d)

        # Interpolate 1D CAM back to full 1D ECG signal duration (T=1000)
        cam_1d = cam_1d.unsqueeze(0).unsqueeze(0) # (1, 1, T_map)
        cam_1d_interp = F.interpolate(cam_1d, size=x_ecg_tensor.shape[-1], mode='linear', align_corners=False)
        cam_1d_interp = cam_1d_interp.squeeze().detach().cpu().numpy()

        # Min-Max Normalization to [0, 1]
        cam_min, cam_max = np.min(cam_1d_interp), np.max(cam_1d_interp)
        if cam_max > cam_min:
            cam_1d_norm = (cam_1d_interp - cam_min) / (cam_max - cam_min)
        else:
            cam_1d_norm = np.zeros_like(cam_1d_interp)

        return cam_1d_norm, float(probs[0, 0].detach().cpu().item())
