"""
Facial Emotion Recognition Model Module
AGB1303 - AI Problem Solving Techniques
Architecture: Deep Convolutional Neural Network (CNN) with Batch Normalization & Dropout.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Tuple
from src.config import NUM_CLASSES, EMOTION_CLASSES, FACE_MODEL_PATH


class FacialCNN(nn.Module):
    """
    Deep CNN for Facial Emotion Recognition.
    Processes 48x48 single-channel normalized face images.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super(FacialCNN, self).__init__()
        
        # Block 1: 48x48 -> 24x24
        self.conv1_1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout2d(0.25)

        # Block 2: 24x24 -> 12x12
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout2d(0.25)

        # Block 3: 12x12 -> 6x6
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop3 = nn.Dropout2d(0.25)

        # Fully Connected Classification Head
        self.fc1 = nn.Linear(128 * 6 * 6, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.drop_fc1 = nn.Dropout(0.50)
        self.fc_out = nn.Linear(256, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts 256-dimensional penultimate representation."""
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = self.drop1(self.pool1(x))

        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = self.drop2(self.pool2(x))

        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = self.drop3(self.pool3(x))

        x = x.view(x.size(0), -1)
        features = F.relu(self.bn_fc1(self.fc1(x)))
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass returning raw logits."""
        features = self.forward_features(x)
        features_dropped = self.drop_fc1(features)
        logits = self.fc_out(features_dropped)
        return logits


class VGGFERNetwork(nn.Module):
    """
    VGG-inspired Deep Convolutional Network for FER (Enes Ozturk VGG19/FER-2013 adaptation).
    Features 4 convolutional blocks with double 3x3 convs, batch normalization,
    max-pooling and dual dropout for high-capacity facial feature representations.
    Input: (B, 1, 48, 48) -> Output: (B, num_classes)
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super(VGGFERNetwork, self).__init__()

        # VGG Block 1: 48x48 -> 24x24 (32 channels)
        self.conv1_1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout2d(0.20)

        # VGG Block 2: 24x24 -> 12x12 (64 channels)
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout2d(0.25)

        # VGG Block 3: 12x12 -> 6x6 (128 channels)
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop3 = nn.Dropout2d(0.30)

        # VGG Block 4: 6x6 -> 3x3 (256 channels)
        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop4 = nn.Dropout2d(0.35)

        # VGG Dense Classification Head: 256 * 3 * 3 = 2304 -> 512 -> 256 -> num_classes
        self.fc1 = nn.Linear(256 * 3 * 3, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.drop_fc1 = nn.Dropout(0.50)

        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.drop_fc2 = nn.Dropout(0.40)

        self.fc_out = nn.Linear(256, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts 256-dimensional penultimate representation."""
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = self.drop1(self.pool1(x))

        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = self.drop2(self.pool2(x))

        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = self.drop3(self.pool3(x))

        x = F.relu(self.bn4_1(self.conv4_1(x)))
        x = F.relu(self.bn4_2(self.conv4_2(x)))
        x = self.drop4(self.pool4(x))

        x = x.view(x.size(0), -1)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.drop_fc1(x)
        features = F.relu(self.bn_fc2(self.fc2(x)))
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass returning raw logits."""
        features = self.forward_features(x)
        features_dropped = self.drop_fc2(features)
        logits = self.fc_out(features_dropped)
        return logits


class SqueezeExcitationBlock(nn.Module):
    """Channel attention block adaptively recalibrating feature maps."""

    def __init__(self, channels: int, reduction: int = 8):
        super(SqueezeExcitationBlock, self).__init__()
        self.fc1 = nn.Linear(channels, max(8, channels // reduction), bias=False)
        self.fc2 = nn.Linear(max(8, channels // reduction), channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MultiScaleAffectNet(nn.Module):
    """
    SOTA Multi-Scale Dual-Branch AffectNet Architecture with Localized Patch Attention.
    Branch 1: Global Context Trunk (full 48x48 normalized facial frame).
    Branch 2: Localized Salient Patch Stream (Upper Eye-Brow 24x48 + Lower Mouth 24x48).
    Cross-Feature Fusion: Squeeze-and-Excitation Channel Attention.
    Dual Multi-Task Output Heads:
      - Head 1: Categorical Emotion Class Logits (5 Classes)
      - Head 2: Continuous 2D Valence-Arousal Coordinates (Russell's Circumplex [-1.0, +1.0]^2)
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super(MultiScaleAffectNet, self).__init__()

        # Branch 1: Global Context Trunk (48x48 -> 24x24 -> 12x12 -> 6x6)
        self.global_conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 24x24
            nn.Dropout2d(0.15)
        )
        self.global_conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 12x12
            nn.Dropout2d(0.20)
        )
        self.global_conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 6x6
            SqueezeExcitationBlock(128),
            nn.Dropout2d(0.25)
        )

        # Branch 2: Localized Salient Patch Feature Extractor
        # Processes upper face crop (eyes/brows) and lower face crop (mouth)
        self.patch_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((6, 6)),
            SqueezeExcitationBlock(64),
            nn.Dropout2d(0.20)
        )

        # Multi-Scale Feature Fusion: Global (128x6x6) + Eye (64x6x6) + Mouth (64x6x6) = 256x6x6
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(128 + 64 + 64, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            SqueezeExcitationBlock(256),
            nn.AdaptiveAvgPool2d((3, 3)),  # 256x3x3 = 2304
            nn.Dropout2d(0.30)
        )

        # Shared Latent Representation
        self.fc_shared = nn.Sequential(
            nn.Linear(256 * 3 * 3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.40),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30)
        )

        # Dual Output Heads:
        # Head A: 5-Class Categorical Emotion Logits
        self.fc_emotion = nn.Linear(256, num_classes)

        # Head B: Continuous Valence-Arousal Regression Head [-1.0, 1.0]
        self.fc_va = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
            nn.Tanh()
        )

    def extract_patches(self, x: torch.Tensor):
        """Splits full face tensor into Upper (Eye-Brow) and Lower (Mouth) patches."""
        h = x.size(2)
        half_h = h // 2
        upper_patch = x[:, :, :half_h, :]  # Upper 50%
        lower_patch = x[:, :, half_h:, :]  # Lower 50%
        return upper_patch, lower_patch

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts fused 256-d penultimate representation."""
        # Global Branch
        g1 = self.global_conv1(x)
        g2 = self.global_conv2(g1)
        g_feat = self.global_conv3(g2)  # (B, 128, 6, 6)

        # Local Patch Branch
        upper_p, lower_p = self.extract_patches(x)
        upper_feat = self.patch_conv(upper_p)  # (B, 64, 6, 6)
        lower_feat = self.patch_conv(lower_p)  # (B, 64, 6, 6)

        # Multi-Branch Attention Fusion
        fused = torch.cat([g_feat, upper_feat, lower_feat], dim=1)  # (B, 256, 6, 6)
        fused_pooled = self.fusion_conv(fused)  # (B, 256, 3, 3)
        flat = fused_pooled.view(fused_pooled.size(0), -1)
        latent = self.fc_shared(flat)  # (B, 256)
        return latent

    def forward(self, x: torch.Tensor, return_va: bool = False):
        latent = self.forward_features(x)
        logits = self.fc_emotion(latent)
        if return_va:
            va = self.fc_va(latent)
            return logits, va
        return logits


# Russell's Circumplex Model Affective Coordinates: (Valence, Arousal) in [-1.0, +1.0]^2
CIRCUMPLEX_EMOTION_ANCHORS = {
    "Happy": (0.82, 0.45),      # High Pleasure, Moderately Activated
    "Surprise": (0.35, 0.88),   # Slight Pleasure, Highly Activated
    "Angry": (-0.72, 0.78),     # Unpleasant, Highly Activated
    "Sad": (-0.85, -0.62),      # Unpleasant, Deactivated
    "Neutral": (0.00, 0.05)     # Balanced Baseline
}


def compute_circumplex_coordinates(probs: Dict[str, float]) -> Tuple[float, float]:
    """
    Computes expected continuous Valence-Arousal coordinates on Russell's Circumplex
    from an emotion probability distribution.
    """
    v_sum = sum(probs.get(e, 0.0) * CIRCUMPLEX_EMOTION_ANCHORS[e][0] for e in CIRCUMPLEX_EMOTION_ANCHORS)
    a_sum = sum(probs.get(e, 0.0) * CIRCUMPLEX_EMOTION_ANCHORS[e][1] for e in CIRCUMPLEX_EMOTION_ANCHORS)
    return float(np.clip(v_sum, -1.0, 1.0)), float(np.clip(a_sum, -1.0, 1.0))


class FacialEmotionModel:
    """Wrapper class managing loading, inference, and predictions for Facial CNN / VGG-FER / MultiScaleAffectNet."""

    def __init__(self, model_path: str = None, device: str = None, model_type: str = "multiscale_affectnet"):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_type = model_type
        if model_type == "multiscale_affectnet":
            self.model = MultiScaleAffectNet(num_classes=NUM_CLASSES).to(self.device)
        elif model_type == "vgg":
            self.model = VGGFERNetwork(num_classes=NUM_CLASSES).to(self.device)
        else:
            self.model = FacialCNN(num_classes=NUM_CLASSES).to(self.device)

        self.model_path = Path(model_path) if model_path else FACE_MODEL_PATH
        self.is_loaded = False

        if self.model_path.exists():
            self.load_weights(self.model_path)
        else:
            print(f"[Info] Facial weights not found at {self.model_path}. Model initialized with default weights.")

    def load_weights(self, path: Path):
        """Loads model state dictionary safely with architecture auto-detection."""
        try:
            state_dict = torch.load(path, map_location=self.device, weights_only=True)
            # Detect architecture:
            if "global_conv1.0.weight" in state_dict:
                if not isinstance(self.model, MultiScaleAffectNet):
                    self.model = MultiScaleAffectNet(num_classes=NUM_CLASSES).to(self.device)
                self.model_type = "multiscale_affectnet"
            elif "conv4_1.weight" in state_dict:
                if not isinstance(self.model, VGGFERNetwork):
                    self.model = VGGFERNetwork(num_classes=NUM_CLASSES).to(self.device)
                self.model_type = "vgg"
            else:
                if not isinstance(self.model, FacialCNN):
                    self.model = FacialCNN(num_classes=NUM_CLASSES).to(self.device)
                self.model_type = "cnn"

            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.is_loaded = True
            print(f"[Facial Model] Successfully loaded {self.model_type.upper()} weights from {path}")
        except Exception as e:
            print(f"[Error] Failed to load facial model weights: {e}")

    def save_weights(self, path: Path = None):
        """Saves current model state dictionary."""
        save_path = path or self.model_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        print(f"[Facial Model] Saved weights to {save_path}")

    @torch.no_grad()
    def predict(self, face_tensor: torch.Tensor, return_va: bool = False):
        """
        Runs inference on face tensor of shape (1, 1, 48, 48).
        Returns:
            top_emotion: str
            confidence: float (0.0 to 1.0)
            probabilities: dict mapping emotion -> prob
            (valence, arousal): optional tuple if return_va=True
        """
        self.model.eval()
        face_tensor = face_tensor.to(self.device)

        if isinstance(self.model, MultiScaleAffectNet):
            logits, va_coords = self.model(face_tensor, return_va=True)
            va = (float(va_coords[0, 0].cpu().item()), float(va_coords[0, 1].cpu().item()))
        else:
            logits = self.model(face_tensor)
            va = None

        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        prob_dict = {EMOTION_CLASSES[i]: float(probs[i]) for i in range(NUM_CLASSES)}
        top_idx = int(probs.argmax())
        top_emotion = EMOTION_CLASSES[top_idx]
        confidence = float(probs[top_idx])

        if va is None:
            va = compute_circumplex_coordinates(prob_dict)

        if return_va:
            return top_emotion, confidence, prob_dict, va
        return top_emotion, confidence, prob_dict

    def get_valence_arousal(self, prob_dict: Dict[str, float]) -> Tuple[float, float]:
        """Utility to retrieve (valence, arousal) from predicted probability distribution."""
        return compute_circumplex_coordinates(prob_dict)

    @torch.no_grad()
    def extract_embedding(self, face_tensor: torch.Tensor) -> torch.Tensor:
        """Extracts penultimate 256-d feature vector."""
        self.model.eval()
        face_tensor = face_tensor.to(self.device)
        return self.model.forward_features(face_tensor)
