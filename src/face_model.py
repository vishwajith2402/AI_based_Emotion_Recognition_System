"""
Facial Emotion Recognition Model Module
AGB1303 - AI Problem Solving Techniques
Architecture: Deep Convolutional Neural Network (CNN) with Batch Normalization & Dropout.
"""

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


class FacialEmotionModel:
    """Wrapper class managing loading, inference, and predictions for Facial CNN."""

    def __init__(self, model_path: str = None, device: str = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = FacialCNN(num_classes=NUM_CLASSES).to(self.device)
        self.model_path = Path(model_path) if model_path else FACE_MODEL_PATH
        self.is_loaded = False

        if self.model_path.exists():
            self.load_weights(self.model_path)
        else:
            print(f"[Info] Facial weights not found at {self.model_path}. Model initialized with default weights.")

    def load_weights(self, path: Path):
        """Loads model state dictionary safely."""
        try:
            state_dict = torch.load(path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.is_loaded = True
            print(f"[Facial Model] Successfully loaded weights from {path}")
        except Exception as e:
            print(f"[Error] Failed to load facial model weights: {e}")

    def save_weights(self, path: Path = None):
        """Saves current model state dictionary."""
        save_path = path or self.model_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        print(f"[Facial Model] Saved weights to {save_path}")

    @torch.no_grad()
    def predict(self, face_tensor: torch.Tensor) -> Tuple[str, float, Dict[str, float]]:
        """
        Runs inference on face tensor of shape (1, 1, 48, 48).
        Returns:
            top_emotion: str
            confidence: float (0.0 to 1.0)
            probabilities: dict mapping emotion -> prob
        """
        self.model.eval()
        face_tensor = face_tensor.to(self.device)
        logits = self.model(face_tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        prob_dict = {EMOTION_CLASSES[i]: float(probs[i]) for i in range(NUM_CLASSES)}
        top_idx = int(probs.argmax())
        top_emotion = EMOTION_CLASSES[top_idx]
        confidence = float(probs[top_idx])

        return top_emotion, confidence, prob_dict

    @torch.no_grad()
    def extract_embedding(self, face_tensor: torch.Tensor) -> torch.Tensor:
        """Extracts penultimate 256-d feature vector."""
        self.model.eval()
        face_tensor = face_tensor.to(self.device)
        return self.model.forward_features(face_tensor)
