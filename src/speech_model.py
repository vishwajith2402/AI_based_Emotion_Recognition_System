"""
Speech Emotion Recognition Model Module
AGB1303 - AI Problem Solving Techniques
Architecture: Bidirectional Long Short-Term Memory (BiLSTM) with Temporal Pooling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from src.config import N_MFCC, SPEECH_FEATURE_DIM, NUM_CLASSES, EMOTION_CLASSES, SPEECH_MODEL_PATH


class SpeechBiLSTM(nn.Module):
    """
    Bidirectional LSTM Network for Speech Emotion Recognition.
    Processes sequential acoustic features of shape (B, T, 82) or (B, T, 40).
    Incorporates MFCCs, Zero Crossing Rate, RMS energy, and Mel Spectrogram filterbanks
    from Shivam Burnwal's Speech Emotion Recognition pipeline.
    """

    def __init__(
        self,
        input_dim: int = SPEECH_FEATURE_DIM,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.25
    ):
        super(SpeechBiLSTM, self).__init__()
        self.input_dim = input_dim
        
        # Projection layer: accommodates 82 multi-feature dimensions (with fallback for legacy 40)
        self.fc_in = nn.Linear(input_dim, 64)
        self.bn_in = nn.BatchNorm1d(64)
        
        # Optional adapter for legacy 40-dim MFCC tensors
        self.legacy_adapter = nn.Linear(N_MFCC, input_dim) if input_dim != N_MFCC else None
        
        # BiLSTM Layers
        self.bilstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # BiLSTM output dimension: hidden_dim * 2 (bidirectional) = 128
        bilstm_out_dim = hidden_dim * 2

        # Fully Connected Classification Head (pooling avg + max = 2 * bilstm_out_dim = 256)
        self.fc1 = nn.Linear(bilstm_out_dim * 2, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.35)
        self.fc_out = nn.Linear(128, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts 128-dimensional pooled acoustic embedding.
        x shape: (Batch, Time_Steps, Features) where Features is 82 or 40.
        """
        B, T, F_dim = x.size()
        
        # Adapt if tensor dimension differs from layer input_dim
        if F_dim != self.input_dim:
            if F_dim == N_MFCC and self.legacy_adapter is not None:
                x = self.legacy_adapter(x)
                F_dim = self.input_dim
            elif F_dim > self.input_dim:
                x = x[:, :, :self.input_dim]
                F_dim = self.input_dim
            elif F_dim < self.input_dim:
                pad = torch.zeros(B, T, self.input_dim - F_dim, device=x.device, dtype=x.dtype)
                x = torch.cat([x, pad], dim=-1)
                F_dim = self.input_dim

        # Project frame features: (B * T, F_dim) -> (B * T, 64)
        x_flat = x.view(-1, F_dim)
        x_proj = F.relu(self.bn_in(self.fc_in(x_flat)))
        x_seq = x_proj.view(B, T, -1)

        # BiLSTM processing: output shape (B, T, 128)
        lstm_out, _ = self.bilstm(x_seq)

        # Temporal Dual-Pooling (Average + Max across time)
        avg_pool = torch.mean(lstm_out, dim=1)           # (B, 128)
        max_pool, _ = torch.max(lstm_out, dim=1)          # (B, 128)
        pooled = torch.cat([avg_pool, max_pool], dim=1)  # (B, 256)

        # Penultimate layer representation
        embedding = F.relu(self.bn1(self.fc1(pooled)))    # (B, 128)
        return embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass returning raw class logits."""
        embedding = self.forward_features(x)
        embedding_dropped = self.drop1(embedding)
        logits = self.fc_out(embedding_dropped)
        return logits


class SpeechEmotionModel:
    """Wrapper class managing loading, inference, and predictions for Speech BiLSTM."""

    def __init__(self, model_path: str = None, device: str = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = SpeechBiLSTM(input_dim=SPEECH_FEATURE_DIM).to(self.device)
        self.model_path = Path(model_path) if model_path else SPEECH_MODEL_PATH
        self.is_loaded = False

        if self.model_path.exists():
            self.load_weights(self.model_path)
        else:
            print(f"[Info] Speech weights not found at {self.model_path}. Model initialized with default weights.")

    def load_weights(self, path: Path):
        """Loads speech model state dictionary safely with shape flexibility."""
        try:
            state_dict = torch.load(path, map_location=self.device, weights_only=True)
            # Detect whether checkpoint was trained with 40-dim or 82-dim input
            if "fc_in.weight" in state_dict:
                ckpt_input_dim = state_dict["fc_in.weight"].size(1)
                if ckpt_input_dim != self.model.input_dim:
                    self.model = SpeechBiLSTM(input_dim=ckpt_input_dim).to(self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.is_loaded = True
            print(f"[Speech Model] Successfully loaded weights ({self.model.input_dim}-d features) from {path}")
        except Exception as e:
            print(f"[Error] Failed to load speech model weights: {e}")

    def save_weights(self, path: Path = None):
        """Saves current model state dictionary."""
        save_path = path or self.model_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        print(f"[Speech Model] Saved weights to {save_path}")

    @torch.no_grad()
    def predict(self, audio_tensor: torch.Tensor) -> Tuple[str, float, Dict[str, float]]:
        """
        Runs inference on MFCC tensor of shape (1, 100, 40).
        Returns:
            top_emotion: str
            confidence: float (0.0 to 1.0)
            probabilities: dict mapping emotion -> prob
        """
        self.model.eval()
        audio_tensor = audio_tensor.to(self.device)
        logits = self.model(audio_tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        prob_dict = {EMOTION_CLASSES[i]: float(probs[i]) for i in range(NUM_CLASSES)}
        top_idx = int(probs.argmax())
        top_emotion = EMOTION_CLASSES[top_idx]
        confidence = float(probs[top_idx])

        return top_emotion, confidence, prob_dict

    @torch.no_grad()
    def extract_embedding(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        """Extracts penultimate 128-d feature vector."""
        self.model.eval()
        audio_tensor = audio_tensor.to(self.device)
        return self.model.forward_features(audio_tensor)
