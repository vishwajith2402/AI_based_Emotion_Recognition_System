"""
Speech Emotion Recognition BiLSTM Training Script
AGB1303 - AI Problem Solving Techniques
Trains the MFCC + BiLSTM on Speech Emotion Audio Datasets.
"""

import os
import sys
from pathlib import Path

# Setup project root path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from src.config import (
    SPEECH_TRAIN_DIR, SPEECH_VAL_DIR, SPEECH_MODEL_PATH,
    EMOTION_CLASSES, CLASS_TO_IDX, NUM_CLASSES, N_MFCC, REPORTS_DIR
)
from src.speech_preprocessing import SpeechPreprocessor
from src.speech_model import SpeechBiLSTM


class SpeechDataset(Dataset):
    """PyTorch Dataset for labeled speech audio files with MFCC extraction."""

    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)
        self.preprocessor = SpeechPreprocessor()
        self.samples = []

        for cls_name in EMOTION_CLASSES:
            # Try both capitalized (real datasets) and lowercase (synthetic) folder names
            for folder_name in [cls_name, cls_name.lower()]:
                cls_folder = self.root_dir / folder_name
                if not cls_folder.exists():
                    continue
                cls_idx = CLASS_TO_IDX[cls_name]
                for ext in ["*.wav", "*.mp3"]:
                    for file_path in cls_folder.glob(ext):
                        self.samples.append((file_path, cls_idx))
                break  # found the folder, no need to try lowercase

        print(f"Loaded {len(self.samples)} speech samples from {self.root_dir}. Extracting MFCCs...", flush=True)
        self.cached_data = []
        for file_path, cls_idx in self.samples:
            try:
                audio = self.preprocessor.load_audio_file(str(file_path))
                mfcc = self.preprocessor.extract_mfcc(audio)  # Shape (100, 40)
                tensor = torch.from_numpy(mfcc).float()
                self.cached_data.append((tensor, cls_idx))
            except Exception as e:
                pass
        print(f"Extracted and cached {len(self.cached_data)} MFCC feature tensors in memory.", flush=True)

    def __len__(self):
        return len(self.cached_data)

    def __getitem__(self, idx):
        tensor, label = self.cached_data[idx]
        return tensor, torch.tensor(label, dtype=torch.long)


def train_speech_model(epochs: int = 15, batch_size: int = 32, lr: float = 1e-3):
    """Executes speech BiLSTM training and checkpointing."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Training] Initializing Speech BiLSTM on device: {device}", flush=True)

    train_dataset = SpeechDataset(SPEECH_TRAIN_DIR)
    val_dataset = SpeechDataset(SPEECH_VAL_DIR)

    if len(train_dataset) == 0:
        print("[Error] No speech training data found! Run data/dataset_builder.py first.", flush=True)
        return

    # num_workers=0 is required on Windows to avoid multiprocessing deadlocks
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = SpeechBiLSTM(input_dim=N_MFCC, num_classes=NUM_CLASSES).to(device)
    
    # Class-weighted loss to ensure balanced representation across all 5 emotions
    class_counts = torch.zeros(NUM_CLASSES)
    for _, lbl in train_dataset.samples:
        class_counts[lbl] += 1
    total_samples = len(train_dataset.samples)
    class_weights = total_samples / (NUM_CLASSES * class_counts)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.03)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print("\n" + "=" * 65, flush=True)
    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Train Acc (%)':^15} | {'Val Loss':^10} | {'Val Acc (%)':^13}", flush=True)
    print("=" * 65, flush=True)

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == targets).sum().item()
            total_train += targets.size(0)

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = (correct_train / total_train) * 100.0

        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == targets).sum().item()
                total_val += targets.size(0)

        epoch_val_loss = val_loss / (total_val if total_val > 0 else 1)
        epoch_val_acc = (correct_val / (total_val if total_val > 0 else 1)) * 100.0

        scheduler.step(epoch_val_acc)

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        print(f"{epoch:^7d} | {epoch_train_loss:^12.4f} | {epoch_train_acc:^15.2f} | {epoch_val_loss:^10.4f} | {epoch_val_acc:^13.2f}", flush=True)

        # Checkpoint Best Model
        if epoch_val_acc >= best_val_acc:
            best_val_acc = epoch_val_acc
            SPEECH_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), SPEECH_MODEL_PATH)

    print("=" * 65, flush=True)
    print(f"[Done] Speech BiLSTM Training Complete! Best Validation Accuracy: {best_val_acc:.2f}%", flush=True)
    print(f"[Model Saved] Weights exported to: {SPEECH_MODEL_PATH}", flush=True)

    # Save history
    history_file = REPORTS_DIR / "speech_training_history.json"
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[Reports] Training log saved to: {history_file}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Speech BiLSTM Emotion Model")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    train_speech_model(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
