"""
Facial Emotion Recognition CNN Training Script
AGB1303 - AI Problem Solving Techniques
Trains the Deep CNN on Facial Expression Image Datasets.
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
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from src.config import (
    FACIAL_TRAIN_DIR, FACIAL_VAL_DIR, FACE_MODEL_PATH,
    EMOTION_CLASSES, CLASS_TO_IDX, NUM_CLASSES, REPORTS_DIR
)
from src.face_preprocessing import FacePreprocessor
from src.face_model import FacialCNN


class FacialDataset(Dataset):
    """PyTorch Dataset for labeled facial expression images."""

    def __init__(self, root_dir: Path, augment: bool = False):
        self.root_dir = Path(root_dir)
        self.augment = augment
        self.preprocessor = FacePreprocessor()
        self.samples = []

        for cls_name in EMOTION_CLASSES:
            # Try both capitalized (real datasets) and lowercase (synthetic) folder names
            for folder_name in [cls_name, cls_name.lower()]:
                cls_folder = self.root_dir / folder_name
                if not cls_folder.exists():
                    continue
                cls_idx = CLASS_TO_IDX[cls_name]
                for ext in ["*.png", "*.jpg", "*.jpeg"]:
                    for file_path in cls_folder.glob(ext):
                        self.samples.append((file_path, cls_idx))
                break  # found the folder, no need to try lowercase

        print(f"Loaded {len(self.samples)} facial samples from {self.root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            img = np.zeros((48, 48), dtype=np.uint8)

        # Preprocess & normalize to (48, 48) in [0, 1]
        img_norm = self.preprocessor.preprocess_face(img)

        # Data augmentation on training set
        if self.augment:
            if np.random.rand() > 0.5:
                img_norm = np.fliplr(img_norm).copy()
            if np.random.rand() > 0.5:
                # Slight brightness scaling
                img_norm = np.clip(img_norm * np.random.uniform(0.85, 1.15), 0.0, 1.0)

        # Convert to tensor (1, 48, 48)
        tensor = torch.from_numpy(img_norm).unsqueeze(0).float()
        return tensor, torch.tensor(label, dtype=torch.long)


def train_facial_model(epochs: int = 8, batch_size: int = 128, lr: float = 1e-3):
    """Executes facial CNN training and checkpointing."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Training] Initializing Facial CNN on device: {device}", flush=True)

    # Datasets and Loaders
    train_dataset = FacialDataset(FACIAL_TRAIN_DIR, augment=True)
    val_dataset = FacialDataset(FACIAL_VAL_DIR, augment=False)

    if len(train_dataset) == 0:
        print("[Error] No training data found! Run data/dataset_builder.py first.", flush=True)
        return

    # num_workers=0 is required on Windows to avoid multiprocessing deadlocks
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Initialize Model, Loss, Optimizer
    model = FacialCNN(num_classes=NUM_CLASSES).to(device)

    # Class-weighted loss to balance representation across all 5 emotions
    class_counts = torch.zeros(NUM_CLASSES)
    for _, lbl in train_dataset.samples:
        class_counts[lbl] += 1
    total_samples = len(train_dataset.samples)
    class_weights = total_samples / (NUM_CLASSES * class_counts)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.03)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

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

        for batch_idx, (inputs, targets) in enumerate(train_loader):
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

            if (batch_idx + 1) % 40 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"  [Epoch {epoch}/{epochs}] Batch {batch_idx+1}/{len(train_loader)} - Batch Loss: {loss.item():.4f}", flush=True)

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
            FACE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), FACE_MODEL_PATH)

    print("=" * 65, flush=True)
    print(f"[Done] Facial CNN Training Complete! Best Validation Accuracy: {best_val_acc:.2f}%", flush=True)
    print(f"[Model Saved] Weights exported to: {FACE_MODEL_PATH}", flush=True)

    # Save history
    history_file = REPORTS_DIR / "facial_training_history.json"
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[Reports] Training log saved to: {history_file}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Facial CNN Emotion Model")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    train_facial_model(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
