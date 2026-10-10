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
from src.face_model import FacialCNN, VGGFERNetwork


class FacialDataset(Dataset):
    """PyTorch Dataset for labeled facial expression images with fast in-memory caching."""

    def __init__(self, root_dir: Path, augment: bool = False):
        self.root_dir = Path(root_dir)
        self.augment = augment
        self.preprocessor = FacePreprocessor()
        self.samples = []

        for cls_name in EMOTION_CLASSES:
            # Try both capitalized and lowercase folder names
            for folder_name in [cls_name, cls_name.lower()]:
                cls_folder = self.root_dir / folder_name
                if not cls_folder.exists():
                    continue
                cls_idx = CLASS_TO_IDX[cls_name]
                for ext in ["*.png", "*.jpg", "*.jpeg"]:
                    for file_path in cls_folder.glob(ext):
                        self.samples.append((file_path, cls_idx))
                break

        print(f"Loading and caching {len(self.samples)} facial samples into memory from {self.root_dir}...", flush=True)
        n = len(self.samples)
        self.images = np.zeros((n, 48, 48), dtype=np.uint8)
        self.labels = np.zeros(n, dtype=np.int64)

        for i, (file_path, cls_idx) in enumerate(self.samples):
            img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                img = np.zeros((48, 48), dtype=np.uint8)
            else:
                img = cv2.equalizeHist(img)
                if img.shape != (48, 48):
                    img = cv2.resize(img, (48, 48), interpolation=cv2.INTER_AREA)
            self.images[i] = img
            self.labels[i] = cls_idx

        print(f"Cached {n} samples ({self.images.nbytes / (1024*1024):.1f} MB) in RAM ready for training.", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        # Rich real-time data augmentation on training set
        if self.augment:
            # 1. Random horizontal flip (50%)
            if np.random.rand() > 0.5:
                img = np.ascontiguousarray(np.fliplr(img))

            # 2. Random subtle rotation (-6 to +6 degrees) (40%)
            if np.random.rand() > 0.6:
                angle = float(np.random.uniform(-6.0, 6.0))
                M = cv2.getRotationMatrix2D((24, 24), angle, 1.0)
                img = cv2.warpAffine(img, M, (48, 48), borderMode=cv2.BORDER_REFLECT)

            # 3. Random subtle translation (-2 to +2 pixels) (40%)
            if np.random.rand() > 0.6:
                dx = int(np.random.randint(-2, 3))
                dy = int(np.random.randint(-2, 3))
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                img = cv2.warpAffine(img, M, (48, 48), borderMode=cv2.BORDER_REFLECT)

            # 4. Random contrast & brightness scaling (40%)
            if np.random.rand() > 0.6:
                alpha = float(np.random.uniform(0.90, 1.10))
                beta = float(np.random.uniform(-8.0, 8.0))
                img = np.clip(img.astype(np.float32) * alpha + beta, 0.0, 255.0).astype(np.uint8)

        # Convert to normalized float32 tensor (1, 48, 48) in [0, 1]
        tensor = torch.from_numpy(img).float().div_(255.0).unsqueeze(0)
        return tensor, torch.tensor(label, dtype=torch.long)


def train_facial_model(epochs: int = 10, batch_size: int = 128, lr: float = 3e-4, resume: bool = True, arch: str = "vgg"):
    """Executes high-precision facial CNN / VGG-FER training and checkpointing."""
    torch.set_num_threads(os.cpu_count() or 8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Training] Initializing Facial {arch.upper()} on device: {device} (threads: {torch.get_num_threads()})", flush=True)

    # Datasets and Loaders
    train_dataset = FacialDataset(FACIAL_TRAIN_DIR, augment=True)
    val_dataset = FacialDataset(FACIAL_VAL_DIR, augment=False)

    if len(train_dataset) == 0:
        print("[Error] No training data found! Run data/dataset_builder.py first.", flush=True)
        return

    # num_workers=0 is required on Windows to avoid multiprocessing deadlocks
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Initialize Model: VGG-FER Deep Backbone or Baseline FacialCNN
    if arch.lower() == "vgg":
        model = VGGFERNetwork(num_classes=NUM_CLASSES).to(device)
    else:
        model = FacialCNN(num_classes=NUM_CLASSES).to(device)

    best_val_acc = 0.0
    if resume and FACE_MODEL_PATH.exists():
        try:
            sd = torch.load(FACE_MODEL_PATH, map_location=device, weights_only=True)
            model.load_state_dict(sd)
            print(f"[Checkpoint] Resumed prior weights from {FACE_MODEL_PATH} for fine-tuning.", flush=True)
            
            # Evaluate baseline accuracy of loaded checkpoint
            model.eval()
            init_correct, init_total = 0, 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    init_correct += (preds == targets).sum().item()
                    init_total += targets.size(0)
            best_val_acc = (init_correct / init_total) * 100.0
            print(f"[Baseline] Prior checkpoint Val Accuracy: {best_val_acc:.2f}%", flush=True)
        except Exception as e:
            print(f"[Warning] Could not load checkpoint ({e}), initializing fresh.", flush=True)

    # Class-weighted loss with label smoothing to prevent overfitting
    class_counts = torch.zeros(NUM_CLASSES)
    for _, lbl in train_dataset.samples:
        class_counts[lbl] += 1
    total_samples = len(train_dataset.samples)
    class_weights = total_samples / (NUM_CLASSES * class_counts)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.04)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print("\n" + "=" * 70, flush=True)
    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Train Acc (%)':^15} | {'Val Loss':^10} | {'Val Acc (%)':^13} | Status", flush=True)
    print("=" * 70, flush=True)

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

        scheduler.step()

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        status_flag = " "
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            FACE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), FACE_MODEL_PATH)
            status_flag = "[BEST]"

        print(f"{epoch:^7d} | {epoch_train_loss:^12.4f} | {epoch_train_acc:^15.2f} | {epoch_val_loss:^10.4f} | {epoch_val_acc:^13.2f} | {status_flag}", flush=True)

    print("=" * 70, flush=True)
    print(f"[Done] Facial CNN Training Complete! Best Validation Accuracy: {best_val_acc:.2f}%", flush=True)
    print(f"[Model Saved] Optimal weights exported to: {FACE_MODEL_PATH}", flush=True)

    # Save history
    history_file = REPORTS_DIR / "facial_training_history.json"
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[Reports] Training log saved to: {history_file}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Facial CNN / VGG-FER Emotion Model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--arch", type=str, default="vgg", choices=["vgg", "cnn"], help="Architecture: vgg (VGG-FER) or cnn")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume prior weights")
    args = parser.parse_args()

    train_facial_model(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, resume=not args.no_resume, arch=args.arch)

