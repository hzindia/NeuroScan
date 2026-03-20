"""
NeuroScan Autoresearch — prepare.py (FIXED — DO NOT MODIFY)

This file contains all fixed components:
  - Constants (image size, time budget, batch size, paths)
  - Data loading (train = NORMAL only, test = NORMAL + PNEUMONIA)
  - Evaluation function (F1 score — the single metric to optimize)

The agent must NEVER modify this file. It is the ground truth.
"""

import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score

# ===========================================================================
# CONSTANTS — these define the experiment constraints
# ===========================================================================

IMAGE_SIZE = 128                  # Input image resolution (128x128 grayscale)
BATCH_SIZE = 8                    # Keep small for 8GB RAM
NUM_WORKERS = 2                   # 4 CPU cores → 2 workers is safe
TIME_BUDGET = 15 * 60             # 15 minutes wall-clock per experiment
EVAL_THRESHOLD_PERCENTILE = 95    # Anomaly threshold for classification
DEVICE = torch.device("cpu")      # No GPU available

# Dataset path — relative to repo root
# Expects: chest_xray/train/NORMAL/, chest_xray/train/PNEUMONIA/
#           chest_xray/test/NORMAL/,  chest_xray/test/PNEUMONIA/
DATASET_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chest_xray")

# ===========================================================================
# DATA LOADING
# ===========================================================================

_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


def get_train_loader():
    """Returns a DataLoader of NORMAL-only training images."""
    train_dir = os.path.join(DATASET_ROOT, "train")
    if not os.path.exists(train_dir):
        raise FileNotFoundError(
            f"Training data not found at {train_dir}\n"
            f"Download from: https://www.kaggle.com/datasets/paultimothythomas/chest-xray-pneumonia\n"
            f"Expected structure: chest_xray/train/NORMAL/ and chest_xray/train/PNEUMONIA/"
        )
    dataset = datasets.ImageFolder(root=train_dir, transform=_transform)
    # Filter NORMAL only (label 0 in alphabetical order: NORMAL < PNEUMONIA)
    normal_class_idx = dataset.class_to_idx.get("NORMAL")
    if normal_class_idx is None:
        raise ValueError(f"No 'NORMAL' class found. Classes: {dataset.classes}")
    normal_indices = [i for i, label in enumerate(dataset.targets) if label == normal_class_idx]
    subset = Subset(dataset, normal_indices)
    loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    print(f"[prepare] Train loader: {len(normal_indices)} NORMAL images, batch_size={BATCH_SIZE}")
    return loader


def get_test_loader():
    """Returns a DataLoader of ALL test images (NORMAL + PNEUMONIA) with labels."""
    test_dir = os.path.join(DATASET_ROOT, "test")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test data not found at {test_dir}")
    dataset = datasets.ImageFolder(root=test_dir, transform=_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    print(f"[prepare] Test loader: {len(dataset)} images ({dataset.classes}), batch_size={BATCH_SIZE}")
    return loader, dataset


# ===========================================================================
# EVALUATION — the ground truth metric (DO NOT MODIFY)
# ===========================================================================

def evaluate_f1(model, test_loader=None):
    """
    Evaluate the model on the test set.
    
    For anomaly detection:
      - Model is trained on NORMAL only, learns to reconstruct healthy images.
      - At test time, per-image MSE reconstruction error is computed.
      - A threshold at the 95th percentile splits predictions.
      - Images with error > threshold → predicted PNEUMONIA (anomaly).
    
    Returns: (accuracy, f1_score)
    
    The PRIMARY metric to optimize is F1 score. Higher is better.
    """
    if test_loader is None:
        test_loader, _ = get_test_loader()

    model.eval()
    model.to(DEVICE)
    errors = []
    labels = []
    criterion = torch.nn.MSELoss(reduction='none')

    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(DEVICE)
            # Handle different model types
            output = model(data)
            if isinstance(output, tuple):
                # VAE returns (recon, mu, logvar)
                recon = output[0]
            else:
                recon = output
            loss = criterion(recon, data).mean(dim=[1, 2, 3])
            errors.extend(loss.cpu().numpy())
            labels.extend(label.numpy())

    # Threshold-based classification
    threshold = np.percentile(errors, EVAL_THRESHOLD_PERCENTILE)

    # In the dataset: NORMAL=0, PNEUMONIA=1
    # High reconstruction error → predicted anomaly (1)
    preds = [1 if e > threshold else 0 for e in errors]

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='binary')

    return acc, f1


# ===========================================================================
# TIMING UTILITIES
# ===========================================================================

class TimeBudget:
    """Enforces a fixed wall-clock training budget."""
    def __init__(self, budget_seconds=TIME_BUDGET):
        self.budget = budget_seconds
        self.start = None

    def start_timer(self):
        self.start = time.time()

    def elapsed(self):
        return time.time() - self.start

    def remaining(self):
        return max(0, self.budget - self.elapsed())

    def is_expired(self):
        return self.elapsed() >= self.budget


# ===========================================================================
# VERIFICATION (run this file directly to check data setup)
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("NeuroScan Autoresearch — Data Verification")
    print("=" * 60)
    print(f"Dataset root: {DATASET_ROOT}")
    print(f"Device: {DEVICE}")
    print(f"Time budget: {TIME_BUDGET}s ({TIME_BUDGET // 60} minutes)")
    print()

    try:
        train_loader = get_train_loader()
        test_loader, test_dataset = get_test_loader()
        print(f"\n✅ Data OK! Ready to run experiments.")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Test batches:  {len(test_loader)}")
        print(f"   Test classes:  {test_dataset.class_to_idx}")
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
