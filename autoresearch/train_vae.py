"""
NeuroScan Autoresearch — train_vae.py
Variational Autoencoder for anomaly detection on Chest X-Rays.

*** THIS IS THE FILE THE AGENT MODIFIES ***

Everything is fair game: architecture, latent_dim, loss function,
optimizer, learning rate schedule, batch strategy, etc.
The only constraint: must finish within the TIME_BUDGET and produce
a model whose forward() returns (recon, mu, logvar) for evaluation.

Usage: python train_vae.py
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Fixed imports from prepare.py — do not modify prepare.py
sys.path.insert(0, os.path.dirname(__file__))
from prepare import (
    IMAGE_SIZE, BATCH_SIZE, DEVICE, TIME_BUDGET,
    get_train_loader, get_test_loader, evaluate_f1, TimeBudget
)

# ===========================================================================
# HYPERPARAMETERS — agent can change all of these
# ===========================================================================
LATENT_DIM = 128
LEARNING_RATE = 1e-3
KL_WEIGHT = 1.0           # Beta-VAE weight on KL divergence term
WEIGHT_DECAY = 0.0         # L2 regularization


# ===========================================================================
# MODEL ARCHITECTURE — agent can change all of this
# ===========================================================================
class MedicalVAE(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super(MedicalVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(),
            nn.Flatten()
        )
        # 256 * 8 * 8 = 16384 for IMAGE_SIZE=128
        self.fc_mu = nn.Linear(16384, latent_dim)
        self.fc_logvar = nn.Linear(16384, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, 16384)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256, 8, 8)),
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1), nn.BatchNorm2d(128), nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1), nn.BatchNorm2d(64), nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1), nn.BatchNorm2d(32), nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 1, 3, 2, 1, 1), nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        z_proj = self.decoder_input(z)
        return self.decoder(z_proj), mu, logvar


# ===========================================================================
# LOSS FUNCTION — agent can change this
# ===========================================================================
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + KL_WEIGHT * kld_loss


# ===========================================================================
# TRAINING LOOP — agent can change optimizer, schedule, etc.
# ===========================================================================
def train():
    print("=" * 60)
    print("NeuroScan Autoresearch — VAE Experiment")
    print("=" * 60)

    # Data
    train_loader = get_train_loader()

    # Model
    model = MedicalVAE(latent_dim=LATENT_DIM).to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # Time budget
    timer = TimeBudget(TIME_BUDGET)
    timer.start_timer()

    epoch = 0
    while not timer.is_expired():
        epoch += 1
        model.train()
        epoch_loss = 0.0
        batches = 0

        for x, _ in train_loader:
            if timer.is_expired():
                break
            x = x.to(DEVICE)
            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            loss = vae_loss(recon, x, mu, logvar)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batches += 1

        avg_loss = epoch_loss / max(batches, 1)
        remaining = timer.remaining()
        print(f"Epoch {epoch:3d} | loss: {avg_loss:.4f} | remaining: {remaining:.0f}s")

    total_time = timer.elapsed()
    print(f"\nTraining complete: {epoch} epochs in {total_time:.1f}s")

    # --- EVALUATION (ground truth metric) ---
    print("\nEvaluating on test set...")
    test_loader, _ = get_test_loader()
    acc, f1 = evaluate_f1(model, test_loader)

    # --- PRINT RESULTS (agent reads these) ---
    print()
    print(f"f1_score: {f1:.6f}")
    print(f"accuracy: {acc:.6f}")
    print(f"parameters: {num_params}")
    print(f"epochs_completed: {epoch}")
    print(f"wall_time_s: {total_time:.1f}")
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    print(f"peak_vram_mb: {peak_mem:.0f}")

    # Save model
    save_dir = os.path.join(os.path.dirname(__file__), "saved_models")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "vae_best.pt")
    torch.save(model.state_dict(), save_path)
    print(f"model_saved: {save_path}")


if __name__ == "__main__":
    train()
