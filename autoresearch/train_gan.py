"""
NeuroScan Autoresearch — train_gan.py
GAN (Adversarial Autoencoder) for anomaly detection on Chest X-Rays.

*** THIS IS THE FILE THE AGENT MODIFIES ***

Everything is fair game: generator/discriminator architecture, loss weighting,
training schedule (D steps per G step), optimizer, spectral norm, etc.
The only constraint: must finish within TIME_BUDGET and the generator's
forward(x) must return a reconstructed image tensor of same shape as input.

Usage: python train_gan.py
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(__file__))
from prepare import (
    IMAGE_SIZE, BATCH_SIZE, DEVICE, TIME_BUDGET,
    get_train_loader, get_test_loader, evaluate_f1, TimeBudget
)

# ===========================================================================
# HYPERPARAMETERS — agent can change all of these
# ===========================================================================
LATENT_DIM = 128
LR_GENERATOR = 2e-4
LR_DISCRIMINATOR = 2e-4
PIXEL_LOSS_WEIGHT = 100.0     # Weight of MSE pixel loss vs adversarial loss
D_STEPS_PER_G = 1             # How many D updates per G update
WEIGHT_DECAY = 0.0


# ===========================================================================
# GENERATOR (Autoencoder) — agent can change architecture
# ===========================================================================
class MedicalGANGenerator(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super(MedicalGANGenerator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(32768, latent_dim)   # 128*16*16 = 32768
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32768),
            nn.Unflatten(1, (128, 16, 16)),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, 2, 1, 1), nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


# ===========================================================================
# DISCRIMINATOR — agent can change architecture
# ===========================================================================
class MedicalGANDiscriminator(nn.Module):
    def __init__(self):
        super(MedicalGANDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(16384, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# ===========================================================================
# TRAINING LOOP — agent can change optimizer, schedule, loss, etc.
# ===========================================================================
def train():
    print("=" * 60)
    print("NeuroScan Autoresearch — GAN Experiment")
    print("=" * 60)

    # Data
    train_loader = get_train_loader()

    # Models
    gen = MedicalGANGenerator(latent_dim=LATENT_DIM).to(DEVICE)
    disc = MedicalGANDiscriminator().to(DEVICE)

    gen_params = sum(p.numel() for p in gen.parameters() if p.requires_grad)
    disc_params = sum(p.numel() for p in disc.parameters() if p.requires_grad)
    total_params = gen_params + disc_params
    print(f"Generator params:     {gen_params:,}")
    print(f"Discriminator params: {disc_params:,}")
    print(f"Total params:         {total_params:,}")

    # Optimizers
    opt_g = optim.Adam(gen.parameters(), lr=LR_GENERATOR, weight_decay=WEIGHT_DECAY)
    opt_d = optim.Adam(disc.parameters(), lr=LR_DISCRIMINATOR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCELoss()

    # Time budget
    timer = TimeBudget(TIME_BUDGET)
    timer.start_timer()

    epoch = 0
    while not timer.is_expired():
        epoch += 1
        gen.train()
        disc.train()
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        batches = 0

        for x, _ in train_loader:
            if timer.is_expired():
                break
            x = x.to(DEVICE)
            batch_size = x.size(0)
            real_labels = torch.ones(batch_size, 1).to(DEVICE)
            fake_labels = torch.zeros(batch_size, 1).to(DEVICE)

            # --- Train Discriminator ---
            for _ in range(D_STEPS_PER_G):
                opt_d.zero_grad()
                d_real = disc(x)
                d_loss_real = criterion(d_real, real_labels)
                fake_img = gen(x)
                d_fake = disc(fake_img.detach())
                d_loss_fake = criterion(d_fake, fake_labels)
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                opt_d.step()

            # --- Train Generator ---
            opt_g.zero_grad()
            fake_img = gen(x)
            d_fake_preds = disc(fake_img)
            g_adv_loss = criterion(d_fake_preds, real_labels)
            g_pixel_loss = F.mse_loss(fake_img, x)
            g_loss = g_adv_loss + PIXEL_LOSS_WEIGHT * g_pixel_loss
            g_loss.backward()
            opt_g.step()

            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            batches += 1

        avg_g = epoch_g_loss / max(batches, 1)
        avg_d = epoch_d_loss / max(batches, 1)
        remaining = timer.remaining()
        print(f"Epoch {epoch:3d} | G_loss: {avg_g:.4f} | D_loss: {avg_d:.4f} | remaining: {remaining:.0f}s")

    total_time = timer.elapsed()
    print(f"\nTraining complete: {epoch} epochs in {total_time:.1f}s")

    # --- EVALUATION (ground truth metric — uses generator only) ---
    print("\nEvaluating on test set...")
    test_loader, _ = get_test_loader()
    acc, f1 = evaluate_f1(gen, test_loader)

    # --- PRINT RESULTS ---
    print()
    print(f"f1_score: {f1:.6f}")
    print(f"accuracy: {acc:.6f}")
    print(f"parameters: {total_params}")
    print(f"epochs_completed: {epoch}")
    print(f"wall_time_s: {total_time:.1f}")
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    print(f"peak_vram_mb: {peak_mem:.0f}")

    # Save generator
    save_dir = os.path.join(os.path.dirname(__file__), "saved_models")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "gan_best.pt")
    torch.save(gen.state_dict(), save_path)
    print(f"model_saved: {save_path}")


if __name__ == "__main__":
    train()
