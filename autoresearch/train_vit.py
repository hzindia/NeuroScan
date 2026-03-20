"""
NeuroScan Autoresearch — train_vit.py
Vision Transformer Autoencoder (ViT-AE) for anomaly detection on Chest X-Rays.

*** THIS IS THE FILE THE AGENT MODIFIES ***

Everything is fair game: patch size, embed dim, depth, num heads,
decoder design, positional encoding, optimizer, schedule, etc.
The only constraint: must finish within TIME_BUDGET and forward(x)
must return a reconstructed image tensor of same shape as input.

Usage: python train_vit.py
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
PATCH_SIZE = 16
EMBED_DIM = 128
DEPTH = 4                # Number of transformer encoder layers
NUM_HEADS = 4
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0


# ===========================================================================
# MODEL ARCHITECTURE — agent can change all of this
# ===========================================================================
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=IMAGE_SIZE, patch_size=PATCH_SIZE,
                 in_channels=1, embed_dim=EMBED_DIM):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)           # (B, Embed, H/P, W/P)
        x = x.flatten(2)           # (B, Embed, N_Patches)
        x = x.transpose(1, 2)     # (B, N_Patches, Embed)
        return x


class MedicalTransformer(nn.Module):
    def __init__(self, img_size=IMAGE_SIZE, patch_size=PATCH_SIZE,
                 embed_dim=EMBED_DIM, depth=DEPTH, num_heads=NUM_HEADS):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, 1, embed_dim)

        # Positional encoding
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True, dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=depth
        )

        # Decoder: project patches back to pixels
        self.decoder_proj = nn.Linear(embed_dim, patch_size * patch_size)
        self.patch_size = patch_size
        self.img_size = img_size

    def forward(self, x):
        B, C, H, W = x.shape

        # 1. Patchify & Embed + positional encoding
        patches = self.patch_embed(x) + self.pos_embed

        # 2. Transformer
        encoded = self.transformer_encoder(patches)

        # 3. Reconstruct patches
        rec_patches = self.decoder_proj(encoded)  # (B, N_Patches, P*P)

        # 4. Reshape back to image
        rec_patches = rec_patches.transpose(1, 2)  # (B, P*P, N)
        rec_img = F.fold(
            rec_patches,
            output_size=(H, W),
            kernel_size=self.patch_size,
            stride=self.patch_size
        )
        return torch.sigmoid(rec_img)


# ===========================================================================
# LOSS FUNCTION — agent can change this
# ===========================================================================
def reconstruction_loss(recon, target):
    return F.mse_loss(recon, target)


# ===========================================================================
# TRAINING LOOP — agent can change optimizer, schedule, etc.
# ===========================================================================
def train():
    print("=" * 60)
    print("NeuroScan Autoresearch — ViT Experiment")
    print("=" * 60)

    # Data
    train_loader = get_train_loader()

    # Model
    model = MedicalTransformer(
        img_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS
    ).to(DEVICE)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    print(f"Patch size: {PATCH_SIZE}, Embed dim: {EMBED_DIM}, Depth: {DEPTH}, Heads: {NUM_HEADS}")

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
            recon = model(x)
            loss = reconstruction_loss(recon, x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batches += 1

        avg_loss = epoch_loss / max(batches, 1)
        remaining = timer.remaining()
        print(f"Epoch {epoch:3d} | loss: {avg_loss:.6f} | remaining: {remaining:.0f}s")

    total_time = timer.elapsed()
    print(f"\nTraining complete: {epoch} epochs in {total_time:.1f}s")

    # --- EVALUATION ---
    print("\nEvaluating on test set...")
    test_loader, _ = get_test_loader()
    acc, f1 = evaluate_f1(model, test_loader)

    # --- PRINT RESULTS ---
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
    save_path = os.path.join(save_dir, "vit_best.pt")
    torch.save(model.state_dict(), save_path)
    print(f"model_saved: {save_path}")


if __name__ == "__main__":
    train()
