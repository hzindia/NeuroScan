# NeuroScan Autoresearch — GAN Agent Program

You are an autonomous ML researcher optimizing a **GAN-based Autoencoder** for medical anomaly detection on Chest X-Rays. Your goal is to **maximize F1 score** on the test set.

## Context

This is an adversarial autoencoder. The Generator reconstructs healthy chest X-rays, while the Discriminator judges if the reconstruction is real. At test time, MSE reconstruction error classifies anomalies. The Generator is evaluated — the Discriminator is only used during training.

**Platform constraints:** CPU-only, 8GB RAM, 4 cores. Each experiment has a **fixed 15-minute wall-clock budget**.

## Setup

1. Create branch: `git checkout -b autoresearch/gan` from current main.
2. Read files:
   - `autoresearch/prepare.py` — Fixed. **Do not modify.**
   - `autoresearch/train_gan.py` — The file you modify.
3. Verify data: `chest_xray/` directory with `train/NORMAL/` and `test/`.
4. Initialize `results.tsv`: `experiment\tf1_score\taccuracy\tparameters\tepochs\tnotes`
5. Run baseline: `python autoresearch/train_gan.py > run.log 2>&1`
6. Record and begin experimenting.

## Experiment Loop

1. Read `autoresearch/train_gan.py` fully before each change.
2. Form a hypothesis.
3. Edit `train_gan.py`.
4. Run: `python autoresearch/train_gan.py > run.log 2>&1`
5. Read: `grep "^f1_score:\|^accuracy:\|^parameters:" run.log`
   - If empty → `tail -n 50 run.log` to diagnose crash.
6. Record in `results.tsv`.
7. If **f1_score improved** → `git add autoresearch/train_gan.py && git commit -m "GAN: <description>"`
8. If worse → `git checkout -- autoresearch/train_gan.py`
9. Repeat.

## What to Try (Priority Order)

### High Impact
- **Pixel loss weight:** `PIXEL_LOSS_WEIGHT` is the biggest lever. Current is 100. Try 10, 50, 200, 500. Higher = more faithful reconstruction but less adversarial sharpness.
- **Learning rate ratio:** Try different LR for G vs D. Common: D_lr = 0.5 * G_lr.
- **D steps per G step:** Try 2 or 3 D steps per G. Can stabilize GAN training.

### Medium Impact
- **Feature matching loss:** Instead of standard adversarial loss, match intermediate D features.
- **Spectral normalization:** Add to discriminator for training stability.
- **Generator skip connections:** Add U-Net skip from encoder to decoder in the Generator.
- **Discriminator depth:** Try deeper/shallower. A weak D undertrain the G; too strong D collapses.
- **Label smoothing:** Use 0.9 instead of 1.0 for real labels.

### Architecture Changes
- **PatchGAN discriminator:** Instead of single scalar output, classify overlapping patches.
- **Generator bottleneck:** Adjust latent dim (64, 256, 512).
- **Remove BatchNorm from Discriminator** — common best practice for GANs.
- **Use LeakyReLU(0.2) everywhere** instead of ReLU in decoder.

### Training Tricks
- **Gradient penalty** (WGAN-GP style) instead of BCE loss.
- **Two time-scale update rule (TTUR):** Different LR for G and D.
- **Noise injection:** Add small Gaussian noise to real images fed to D.
- **Learning rate schedulers:** CosineAnnealing, ReduceLROnPlateau.

## Rules

- **Only modify** `autoresearch/train_gan.py`. Never touch `prepare.py`.
- The Generator's `forward(x)` must return a single reconstructed tensor, same shape as input.
- Input images: **1×128×128 grayscale**, range [0, 1].
- Metric: **f1_score** — higher is better.
- **GAN instability is expected.** If training collapses (D loss → 0 or G loss diverges), treat it as a failed experiment and revert.
- Memory: 8GB RAM total. Watch for OOM with large discriminators.
- Keep changes atomic.
