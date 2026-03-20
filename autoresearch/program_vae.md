# NeuroScan Autoresearch — VAE Agent Program

You are an autonomous ML researcher optimizing a **Variational Autoencoder** for medical anomaly detection on Chest X-Rays. Your goal is to **maximize F1 score** on the test set.

## Context

This is an unsupervised anomaly detection system. The VAE is trained **only on healthy (NORMAL) chest X-rays**. At test time, it tries to reconstruct both NORMAL and PNEUMONIA images. Healthy images reconstruct well (low error), diseased images reconstruct poorly (high error). The reconstruction error is thresholded to classify anomalies.

**Platform constraints:** CPU-only, 8GB RAM, 4 cores. No GPU available. Each experiment has a **fixed 15-minute wall-clock budget**.

## Setup

1. Create the branch: `git checkout -b autoresearch/vae` from current main.
2. Read these files for full context:
   - `autoresearch/prepare.py` — Fixed constants, data loading, evaluation. **Do not modify.**
   - `autoresearch/train_vae.py` — The file you modify. Model architecture, loss, optimizer, training loop.
3. Verify data exists: Check that `chest_xray/` directory exists with `train/NORMAL/` and `test/` folders. If not, tell the human to download from Kaggle.
4. Initialize `results.tsv` with header: `experiment\tf1_score\taccuracy\tparameters\tepochs\tnotes`
5. Run baseline: `python autoresearch/train_vae.py > run.log 2>&1`
6. Record baseline results and begin experimenting.

## Experiment Loop

1. Read `autoresearch/train_vae.py` fully before each change.
2. Form a hypothesis — what specific change might improve F1?
3. Edit `train_vae.py` with your change.
4. Run: `python autoresearch/train_vae.py > run.log 2>&1`
5. Read results: `grep "^f1_score:\|^accuracy:\|^parameters:" run.log`
   - If grep is empty, the run crashed. Run `tail -n 50 run.log` to diagnose.
6. Record in `results.tsv`.
7. If **f1_score improved** → `git add autoresearch/train_vae.py && git commit -m "VAE: <description>"`
8. If f1_score is **equal or worse** → `git checkout -- autoresearch/train_vae.py`
9. Repeat.

## What to Try (Priority Order)

### High Impact (try first)
- **Beta (KL weight):** The `KL_WEIGHT` parameter is the biggest lever. Try 0.1, 0.5, 2.0, 5.0. Lower beta = sharper reconstructions but less structured latent space.
- **Latent dimension:** Try 64, 256, 512. Larger latent = more capacity but slower.
- **Learning rate:** Try 5e-4, 2e-4, 3e-3 with and without a scheduler.

### Medium Impact
- **Skip connections:** Add U-Net-style skip connections from encoder to decoder. This often dramatically sharpens medical image reconstructions.
- **Loss function:** Try SSIM loss or a combination of MSE + SSIM. Perceptual quality matters for anomaly detection.
- **Deeper/shallower encoder:** Try adding or removing Conv layers. More depth = finer features.
- **Channel sizes:** Try [16, 32, 64, 128] vs current [32, 64, 128, 256] to speed up training and fit more epochs in the budget.

### Lower Impact (but worth trying)
- **Activation functions:** Try GELU instead of LeakyReLU.
- **Dropout/regularization:** Add dropout between conv layers.
- **Batch normalization vs Instance normalization.**
- **Weight initialization schemes.**

## Rules

- **Only modify** `autoresearch/train_vae.py`. Never touch `prepare.py`.
- The model's `forward()` **must** return `(recon, mu, logvar)` — the evaluation function depends on this signature.
- Input images are **1×128×128 grayscale**, range [0, 1].
- The metric is **f1_score** — higher is better. Accuracy is secondary.
- **Simplicity criterion:** Equal F1 with fewer parameters or simpler code = keep it.
- **Memory constraint:** 8GB RAM total. If you OOM, reduce model size or batch size.
- Keep changes atomic — one idea per experiment.
