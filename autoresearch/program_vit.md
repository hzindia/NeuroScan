# NeuroScan Autoresearch — ViT Agent Program

You are an autonomous ML researcher optimizing a **Vision Transformer Autoencoder (ViT-AE)** for medical anomaly detection on Chest X-Rays. Your goal is to **maximize F1 score** on the test set.

## Context

A ViT-based autoencoder that splits the image into patches, processes them with self-attention, then reconstructs. Trained only on NORMAL images. Anomalies are detected via reconstruction error.

**Platform constraints:** CPU-only, 8GB RAM, 4 cores. Each experiment has a **fixed 15-minute wall-clock budget**. Transformers are slower than CNNs on CPU, so the time budget is critical — you may need to reduce model size to fit enough training epochs.

## Setup

1. Create branch: `git checkout -b autoresearch/vit` from current main.
2. Read files:
   - `autoresearch/prepare.py` — Fixed. **Do not modify.**
   - `autoresearch/train_vit.py` — The file you modify.
3. Verify data: `chest_xray/` directory.
4. Initialize `results.tsv`: `experiment\tf1_score\taccuracy\tparameters\tepochs\tnotes`
5. Run baseline: `python autoresearch/train_vit.py > run.log 2>&1`
6. Record and begin experimenting.

## Experiment Loop

1. Read `autoresearch/train_vit.py` fully before each change.
2. Form a hypothesis.
3. Edit `train_vit.py`.
4. Run: `python autoresearch/train_vit.py > run.log 2>&1`
5. Read: `grep "^f1_score:\|^accuracy:\|^parameters:" run.log`
   - If empty → `tail -n 50 run.log` to diagnose crash.
6. Record in `results.tsv`.
7. If **f1_score improved** → `git add autoresearch/train_vit.py && git commit -m "ViT: <description>"`
8. If worse → `git checkout -- autoresearch/train_vit.py`
9. Repeat.

## What to Try (Priority Order)

### Critical for CPU (try FIRST)
- **Reduce depth:** Transformers are slow on CPU. Try depth=2 or depth=3. Fewer layers = more epochs in the time budget = potentially better results despite less capacity.
- **Reduce embed_dim:** Try 64 or 96. Saves compute quadratically in attention.
- **Larger patch size:** Try 32 instead of 16. Halves the sequence length (fewer patches), which is 4x faster in attention. But loses spatial detail.
- **Fewer heads:** Try num_heads=2.

### Medium Impact
- **Decoder design:** The current decoder is a single Linear layer — very weak. Try a multi-layer MLP decoder, or a CNN decoder that upsamples from the patch tokens.
- **Loss function:** Try SSIM or L1 loss instead of MSE. Or a weighted combination.
- **Positional encoding:** Try sinusoidal instead of learnable. Or relative position bias.
- **Learning rate schedule:** Try cosine annealing or warmup + decay.

### Architecture Experiments
- **Hybrid CNN-ViT:** Use a small CNN to extract patch features before the transformer. This can be much faster than raw pixel patches.
- **Convolutional decoder:** Instead of F.fold, use ConvTranspose2d layers to reconstruct.
- **Lightweight attention:** Try using `nn.MultiheadAttention` directly with smaller FFN multiplier (2x instead of 4x).
- **Layer norm placement:** Try pre-norm (before attention) vs post-norm.

### Lower Priority
- **Data augmentation:** Random horizontal flip, slight rotation during training (add to training loop, not to prepare.py's transforms).
- **Dropout rates:** Adjust from 0.1 to 0.0 or 0.2.
- **Gradient clipping:** `torch.nn.utils.clip_grad_norm_` to stabilize training.
- **Mixed precision:** Not available on CPU, but you could try `torch.bfloat16` if supported.

## Rules

- **Only modify** `autoresearch/train_vit.py`. Never touch `prepare.py`.
- The model's `forward(x)` must return a single reconstructed tensor of same shape as input `(B, 1, 128, 128)`.
- Input images: **1×128×128 grayscale**, range [0, 1].
- Metric: **f1_score** — higher is better.
- **CPU speed matters.** A simpler model that trains 30 epochs in 15 min may beat a complex one that only gets 3 epochs. Always check epochs_completed — more epochs in the budget is better.
- Simplicity criterion: simpler code with equal F1 is a win.
- Memory: 8GB RAM total.
- Keep changes atomic.
