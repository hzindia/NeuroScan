# 🧪 NeuroScan Autoresearch

> Autonomous AI-driven optimization of VAE, GAN, and ViT anomaly detectors for Chest X-Rays.
> Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

This folder applies the **autoresearch pattern** to NeuroScan's three algorithms. An AI agent (Claude Code, Codex, etc.) autonomously runs experiments, edits the training code, measures F1 score, keeps improvements, and reverts failures — all while you sleep.

## How It Works

```
Human writes strategy ──→ program_{vae,gan,vit}.md
                              │
                              ▼
AI Agent reads instructions + code
                              │
                              ▼
Agent edits train_{vae,gan,vit}.py ──→ Runs 15-min experiment
                              │                    │
                              │                    ▼
                              │             Reads f1_score
                              │                    │
                              ▼                    ▼
                    Improved? ──YES──→ git commit (new baseline)
                         │
                         NO──→ git reset (discard change)
                         │
                         └──→ Repeat (~4 experiments/hour)
```

## Design (One Algo, One File, One Metric)

| File | Role | Who edits? |
|------|------|------------|
| `prepare.py` | Data loading, evaluation, constants | **Nobody** (locked) |
| `train_vae.py` | VAE architecture + training loop | **AI Agent** |
| `train_gan.py` | GAN architecture + training loop | **AI Agent** |
| `train_vit.py` | ViT architecture + training loop | **AI Agent** |
| `program_vae.md` | Strategy instructions for VAE agent | **Human** |
| `program_gan.md` | Strategy instructions for GAN agent | **Human** |
| `program_vit.md` | Strategy instructions for ViT agent | **Human** |

**Metric:** F1 Score on test set (higher = better). Computed by `prepare.py` — never modified.

**Time budget:** 15 minutes per experiment (wall clock). On your 4-core CPU, this gives ~4 experiments/hour or ~30+ overnight.

## Prerequisites

- Python 3.10+
- 8GB RAM, 4 CPU cores (no GPU needed)
- Chest X-Ray dataset from Kaggle

## Quick Start

### 1. Install Dependencies

```bash
# From repo root (NeuroScan/)
pip install torch torchvision scikit-learn numpy
```

### 2. Download Dataset

Download [Chest X-Ray Pneumonia dataset](https://www.kaggle.com/datasets/paultimothythomas/chest-xray-pneumonia) and extract so the structure is:

```
NeuroScan/
├── chest_xray/
│   ├── train/
│   │   ├── NORMAL/       ← Training data (healthy only)
│   │   └── PNEUMONIA/    ← Ignored during training
│   └── test/
│       ├── NORMAL/       ← Used for evaluation
│       └── PNEUMONIA/    ← Used for evaluation
├── autoresearch/         ← This folder
│   ├── prepare.py
│   ├── train_vae.py
│   ├── train_gan.py
│   ├── train_vit.py
│   ├── program_vae.md
│   ├── program_gan.md
│   └── program_vit.md
├── app.py                ← Original Streamlit app (unchanged)
├── model.py
└── utils.py
```

### 3. Verify Data Setup

```bash
cd NeuroScan
python autoresearch/prepare.py
```

You should see `✅ Data OK!` — if not, fix your dataset path.

### 4. Manual Test Run (Optional)

```bash
# Test one experiment manually (~15 min)
python autoresearch/train_vae.py
```

Check it prints `f1_score: X.XXXXXX` at the end.

---

## 🤖 Connecting the AI Agent

### Option A: Claude Code (Recommended)

[Claude Code](https://docs.anthropic.com/en/docs/claude-code) is a command-line agent that can edit files and run commands.

```bash
# 1. Install Claude Code
npm install -g @anthropic-ai/claude-code

# 2. Navigate to repo
cd NeuroScan

# 3. Launch Claude Code
claude

# 4. Give it the prompt:
```

**Prompt for VAE:**
```
Read autoresearch/program_vae.md and let's start optimizing the VAE.
Do the setup first, then kick off experiments autonomously.
```

**Prompt for GAN:**
```
Read autoresearch/program_gan.md and let's start optimizing the GAN.
Do the setup first, then kick off experiments autonomously.
```

**Prompt for ViT:**
```
Read autoresearch/program_vit.md and let's start optimizing the ViT.
Do the setup first, then kick off experiments autonomously.
```

Claude Code will:
1. Read the program file + training script
2. Create a git branch
3. Run the baseline experiment
4. Start proposing and testing changes
5. Keep improvements, revert failures
6. Repeat until you stop it

### Option B: OpenAI Codex CLI

```bash
# Install
npm install -g @openai/codex

# Launch in the repo
cd NeuroScan
codex

# Same prompts as above
```

### Option C: Cursor / Windsurf / Aider

Any AI coding agent that can edit files and run shell commands works. The key is:

1. Point it at the appropriate `program_*.md` file
2. Let it edit only the matching `train_*.py` file
3. Make sure it can run `python autoresearch/train_*.py` and read the output

### Running All Three Overnight

You can run three separate terminal sessions, one per algorithm:

```bash
# Terminal 1 — VAE
cd NeuroScan && claude
# → "Read autoresearch/program_vae.md and start experiments"

# Terminal 2 — GAN  
cd NeuroScan && claude
# → "Read autoresearch/program_gan.md and start experiments"

# Terminal 3 — ViT
cd NeuroScan && claude
# → "Read autoresearch/program_vit.md and start experiments"
```

⚠️ **Memory warning:** Running all three simultaneously on 8GB RAM may cause OOM. Run them **one at a time** or sequentially:

```bash
# Sequential approach (safest for 8GB RAM):
# Run VAE overnight → review in morning
# Run GAN next night → review
# Run ViT third night → review
```

---

## Reviewing Results

After the agent runs, check:

```bash
# See all commits (each = one improvement)
git log --oneline autoresearch/

# See the results log
cat results.tsv

# Compare branches
git diff main..autoresearch/vae -- autoresearch/train_vae.py
```

## Applying Improvements Back to Main App

Once the agent finds better architectures/hyperparameters, manually port the key changes from `autoresearch/train_*.py` back into your main `model.py` and `app.py`. The autoresearch scripts are standalone experiments — the original Streamlit app stays untouched.

## Tips for Your Machine

Since you're on CPU with 8GB RAM:

1. **VAE will be fastest** — expect 15-20 epochs per 15-min run. Start here.
2. **GAN is trickiest** — adversarial training is inherently unstable. The agent may need several attempts.
3. **ViT is slowest on CPU** — the agent should prioritize reducing model size first (fewer layers, smaller embed dim, larger patches).
4. Close other apps while running to free up RAM.
5. The agent should get **~4 experiments per hour** per algorithm.

## License

Same as NeuroScan repo (MIT).
