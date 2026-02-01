# 🩻 NeuroScan Pro: Multi-Model Anomaly Detection

> **Advanced Unsupervised Anomaly Detection for Medical Imaging using VAEs, GANs, and Vision Transformers.**

NeuroScan Pro is a state-of-the-art benchmarking workbench for medical AI. It uses **One-Class Classification** to detect pathologies (like Pneumonia) by training strictly on healthy tissue.

Unlike standard classifiers, NeuroScan Pro learns the "manifold of health." When presented with disease, the models fail to reconstruct the anatomy correctly, creating a "Difference Map" that highlights the tumor or infection.

---

## 🌟 New Pro Features

* **Three Architectures:**
    * **ConvVAE (Variational Autoencoder):** The classic probabilistic baseline.
    * **GAN (Generative Adversarial Network):** Uses adversarial training to generate sharper, more realistic healthy tissue reconstructions.
    * **ViT (Vision Transformer):** Uses Self-Attention mechanisms to capture long-range dependencies in anatomical structures.
* **⚔️ Benchmark Suite:**
    * **Visual Comparison:** See how VAE, GAN, and ViT reconstruct the same X-Ray side-by-side.
    * **Anomaly Localization:** Compare "Difference Maps" to see which model best highlights the disease.
    * **Metric Table:** Auto-calculates **Accuracy** and **F1-Score** for all loaded models on the Test set.
* **Persistent History:** Training loss graphs are saved in memory, allowing you to train Model A, then Model B, and compare their learning curves on the same plot.
* **Auto-Save & Load:** Models are automatically timestamped and saved. The "Diagnostics" tab lets you mix and match versions (e.g., "Load VAE from yesterday vs. GAN from today").

---

## 🛠️ Technical Stack

* **Core:** PyTorch, TorchVision
* **UI:** Streamlit
* **Algorithms:**
    * **VAE:** Standard Encoder-Decoder with KL Divergence loss.
    * **GAN:** Autoencoder-style Generator with a PatchGAN Discriminator.
    * **Transformer:** Patch-based ViT Encoder with a linear projection Decoder.
* **Metrics:** Scikit-Learn (Accuracy, F1).

---

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/NeuroScanPro.git](https://github.com/yourusername/NeuroScanPro.git)
cd NeuroScanPro
```

### 2. Set up Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 📂 Dataset Setup

**Required:** [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothythomas/chest-xray-pneumonia) from Kaggle.

1.  Download and Unzip.
2.  Ensure folder structure:
    ```text
    chest_xray/
    ├── train/
    │   ├── NORMAL/     <-- Training Data (Healthy)
    │   └── PNEUMONIA/  <-- Ignored during training
    └── test/           <-- Used for Benchmarking (Contains Both)
    ```
3.  **Note Path:** You will select this folder using the "📂" button in the app.

---

## 🕹️ Usage Guide

### 1. Launch the Workbench
```bash
streamlit run app.py
```

### 2. Train Models (Tab 1)
* **Select Architecture:** Choose VAE, GAN, Transformer, or "Train ALL Sequentially".
* **Configure:** Set Epochs (Rec: 20+) and Learning Rate.
* **Train:** Click Start.
    * *The Loss Graph will update in real-time and persists across different training runs.*
    * *Models are auto-saved to `saved_models/`.*

### 3. Run Diagnostics & Benchmark (Tab 2)
* **Load Models:** Use the dropdowns to select specific `.pt` files for VAE, GAN, and ViT. (Select "None" if you only want to test one).
* **Click "RUN BENCHMARK":**
    * **Visuals:** Shows the Input X-Ray, Reconstructions, and **Difference Maps** (Heatmaps) for all loaded models.
    * **Metrics:** Displays a table with **Accuracy** and **F1 Score** to numerically prove which architecture performs best.

---

## 🧠 Model Architectures

| Model | Mechanism | Strengths | Weaknesses |
| :--- | :--- | :--- | :--- |
| **VAE** | Compresses input to Gaussian latent space. | Smooth, stable training. | Blurry reconstructions. |
| **GAN** | Generator fights Discriminator. | Sharp, realistic details. | Unstable training (Mode Collapse). |
| **ViT** | Splits image into 16x16 patches + Self-Attention. | Understands global structure. | Data hungry; heavy compute. |

### Main Architecture
<img width="1737" height="1104" alt="Architecture" src="https://github.com/user-attachments/assets/6a10e15f-3943-492c-8492-ec5c6413e9b2" />

#### VAE
<img width="1737" height="1104" alt="VAE" src="https://github.com/user-attachments/assets/f47c3e7b-d8dd-4d4f-8328-ad53902fbbc4" />

#### GAN
<img width="1737" height="1104" alt="GAN" src="https://github.com/user-attachments/assets/463168d9-f5fd-446b-81ab-13f79ca7d2fc" />

#### ViT
<img width="1737" height="1104" alt="VIT" src="https://github.com/user-attachments/assets/716d97e5-74b5-4f8c-ab5d-27cf167b2b50" />

---

## 📂 Project Structure

```text
NeuroScanPro/
├── saved_models/       # Auto-created folder for trained weights
├── app.py              # Main Benchmark Dashboard
├── model.py            # PyTorch Architectures (VAE, GAN, ViT)
├── utils.py            # Metrics, Heatmaps & Plotting Logic
├── requirements.txt    # Dependencies
└── README.md           # Documentation
```

---
## Images
<img width="1286" height="614" alt="neuroscan train models" src="https://github.com/user-attachments/assets/802c6338-368c-45ca-963a-3e766a14dbe8" />
<img width="1288" height="356" alt="diagnostic and benchmark" src="https://github.com/user-attachments/assets/971569a5-e788-4be1-8a50-a2b4de93e60a" />
<img width="1006" height="511" alt="comparative diagnostics" src="https://github.com/user-attachments/assets/6a7ff3dd-b467-4c29-9bc9-7c23f1016075" />
<img width="991" height="452" alt="visual reconstruction inspection" src="https://github.com/user-attachments/assets/e76644d4-e4f3-412f-b736-7eaf459df909" />


