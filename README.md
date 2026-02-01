# 🩻 NeuroScan Pro: Multi-Model Anomaly Detection

> **Next-Gen Unsupervised Pathology Detection using Generative AI (VAE, GAN, ViT).**

NeuroScan Pro is a research workbench designed to tackle the "Black Box" problem in medical diagnosis. Instead of training models to recognize specific diseases (which requires massive labeled datasets), NeuroScan Pro learns the **"Manifold of Health."** By understanding healthy anatomy at a pixel level, it can detect **any** anomaly—known or unknown—by flagging deviations from the norm.

---

## ❓ The Challenge: Why NeuroScan?

### 1. The Data Scarcity Problem
In medical imaging, "Normal" data is abundant, but "Pathological" (Disease) data is rare, expensive to label, and strictly regulated (HIPAA/GDPR). Traditional Supervised Learning requires thousands of labeled examples for *every* specific disease (Pneumonia, TB, COVID-19). If a new virus emerges, traditional models fail until new datasets are curated.

### 2. The "Black Swan" Failure
Supervised classifiers (e.g., ResNet50 trained on Pneumonia) are binary: they output `0` or `1`. They cannot detect *novel* anomalies. If a patient has a rare lung tumor but the model was only trained on Pneumonia, it will likely misclassify the patient as "Healthy" because it doesn't know what a tumor looks like.

### 3. Lack of Explainability
Standard "Black Box" AI gives a probability score (e.g., "98% Pneumonia") but rarely tells the doctor *where* to look. This lack of localization makes clinicians hesitant to trust AI predictions.

---

## 💡 The Solution: Generative Anomaly Detection

NeuroScan Pro flips the paradigm using **Unsupervised One-Class Classification**:

1.  **Train on "Normal" Only:** We feed the models (VAE, GAN, ViT) *only* healthy Chest X-Rays. They learn to reconstruct healthy anatomy perfectly.
2.  **Test on Everything:** When a patient scan is input:
    * If **Healthy**: The model reconstructs it accurately (Low Error).
    * If **Diseased**: The model *fails* to reconstruct the lesion (High Error) because it has never seen a disease before.
3.  **The "Difference Map" Breakthrough:** By subtracting the **Reconstruction** from the **Original Input**, we generate a pixel-perfect **Heatmap** that highlights exactly *where* the anomaly is. This provides instant visual explainability for doctors.

### 🚀 Key Breakthroughs & Significance
* **Zero-Shot Detection:** Can theoretically detect *any* lung pathology (Pneumonia, COVID, Tumors) without ever training on them.
* **Data Privacy:** Models can be trained on purely healthy data, which is less sensitive and easier to acquire.
* **Architecture Benchmarking:** A first-of-its-kind workbench to compare **Probabilistic (VAE)** vs. **Adversarial (GAN)** vs. **Attention-based (ViT)** approaches for medical reconstruction side-by-side.

---

## 🌟 Key Features

* **Multi-Model Engine:**
    * **ConvVAE (Variational Autoencoder):** Probabilistic baseline for smooth latent representations.
    * **GAN (Generative Adversarial Network):** Uses adversarial loss to generate sharper, high-frequency details.
    * **ViT (Vision Transformer):** Leverages Self-Attention to understand global anatomical structure.
* **🏆 Live Leaderboard:**
    * Real-time tracking of **Accuracy**, **F1-Score**, and **Parameter Count**.
    * Instant comparison of training stability via **Log-Scale Loss Curves**.
* **🔎 Comparative Diagnostics:**
    * **Visual Inspection:** Randomly loads Normal vs. Pneumonia samples to visually compare Difference Maps across architectures.
    * **Metric Benchmarking:** Auto-calculates performance stats on the Test Set.

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


