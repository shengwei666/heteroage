<div align="center">

# 🧬 HeteroAge-HAB (Hierarchical Attention Branch Network)

**A Deep Learning Framework for Biological Age Prediction with Hallmark-Level Interpretability.**

[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

[Architecture](#-1-architecture) •
[Installation](#-2-installation) •
[Quick Start](#-3-quick-start) •
[Inference](#-4-inference) •
[Repository Layout](#-5-repository-layout)

</div>

---

Traditional epigenetic clocks predominantly rely on linear models using exclusively $\beta$ values, which faces three critical limitations. First, as arithmetic means of site-specific methylation, $\beta$ values are inherently blinded to stochastic methylation heterogeneity—a vital dimension of aging that occurs within cell populations. Second, the reliance on linear frameworks fails to capture the complex, non-linear dynamics and multi-layered interactions characteristic of biological aging. Finally, many existing models function as opaque 'black boxes,' offering limited biological interpretability.To address these gaps, we present HeteroAge-HAB, a novel Pan-Tissue deep learning framework trained on a diverse atlas of 19 tissue types. By integrating our previously developed heterogeneity adjustment model, HeteroAge-HAB derives high-dimensional methylation heterogeneity metrics (CHALM and CAMDA) directly from $\beta$ values, capturing information previously lost in the noise. Architecturally structured around the Hallmarks of Aging, the model processes multi-modal inputs through specialized attention branches, bridging the gap between deep learning performance and biological insight to provide a transparent, hallmark-specific breakdown of the aging process.

## ✨ 1. Architecture

The framework uses a unified, end-to-end differentiable network:

* **🧬 Multi-Modal Input**: Simultaneously processes **Beta** , **CHALM**, and **CAMDA** to form a complete 3D epigenetic view.
* **🔍 Attention Gating Mechanism**: Dynamically filters stochastic noise and amplifies biologically relevant heterogeneity signals based on local genomic context.
* **🌿 Hallmark-Specific Branches**: Parallel neural networks ("branches") specialize in distinct aging hallmarks (e.g., Inflammation, Mitochondrial Dysfunction), guided by a priori biological knowledge.
* **⚖️ Pairwise Optimization**: Trains using a **Siamese-style Pairwise Loss**, directly learning the *relative* aging rate between samples to eliminate batch effects and improve cross-cohort generalization.
* **🧠 Attribution Layer**: An interpretable final layer that dynamically weights each hallmark's contribution, offering personalized "Aging Reports" rather than just a single number.

---

## ✨ 2. Installation


### Prerequisites
* Python 3.8+
* PyTorch 1.12+ (CUDA recommended for training)


### Installation
Clone the repository and install in editable mode:


```bash
# 1. Clone the repository
git clone https://github.com/shengwei666/heteroage.git
cd Heteroage


# 2. Create environment (Recommended)
conda create -n heteroage_env python=3.9 -y
conda activate heteroage_env


# 3. Install dependencies
pip install -e .


# 4. Verify installation
heteroage --help
```
---

## ✨ 3. Quick Start: Training

The training workflow consists of **Data Assembly** (Multi-modal alignment) and **Hybrid Loss Training**.


### Step 1: Data Assembly

Align raw tri-modal matrices (Beta, CHALM, CAMDA) and generate a high-performance tensor cache (`.pt`). This process ensures all modalities share the same sample IDs and the feature order strictly follows the Hallmark-CpG dictionary.


```
heteroage assemble \
  --beta_path ./data/raw/Beta_Matrix/Train \
  --chalm_path ./data/raw/Chalm_Matrix/Train \
  --camda_path ./data/raw/Camda_Matrix/Train \
  --hallmark_json ./data/hallmarks.json \
  --output_root ./data/processed \
  --split Train
```

**Output:**

- `./data/processed/cached_tensor_data/merged_Train.pt`: Unified tri-modal tensor dictionary.


### Step 2: Hybrid-Loss Training

Launch the training loop using the **Adaptive Hallmark Cascade** architecture. This mode optimizes both absolute regression accuracy (MAE) and relative aging rank consistency.


```
heteroage train \
  --data_root ./data/processed \
  --hallmark_json ./data/hallmarks.json \
  --output_dir ./models/v1 \
  --batch_size 64 \
  --lr 1e-4 \
  --rank_weight 1.0 \
  --rank_margin 2.0 \
  --use_amp
```

**Key Arguments:**

- `--rank_weight`: Balances MAE (Precision) and Rank-Consistent Loss (Generalization).
- `--rank_margin`: Minimum age gap (years) required to enforce ordering between samples.
- `--use_amp`: Enables Automatic Mixed Precision to reduce VRAM usage.


---


## ✨ 4. Inference & Explainability

The `predict` engine automatically reconstructs the biological topology from the checkpoint and generates hallmark-specific scores.


### 4.1 Cohort Prediction

Process an entire test cohort using the assembled tensor cache.


```
heteroage predict \
  --checkpoint ./models/v1/best_model.pth \
  --data_root ./data/processed \
  --hallmark_json ./data/hallmarks.json \
  --split Test \
  --batch_size 128 \
  --hidden_dim 64
```


### 4.2 Understanding the Aging Report

The model outputs a CSV containing the `Predicted_Age` and a contribution breakdown for each hallmark.


| sample_id | Predicted_Age | True_Age | Inflammation_Score | ... | Inflammation_Weight |
| --- | --- | --- | --- | --- | --- |
| Sample_01 | 45.2 | 43.0 | 52.1 | ... | 0.65 |
| Sample_02 | 45.2 | 46.5 | 30.2 | ... | 0.20 |


> **Biological Insight:** While both samples have a predicted age of 45.2, **Sample_01** shows accelerated aging driven primarily by **Inflammation** (65% weight), whereas **Sample_02** exhibits a more balanced aging profile.

---

## ✨ 5. Repository Layout

The project follows a clean, PyTorch-style modular architecture.

```text
heteroage/
├── configs/                  # Hyperparameter YAML files
├── data/                     # Local data storage & Hallmark masks
├── src/
│   └── heteroage/            # Main Package
│       ├── artifacts/        # Model saving & checkpoint logic
│       ├── models/           # HAB-Net & Bio-Sparse Layers
│       ├── data/             # Dataset loading & Sampler logic
│       ├── engine/           # Trainer engine & Hybrid Loss
│       ├── utils/            # Bio-topology & Logger utilities
│       ├── cli.py            # Unified Entry Point (Assemble/Train/Tune/Predict)
│       └── config.py         # Global TIER & Hardware settings
├── tests/                    # Unit tests (Functional & Gradient checks)
├── pyproject.toml            # Build system & Dependency specs
└── README.md
```

---

## ✨ 6. Key Features


| Feature | Description |
| --- | --- |
| Deep Bio-Interpretability | Unlocks the "Black Box" of aging. Provides a systemic breakdown of drivers across 12 aging hallmarks. |
| Adaptive Signaling Cascades | Dynamically scales network depth based on the biological complexity of each hallmark's CpG count. |
| Rank-Consistent Manifold | Learns relative aging trajectories, making the model inherently robust to batch effects and technical noise. |
| High-Performance Engineering | Built with Mixed Precision (AMP) and Tensor Caching to handle 100k+ features with minimal memory usage. |


---


## 🛠️ Troubleshooting


### ⚠️ OOM (Out of Memory)

- **Solution**: Reduce `--batch_size` or decrease `--hidden_dim` in `configs/habnet_v1.yaml`. Ensure `--use_amp` is enabled.


### ⚠️ Data Alignment Failure

- **Solution**: Ensure your sample IDs (index) match across Beta, CHALM, and CAMDA files. `assemble` will skip any samples not present in all three modalities.