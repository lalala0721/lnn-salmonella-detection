# LNN Salmonella Detection: Deep Learning for Genomic Pathogen Detection

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20597488.svg)](https://doi.org/10.5281/zenodo.20597488)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Comprehensive benchmark of Liquid Neural Networks (LNN), Convolutional Neural Networks (CNN), and other deep learning architectures for detecting *Salmonella enterica* from whole-genome sequences using k-mer frequency encoding.

## Paper

This repository accompanies the manuscript:

> **"Deep Learning Outperforms Exact k-mer Matching for *Salmonella enterica* Detection from Whole-Genome Sequences"**

The preprint is available on bioRxiv and the published version in *Microbial Genomics* / *BMC Genomics*. See the [Zenodo record](https://doi.org/10.5281/zenodo.20597488) for the full dataset and trained models.

## Key Results

| Model | Accuracy | F1 | AUC | Params | Training |
|-------|:--------:|:--:|:---:|:------:|:--------:|
| **CNN** | **99.64%** | 0.998 | 1.000 | 174K | 2s |
| GRU | 98.27% | 0.989 | 0.999 | 196K | 4s |
| LNN+Attn | 97.84% | 0.986 | 0.994 | 413K | 130s |
| XGBoost | 94.16% | 0.964 | 0.989 | ~500K | 2s |
| LSTM | 88.82% | 0.935 | 0.843 | 824K | 3s |
| Transformer | — | — | — | 851K | — |

**Key findings:**
- **1D-CNN achieves 99.64% accuracy** — matching or exceeding published Kraken2 and Centrifuge benchmarks without any reference database
- **Attention pooling** is the critical component: improves LNN by 18 percentage points (79.9% → 97.8%)
- **LNN is 6× more stable** than CNN at low data volumes (std ±0.9% vs ±5.5% at 10% data)
- Dataset: **14,156 bacterial genomes** across 12 species and 6 *Salmonella* serotypes

## Project Structure

```
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── .gitignore
│
├── lnn_salmonella/                  # Core package
│   ├── config.py                    # Global configuration
│   ├── train.py                     # Training pipeline
│   ├── evaluate.py                  # Evaluation pipeline
│   ├── utils.py                     # Metric tracking, LR scheduler, etc.
│   │
│   ├── data/                        # Data processing subpackage
│   │   ├── encoding.py              # K-mer frequency & one-hot encoding
│   │   ├── tokenizer.py             # DNA tokenization
│   │   ├── dataset.py               # PyTorch Dataset classes
│   │   ├── preprocessing.py         # Data loading & splitting
│   │   ├── preprocess_cache.py      # Build k-mer cache (binary classification)
│   │   ├── preprocess_serotype_cache.py   # Build serotype cache
│   │   ├── preprocess_full_cache.py       # Build full dataset cache
│   │   ├── preprocess_enhanced_cache.py   # Build enhanced cache (hard negatives)
│   │   ├── preprocess_token_cache.py      # Build token-level cache
│   │   └── cache/                   # Cache output directory
│   │
│   ├── models/                      # Model definitions
│   │   ├── lnn_classifier.py        # LNN (CfC) with attention/last/mean pooling
│   │   ├── baselines.py             # CNN, BiLSTM, Transformer baselines
│   │   └── embedding_models.py      # Hierarchical embedding models
│   │
│   ├── results/                     # Experiment outputs (local only)
│   │
│   ├── ablation.py                  # Ablation experiments
│   ├── data_efficiency.py           # Data efficiency experiments
│   ├── plot_efficiency.py           # Publication-quality efficiency charts
│   ├── ood_experiment.py            # Out-of-distribution generalization
│   ├── ood_enhanced.py              # Enhanced OOD with hard negative mining
│   ├── repeat3.py                   # 3-repeat statistical significance tests
│   ├── interpretability.py          # t-SNE, PCA, hidden state analysis
│   ├── tool_comparison.py           # Comparison with Kraken2/SeqSero2-like tools
│   ├── fig1_pipeline.py             # Figure 1: Pipeline schematic
│   ├── fig2_cfc_mechanism.py        # Figure 2: CfC mechanism
│   └── fig3_results.py              # Figure 3: Results comparison
│
└── scripts/                         # Utility scripts
    └── download_data.py             # Download genomes from NCBI (optional)
```

## Installation

```bash
# Clone the repository
git clone https://github.com/lalala0721/lnn-salmonella-detection.git
cd lnn-salmonella-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

**Requirements:**
- Python ≥ 3.10
- PyTorch ≥ 2.0 (CUDA recommended but not required)
- GPU with ≥ 8GB VRAM recommended for full training

## Quick Start

### 1. Download Data

Raw genomes are publicly available from NCBI GenBank. Accession lists are provided in [`scripts/accession_list.csv`](scripts/).

```bash
# Download genomes from NCBI using accession list
python scripts/download_data.py
```

Alternatively, use our pre-processed k-mer cache files from Zenodo ([10.5281/zenodo.20597488](https://doi.org/10.5281/zenodo.20597488)):

```bash
# Download cache from Zenodo and place in lnn_salmonella/data/cache/
# Files: kmer4_chunks32.npz, meta.pkl, etc.
```

### 2. Build k-mer Cache (if using raw genomes)

```bash
# Binary classification cache (Salmonella vs non-Salmonella)
python -m lnn_salmonella.data.preprocess_cache

# Serotype multi-class cache
python -m lnn_salmonella.data.preprocess_serotype_cache

# Full dataset cache
python -m lnn_salmonella.data.preprocess_full_cache
```

### 3. Train Models

```bash
# Train LNN (small variant)
python lnn_salmonella/train.py --model lnn-small --epochs 100

# Train all baselines
python lnn_salmonella/train.py --model cnn
python lnn_salmonella/train.py --model lstm
python lnn_salmonella/train.py --model transformer

# Multi-class serotype classification
python lnn_salmonella/train.py --model lnn-medium --multiclass --epochs 100
```

### 4. Evaluate

```bash
# Evaluate a trained model
python lnn_salmonella/evaluate.py \
    --model lnn-small \
    --checkpoint lnn_salmonella/results/lnn-small_best.pt

# Full evaluation with all metrics
python lnn_salmonella/evaluate.py \
    --model cnn \
    --checkpoint lnn_salmonella/results/cnn_best.pt \
    --ood  # Out-of-distribution evaluation
```

### 5. Run Experiments

```bash
# Ablation studies
python lnn_salmonella/ablation.py

# Data efficiency analysis
python lnn_salmonella/data_efficiency.py

# OOD generalization
python lnn_salmonella/ood_experiment.py
python lnn_salmonella/ood_enhanced.py

# Statistical significance (3 repeats)
python lnn_salmonella/repeat3.py

# Interpretability analysis
python lnn_salmonella/interpretability.py

# External tool comparison
python lnn_salmonella/tool_comparison.py
```

### 6. Generate Figures

```bash
python lnn_salmonella/fig1_pipeline.py       # Fig 1: Experimental pipeline
python lnn_salmonella/fig2_cfc_mechanism.py   # Fig 2: CfC ODE mechanism
python lnn_salmonella/fig3_results.py         # Fig 3: Model comparison results
python lnn_salmonella/plot_efficiency.py      # Fig 4-7: Data efficiency analysis
```

## Configuration

All hyperparameters and paths are configured in [`lnn_salmonella/config.py`](lnn_salmonella/config.py).

Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `KMER_K` | 4 | K-mer length (4 → 256-dim) |
| `NUM_CHUNKS_PER_GENOME` | 32 | Chunks per genome (sequence length) |
| `BATCH_SIZE` | 256 | Training batch size |
| `LEARNING_RATE` | 1e-3 | Initial learning rate |
| `EPOCHS` | 100 | Maximum training epochs |
| `CFC_HIDDEN_SIZES` | [128, 64, 32] | CfC layer hidden sizes |

### Data Path Configuration

Before training, set your data directories in `config.py` or via environment variables:

```python
# In config.py, update these paths:
DATA_DIR = Path("/path/to/your/data")
SEROTYPE_DIR = DATA_DIR / "serotype_data"
NEGATIVE_DIR = DATA_DIR / "negative_species"
```

## Model Architectures

### LNN (Liquid Neural Network)
```
Input (32, 256) → Linear(256,128) → CfC Stack [128→64→32]
→ Attention Pooling → Linear(32,16) → ReLU → Dropout(0.1) → Linear(16,1)
```
Uses Closed-form Continuous-time (CfC) cells from `ncps-torch`.

### CNN Baseline
```
Input (32, 256) → Conv1D [256→64→128→256, k=3] + BatchNorm + MaxPool
→ AdaptiveMaxPool → Linear(256,1)
```
174K parameters, fastest training (2 seconds).

### BiLSTM Baseline
```
Input (32, 256) → BiLSTM(2×128) → Attention Pooling → Linear(256,1)
```
824K parameters.

### Transformer Baseline
```
Input (32, 256) → PositionalEncoding → TransformerEncoder(4L, 4H, d=128)
→ Attention Pooling → Linear(128,1)
```
851K parameters.

## Data

| Data Source | Type | Genomes | Chunks |
|-------------|------|:-------:|:------:|
| `serotype_data/` | *Salmonella* (6 serotypes) | 5,756 | ~298K |
| `negative_species/` | Non-*Salmonella* (6 species) | 7,885 | ~123K |
| `zheng-yangpin/` | *Salmonella* complete genomes | 354 | — |
| `fu-yangpin/` | Non-*Salmonella* complete genomes (incl. 63 *E. coli*) | 161 | — |
| **Total** | | **14,156** | **~422K** |

All raw genome sequences are publicly available from NCBI GenBank. Pre-processed k-mer frequency vectors and trained model weights are deposited at [Zenodo](https://doi.org/10.5281/zenodo.20597488).

## Reproducibility

All experiments are reproducible with fixed random seeds (`RANDOM_SEED = 42`). The `repeat3.py` script runs each model 3 times with different seeds (42, 123, 999) and reports mean ± standard deviation.

Key reproducibility commands:
```bash
# Full reproduction pipeline
python lnn_salmonella/data/preprocess_full_cache.py   # Step 1: Build cache
python lnn_salmonella/repeat3.py                      # Step 2: 3-repeat experiments
python lnn_salmonella/data_efficiency.py              # Step 3: Data efficiency
python lnn_salmonella/ablation.py                     # Step 4: Ablation
python lnn_salmonella/ood_enhanced.py                 # Step 5: OOD tests
python lnn_salmonella/interpretability.py             # Step 6: Interpretability
python lnn_salmonella/plot_efficiency.py              # Step 7: Generate figures
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{lnn_salmonella_2025,
  author       = {[Author Names]},
  title        = {LNN Salmonella Detection: Deep Learning for Genomic Pathogen Detection},
  year         = 2025,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.20597488},
  url          = {https://github.com/lalala0721/lnn-salmonella-detection}
}
```

See also [`CITATION.cff`](CITATION.cff) for structured citation metadata.

## References

- Hasani et al. "Closed-form Continuous-time Neural Networks." *Nature Machine Intelligence*, 2022.
- Wood & Salzberg. "Kraken: ultrafast metagenomic sequence classification." *Genome Biology*, 2014.
- Zhang et al. "SeqSero2: Rapid and Improved Salmonella Serotyping." *Applied and Environmental Microbiology*, 2019.
- `ncps-torch`: [https://github.com/mlech26l/ncps](https://github.com/mlech26l/ncps)

## License

This project is licensed under the MIT License — see [`LICENSE`](LICENSE) for details.

## Contact

[Author Names] — [Institution] — [Email]
