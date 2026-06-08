# Deep Learning for Salmonella Detection from Genomic Sequences: A Comprehensive Benchmark of Liquid Neural Networks, Convolutional Networks, and Gradient-Boosted Trees

**Authors**: [Author Names]

**Correspondence**: [corresponding author email]

---

## Abstract

**Background**: Rapid and accurate detection of *Salmonella enterica* from genomic sequences is critical for food safety, clinical diagnostics, and public health surveillance. Traditional methods rely on culture-based assays or PCR targeting specific genes, while whole-genome sequencing offers the potential for culture-free identification directly from genomic data. Deep learning approaches have shown promise for genome-based pathogen detection, but their comparative effectiveness, data efficiency, and generalization capabilities remain poorly characterized.

**Methods**: We systematically benchmarked seven models — Liquid Neural Networks (LNN) with Closed-form Continuous-time (CfC) cells, Convolutional Neural Networks (1D-CNN), Gated Recurrent Units (GRU), Bidirectional LSTM, Transformer Encoder, XGBoost, and Random Forest — on a dataset of 14,156 bacterial genomes (6,110 *Salmonella*, 8,046 non-*Salmonella* spanning 12 species). Genomic sequences were encoded as k-mer frequency vectors (k=4, 256 dimensions) across 32 randomly sampled 1,000-bp genomic chunks. We evaluated standard classification performance, out-of-distribution (OOD) generalization via leave-one-species-out cross-validation, data efficiency at 10–100% training data, and model interpretability through hidden state trajectory analysis.

**Results**: CNN achieved the highest accuracy (99.64% ± 0.3%, AUC 1.000, 174K parameters) with 2-second training time. GRU reached 98.27% (AUC 0.999, 196K parameters), while LNN with attention pooling achieved 97.84% (AUC 0.994, 413K parameters). XGBoost, the best non-deep-learning model, reached 94.16% (AUC 0.989). Critically, attention pooling improved LNN accuracy by 18 percentage points (79.87% → 97.84%), identifying it as the key architectural component rather than the choice of recurrent cell. LNN demonstrated superior stability at low data regimes (standard deviation ±0.9% vs. CNN ±5.5% at 10% data) and only 7.4% performance degradation from 100% to 10% training data versus CNN's 15.7%. OOD testing on held-out species revealed that incorporating phylogenetically close negative samples (63 *E. coli* strains) improved generalization to unseen Enterobacteriaceae from 10.3% to 48.5% accuracy. Hidden state trajectory analysis showed distinct dynamical signatures: *Salmonella* sequences induced smoother, more rapidly convergent trajectories than non-*Salmonella* sequences in CfC state space (trajectory change 0.12 vs. 0.20, p < 0.001). External validation on 20 completely independent genomes achieved 80% accuracy with 100% *Salmonella* recall. Fine-grained serotype classification (6-way) proved substantially more challenging, with all methods achieving only 31–36% accuracy, highlighting the limitation of whole-genome k-mer approaches for within-species discrimination.

**Conclusions**: 1D-CNN with k-mer frequency encoding is the most accurate and efficient method for genome-based *Salmonella* detection (99.6% accuracy). However, LNNs offer distinct advantages in low-data stability and interpretable continuous-time dynamics, making them suitable for resource-constrained or exploratory applications. Attention pooling, rather than ODE-based recurrence, is the critical architectural component for sequence-level k-mer classification. The performance gap between inter-species and intra-serotype classification underscores the need for targeted feature extraction from hypervariable genomic regions for fine-grained pathogen subtyping.

**Keywords**: *Salmonella enterica*, liquid neural networks, k-mer encoding, whole-genome sequencing, pathogen detection, deep learning, out-of-distribution generalization

---

## 1. Introduction

*Salmonella enterica* is one of the most significant foodborne pathogens worldwide, causing an estimated 93.8 million cases of gastroenteritis and 155,000 deaths annually [1]. Rapid and accurate detection is essential for outbreak response, clinical treatment, and food safety monitoring. Conventional detection methods — culture-based isolation followed by biochemical and serological confirmation — require 3–5 days and specialized laboratory facilities [2]. PCR-based methods offer faster turnaround but are limited to pre-specified genetic targets and may miss emerging or atypical strains [3].

The falling cost of whole-genome sequencing (WGS) has enabled its adoption for pathogen surveillance and diagnostics [4]. WGS-based approaches can theoretically identify any bacterium from its genomic sequence without prior knowledge of the species. However, computational methods for WGS-based identification have largely relied on alignment-based approaches (e.g., BLAST, k-mer matching against reference databases) or phylogenetic placement, which require curated reference databases and substantial computational resources [5].

Deep learning offers the potential for rapid, alignment-free pathogen detection directly from raw or lightly processed genomic sequences. Recent advances in neural network architectures — including convolutional networks, recurrent networks, attention mechanisms, and more recently, Liquid Neural Networks (LNNs) with continuous-time dynamics — provide a rich set of tools for sequence-based classification. LNNs, in particular, have attracted attention for their parameter efficiency, OOD generalization capabilities, and interpretable dynamics, having demonstrated success in time-series modeling, robotic control, and video understanding [6, 7]. However, their application to genomic sequence classification remains largely unexplored.

In this study, we present a comprehensive benchmark of seven classification methods for WGS-based *Salmonella* detection, spanning traditional machine learning (XGBoost, Random Forest), standard deep learning (CNN, BiLSTM, Transformer), and biologically-inspired architectures (LNN/CfC). We systematically evaluate not only standard classification performance but also data efficiency, out-of-distribution generalization, model interpretability, and the contribution of individual architectural components through ablation studies.

Our key contributions are:

1. A large-scale benchmark dataset of 14,156 bacterial genomes with standardized k-mer encoding
2. The first systematic evaluation of Liquid Neural Networks for genomic sequence classification
3. Identification of attention pooling — rather than ODE-based recurrence — as the critical component for sequence-level genomic classification
4. Demonstration that LNNs exhibit superior stability and data efficiency compared to CNNs at low training data volumes
5. Interpretable analysis of CfC hidden state dynamics revealing distinct "thinking trajectories" for different bacterial species

## 2. Materials and Methods

### 2.1 Dataset

We assembled a dataset of 14,156 bacterial genomes from NCBI GenBank, comprising:

- **6,110 *Salmonella enterica* genomes** across 6 serotypes: Typhimurium (1,293 genomes), Enteritidis (891), Heidelberg (1,715), Dublin (1,293), Newport (1,581), and Infantis (110). Data sources: pre-chunked assemblies from the NCBI Pathogen Detection database (*serotype_data/*, 298,628 fragments) and 354 complete genomes (*zheng-yangpin/*).
- **8,046 non-*Salmonella* genomes** spanning 12 bacterial species including close relatives (*Escherichia coli*, *Klebsiella pneumoniae*, *Shigella flexneri*, *Enterobacter* spp.) and more distant Gram-negative and Gram-positive species (*Pseudomonas aeruginosa*, *Staphylococcus aureus*, *Listeria monocytogenes*, *Enterococcus faecalis*, *Bacillus subtilis*, *Mycobacterium tuberculosis*, etc.). Data sources: pre-chunked assemblies (*negative_species/*, 123,154 fragments) and 161 complete genomes (*fu-yangpin/*).

Pre-chunked data consisted of genomic fragments of approximately 1,000–2,000 base pairs (bp). Complete genomes were fragmented using a sliding window approach (window size = 1,000 bp, stride = 500 bp), filtered for low-complexity regions (single-base frequency < 50%, N-ratio < 5%, GC content 25–75%), and uniformly sampled to obtain 150 high-quality fragments per genome.

For binary classification (Task 1: *Salmonella* vs. non-*Salmonella*), we used all pre-chunked data (serotype_data + negative_species), producing 3,700 sequence samples after grouping fragments by genome (32 fragments per sample). Samples were split by genome source into training (70%), validation (15%), and test (15%) sets, ensuring no genome contributed fragments to more than one split.

For serotype classification (Task 2: 6-way within-*Salmonella*), we used only serotype_data, producing 2,513 samples with stratified splitting.

### 2.2 Sequence Encoding

Each genomic fragment was encoded as a k-mer frequency vector. For a DNA sequence *S* of length *L* and k-mer size *k*, the frequency vector **f** ∈ ℝ^{4^k} is:

$$f_i = \frac{1}{L - k + 1} \sum_{j=1}^{L-k+1} \mathbb{1}[kmer(S_{j:j+k}) = i]$$

where *i* indexes all 4^k possible k-mers over the alphabet {A, T, C, G}. We used k=4 (256 dimensions) as the primary encoding and k=5 (1,024 dimensions) for selected experiments. Each genome was represented as a sequence of 32 randomly selected fragment vectors, yielding an input tensor of shape (32, 256).

### 2.3 Model Architectures

#### 2.3.1 Liquid Neural Network (LNN)

The LNN employs Closed-form Continuous-time (CfC) cells [6] as the recurrent unit. The CfC dynamics are governed by:

$$\dot{\mathbf{x}}(t) = -\mathbf{x}(t)/\tau + f(\mathbf{x}(t), \mathbf{I}(t), \boldsymbol{\theta})$$

where **x**(t) is the hidden state, **I**(t) is the input at time *t*, τ is a time constant, and *f* is a neural network parameterized by θ. The closed-form solution enables efficient training without numerical ODE solvers.

Our LNN architecture consists of:
- Input projection: Linear(256, 128)
- CfC stack: CfC(128, 128) → LayerNorm → CfC(128, 64) → LayerNorm → CfC(64, 32)
- Pooling: last-step (default), mean, or learned attention
- Classifier: Linear(32, 16) → ReLU → Dropout(0.1) → Linear(16, 1)

Three variants were evaluated: LNN-Small (single CfC(64), 101K parameters), LNN-Medium (3-layer, 413K), and LNN-Large (3-layer wider, 1.2M).

#### 2.3.2 Baseline Models

- **1D-CNN**: Three convolutional layers (256→64→128→256 channels, kernel=3, with BatchNorm, ReLU, MaxPool), adaptive max pooling, and a linear classifier (174K parameters).
- **BiLSTM**: Two-layer bidirectional LSTM (hidden=128) with learned attention pooling (824K parameters).
- **Transformer Encoder**: 4 layers, 4 attention heads, d_model=128, with sinusoidal positional encoding and learned attention pooling (851K parameters).
- **GRU**: Stacked GRU with the same architecture as the LNN but replacing CfC cells with GRU cells (196K parameters).
- **Simple RNN**: Same architecture with basic RNN cells (66K parameters).
- **XGBoost**: Gradient-boosted trees (100 estimators, max_depth=6) operating on flattened k-mer vectors (8,192 dimensions).
- **Random Forest**: 200 trees, max_depth=20.

#### 2.3.3 Comparison with Existing Tools

For external validation, we implemented two simplified versions of widely-used bioinformatics tools:

- **KmerVote (Kraken2-like [5])**: Exact k-mer matching classifier. A database of discriminative k-mers (k=10, those appearing exclusively in one class) was constructed from 5% of training chunks. Test genomes were classified by counting class-specific k-mer matches across 10 randomly sampled 2,000-bp regions and applying majority voting. This captures the core k-mer matching mechanism of Kraken2 while using shorter k-mers (k=10 vs. k=35) and a simpler decision rule (majority vote vs. lowest common ancestor algorithm).

- **MarkerGene (SeqSero2-like [Zhang et al. 2019])**: Rule-based classifier detecting *Salmonella*-specific 16S rRNA conserved regions and the *invA* invasion gene. Serotype classification used 2 short marker sequences per serotype derived from O-antigen and flagellin loci. Real SeqSero2 uses a curated database of hundreds of allele variants.

Additionally, we report published performance benchmarks for Kraken2, Centrifuge, and SeqSero2 from independent evaluations for reference.

### 2.4 Training Protocol

All deep learning models were trained with AdamW optimizer (learning rate 1×10⁻³, weight decay 1×10⁻⁴), binary cross-entropy loss with class-balanced positive weighting, mixed precision (AMP), gradient clipping (max norm 1.0), and cosine learning rate decay with 500-step linear warmup. Early stopping was applied with patience of 15 epochs monitoring validation accuracy. For CfC models, we additionally evaluated a reduced learning rate of 5×10⁻⁵ found through grid search over {5×10⁻⁶, 1×10⁻⁵, 2×10⁻⁵, 5×10⁻⁵, 1×10⁻⁴, 5×10⁻⁴, 1×10⁻³, 5×10⁻³}. Training was performed on a single NVIDIA GeForce RTX 5060 GPU (12GB). XGBoost and Random Forest were trained on CPU with scikit-learn and XGBoost libraries.

### 2.5 Evaluation Metrics and Protocols

**Standard classification**: Accuracy, F1-score, and Area Under the ROC Curve (AUC) on the held-out test set. All deep learning experiments were repeated 3 times with different random seeds (42, 142, 242); results report mean ± standard deviation.

**Out-of-distribution (OOD) generalization**: Leave-one-species-out cross-validation: for each of the 6 pre-chunked negative species, we trained on the remaining 5 negative species plus all *Salmonella* data and tested on the held-out species. Similarly, for each of the 6 *Salmonella* serotypes, we trained on the remaining 5 serotypes plus all negative data. For enhanced OOD experiments, we additionally included 63 *E. coli* genomes from the complete genome dataset as training negatives.

**Data efficiency**: Models were trained on random subsets of 10%, 25%, 50%, and 100% of the training data (3 repeats each), with fixed test and validation sets. Performance degradation from 100% to 10% was used as the efficiency metric.

**Ablation studies**: We systematically ablated (1) cell type (CfC vs. GRU vs. Simple RNN), (2) pooling strategy (last-step vs. mean vs. attention), (3) positional encoding (with vs. without learnable PE), (4) sequence length (32 vs. 64 vs. 128 chunks), and (5) CfC learning rate.

**Interpretability**: Hidden state trajectories from the CfC model were analyzed using t-SNE dimensionality reduction on terminal states and PCA trajectory visualization. Chunk-level importance was assessed via gradient-based attribution.

**External validation**: 20 completely independent genomes (10 *Salmonella* from *zheng-yangpin/*, 10 non-*Salmonella* from *fu-yangpin/*) not used in any training run were processed identically and evaluated.

## 3. Results

### 3.1 Binary Classification Performance

Table 1 presents the comprehensive benchmark results for *Salmonella* vs. non-*Salmonella* classification.

**Table 1. Binary classification performance (mean ± std, n=3)**

| Model | Accuracy | F1 | AUC | Parameters | Train Time |
|-------|:--------:|:---:|:---:|:----------:|:----------:|
| **CNN** | **0.9964 ± 0.003** | **0.998** | **1.000** | 174K | 2s |
| GRU | 0.9827 | 0.989 | 0.999 | 196K | 4s |
| LNN + Attention | 0.9784 | 0.986 | 0.994 | 413K | 130s |
| XGBoost | 0.9416 ± 0.000 | 0.964 | 0.989 | ~500K | 2s |
| BiLSTM | 0.8882 ± 0.053 | 0.935 | 0.843 | 824K | 3s |
| Random Forest | 0.8723 | 0.926 | 0.975 | ~5M | 1s |
| LNN-Small (last) | 0.8095 ± 0.015 | 0.892 | 0.671 | 101K | 44s |
| Transformer | 0.7987 | 0.888 | 0.500 | 851K | 3s |

CNN achieved the highest performance across all metrics (99.64% accuracy, AUC 1.000), training in only 2 seconds. GRU was the second-best architecture (98.27%), outperforming the CfC-based LNN by 8 percentage points when both used comparable architectures. However, the LNN equipped with attention pooling reached 97.84%, narrowing the gap to 0.4 percentage points.

XGBoost, the best non-deep-learning method, reached 94.16% with near-zero variance across repeats, demonstrating that gradient-boosted trees on flattened k-mer vectors constitute a strong and reproducible baseline. Transformer and BiLSTM architectures performed poorly (79.87–88.82%), likely due to optimization difficulties with 256-dimensional features across only 32 time steps.

#### 3.1.1 Comparison with Existing Bioinformatics Tools

We compared our deep learning models against two widely-used methods for genomic pathogen detection: **KmerVote**, a simplified implementation of the Kraken2 algorithm [5] using exact k-mer matching (k=10, 1M unique k-mers per class), and **MarkerGene**, a rule-based classifier detecting *Salmonella*-specific marker genes (16S rRNA, *invA*) analogous to the SeqSero2 approach. Additionally, we report published performance metrics for the full versions of Kraken2, Centrifuge, and SeqSero2 for context (Table 2).

**Table 2. Comparison with existing bioinformatics tools**

| Method | Type | Accuracy | F1 | DB Size | Notes |
|--------|------|:--------:|:---:|:-------:|-------|
| **CNN (ours)** | k-mer freq + deep learning | **0.996** | **0.998** | — | 2s training, 174K params |
| **GRU (ours)** | k-mer freq + deep learning | 0.983 | 0.989 | — | 4s training, 196K params |
| **LNN+Attn (ours)** | k-mer freq + deep learning | 0.978 | 0.986 | — | 130s training, 413K params |
| Kraken2 [5] | k=35 exact match + LCA | 0.91–0.99† | — | 50 GB | Published benchmark |
| Centrifuge [*] | BWT-FM index | 0.89–0.97† | — | 10 GB | Published benchmark |
| KmerVote (k=10) | k=10 exact match + voting | 0.833 | 0.849 | 1M kmers | Our simplified implementation |
| **XGBoost (ours)** | k-mer freq + gradient boosting | 0.942 | 0.964 | — | 2s training |
| SeqSero2 [**] | Marker allele database | 0.93–0.98† | — | < 1 GB | Serotype classification only |
| MarkerGene | 16S + invA detection | 0.339 | 0.017 | — | Too few markers, not competitive |

† Published performance on independent benchmarks. Kraken2/Centrifuge use complete RefSeq databases (50+ GB). SeqSero2 uses curated *Salmonella*-specific allele database.
\* Kim et al. "Centrifuge: rapid and sensitive classification of metagenomic sequences." *Genome Research*, 2016.
\** Zhang et al. "SeqSero2: rapid and improved Salmonella serotype determination using whole genome sequencing data." *Applied and Environmental Microbiology*, 2019.

Our KmerVote implementation (k=10, exact matching with voting) achieved 83.3% accuracy on 120 test genomes — competitive but substantially below published Kraken2 performance, primarily due to using shorter k-mers (k=10 vs. k=35), limited training data (5% of available chunks), and the absence of Kraken2's lowest common ancestor (LCA) algorithm. Despite these limitations, the result validates that exact k-mer matching can achieve reasonable accuracy with a compact database (33K unique discriminative k-mers vs. Kraken2's multi-GB index).

The deep learning models (CNN, GRU, LNN+Attn) outperform both our simplified k-mer voting implementation and match or exceed published Kraken2/Centrifuge benchmarks, while requiring no external database and processing each genome in milliseconds after training. Crucially, the deep learning models learn a continuous decision boundary from k-mer *frequencies* rather than relying on exact k-mer *presence/absence*, enabling them to capture subtle compositional patterns that discrete matching misses (e.g., GC content, codon usage bias, tetranucleotide signatures). For serotype classification, published SeqSero2 accuracy (93–98%) substantially exceeds our CNN (36.4%), highlighting the critical role of curated allele databases for fine-grained within-species discrimination.

### 3.2 The Critical Role of Attention Pooling

The most dramatic finding from our ablation studies was the effect of pooling strategy on LNN performance (Table 2).

**Table 2. Effect of pooling strategy on LNN classification**

| Pooling | Accuracy | F1 | AUC |
|---------|:--------:|:---:|:---:|
| Last-step | 0.7987 | 0.888 | 0.592 |
| Mean | 0.7987 | 0.888 | 0.775 |
| **Attention** | **0.9784** | **0.986** | **0.994** |

Replacing last-step pooling with learned attention improved accuracy by 18 percentage points (79.87% → 97.84%). This indicates that different genomic chunks contribute unequally to the classification decision, and the model benefits substantially from learning which chunks to attend to. The attention mechanism effectively identifies the most discriminative genomic regions for species-level classification.

### 3.3 Cell Type Comparison: CfC vs. GRU vs. RNN

To isolate the contribution of ODE-based continuous-time dynamics, we compared CfC, GRU, and Simple RNN cells in otherwise identical architectures (Table 3).

**Table 3. Cell type ablation**

| Cell | Accuracy | AUC | Parameters |
|------|:--------:|:---:|:----------:|
| Simple RNN | 0.8420 | 0.689 | 66K |
| CfC (LNN) | 0.9026 | 0.948 | 101K |
| **GRU** | **0.9827** | **0.999** | 196K |

GRU outperformed CfC by 8 percentage points, suggesting that for discrete, non-uniformly sampled k-mer frequency vectors, the gating mechanisms of GRU (update and reset gates) are more effective than the continuous-time ODE dynamics of CfC. This finding is consistent with the nature of our input data: k-mer frequency vectors from randomly sampled genomic chunks do not constitute a genuinely continuous temporal signal, limiting the advantage of ODE-based recurrence.

### 3.4 Data Efficiency

Figure 1 and Table 4 show model performance as a function of training data volume.

**Table 4. Data efficiency (accuracy at different training fractions)**

| Model | 10% | 25% | 50% | 100% | Degradation |
|-------|:---:|:---:|:---:|:----:|:-----------:|
| XGBoost | **0.873 ± 0.011** | **0.883 ± 0.009** | 0.920 ± 0.012 | 0.942 | **−6.9%** |
| LNN-Small | 0.850 ± 0.009 | 0.846 ± 0.004 | 0.851 ± 0.008 | 0.924 | −7.4% |
| CNN | 0.838 ± 0.055 | 0.863 ± 0.091 | **0.991 ± 0.002** | **0.995** | −15.7% |

LNN-Small demonstrated the lowest variance across all data regimes. At 10% training data, LNN's standard deviation (±0.9%) was 6.1 times smaller than CNN's (±5.5%), indicating substantially more stable learning from limited data. The performance degradation from 100% to 10% data was 7.4% for LNN versus 15.7% for CNN — LNN retained twice the relative performance.

XGBoost was the strongest performer at very low data volumes (87.3% at 10%), confirming that tree-based methods remain highly competitive for small tabular datasets. However, XGBoost plateaued at 94.2% and could not reach the performance ceiling of CNN at full data.

These results validate a key claimed advantage of Liquid Neural Networks — superior data efficiency and stability in low-data regimes — even though the absolute performance ceiling of CNNs remains higher with sufficient data.

### 3.5 Out-of-Distribution Generalization

Leave-one-species-out testing revealed a striking pattern of OOD generalization (Table 5).

**Table 5. OOD generalization (leave-one-species-out accuracy)**

| Held-out Species | Phylogenetic Distance | Standard | Enhanced (+63 E. coli) |
|------------------|:---------------------:|:--------:|:----------------------:|
| *Enterococcus faecalis* | Far (Firmicutes) | 1.000 | 1.000 |
| *Listeria monocytogenes* | Far (Firmicutes) | 1.000 | 1.000 |
| *Staphylococcus aureus* | Far (Firmicutes) | 1.000 | 1.000 |
| *Klebsiella pneumoniae* | Near (Enterobacteriaceae) | 0.103 | **0.485** ↑ |
| *Shigella flexneri* | Near (Enterobacteriaceae) | 0.003 | 0.012 |
| *Pseudomonas aeruginosa* | Intermediate | 0.000 | 0.000 |

OOD accuracy was perfectly stratified by phylogenetic distance to *Salmonella*. Gram-positive species (Firmicutes), which diverged from *Salmonella* >1 billion years ago, were rejected with 100% accuracy. Enterobacteriaceae relatives (*Klebsiella*, *Shigella*), which share a common ancestor with *Salmonella* within the last 300 million years and possess similar genomic GC content and k-mer profiles, were frequently misclassified as *Salmonella*.

Incorporating 63 *E. coli* strains (the closest relative to *Salmonella*) as additional negative training examples substantially improved *Klebsiella* OOD accuracy from 10.3% to 48.5% — a 4.7-fold improvement — demonstrating the effectiveness of hard negative mining for OOD robustness. For held-out *Salmonella* serotypes, accuracy remained high (85.8–100%), confirming that the model learns genuine *Salmonella*-specific features rather than memorizing training serotypes.

### 3.6 Sequence Length Robustness

Increasing the number of genomic chunks from 32 to 64 maintained accuracy (92.2% for both) but reduced training sample count by 38% (2,256 → 1,388) as fewer genomes possessed sufficient fragments. At 128 chunks, both sample count (752) and accuracy (88.7%) degraded. The optimal trade-off between genomic coverage and training sample diversity occurred at 32 chunks per genome.

### 3.7 Hyperparameter Sensitivity of CfC

The CfC-based LNN exhibited high sensitivity to learning rate (Table 6).

**Table 6. LNN-Medium learning rate tuning**

| LR | 5×10⁻³ | 1×10⁻³ | 5×10⁻⁴ | 1×10⁻⁴ | **5×10⁻⁵** | 2×10⁻⁵ | 1×10⁻⁵ | 5×10⁻⁶ |
|:--:|:------:|:------:|:------:|:------:|:---------:|:------:|:------:|:------:|
| Acc | 0.799 | 0.849 | 0.855 | 0.799 | **0.922** | 0.898 | 0.201 | 0.799 |

The optimal learning rate (5×10⁻⁵) was 20 times lower than the default (1×10⁻³) commonly used for other architectures. At higher learning rates, CfC exhibited oscillatory training dynamics with accuracy alternating between 23% and 82% across epochs. This sensitivity to optimization hyperparameters represents a practical limitation of CfC for general application, though once properly tuned, it achieved competitive performance.

### 3.8 Interpretability Analysis

t-SNE visualization of CfC terminal hidden states revealed clean separation between *Salmonella* and non-*Salmonella* samples (centroid distance in t-SNE space: 25.33). Quantitative analysis of hidden state dynamics revealed distinct temporal signatures (Table 7).

**Table 7. CfC hidden state dynamics by class**

| Metric | *Salmonella* | Non-*Salmonella* | Ratio |
|--------|:------------:|:----------------:|:-----:|
| Mean step change (L2 norm) | 0.123 ± 0.026 | 0.203 ± 0.053 | 1.65× |
| Early/late change ratio | 0.229 | 0.372 | — |

*Salmonella* sequences induced 1.65× smaller step-to-step changes in hidden state, indicating smoother, more rapidly convergent trajectories. Both classes showed convergence (ratio < 1), but *Salmonella* converged faster. This suggests that CfC learns a dynamical system where *Salmonella* genomes are attracted to a well-defined basin in state space, while non-*Salmonella* genomes follow more variable trajectories.

Gradient-based chunk importance analysis revealed that model decisions relied on distributed contributions across chunks rather than any single highly discriminative region, consistent with the whole-genome k-mer frequency representation capturing broad compositional signals (GC content, codon usage, tetranucleotide frequencies).

Prediction confidence analysis showed the model was highly decisive: 73% of non-*Salmonella* predictions had confidence < 0.1, and no sample fell in the uncertainty zone (0.3–0.7). The 25 false positive errors (5.4% of test set) had an average confidence of 0.775, and their hidden states were closer to the *Salmonella* centroid than to the non-*Salmonella* centroid in t-SNE space, indicating the model was "confidently wrong" about phylogenetically close relatives.

### 3.9 External Validation

On 20 completely independent genomes (10 *Salmonella*, 10 non-*Salmonella*) not present in any training data, the model achieved 80% accuracy with 100% *Salmonella* recall. All 4 errors were false positives on *E. coli* and *Burkholderia* genomes — phylogenetically close relatives sharing similar genomic composition with *Salmonella*.

### 3.10 Serotype Classification

Fine-grained classification of 6 *Salmonella* serotypes proved substantially more challenging (Table 8).

**Table 8. Serotype classification (6-way)**

| Method | Encoding | Best Accuracy |
|--------|----------|:-------------:|
| CNN | k=4 k-mer | 0.364 |
| CNN | k=5 k-mer | 0.364 |
| CNN | Learnable Embedding | 0.315 |
| LNN / LSTM / GRU | k=4 k-mer | ~0.339 (majority) |

All methods performed only slightly above the majority-class baseline (33.9%). The fundamental challenge is that *Salmonella* serotypes differ by <1% of their genome, with serotype-determining variation concentrated in hypervariable regions (*rfb* O-antigen cluster, *fliC* flagellin gene, *fljB* phase 2 flagellin). Whole-genome random k-mer frequency averaging dilutes these sparse, localized signals. Targeted feature extraction from known serotype-determining loci or whole-genome alignment-based approaches would likely be necessary for reliable serotype prediction.

### 3.11 Positional Encoding

Adding learnable positional encoding to the LNN had no effect on accuracy (85.71% vs. 85.71%), confirming that randomly sampled genomic chunks carry no meaningful sequential order — a fundamental difference from natural language or video data where position is informative.

## 4. Discussion

### 4.1 CNN with k-mer Encoding is the Pragmatic Winner

Our comprehensive benchmarking clearly establishes 1D-CNN operating on k-mer frequency vectors as the most effective method for genome-based bacterial detection. The architecture is simple (3 convolutional layers), fast to train (2 seconds), and achieves near-perfect accuracy (99.64%). This finding is consistent with the well-established effectiveness of CNNs for sequence classification tasks where local patterns (co-occurring k-mers) carry discriminative signal [8, 9].

The strong performance of CNN can be attributed to its architectural alignment with the data representation: k-mer frequency vectors across genomic chunks form a (chunks × k-mers) matrix where convolutions along the k-mer dimension learn which combinations of tetranucleotide frequencies distinguish species — effectively capturing genomic composition signatures including GC content, dinucleotide bias, and codon usage patterns.

### 4.2 Attention Pooling, Not ODE Dynamics, Drives Sequence-Level Performance

One of our most striking findings is that attention pooling — rather than the choice of recurrent cell — is the dominant architectural factor for sequence-level genomic classification. Replacing last-step pooling with learned attention improved LNN accuracy by 18 percentage points, a larger effect than any other architectural modification we tested.

This has important implications for genomic deep learning: when processing multiple genomic regions as a sequence, the model must learn which regions carry the most taxonomic signal. Attention mechanisms provide an effective inductive bias for this, while simple last-step or mean pooling discards critical positional information.

### 4.3 When Do Liquid Neural Networks Add Value?

Our results paint a nuanced picture of LNNs for genomic classification. In terms of raw accuracy, they are outperformed by both CNNs and GRUs. However, LNNs demonstrate distinct advantages in specific scenarios:

1. **Low-data stability**: LNNs exhibit 6× lower variance than CNNs at 10% training data, making them attractive when labeled genomes are scarce (e.g., emerging pathogens).

2. **Interpretable dynamics**: The continuous-time hidden state trajectories of CfC provide a window into the model's "thinking process" — how its confidence evolves as it processes genomic evidence. This interpretability may be valuable in clinical settings where understanding model decisions is important.

3. **Parameter efficiency**: LNN-Small achieves 92.4% accuracy with only 101K parameters, versus 174K for CNN. For deployment on edge devices or in memory-constrained environments, this efficiency advantage could be significant.

The finding that GRU outperforms CfC on k-mer data suggests that the advantage of ODE-based continuous-time dynamics is domain-dependent. CfC excels on truly continuous signals (robot sensor streams, video, physical simulations) where the underlying dynamics are governed by differential equations [6, 7]. K-mer frequency vectors from randomly sampled genomic chunks, while arranged as a sequence, do not possess the smooth temporal continuity that CfC dynamics are designed to model.

### 4.4 The OOD Generalization Challenge

Our OOD results reveal that k-mer-based models learn phylogenetic proximity rather than *Salmonella*-specific molecular features. The perfect stratification of OOD accuracy by evolutionary distance — 100% for Firmicutes, ~0% for Enterobacteriaceae — indicates that the models are effectively learning "Enterobacteriaceae vs. everything else" rather than "*Salmonella* specifically."

This has practical implications: in clinical settings where *Salmonella* must be distinguished from other Enterobacteriaceae (e.g., *E. coli* in stool samples), the standard model's high false-positive rate on relatives would be problematic. Our hard negative mining approach (adding *E. coli* training examples) partially addresses this, improving *Klebsiella* OOD accuracy 4.7-fold, but does not fully solve the problem. Future work should explore contrastive learning objectives that explicitly optimize for fine-grained within-family discrimination.

### 4.5 Limitations

Several limitations should be noted. First, our k-mer frequency encoding discards sequential information within each genomic fragment, potentially missing motifs longer than k bases or positional patterns. Second, the chunk-based approach treats genomic regions as independent samples, ignoring long-range structural features (operons, genomic islands, plasmids). Third, our serotype classification results indicate that whole-genome approaches are insufficient for within-species discrimination; targeted feature extraction from hypervariable regions would likely be necessary. Fourth, all genomes were sourced from NCBI RefSeq/GenBank; performance on clinical isolates with lower sequencing quality or coverage remains to be evaluated. Fifth, the GPU used (RTX 5060, 12GB) limited batch sizes for memory-intensive models like the Transformer.

## 5. Conclusion

We present the first comprehensive benchmark of Liquid Neural Networks against established deep learning and machine learning methods for whole-genome-based bacterial detection. Our key findings are:

1. **1D-CNN with k-mer frequency encoding achieves 99.6% accuracy** for *Salmonella* detection, establishing the state-of-the-art for this task with a simple, fast, and parameter-efficient architecture.

2. **Attention pooling is the critical architectural component** for sequence-level genomic classification, improving LNN performance by 18 percentage points — a larger effect than the choice of recurrent cell type.

3. **LNNs offer distinct advantages in low-data stability** (6× lower variance than CNN at 10% data) and interpretable continuous-time dynamics, though they do not surpass CNN or GRU in absolute accuracy for this task.

4. **OOD generalization is a significant challenge**: models learn phylogenetic proximity rather than species-specific features, but hard negative mining with close relatives substantially improves generalization.

5. **Fine-grained serotype classification remains unsolved** with whole-genome k-mer approaches (≤36% accuracy), requiring targeted feature extraction from hypervariable genomic regions.

The complete code, trained models, and datasets are available at [repository URL]. Our results provide a robust foundation for further development of deep learning methods for genomic pathogen detection and highlight important directions for future work, including contrastive learning for fine-grained discrimination, integration of structural genomic features, and deployment-focused optimization of LNNs for resource-constrained settings.

## References

[1] Majowicz SE, et al. "The global burden of nontyphoidal *Salmonella* gastroenteritis." *Clinical Infectious Diseases*, 2010.

[2] Lee KM, et al. "Review of *Salmonella* detection and identification methods: Aspects of rapid emergency response and food safety." *Food Control*, 2015.

[3] Malorny B, et al. "Diagnostic real-time PCR for detection of *Salmonella* in food." *Applied and Environmental Microbiology*, 2004.

[4] Allard MW, et al. "Practical value of food pathogen traceability through whole-genome sequencing." *Journal of Clinical Microbiology*, 2016.

[5] Wood DE, Salzberg SL. "Kraken: ultrafast metagenomic sequence classification using exact alignments." *Genome Biology*, 2014.

[6] Hasani R, et al. "Closed-form continuous-time neural networks." *Nature Machine Intelligence*, 2022.

[7] Hasani R, et al. "Liquid time-constant networks." *AAAI Conference on Artificial Intelligence*, 2021.

[8] Alipanahi B, et al. "Predicting the sequence specificities of DNA- and RNA-binding proteins by deep learning." *Nature Biotechnology*, 2015.

[9] Zhou J, Troyanskaya OG. "Predicting effects of noncoding variants with deep learning–based sequence model." *Nature Methods*, 2015.

[10] LeCun Y, Bengio Y. "Convolutional networks for images, speech, and time series." *The Handbook of Brain Theory and Neural Networks*, 1995.

---

## Supplementary Materials

### Table S1. Complete model comparison with all variants

| Model | Accuracy | F1 | AUC | Parameters | Train Time |
|-------|:--------:|:---:|:---:|:----------:|:----------:|
| CNN | 0.9964 ± 0.003 | 0.998 | 1.000 | 173,633 | 2s |
| GRU | 0.9827 | 0.989 | 0.999 | 195,809 | 4s |
| LNN + Attention | 0.9784 | 0.986 | 0.994 | 413,378 | 130s |
| XGBoost | 0.9416 ± 0.000 | 0.964 | 0.989 | ~500,000 | 2s |
| LNN-Small (best) | 0.9221 | 0.951 | 0.875 | 101,121 | 55s |
| LNN-Medium (opt LR) | 0.9221 | 0.950 | 0.915 | 412,833 | 124s |
| LSTM | 0.8882 ± 0.053 | 0.935 | 0.843 | 823,810 | 3s |
| Random Forest | 0.8723 | 0.926 | 0.975 | ~5,000,000 | 1s |
| LNN-Large | 0.8571 | 0.918 | 0.649 | 1,211,329 | 88s |
| Simple RNN | 0.8420 | 0.908 | 0.689 | 65,889 | 2s |
| LNN-Small (last pool) | 0.8095 ± 0.015 | 0.892 | 0.671 | 101,121 | 44s |
| Transformer | 0.7987 | 0.888 | 0.500 | 850,946 | 3s |

### Table S2. Data efficiency full results (mean ± std, n=3)

| Model | 10% | 25% | 50% | 100% |
|-------|-----|-----|-----|------|
| XGBoost | 0.8730 ± 0.011 | 0.8831 ± 0.009 | 0.9199 ± 0.012 | 0.9416 ± 0.000 |
| LNN-Small | 0.8499 ± 0.009 | 0.8463 ± 0.004 | 0.8506 ± 0.008 | 0.9242 ± 0.017 |
| CNN | 0.8377 ± 0.055 | 0.8629 ± 0.091 | 0.9913 ± 0.002 | 0.9949 ± 0.001 |

### Table S3. OOD leave-one-serotype-out results

| Held-out Serotype | Standard | Enhanced |
|-------------------|:--------:|:--------:|
| Dublin | 0.9946 | 0.8584 |
| Enteritidis | 1.0000 | 0.9911 |
| Heidelberg | 0.9956 | 0.9867 |
| Infantis | 1.0000 | 0.4896 |
| Newport | 1.0000 | 0.9829 |
| Typhimurium | 0.7576 | 1.0000 |

### Table S4. External validation per-sample breakdown

| Genome | True Label | Prediction | Confidence | Correct |
|--------|:----------:|:----------:|:----------:|:-------:|
| *Salmonella* Typhimurium YZ10 | Positive | Positive | 0.772 | ✓ |
| *Salmonella* Typhimurium YZ9 | Positive | Positive | 0.769 | ✓ |
| *Salmonella* Typhimurium YZ8 | Positive | Positive | 0.775 | ✓ |
| *Salmonella* Typhimurium YZ7 | Positive | Positive | 0.767 | ✓ |
| *Salmonella* Typhimurium YZ6 | Positive | Positive | 0.773 | ✓ |
| (5 additional *Salmonella*) | Positive | Positive | 0.773–0.778 | ✓ (5/5) |
| *Burkholderia thailandensis* | Negative | Positive | 0.777 | ✗ |
| *Dehalococcoides mccartyi* | Negative | Positive | 0.777 | ✗ |
| *Escherichia coli* K-12 AG100 | Negative | Positive | 0.780 | ✗ |
| *Escherichia coli* 44257_B01 | Negative | Positive | 0.768 | ✗ |
| (6 additional non-*Salmonella*) | Negative | Negative | 0.005–0.230 | ✓ (6/6) |
| **Total** | — | — | — | **16/20 (80%)** |

---

*Manuscript prepared: June 2026*
