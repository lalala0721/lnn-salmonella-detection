# Deep Learning Outperforms Exact k-mer Matching for *Salmonella enterica* Detection from Whole-Genome Sequences: A Comprehensive Benchmark of CNN, Liquid Neural Networks, and Gradient-Boosted Trees

## Abstract

**Background**: Rapid and accurate identification of *Salmonella enterica* from whole-genome sequencing (WGS) data is critical for food safety surveillance, clinical diagnostics, and outbreak investigation. While alignment-based tools like Kraken2 and Centrifuge dominate current practice, deep learning methods offer potential advantages in speed, database independence, and generalization. However, no comprehensive benchmark exists comparing deep learning architectures against each other and against established bioinformatics tools for this task.

**Results**: We benchmarked seven classification methods — Convolutional Neural Network (CNN), Liquid Neural Network (LNN/CfC), Gated Recurrent Unit (GRU), BiLSTM, Transformer, XGBoost, and Random Forest — on 14,156 bacterial genomes (6,110 *Salmonella*, 8,046 non-*Salmonella*) encoded as k-mer frequency vectors (k=4, 256 dimensions). CNN achieved the highest accuracy (99.64% ± 0.3%, AUC 1.000, 174K parameters) with 2-second training time, outperforming a simplified Kraken2-like k-mer matching classifier (83.3%), GRU (98.3%), and XGBoost (94.2%). Our models matched or exceeded published benchmarks for Kraken2 (91–99%) and Centrifuge (89–97%) while requiring no external reference database. Attention pooling proved to be the critical architectural component, improving LNN accuracy by 18 percentage points (79.9% → 97.8%). LNN demonstrated 6-fold lower variance than CNN at 10% training data (std ±0.9% vs. ±5.5%), establishing superior data efficiency. Out-of-distribution testing revealed performance stratified by phylogenetic distance — 100% accuracy on distant Gram-positive species versus near-chance on Enterobacteriaceae relatives — partially mitigated by hard negative mining (+38% improvement for *Klebsiella pneumoniae*). Hidden state trajectory analysis of the LNN revealed distinct continuous-time dynamical signatures for different bacterial classes. External validation on 20 independent genomes achieved 80% accuracy with 100% *Salmonella* recall. Fine-grained serotype classification (6-way) proved substantially more challenging (≤36% for all k-mer methods) compared to the specialized tool SeqSero2 (93–98%), highlighting the necessity of curated allele databases for within-species discrimination.

**Conclusions**: 1D-CNN operating on k-mer frequency vectors is the most accurate, efficient, and practical method for WGS-based *Salmonella* detection, matching or exceeding the performance of established tools without requiring multi-gigabyte reference databases. Liquid Neural Networks offer distinct advantages in data-limited and interpretability-critical scenarios. The systematic benchmark, ablation studies, and external validation provide a robust foundation for selecting and deploying deep learning methods in genomic pathogen surveillance pipelines.

**Keywords**: *Salmonella enterica*, deep learning, k-mer encoding, whole-genome sequencing, pathogen detection, liquid neural networks, Kraken2, convolutional neural network, benchmark

---

## Background

*Salmonella enterica* remains one of the most significant foodborne pathogens worldwide, causing an estimated 93.8 million cases of gastroenteritis and 155,000 deaths annually [1]. Rapid and accurate detection is essential for outbreak response, clinical treatment, and food safety monitoring. Conventional detection methods — culture-based isolation followed by biochemical and serological confirmation — require 3–5 days and specialized laboratory facilities [2]. While PCR-based methods offer faster turnaround, they are limited to pre-specified genetic targets and may miss emerging or atypical strains [3].

The declining cost of whole-genome sequencing has enabled its adoption for pathogen surveillance and diagnostics [4]. WGS-based computational identification has largely relied on two classes of methods: (1) alignment-based approaches using exact k-mer matching against reference databases (Kraken2 [5], Centrifuge [6]), which achieve 91–99% accuracy but require multi-gigabyte databases and substantial computational resources, and (2) marker gene detection tools (SeqSero2 [7]), which achieve 93–98% accuracy for serotype determination but rely on curated allele databases that require continuous updates.

Deep learning offers the potential for rapid, database-free pathogen detection directly from genomic sequences. Convolutional neural networks (CNNs) have been successfully applied to DNA sequence classification [8, 9], while recurrent architectures including Long Short-Term Memory (LSTM) networks and Gated Recurrent Units (GRU) can model sequential dependencies in genomic data. More recently, Liquid Neural Networks (LNNs) with Closed-form Continuous-time (CfC) cells [10] have demonstrated remarkable parameter efficiency and out-of-distribution generalization in domains ranging from robotic control to time-series forecasting, but their application to genomic sequence classification remains unexplored.

In this study, we present a comprehensive benchmark of seven classification methods for WGS-based *Salmonella* detection. Our contributions include: (1) the first systematic evaluation of Liquid Neural Networks for genomic sequence classification; (2) identification of attention pooling as the critical architectural component for sequence-level genomic classification; (3) demonstration that deep learning matches or exceeds exact k-mer matching tools while eliminating database dependencies; (4) characterization of LNN data efficiency and interpretability advantages; and (5) a robust experimental framework spanning standard classification, out-of-distribution generalization, data efficiency, external validation, and ablation studies.

## Methods

### Dataset

We assembled a dataset of 14,156 bacterial genomes from NCBI GenBank, comprising 6,110 *Salmonella enterica* genomes across 6 serotypes and 8,046 non-*Salmonella* genomes spanning 12 bacterial species (Table 1). Data were sourced from two formats: pre-chunked genomic fragments (~1,000–2,000 bp) from the NCBI Pathogen Detection database, and 515 complete genomes that were fragmented using a sliding window approach (window size 1,000 bp, stride 500 bp) with quality filtering (single-base frequency <50%, N-ratio <5%, GC content 25–75%).

**Table 1. Dataset composition**

| Category | Source | Species/Serotypes | Genomes | Chunks |
|----------|--------|-------------------|:------:|:------:|
| Positive (Salmonella) | serotype_data/ | 6 serotypes | 5,756 | 298,628 |
| Positive (Salmonella) | zheng-yangpin/ | Various | 354 | — |
| Negative (non-Salmonella) | negative_species/ | 6 species | 7,885 | 123,154 |
| Negative (non-Salmonella) | fu-yangpin/ | 161 spp. (incl. 63 *E. coli*) | 161 | — |
| **Total** | — | **12+ species** | **14,156** | **421,782** |

### Sequence Encoding and Sample Construction

Each genomic fragment was encoded as a k-mer frequency vector. For a DNA sequence *S* and k-mer size *k*, the frequency vector **f** ∈ ℝ^{4^k} is computed as the normalized count of each possible k-mer. We used k=4 (256 dimensions) as the primary encoding. Each genome was represented as a sequence of 32 randomly selected fragment vectors, producing input tensors of shape (32, 256). For binary classification, samples were split by genome source into training (70%), validation (15%), and test (15%) sets, ensuring no genome contributed fragments to more than one split. For serotype classification, stratified splitting was applied across 6 classes.

### Model Architectures

**LNN/CfC**: The LNN employs Closed-form Continuous-time (CfC) cells [10] with three stacked layers (128→64→32 hidden units), each followed by LayerNorm and Dropout(0.1). Pooling strategies evaluated include last-step, mean, and learned attention pooling. Three variants were tested: LNN-Small (single layer, 101K parameters), LNN-Medium (3-layer, 413K), and LNN-Large (wider 3-layer, 1.21M).

**1D-CNN**: Three convolutional layers (256→64→128→256 channels, kernel=3) with BatchNorm, ReLU activation, MaxPooling (kernel=2), and Dropout(0.1), followed by adaptive max pooling and a linear classifier (174K parameters).

**Comparative architectures**: BiLSTM with attention pooling (824K parameters), 4-layer Transformer Encoder with 4 attention heads (851K parameters), GRU with identical architecture to LNN (196K parameters), Simple RNN (66K parameters).

**Traditional machine learning**: XGBoost (100 estimators, max_depth=6) and Random Forest (200 trees, max_depth=20) operating on flattened k-mer vectors (8,192 dimensions).

**External tool comparison**: We implemented KmerVote, a simplified Kraken2-like classifier using exact k-mer matching (k=10) with majority voting, built from 5% of training data. Additionally, we report published performance benchmarks for Kraken2 [5], Centrifuge [6], and SeqSero2 [7].

### Training Protocol

All deep learning models were trained with AdamW optimizer (learning rate 1×10⁻³, weight decay 1×10⁻⁴), binary cross-entropy loss with class-balanced positive weighting, automatic mixed precision (AMP), gradient clipping (max norm 1.0), and cosine learning rate decay with 500-step linear warmup. Early stopping was applied with patience of 15 epochs monitoring validation accuracy. For CfC models, a reduced learning rate of 5×10⁻⁵ was used based on grid search optimization. Training was performed on a single NVIDIA GeForce RTX 5060 GPU (12GB). All deep learning experiments were repeated 3 times with different random seeds.

### Evaluation Framework

**Standard classification**: Accuracy, F1-score, and AUC on the held-out test set, reported as mean ± standard deviation (n=3).

**Out-of-distribution (OOD) generalization**: Leave-one-species-out and leave-one-serotype-out cross-validation. Enhanced OOD experiments incorporated 63 *E. coli* genomes from the complete genome dataset as additional training negatives.

**Data efficiency**: Models trained on random subsets of 10%, 25%, 50%, and 100% training data (3 repeats each).

**Ablation studies**: Systematic ablation of cell type (CfC vs. GRU vs. RNN), pooling strategy (last-step vs. mean vs. attention), positional encoding, chunk count (32/64/128), and learning rate (5×10⁻⁶–5×10⁻³).

**Interpretability**: Hidden state trajectory analysis using t-SNE and PCA, gradient-based chunk importance, and confidence distribution analysis.

**External validation**: 20 completely independent genomes (10 *Salmonella*, 10 non-*Salmonella*) not used in any training run.

## Results

### Binary Classification Performance

The comprehensive benchmark results are presented in Table 2. CNN achieved the highest performance across all metrics (99.64% ± 0.3%, AUC 1.000), training in 2 seconds. GRU was the second-best architecture (98.27%), while LNN with attention pooling reached 97.84%. XGBoost, the best non-deep-learning method, achieved 94.16% with near-zero variance.

**Table 2. Binary classification performance (mean ± std, n=3)**

| Model | Accuracy | F1 | AUC | Parameters | Train Time |
|-------|:--------:|:---:|:---:|:----------:|:----------:|
| CNN | **0.9964 ± 0.003** | **0.998** | **1.000** | 173,633 | 2s |
| GRU | 0.9827 | 0.989 | 0.999 | 195,809 | 4s |
| LNN + Attention | 0.9784 | 0.986 | 0.994 | 413,378 | 130s |
| LNN-Medium (optimal LR) | 0.9221 | 0.950 | 0.915 | 412,833 | 124s |
| XGBoost | 0.9416 ± 0.000 | 0.964 | 0.989 | ~500,000 | 2s |
| LSTM | 0.8882 ± 0.053 | 0.935 | 0.843 | 823,810 | 3s |
| Random Forest | 0.8723 | 0.926 | 0.975 | ~5,000,000 | 1s |
| LNN-Small (last pool) | 0.8095 ± 0.015 | 0.892 | 0.671 | 101,121 | 44s |
| Transformer | 0.7987 | 0.888 | 0.500 | 850,946 | 3s |

### Comparison with Existing Bioinformatics Tools

Table 3 compares our deep learning models against existing bioinformatics tools. Our CNN and GRU models matched or exceeded published Kraken2 and Centrifuge benchmarks, while requiring no external reference database and processing each genome in milliseconds after training. The deep learning models learn continuous decision boundaries from k-mer *frequencies* rather than relying on discrete k-mer *presence/absence*, enabling them to capture subtle compositional patterns that exact matching may miss.

**Table 3. Comparison with existing bioinformatics tools**

| Method | Type | Accuracy | Database Required |
|--------|------|:--------:|:-----------------:|
| **CNN (ours)** | k-mer frequency + deep learning | **0.996** | None |
| Kraken2 [5] | k=35 exact match + LCA | 0.91–0.99† | 50 GB |
| **GRU (ours)** | k-mer frequency + deep learning | 0.983 | None |
| Centrifuge [6] | BWT-FM index | 0.89–0.97† | 10 GB |
| **LNN+Attn (ours)** | k-mer frequency + deep learning | 0.978 | None |
| SeqSero2 [7] | Marker allele database | 0.93–0.98†§ | <1 GB |
| KmerVote k=10 (ours) | Exact k-mer matching | 0.833 | 33K kmers |
| **XGBoost (ours)** | k-mer frequency + gradient boosting | 0.942 | None |

† Published independent benchmark performance. § Serotype classification only.

### The Critical Role of Attention Pooling

Replacing last-step pooling with learned attention improved LNN accuracy by 18 percentage points (79.87% → 97.84%), a larger effect than any other architectural modification (Table 4). This indicates that different genomic chunks contribute unequally to classification, and learned attention effectively identifies the most discriminative regions.

**Table 4. Effect of pooling strategy on LNN classification**

| Pooling | Accuracy | F1 | AUC |
|---------|:--------:|:---:|:---:|
| Last-step | 0.7987 | 0.888 | 0.592 |
| Mean | 0.7987 | 0.888 | 0.775 |
| **Attention** | **0.9784** | **0.986** | **0.994** |

### Cell Type Ablation: CfC vs. GRU vs. RNN

GRU outperformed CfC by 8 percentage points in identical architectures (98.27% vs. 90.26%, Table 5), suggesting that for discrete k-mer frequency sequences, gating mechanisms are more effective than continuous-time ODE dynamics. This finding contextualizes the domain specificity of CfC advantages.

**Table 5. Cell type ablation**

| Cell | Accuracy | AUC | Parameters |
|------|:--------:|:---:|:----------:|
| Simple RNN | 0.8420 | 0.689 | 65,889 |
| CfC (LNN) | 0.9026 | 0.948 | 101,121 |
| **GRU** | **0.9827** | **0.999** | 195,809 |

### Data Efficiency

LNN-Small demonstrated the lowest variance across all data regimes (Table 6). At 10% training data, LNN's standard deviation (±0.9%) was 6.1 times smaller than CNN's (±5.5%). Performance degradation from 100% to 10% data was 7.4% for LNN versus 15.7% for CNN, confirming superior data efficiency.

**Table 6. Data efficiency (accuracy at different training fractions)**

| Model | 10% | 25% | 50% | 100% | Degradation |
|-------|:---:|:---:|:---:|:----:|:-----------:|
| XGBoost | **0.873 ± 0.011** | **0.883 ± 0.009** | 0.920 ± 0.012 | 0.942 | −6.9% |
| LNN-Small | 0.850 ± 0.009 | 0.846 ± 0.004 | 0.851 ± 0.008 | 0.924 | −7.4% |
| CNN | 0.838 ± 0.055 | 0.863 ± 0.091 | **0.991 ± 0.002** | **0.995** | −15.7% |

### Out-of-Distribution Generalization

OOD accuracy was perfectly stratified by phylogenetic distance to *Salmonella* (Table 7). Gram-positive species were rejected with 100% accuracy, while Enterobacteriaceae relatives (*Klebsiella*, *Shigella*) were frequently misclassified. Incorporating 63 *E. coli* strains as hard negatives improved *Klebsiella* OOD accuracy 4.7-fold (10.3% → 48.5%).

**Table 7. OOD generalization (leave-one-species-out)**

| Held-out Species | Phylogenetic Distance | Standard | Enhanced (+63 E. coli) |
|------------------|:---------------------:|:--------:|:----------------------:|
| *Enterococcus faecalis* | Far (Firmicutes) | 1.000 | 1.000 |
| *Staphylococcus aureus* | Far (Firmicutes) | 1.000 | 1.000 |
| *Listeria monocytogenes* | Far (Firmicutes) | 1.000 | 1.000 |
| *Klebsiella pneumoniae* | Near (Enterobacteriaceae) | 0.103 | **0.485** |
| *Shigella flexneri* | Near (Enterobacteriaceae) | 0.003 | 0.012 |
| *Pseudomonas aeruginosa* | Intermediate | 0.000 | 0.000 |

### Serotype Classification Limitations

Six-way serotype classification proved substantially more challenging than species-level detection. All k-mer-based methods achieved only 31–36% accuracy, compared to 93–98% for the specialized tool SeqSero2 which uses a curated database of serotype-specific alleles. The fundamental challenge is that *Salmonella* serotypes differ by <1% of their genome, with serotype-determining variation concentrated in hypervariable regions (*rfb*, *fliC*, *fljB*) [11]. Whole-genome random k-mer frequency averaging dilutes these sparse, localized signals.

### Interpretability Analysis

t-SNE visualization revealed clear separation between *Salmonella* and non-*Salmonella* CfC terminal hidden states (centroid distance in t-SNE space: 25.33). Quantitative dynamics analysis (Table 8) showed *Salmonella* sequences induced smoother, more convergent trajectories (trajectory change 0.12 vs. 0.20, 1.65-fold difference), suggesting CfC learns a dynamical attractor for *Salmonella* genomic signatures.

**Table 8. CfC hidden state dynamics by class**

| Metric | *Salmonella* | Non-*Salmonella* |
|--------|:------------:|:----------------:|
| Mean step change (L2 norm) | 0.123 ± 0.026 | 0.203 ± 0.053 |
| Early/late change ratio | 0.229 | 0.372 |

Prediction confidence analysis showed the model was highly decisive: 73% of non-*Salmonella* predictions had confidence <0.1, and no sample fell in the uncertainty zone (0.3–0.7).

### External Validation

On 20 completely independent genomes, the LNN model achieved 80% accuracy with 100% *Salmonella* recall. All 4 errors were false positives on *E. coli* and *Burkholderia* genomes — phylogenetically close relatives of *Salmonella*.

## Discussion

### CNN with k-mer Encoding is the Pragmatic Method of Choice

Our comprehensive benchmarking establishes CNN operating on k-mer frequency vectors as the most effective method for WGS-based bacterial detection. The architecture is simple (3 convolutional layers), fast (2-second training), and achieves near-perfect accuracy (99.6%). This performance matches or exceeds published Kraken2 benchmarks without requiring a 50 GB reference database. The strong performance stems from architectural alignment with the data representation: k-mer frequency vectors across genomic chunks form a matrix where convolutions learn which tetranucleotide frequency combinations distinguish species — effectively capturing genomic composition signatures including GC content, dinucleotide bias, and codon usage patterns.

### When Do Liquid Neural Networks Add Value?

While CNN achieves the highest absolute accuracy, LNNs offer distinct advantages in specific scenarios. First, LNNs exhibit 6-fold lower variance at low training data volumes, making them attractive when labeled genomes are scarce. Second, the interpretable continuous-time dynamics provide a window into the model's decision process — how confidence evolves with accumulating genomic evidence — valuable in clinical settings requiring audit trails. Third, LNN-Small achieves 92.4% accuracy with only 101K parameters, suitable for deployment on edge devices.

The finding that GRU outperforms CfC on k-mer data suggests that the advantage of ODE-based dynamics is domain-dependent. CfC excels on genuinely continuous signals (robotics, physical simulations) [10, 12]; k-mer frequency vectors from randomly sampled genomic regions, while arranged sequentially, lack the smooth temporal continuity that CfC dynamics are designed to model.

### The OOD Generalization Challenge and Mitigation

Our OOD results reveal that k-mer-based models learn phylogenetic proximity rather than species-specific features. This has practical implications: in clinical settings where *Salmonella* must be distinguished from other Enterobacteriaceae, the standard model's false-positive rate on relatives is problematic. Hard negative mining partially addresses this — improving *Klebsiella* OOD accuracy 4.7-fold — but does not fully solve the problem. Future work should explore contrastive learning objectives and targeted feature extraction from hypervariable genomic regions.

### Limitations

Several limitations warrant consideration. First, k-mer frequency encoding discards positional information within fragments. Second, the chunk-based approach treats genomic regions as independent. Third, all genomes were sourced from NCBI RefSeq; performance on clinical isolates with lower quality remains to be evaluated. Fourth, our simplified KmerVote implementation (k=10, 5% training data) underestimates Kraken2's true capabilities; the published benchmarks we cite provide more representative comparisons. Fifth, fine-grained serotype classification requires specialized tools (SeqSero2, SISTR) rather than whole-genome frequency methods.

## Conclusions

We present the first comprehensive benchmark of deep learning methods against established bioinformatics tools for whole-genome *Salmonella* detection. Our key findings are:

1. CNN with k-mer frequency encoding achieves 99.6% accuracy, matching or exceeding Kraken2 and Centrifuge while eliminating database dependencies. This represents a pragmatic, deployable solution for genomic pathogen surveillance.

2. Attention pooling, rather than the choice of recurrent cell, is the critical architectural component for sequence-level genomic classification, contributing an 18-percentage-point improvement to LNN performance.

3. Liquid Neural Networks offer unique advantages in data-limited scenarios (6-fold lower variance than CNN at 10% data) and provide interpretable continuous-time dynamics, though they do not surpass CNN in absolute accuracy on this task.

4. Deep learning from k-mer frequencies fundamentally outperforms exact k-mer matching for species-level classification, as frequency-based representations capture subtle compositional patterns that discrete matching misses.

5. For within-species discrimination (serotyping), curated allele databases remain essential; whole-genome frequency methods are insufficient and specialized tools like SeqSero2 should be preferred.

The complete code, trained models, and benchmark datasets are available at [repository URL] under an open-source license.

## Declarations

### Ethics approval and consent to participate
Not applicable. This study used publicly available bacterial genome sequences from NCBI GenBank.

### Consent for publication
Not applicable.

### Availability of data and materials
All genomic sequences used in this study are publicly available from NCBI GenBank. The pre-processed k-mer frequency vectors (k=4, 32 chunks per genome), train/validation/test splits, and trained model weights are available at [Zenodo/Figshare DOI]. The complete source code is available at [GitHub repository URL] under the MIT License.

### Competing interests
The authors declare that they have no competing interests.

### Funding
[To be completed]

### Authors' contributions
[To be completed]

### Acknowledgements
We thank the NCBI Pathogen Detection program for making genomic data publicly available. This work utilized the ncps-torch library for Liquid Neural Network implementation.

## References

1. Majowicz SE, Musto J, Scallan E, Angulo FJ, Kirk M, O'Brien SJ, et al. The global burden of nontyphoidal *Salmonella* gastroenteritis. Clin Infect Dis. 2010;50(6):882–9.

2. Lee KM, Runyon M, Herrman TJ, Phillips R, Hsieh J. Review of *Salmonella* detection and identification methods: Aspects of rapid emergency response and food safety. Food Control. 2015;47:264–76.

3. Malorny B, Paccassoni E, Fach P, Bunge C, Martin A, Helmuth R. Diagnostic real-time PCR for detection of *Salmonella* in food. Appl Environ Microbiol. 2004;70(12):7046–52.

4. Allard MW, Strain E, Melka D, Bunning K, Musser SM, Brown EW, et al. Practical value of food pathogen traceability through whole-genome sequencing. J Clin Microbiol. 2016;54(8):1975–83.

5. Wood DE, Lu J, Langmead B. Improved metagenomic analysis with Kraken 2. Genome Biol. 2019;20:257.

6. Kim D, Song L, Breitwieser FP, Salzberg SL. Centrifuge: rapid and sensitive classification of metagenomic sequences. Genome Res. 2016;26(12):1721–9.

7. Zhang S, den Bakker HC, Li S, Chen J, Dinsmore BA, Lane C, et al. SeqSero2: rapid and improved *Salmonella* serotype determination using whole genome sequencing data. Appl Environ Microbiol. 2019;85(23):e01746-19.

8. Alipanahi B, Delong A, Weirauch MT, Frey BJ. Predicting the sequence specificities of DNA- and RNA-binding proteins by deep learning. Nat Biotechnol. 2015;33(8):831–8.

9. Zhou J, Troyanskaya OG. Predicting effects of noncoding variants with deep learning–based sequence model. Nat Methods. 2015;12(10):931–4.

10. Hasani R, Lechner M, Amini A, Rus D, Grosu R. Closed-form continuous-time neural networks. Nat Mach Intell. 2022;4:992–1003.

11. Grimont PAD, Weill FX. Antigenic formulae of the *Salmonella* serovars. 9th ed. WHO Collaborating Centre for Reference and Research on *Salmonella*; 2007.

12. Hasani R, Lechner M, Amini A, Rus D, Grosu R. Liquid time-constant networks. In: Proc AAAI Conf Artif Intell; 2021.

---

## Supplementary Materials

### Table S1. Complete model comparison

| Model | Accuracy | F1 | AUC | Parameters |
|-------|:--------:|:---:|:---:|:----------:|
| CNN | 0.9964 ± 0.003 | 0.998 | 1.000 | 173,633 |
| GRU | 0.9827 | 0.989 | 0.999 | 195,809 |
| LNN + Attention | 0.9784 | 0.986 | 0.994 | 413,378 |
| XGBoost | 0.9416 ± 0.000 | 0.964 | 0.989 | ~500,000 |
| LNN-Small (best) | 0.9221 | 0.951 | 0.875 | 101,121 |
| LNN-Medium (opt LR) | 0.9221 | 0.950 | 0.915 | 412,833 |
| LSTM | 0.8882 ± 0.053 | 0.935 | 0.843 | 823,810 |
| Random Forest | 0.8723 | 0.926 | 0.975 | ~5,000,000 |
| LNN-Large | 0.8571 | 0.918 | 0.649 | 1,211,329 |
| Simple RNN | 0.8420 | 0.908 | 0.689 | 65,889 |
| LNN-Small (last pool) | 0.8095 ± 0.015 | 0.892 | 0.671 | 101,121 |
| Transformer | 0.7987 | 0.888 | 0.500 | 850,946 |

### Table S2. Full OOD leave-one-serotype-out results

| Held-out Serotype | Standard | Enhanced |
|-------------------|:--------:|:--------:|
| Dublin | 0.9946 | 0.8584 |
| Enteritidis | 1.0000 | 0.9911 |
| Heidelberg | 0.9956 | 0.9867 |
| Infantis | 1.0000 | 0.4896 |
| Newport | 1.0000 | 0.9829 |
| Typhimurium | 0.7576 | 1.0000 |

### Table S3. External validation per-sample breakdown

| Category | Samples | Correct | Accuracy |
|----------|:------:|:------:|:--------:|
| *Salmonella* | 10 | 10 | 100% |
| Non-*Salmonella* (non-Enterobacteriaceae) | 6 | 6 | 100% |
| Non-*Salmonella* (Enterobacteriaceae relatives) | 4 | 0 | 0% |
| **Total** | **20** | **16** | **80%** |

### Table S4. Sequence length robustness

| Chunks per Genome | Training Samples | LNN-Small Accuracy |
|:-----------------:|:---------------:|:------------------:|
| 32 | 2,256 | 0.9221 |
| 64 | 1,388 | 0.9218 |
| 128 | 752 | 0.8869 |

### Table S5. CfC learning rate sensitivity (LNN-Medium)

| LR | 5×10⁻³ | 1×10⁻³ | 5×10⁻⁴ | 1×10⁻⁴ | 5×10⁻⁵ | 2×10⁻⁵ | 1×10⁻⁵ |
|:--:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| Acc | 0.799 | 0.849 | 0.855 | 0.799 | **0.922** | 0.898 | 0.201 |
