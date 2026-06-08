# 基于液态神经网络的沙门氏菌基因组预测

利用液态神经网络 (Liquid Neural Network, LNN) 根据细菌基因序列预测是否为沙门氏菌（*Salmonella enterica*），并区分 6 种血清型。

## 项目结构

```
e:\LNN\
├── lnn_salmonella/           # 核心代码
│   ├── data/                 # 数据处理
│   │   ├── encoding.py       # k-mer / one-hot 编码
│   │   ├── dataset.py        # PyTorch Dataset
│   │   ├── tokenizer.py      # DNA → token 序列
│   │   └── preprocessing*.py # 缓存构建 (预切/原始/血清型/增强/OOD)
│   ├── models/               # 模型
│   │   ├── lnn_classifier.py # LNN (CfC) 分类器
│   │   ├── baselines.py      # CNN / LSTM / Transformer
│   │   └── embedding_models.py # 可学习 Embedding 模型
│   ├── train.py              # 训练脚本 (含缓存模式/多分类/Embedding)
│   ├── evaluate.py           # 评估脚本
│   ├── ood_experiment.py     # OOD 留一法测试
│   ├── ood_enhanced.py       # 增强 OOD 测试
│   ├── data_efficiency.py    # 数据效率实验
│   ├── ablation.py           # 消融实验
│   ├── repeat3.py            # 3 次重复实验
│   ├── interpretability.py   # 可解释性分析
│   ├── plot_efficiency.py    # 可视化图表
│   └── config.py             # 全局配置
├── fu-yangpin/               # 161 个阴性完整基因组 (含 63 E. coli)
├── zheng-yangpin/            # 354 个沙门氏菌完整基因组
├── negative_species/         # 6 种阴性菌预切 chunks
├── serotype_data/            # 6 种血清型预切 chunks
├── venv/                     # Python 3.14 + PyTorch 2.11 + CUDA 12.8
└── results/                  # 结果输出
```

## 环境

| 组件 | 版本 |
|------|------|
| Python | 3.14.4 |
| PyTorch | 2.11.0+cu128 |
| CUDA | 12.8 |
| GPU | NVIDIA GeForce RTX 5060 (12GB) |
| ncps (CfC) | 0.0.2 |

## 数据

| 数据源 | 类型 | 基因组数 | Chunks |
|--------|------|:------:|:------:|
| `serotype_data/` | 阳性 (6 血清型) | 5,756 | 298,628 |
| `negative_species/` | 阴性 (6 物种) | 7,885 | 123,154 |
| `zheng-yangpin/` | 阳性 (完整基因组) | 354 | — |
| `fu-yangpin/` | 阴性 (完整基因组, 含 63 E. coli) | 161 | — |

## 实验一：二分类（是否沙门氏菌）

**k=4 k-mer 频率向量 (256维) × 32 chunks/基因组**

### 完整结果 (mean ± std, n=3)

| 模型 | Accuracy | F1 | AUC | 参数量 | 训练 |
|------|:---:|:---:|:---:|:---:|:---:|
| **CNN** | **99.64%** ±0.3 | 0.998 | 1.000 | 174k | 2s |
| **GRU** | 98.27% | 0.989 | 0.999 | 196k | 4s |
| **LNN+Attn** | 97.84% | 0.986 | 0.994 | 413k | 130s |
| XGBoost | 94.16% ±0.0 | 0.964 | 0.989 | ~500K | 2s |
| LSTM | 88.82% ±5.3 | 0.935 | 0.843 | 824k | 3s |
| LNN-Small (last) | 80.95% ±1.5 | 0.892 | 0.671 | 101k | 44s |
| Random Forest | 87.23% | 0.926 | 0.975 | ~5M | 1s |

### 数据效率

| 模型 | 10% 数据 | 25% | 50% | 100% | 衰减 |
|------|:---:|:---:|:---:|:---:|:---:|
| XGBoost | **87.3%** | **88.3%** | 92.0% | 94.2% | **-6.9%** |
| LNN-Small | 85.0% ±0.9 | 84.6% | 85.1% | 92.4% | -7.4% |
| CNN | 83.8% ±5.5 | 86.3% | **99.1%** | **99.5%** | -15.7% |

> LNN 在低数据下最稳定（std ±0.9% vs CNN ±5.5%），衰减仅 CNN 的一半。

### OOD 泛化

| 留一血清型 | Acc | 留一物种 | Acc |
|:---|:---:|:---|:---:|
| enteritidis | 100% | enterococcus_faecalis | 100% |
| newport | 100% | listeria_monocytogenes | 100% |
| heidelberg | 99.6% | staphylococcus_aureus | 100% |
| dublin | 99.5% | klebsiella_pneumoniae | 10% → **49%** (增强后) |
| typhimurium | 75.8% → **100%** | pseudomonas_aeruginosa | 0% |
| infantis | 100% | shigella_flexneri | 0.3% |

> 加入 E. coli 难负样本后，Klebsiella 召回从 10% 提升到 49%，typhimurium 从 76% 到 100%。

### 消融实验

| 实验 | 方案 | Acc | 结论 |
|------|------|:---:|------|
| ODE 动力学 | CfC vs GRU vs RNN | 90.3% / **98.3%** / 84.2% | GRU > CfC for k-mer |
| 池化策略 | last / mean / **attention** | 79.9% / 79.9% / **97.8%** | 注意力池化是关键 |
| 位置编码 | w/ vs w/o PE | 85.7% vs 85.7% | 无影响 (random chunks) |
| 序列长度 | 32 / 64 / 128 chunks | **92.2%** / 92.2% / 88.7% | 32 最优 |
| CfC 学习率 | 5e-3 → 5e-6 | **5e-5** 最优 | 比默认低 20x |

### 可解释性

- **t-SNE**: 沙门氏菌与非沙门氏菌在隐藏空间中清晰分离 (中心距 25.3)
- **隐藏动力学**: 沙门氏菌轨迹更平滑收敛 (变化量 0.12 vs 0.20)
- **假阳性**: 25 个 FP 样本在隐藏空间中靠近沙门氏菌质心 (confidence 0.78)
- **置信度**: 73% 非沙门氏菌以 >90% 置信度正确分类

### 外部验证

20 个未参与训练的独立基因组 (10 Salmonella + 10 non-Salmonella):
- **Accuracy**: 80% (16/20)
- **Salmonella 召回**: 100% (10/10)
- **假阳性**: 4 个 E. coli 被误判

## 实验二：血清型多分类

6 类血清型区分: dublin, enteritidis, heidelberg, infantis, newport, typhimurium

| 方法 | 模型 | Acc |
|------|------|:---:|
| k=4 k-mer | CNN | 36.4% |
| k=5 k-mer | CNN | 36.4% |
| 可学习 Embedding | CNN | 31.5% |
| 所有方法 | LNN/LSTM | 33.9% (预测多数类) |

> 血清型级的 k-mer 差异太小 (<1% 基因组差异)，全基因组随机 chunk 无法可靠区分。需要针对血清型决定基因区域 (*fliC*, *rfb*) 进行特征提取。

## 关键结论

1. **CNN 是最优模型**: 99.6% 准确率，174K 参数，2 秒训练 — k-mer 频率 + 卷积的高效组合
2. **注意力池化是关键**: LNN 从 80% → 98%，证明了序列加权的重要性
3. **LNN 在小数据下更稳定**: 10% 数据时方差仅 CNN 的 1/6
4. **GRU ≥ CfC** for this task: ODE 动力学的优势在离散 k-mer 序列上未体现
5. **OOD 可通过难负样本改善**: 加入近缘种可显著提升泛化
6. **血清型多分类需要更精细的特征**: 全基因组 k-mer 不足以区分同种亚型

## 复现

```bash
source venv/Scripts/activate

# 构建缓存
python -m lnn_salmonella.data.preprocess_serotype_cache  # 预切数据缓存

# 训练
python -c "from lnn_salmonella.train import train_from_cache; train_from_cache('lnn-small', cache_path='lnn_salmonella/data/cache/kmer4_chunks32.npz')"

# 评估
python lnn_salmonella/evaluate.py --model lnn-small --checkpoint lnn_salmonella/results/lnn-small_best.pt

# 实验
python lnn_salmonella/data_efficiency.py    # 数据效率
python lnn_salmonella/ablation.py           # 消融
python lnn_salmonella/interpretability.py   # 可解释性
```

## 引用

- Hasani et al. "Closed-form Continuous-time Neural Networks" (2022)
- ncps-torch: https://github.com/mlech26l/ncps
