"""
全局配置文件
"""
from pathlib import Path

# === 项目路径 ===
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT
SEROTYPE_DIR = DATA_DIR / "serotype_data"
NEGATIVE_DIR = DATA_DIR / "negative_species"
ZHENG_DIR = DATA_DIR / "zheng-yangpin"
FU_DIR = DATA_DIR / "fu-yangpin"

# === k-mer 编码 ===
KMER_K = 4                        # k-mer 长度
KMER_DIM = 4 ** KMER_K            # k-mer 特征维度 (256 for k=4)
SEQ_LENGTH = 1024                 # 每个 chunk 的 DNA 序列固定长度 (用于 one-hot)

# === 序列建模参数 ===
NUM_CHUNKS_PER_GENOME = 32        # 每个基因组取多少个 chunk 作为序列长度
                                   # CfC/LSTM/Transformer 的时间步数

# === 训练参数 ===
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.1
EPOCHS = 100
EARLY_STOP_PATIENCE = 15
GRAD_CLIP_NORM = 1.0
WARMUP_STEPS = 500

# === 数据划分 ===
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# === 模型 ===
CFC_HIDDEN_SIZES = [128, 64, 32]  # CfC 各层隐状态维度
LSTM_HIDDEN = 128
CNN_CHANNELS = [64, 128, 256]
TRANSFORMER_DIM = 128
TRANSFORMER_HEADS = 4
TRANSFORMER_LAYERS = 4

# === 设备 ===
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === 类别映射 ===
SEROTYPE_CLASSES = ["dublin", "enteritidis", "heidelberg", "infantis", "newport", "typhimurium"]
NEGATIVE_CLASSES = ["enterococcus_faecalis", "klebsiella_pneumoniae", "listeria_monocytogenes",
                     "pseudomonas_aeruginosa", "shigella_flexneri", "staphylococcus_aureus"]
NUM_SEROTYPES = len(SEROTYPE_CLASSES)
