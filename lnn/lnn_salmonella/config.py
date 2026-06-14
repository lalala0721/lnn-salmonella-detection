"""
Global configuration file for LNN Salmonella Detection.

All hyperparameters and data paths are configured here.
Set environment variable LNN_DATA_DIR to override the default data directory.
"""
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent

DATA_DIR = Path(os.environ.get("LNN_DATA_DIR", ROOT / "data"))

SEROTYPE_DIR = DATA_DIR / "serotype_data"
NEGATIVE_DIR = DATA_DIR / "negative_species"
ZHENG_DIR = DATA_DIR / "zheng-yangpin"
FU_DIR = DATA_DIR / "fu-yangpin"

CACHE_DIR = Path(__file__).parent / "data" / "cache"

RESULTS_DIR = Path(__file__).parent / "results"

KMER_K = 4
KMER_DIM = 4 ** KMER_K
SEQ_LENGTH = 1024

NUM_CHUNKS_PER_GENOME = 32

BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.1
EPOCHS = 100
EARLY_STOP_PATIENCE = 15
GRAD_CLIP_NORM = 1.0
WARMUP_STEPS = 500

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

CFC_HIDDEN_SIZES = [128, 64, 32]
LSTM_HIDDEN = 128
CNN_CHANNELS = [64, 128, 256]
TRANSFORMER_DIM = 128
TRANSFORMER_HEADS = 4
TRANSFORMER_LAYERS = 4

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEROTYPE_CLASSES = [
    "dublin", "enteritidis", "heidelberg",
    "infantis", "newport", "typhimurium"
]
NEGATIVE_CLASSES = [
    "enterococcus_faecalis", "klebsiella_pneumoniae",
    "listeria_monocytogenes", "pseudomonas_aeruginosa",
    "shigella_flexneri", "staphylococcus_aureus"
]
NUM_SEROTYPES = len(SEROTYPE_CLASSES)
