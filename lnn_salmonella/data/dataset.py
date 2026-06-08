"""
PyTorch Dataset 类
- KmerSequenceDataset: k-mer 频率序列 (用于 LNN/LSTM/Transformer)
- KmerFlatDataset: k-mer 频率平铺 (用于 XGBoost/RF/MLP)
"""
import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from typing import List, Dict, Optional, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import KMER_K, NUM_CHUNKS_PER_GENOME
from data.encoding import KmerEncoder


def read_fasta(file_path: str) -> str:
    """读取 FASTA 文件，返回序列字符串（跳过头部行）"""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    # 跳过以 > 开头的描述行，拼接剩余行
    seq_lines = [l.strip() for l in lines if not l.startswith('>')]
    return ''.join(seq_lines)


class KmerSequenceDataset(Dataset):
    """
    k-mer 频率序列数据集

    每个样本 = 同一基因组的 N 个 chunk，每个 chunk 编码为 k-mer 频率向量
    输出形状: (num_chunks, kmer_dim)

    适用于: CfC, LSTM, Transformer 等序列模型
    """

    def __init__(
        self,
        samples: List[Dict],
        k: int = KMER_K,
        num_chunks: int = NUM_CHUNKS_PER_GENOME,
        cache_sequences: bool = True,
    ):
        """
        Args:
            samples: build_sequence_samples() 的输出
            k: k-mer 长度
            num_chunks: 每个基因组的 chunk 数量（序列长度/时间步数）
            cache_sequences: 是否预加载所有序列到内存
        """
        self.samples = samples
        self.num_chunks = num_chunks
        self.encoder = KmerEncoder(k=k)
        self.kmer_dim = self.encoder.dim
        self.cache = cache_sequences

        # 预加载
        self._cached_data = []
        if cache_sequences:
            print(f"预加载 {len(samples)} 个样本到内存...")
            for i, sample in enumerate(samples):
                if (i + 1) % 5000 == 0:
                    print(f"  已加载 {i + 1}/{len(samples)}")
                self._cached_data.append(self._load_sample(sample))
            print("预加载完成")

    def _load_sample(self, sample: Dict) -> Tuple[np.ndarray, int]:
        """加载单个样本: 读取 N 个 chunk 并编码为 k-mer 频率向量"""
        kmer_vecs = []
        for chunk_path in sample['chunk_paths']:
            try:
                seq = read_fasta(chunk_path)
                vec = self.encoder.encode(seq, normalize=True)
                kmer_vecs.append(vec)
            except Exception as e:
                # 文件读取失败时用零向量
                kmer_vecs.append(np.zeros(self.kmer_dim, dtype=np.float32))

        # 补齐到 num_chunks（如果不足）
        while len(kmer_vecs) < self.num_chunks:
            kmer_vecs.append(np.zeros(self.kmer_dim, dtype=np.float32))

        data = np.stack(kmer_vecs[:self.num_chunks], axis=0)  # (num_chunks, kmer_dim)
        label = sample['label']
        return data, label

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cache:
            data, label = self._cached_data[idx]
        else:
            data, label = self._load_sample(self.samples[idx])

        return torch.from_numpy(data), torch.tensor(label, dtype=torch.float32)

    def get_metadata(self, idx: int) -> Dict:
        return self.samples[idx].get('metadata', {})


class KmerFlatDataset(Dataset):
    """
    平铺 k-mer 频率数据集

    每个样本 = 单个 chunk 的 k-mer 频率向量（无序列维度）
    输出形状: (kmer_dim,)

    适用于: XGBoost, Random Forest, MLP
    """

    def __init__(
        self,
        df,  # pandas DataFrame with 'abs_path' and 'label' columns
        k: int = KMER_K,
        cache_sequences: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.encoder = KmerEncoder(k=k)
        self.kmer_dim = self.encoder.dim
        self.cache = cache_sequences

        if cache_sequences:
            self._cached_data = []
            print(f"预加载 {len(self.df)} 个 chunk 到内存...")
            for i, row in self.df.iterrows():
                try:
                    seq = read_fasta(row['abs_path'])
                    vec = self.encoder.encode(seq, normalize=True)
                    self._cached_data.append(vec)
                except Exception:
                    self._cached_data.append(np.zeros(self.kmer_dim, dtype=np.float32))
            print("预加载完成")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cache:
            vec = self._cached_data[idx]
        else:
            row = self.df.iloc[idx]
            seq = read_fasta(row['abs_path'])
            vec = self.encoder.encode(seq, normalize=True)

        label = self.df.iloc[idx]['label']
        return torch.from_numpy(vec), torch.tensor(label, dtype=torch.float32)


class CachedKmerDataset(Dataset):
    """
    预计算缓存的 k-mer 序列数据集

    从 .npz 文件加载预编码的 k-mer 数据，避免重复读取 FASTA 文件。
    加载速度极快（秒级），适合多次训练不同模型。
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: (n_samples, num_chunks, kmer_dim) numpy 数组
            y: (n_samples,) numpy 数组
        """
        self.X = X
        self.y = y
        self.kmer_dim = X.shape[-1]
        self.num_chunks = X.shape[1]

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.float32)


def collate_sequence_batch(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """序列数据的 collate function"""
    data, labels = zip(*batch)
    return torch.stack(data, dim=0), torch.stack(labels, dim=0)
