"""
DNA 序列编码模块
- k-mer 频率向量 (主要方法)
- one-hot 编码 (备选)
"""
import numpy as np
from itertools import product
from typing import Dict, List, Optional


class KmerEncoder:
    """k-mer 频率编码器：将 DNA 序列转换为 k-mer 频率向量"""

    VALID_BASES = {'A', 'T', 'C', 'G'}

    def __init__(self, k: int = 4):
        """
        Args:
            k: k-mer 长度，推荐 3-6
        """
        self.k = k
        self.all_kmers = [''.join(p) for p in product('ATCG', repeat=k)]
        self.kmer_to_idx = {kmer: i for i, kmer in enumerate(self.all_kmers)}
        self.dim = 4 ** k  # 256 for k=4, 1024 for k=5

    def encode(self, sequence: str, normalize: bool = True) -> np.ndarray:
        """
        将一条 DNA 序列转换为 k-mer 频率向量

        Args:
            sequence: DNA 序列字符串 (A/T/C/G)
            normalize: 是否归一化（除以总 k-mer 数量）

        Returns:
            shape (4**k,) 的 numpy 数组
        """
        sequence = sequence.upper().strip()
        freqs = np.zeros(self.dim, dtype=np.float32)

        total = 0
        for i in range(len(sequence) - self.k + 1):
            kmer = sequence[i:i + self.k]
            if kmer in self.kmer_to_idx:
                freqs[self.kmer_to_idx[kmer]] += 1.0
                total += 1

        if total == 0:
            return freqs  # return zero vector instead of NaN

        if normalize:
            freqs /= total

        return freqs

    def encode_batch(self, sequences: List[str], normalize: bool = True) -> np.ndarray:
        """批量编码"""
        return np.stack([self.encode(s, normalize) for s in sequences], axis=0)

    def get_kmer_list(self) -> List[str]:
        """返回所有 k-mer 名称列表"""
        return self.all_kmers


class OneHotEncoder:
    """One-hot 编码：每个碱基映射为 4 维向量 (备选方案)"""

    BASE_TO_IDX = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
    IDX_TO_BASE = {0: 'A', 1: 'T', 2: 'C', 3: 'G'}

    def __init__(self, seq_length: int = 1024):
        self.seq_length = seq_length
        self.dim = 4

    def encode(self, sequence: str) -> np.ndarray:
        """
        Args:
            sequence: DNA 序列字符串

        Returns:
            shape (seq_length, 4) 的 one-hot 矩阵
        """
        sequence = sequence.upper().strip()
        matrix = np.zeros((self.seq_length, 4), dtype=np.float32)

        for i, base in enumerate(sequence[:self.seq_length]):
            if base in self.BASE_TO_IDX:
                matrix[i, self.BASE_TO_IDX[base]] = 1.0

        return matrix

    def encode_batch(self, sequences: List[str]) -> np.ndarray:
        return np.stack([self.encode(s) for s in sequences], axis=0)
