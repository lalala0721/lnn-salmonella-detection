"""
DNA 序列 Tokenizer
- 字符级: A/T/C/G → 0/1/2/3, N/其他 → 4
- k-mer 级: 滑动窗口 k-mer → token index (0 ~ 4^k-1)
"""
import numpy as np
from typing import List


class DNATokenizer:
    """字符级 DNA Tokenizer"""

    BASE_TO_IDX = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
    IDX_TO_BASE = {0: 'A', 1: 'T', 2: 'C', 3: 'G'}
    VOCAB_SIZE = 5
    PAD_IDX = 4

    def encode(self, sequence: str, max_len: int = 500) -> np.ndarray:
        """
        将 DNA 序列转换为 token 索引序列

        Args:
            sequence: ATCG 字符串
            max_len: 最大长度 (截断或补齐)

        Returns:
            (max_len,) int64 数组
        """
        sequence = sequence.upper().strip()
        tokens = np.full(max_len, self.PAD_IDX, dtype=np.int64)

        for i, base in enumerate(sequence[:max_len]):
            tokens[i] = self.BASE_TO_IDX.get(base, self.PAD_IDX)

        return tokens

    def encode_batch(self, sequences: List[str], max_len: int = 500) -> np.ndarray:
        return np.stack([self.encode(s, max_len) for s in sequences], axis=0)

    def decode(self, tokens: np.ndarray) -> str:
        return ''.join(self.IDX_TO_BASE.get(int(t), 'N') for t in tokens if t != self.PAD_IDX)


class KmerTokenizer:
    """k-mer 级 Tokenizer: 将 DNA 序列转为 k-mer token 序列"""

    def __init__(self, k: int = 4, stride: int = 4):
        """
        Args:
            k: k-mer 长度
            stride: 滑动步长
        """
        self.k = k
        self.stride = stride
        self.vocab_size = 4 ** k
        self.pad_idx = self.vocab_size

        from itertools import product
        self.kmer_to_idx = {}
        for i, kmer in enumerate(product('ATCG', repeat=k)):
            self.kmer_to_idx[''.join(kmer)] = i

    def encode(self, sequence: str, max_len: int = 200) -> np.ndarray:
        """
        Args:
            sequence: DNA 序列
            max_len: 最大 token 数

        Returns:
            (max_len,) int64 数组
        """
        sequence = sequence.upper().strip()
        tokens = np.full(max_len, self.pad_idx, dtype=np.int64)

        idx = 0
        for i in range(0, len(sequence) - self.k + 1, self.stride):
            if idx >= max_len:
                break
            kmer = sequence[i:i + self.k]
            tokens[idx] = self.kmer_to_idx.get(kmer, self.pad_idx)
            idx += 1

        return tokens
