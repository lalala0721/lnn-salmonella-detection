"""
可学习 Embedding 模型 — 分层架构
- Per-chunk: Embedding → 小型 CfC 编码每个 chunk → chunk_vector
- Cross-chunk: 堆叠 chunk vectors → CfC 聚合 → 分类
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DROPOUT
from ncps.torch import CfC


class HierarchicalEmbeddingLNN(nn.Module):
    """
    分层 Embedding LNN

    输入: (batch, num_chunks, seq_len) token indices
    Step 1: 每个 chunk 独立通过 Embedding + 小型 CfC → chunk embedding (chunk_dim)
    Step 2: 跨 chunk CfC 聚合 → 池化 → 分类
    """

    def __init__(self, vocab_size: int = 5, embed_dim: int = 32,
                 chunk_hidden: int = 32, cross_hidden: list = None,
                 num_classes: int = 6, dropout: float = DROPOUT,
                 pad_idx: int = 4):
        super().__init__()
        if cross_hidden is None:
            cross_hidden = [64, 32]

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.embed_dim = embed_dim
        self.chunk_hidden = chunk_hidden

        # Per-chunk 编码器 (轻量)
        self.chunk_cfc = CfC(embed_dim, chunk_hidden)

        # Cross-chunk 聚合器
        cfc_layers = []
        norms = []
        in_size = chunk_hidden
        for i, h_size in enumerate(cross_hidden):
            cfc_layers.append(CfC(in_size, h_size))
            in_size = h_size
            norms.append(nn.LayerNorm(h_size) if i < len(cross_hidden) - 1 else nn.Identity())
        self.cross_cfc_layers = nn.ModuleList(cfc_layers)
        self.cross_norms = nn.ModuleList(norms)
        self.dropout = nn.Dropout(dropout)

        # 分类头
        final_hidden = cross_hidden[-1]
        out_dim = 1 if num_classes <= 2 else num_classes
        self.classifier = nn.Sequential(
            nn.Linear(final_hidden, final_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_hidden // 2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, num_chunks, seq_len) int64
        """
        batch, num_chunks, seq_len = x.shape

        # === Per-chunk 编码 (并行) ===
        # 展平 chunks: (batch * num_chunks, seq_len)
        x = x.reshape(batch * num_chunks, seq_len)
        x = self.embedding(x)  # (batch*nc, seq_len, embed_dim)
        x, _ = self.chunk_cfc(x)  # (batch*nc, seq_len, chunk_hidden)
        x = x[:, -1, :]  # 每 chunk 取最后时间步 → (batch*nc, chunk_hidden)

        # === Cross-chunk 聚合 ===
        x = x.reshape(batch, num_chunks, self.chunk_hidden)  # (batch, nc, chunk_dim)
        for cfc, norm in zip(self.cross_cfc_layers, self.cross_norms):
            h, _ = cfc(x)
            h = norm(h)
            h = self.dropout(h)
            x = h

        # 池化 + 分类
        pooled = x.mean(dim=1)  # (batch, final_hidden)
        return self.classifier(pooled)


class HierarchicalEmbeddingCNN(nn.Module):
    """
    分层 Embedding CNN
    Per-chunk: Embedding → 1D-CNN → chunk vector
    Cross-chunk: 1D-CNN → pool → classify
    """

    def __init__(self, vocab_size: int = 5, embed_dim: int = 32,
                 chunk_dim: int = 64, cross_channels: list = None,
                 num_classes: int = 6, dropout: float = DROPOUT,
                 pad_idx: int = 4):
        super().__init__()
        if cross_channels is None:
            cross_channels = [128, 256]

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # Per-chunk CNN
        self.chunk_conv = nn.Sequential(
            nn.Conv1d(embed_dim, chunk_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(chunk_dim), nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )

        # Cross-chunk CNN
        cross_layers = []
        in_ch = chunk_dim
        for out_ch in cross_channels:
            cross_layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_ch), nn.ReLU(),
                nn.MaxPool1d(2), nn.Dropout(dropout),
            ])
            in_ch = out_ch
        self.cross_conv = nn.Sequential(*cross_layers)
        self.global_pool = nn.AdaptiveMaxPool1d(1)

        out_dim = 1 if num_classes <= 2 else num_classes
        self.fc = nn.Linear(cross_channels[-1], out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, num_chunks, seq_len = x.shape

        # Per-chunk: (batch*nc, seq_len) → embed → conv → (batch*nc, chunk_dim)
        x = x.reshape(batch * num_chunks, seq_len)
        x = self.embedding(x)  # (batch*nc, seq_len, embed_dim)
        x = x.transpose(1, 2)  # (batch*nc, embed_dim, seq_len)
        x = self.chunk_conv(x).squeeze(-1)  # (batch*nc, chunk_dim)

        # Cross-chunk: (batch, chunk_dim, num_chunks)
        x = x.reshape(batch, num_chunks, -1).transpose(1, 2)  # (batch, chunk_dim, nc)
        x = self.cross_conv(x)  # (batch, cross_ch, reduced_nc)
        x = self.global_pool(x).squeeze(-1)  # (batch, cross_ch)
        return self.fc(x)


def create_embedding_model(model_type: str, vocab_size: int = 5,
                           num_classes: int = 6, **kwargs) -> nn.Module:
    if model_type == 'emb-lnn':
        return HierarchicalEmbeddingLNN(vocab_size=vocab_size, num_classes=num_classes, **kwargs)
    elif model_type == 'emb-cnn':
        return HierarchicalEmbeddingCNN(vocab_size=vocab_size, num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown: {model_type}")
