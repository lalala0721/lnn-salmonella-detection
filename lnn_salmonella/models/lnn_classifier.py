"""
LNN (Liquid Neural Network) 分类器
基于 CfC (Closed-form Continuous-time) 细胞

架构:
  k-mer 频率序列 (num_chunks, kmer_dim)
    → 输入投影 (Linear)
    → CfC Stack (多层连续时间动力学)
    → 序列池化 (最后步 / 注意力)
    → 分类头
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ncps.torch import CfC
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import KMER_DIM, NUM_CHUNKS_PER_GENOME, DROPOUT


class LNNClassifier(nn.Module):
    """
    液态神经网络分类器 — CfC Stack

    输入: (batch, num_chunks, kmer_dim) k-mer 频率序列
    输出: (batch, num_classes) 预测 logits
    """

    def __init__(
        self,
        input_dim: int = KMER_DIM,          # k-mer 维度 (256 for k=4)
        hidden_sizes: list = None,           # CfC 各层隐状态, e.g. [128, 64, 32]
        num_classes: int = 1,                # 1 = 二分类, >2 = 多分类
        dropout: float = DROPOUT,
        pool_mode: str = 'last',             # 'last' | 'attention' | 'mean'
        mixed_memory: bool = True,           # CfC 混合记忆模式
        use_layer_norm: bool = True,         # 层间 LayerNorm
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [128, 64, 32]
        self.hidden_sizes = hidden_sizes
        self.pool_mode = pool_mode
        self.num_classes = num_classes

        # 输入投影：k-mer 空间 → LNN 隐空间
        self.input_proj = nn.Linear(input_dim, hidden_sizes[0])

        # CfC 层堆叠
        cfc_layers = []
        norms = []
        in_size = hidden_sizes[0]
        for i, h_size in enumerate(hidden_sizes):
            cfc_layers.append(
                CfC(in_size, h_size, mixed_memory=mixed_memory)
            )
            in_size = h_size
            if use_layer_norm and i < len(hidden_sizes) - 1:
                norms.append(nn.LayerNorm(h_size))
            else:
                norms.append(nn.Identity())

        self.cfc_layers = nn.ModuleList(cfc_layers)
        self.norms = nn.ModuleList(norms)
        self.dropout = nn.Dropout(dropout)

        # 池化
        final_hidden = hidden_sizes[-1]
        if pool_mode == 'attention':
            self.pool_attn = nn.Sequential(
                nn.Linear(final_hidden, final_hidden // 2),
                nn.Tanh(),
                nn.Linear(final_hidden // 2, 1),
            )

        # 分类头
        out_dim = 1 if num_classes <= 2 else num_classes
        self.classifier = nn.Sequential(
            nn.Linear(final_hidden, final_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_hidden // 2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_chunks, kmer_dim) k-mer 频率序列

        Returns:
            logits: (batch, 1) or (batch, num_classes)
        """
        # 输入投影
        x = self.input_proj(x)  # (batch, num_chunks, hidden[0])

        # CfC 层堆叠 — 连续时间动力学
        for cfc, norm in zip(self.cfc_layers, self.norms):
            h, _ = cfc(x)            # CfC returns (output, hidden_state)
            h = norm(h)
            h = self.dropout(h)
            x = h                    # 残差连接隐式通过 CfC 内部实现

        # 序列池化
        if self.pool_mode == 'last':
            pooled = x[:, -1, :]     # 最后时间步
        elif self.pool_mode == 'mean':
            pooled = x.mean(dim=1)   # 均值池化
        elif self.pool_mode == 'attention':
            scores = self.pool_attn(x).squeeze(-1)  # (batch, num_chunks)
            weights = F.softmax(scores, dim=-1).unsqueeze(-1)
            pooled = (x * weights).sum(dim=1)
        else:
            pooled = x[:, -1, :]

        # 分类
        out = self.classifier(pooled)  # (batch, out_dim)
        return out

    def get_hidden_trajectory(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取隐藏状态轨迹（用于可解释性分析）

        Returns:
            trajectory: (batch, num_chunks, final_hidden)
        """
        x = self.input_proj(x)
        for cfc, norm in zip(self.cfc_layers, self.norms):
            h, _ = cfc(x)
            h = norm(h)
            x = h
        return x  # 最后一层的完整时间轨迹


def count_parameters(model: nn.Module) -> int:
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_lnn_model(
    variant: str = 'medium',
    num_classes: int = 1,
    pool_mode: str = 'last',
    input_dim: int = None,
) -> LNNClassifier:
    """
    创建不同规模的 LNN 模型

    Args:
        variant: 'small' | 'medium' | 'large'
        num_classes: 类别数
        pool_mode: 池化方式
        input_dim: 输入维度 (None 则用默认)
    """
    from config import KMER_DIM
    if input_dim is None:
        input_dim = KMER_DIM

    configs = {
        'small': {'hidden_sizes': [64]},
        'medium': {'hidden_sizes': [128, 64, 32]},
        'large': {'hidden_sizes': [256, 128, 64]},
    }
    cfg = configs.get(variant, configs['medium'])
    return LNNClassifier(
        input_dim=input_dim,
        hidden_sizes=cfg['hidden_sizes'],
        num_classes=num_classes,
        pool_mode=pool_mode,
    )
