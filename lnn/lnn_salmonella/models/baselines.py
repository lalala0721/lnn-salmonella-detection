"""
基线模型
- CNN1D: 卷积 + 全局池化
- BiLSTM: 双向 LSTM + 注意力池化
- TransformerEncoder: 自注意力编码器

所有模型输入: (batch, num_chunks, kmer_dim)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CNN1DBaseline(nn.Module):
    """1D-CNN 基线"""

    def __init__(self, input_dim: int = 256, num_classes: int = 1,
                 channels: list = None, dropout: float = 0.1):
        super().__init__()
        if channels is None:
            channels = [64, 128, 256]

        layers = []
        in_ch = input_dim
        for out_ch in channels:
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Dropout(dropout),
            ])
            in_ch = out_ch

        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(channels[-1], num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x


class BiLSTMBaseline(nn.Module):
    """双向 LSTM + 注意力池化"""

    def __init__(self, input_dim: int = 256, hidden_dim: int = 128,
                 num_layers: int = 2, num_classes: int = 1, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        lstm_out = hidden_dim * 2
        self.attn = nn.Linear(lstm_out, 1)
        self.fc = nn.Sequential(
            nn.Linear(lstm_out, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes if num_classes > 2 else 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)

        attn_scores = self.attn(lstm_out).squeeze(-1)
        attn_weights = F.softmax(attn_scores, dim=-1).unsqueeze(-1)
        pooled = (lstm_out * attn_weights).sum(dim=1)

        return self.fc(pooled)


class PositionalEncoding(nn.Module):
    """正弦位置编码"""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerBaseline(nn.Module):
    """Transformer Encoder 基线"""

    def __init__(self, input_dim: int = 256, d_model: int = 128,
                 nhead: int = 4, num_layers: int = 4,
                 num_classes: int = 1, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=512, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1),
        )
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes if num_classes > 2 else 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.encoder(x)

        attn_scores = self.pool(x).squeeze(-1)
        attn_weights = F.softmax(attn_scores, dim=-1).unsqueeze(-1)
        pooled = (x * attn_weights).sum(dim=1)

        return self.fc(pooled)
