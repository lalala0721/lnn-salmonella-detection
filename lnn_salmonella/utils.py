"""
工具函数：指标计算、学习率调度、日志等
"""
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from typing import Dict, List, Tuple, Optional
import time
from pathlib import Path
import json


class MetricTracker:
    """训练指标追踪器"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.losses = []
        self.predictions = []
        self.targets = []
        self.probabilities = []

    def update(self, loss: float, preds: torch.Tensor, targets: torch.Tensor,
               probs: Optional[torch.Tensor] = None):
        self.losses.append(loss)
        self.predictions.append(preds.detach().cpu().numpy())
        self.targets.append(targets.detach().cpu().numpy())
        if probs is not None:
            self.probabilities.append(probs.detach().cpu().numpy())

    def compute(self, threshold: float = 0.5) -> Dict[str, float]:
        preds = np.concatenate(self.predictions)
        targets = np.concatenate(self.targets)

        # 概率二值化
        if preds.ndim == 2 and preds.shape[1] > 1:
            # 多分类
            pred_labels = preds.argmax(axis=1)
            probs_flat = None
        else:
            preds_flat = preds.flatten()
            pred_labels = (preds_flat >= threshold).astype(int)
            probs_flat = preds_flat
            preds = preds_flat

        metrics = {
            'loss': float(np.mean(self.losses)),
            'accuracy': accuracy_score(targets, pred_labels),
        }

        if len(np.unique(targets)) == 2:
            metrics['precision'] = precision_score(targets, pred_labels, zero_division=0)
            metrics['recall'] = recall_score(targets, pred_labels, zero_division=0)
            metrics['f1'] = f1_score(targets, pred_labels, zero_division=0)
            if probs_flat is not None:
                try:
                    metrics['auc'] = roc_auc_score(targets, probs_flat)
                except ValueError:
                    metrics['auc'] = 0.0

        return metrics


class EarlyStopping:
    """早停"""

    def __init__(self, patience: int = 15, min_delta: float = 0.0, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.best_state = None

    def __call__(self, score: float, model: nn.Module) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False

        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1

        return self.counter >= self.patience

    def load_best(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def get_lr_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """warmup + cosine decay 学习率调度"""
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def save_results(results: Dict, filepath: Path):
    """保存结果为 JSON"""
    # 转换 numpy 类型
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    results = convert(results)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def format_time(seconds: float) -> str:
    """格式化时间"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m{secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h{mins}m"
