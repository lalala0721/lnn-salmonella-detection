"""
训练脚本
- 支持 LNN (CfC) 和基线模型 (CNN, LSTM, Transformer)
- 二分类 (是否沙门氏菌) 和 多分类 (血清型)
- Warmup + Cosine 学习率, Early Stopping, AMP 混合精度
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import numpy as np
from pathlib import Path
import sys
import time
import argparse
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, EPOCHS,
    EARLY_STOP_PATIENCE, GRAD_CLIP_NORM, WARMUP_STEPS,
    DEVICE, KMER_DIM,
)
from data.dataset import KmerSequenceDataset, collate_sequence_batch
from data.preprocessing import load_all_data, split_by_genome, build_sequence_samples
from models.lnn_classifier import LNNClassifier, create_lnn_model, count_parameters
from models.baselines import CNN1DBaseline, BiLSTMBaseline, TransformerBaseline
from utils import MetricTracker, EarlyStopping, get_lr_scheduler, save_results, format_time


def create_model(model_name: str, num_classes: int = 1, input_dim: int = None, **kwargs) -> nn.Module:
    """创建模型"""
    if input_dim is None:
        input_dim = KMER_DIM
    model_name = model_name.lower()

    if model_name == 'lnn-small':
        return create_lnn_model('small', num_classes=num_classes, input_dim=input_dim)
    elif model_name == 'lnn-medium':
        return create_lnn_model('medium', num_classes=num_classes, input_dim=input_dim)
    elif model_name == 'lnn-large':
        return create_lnn_model('large', num_classes=num_classes, input_dim=input_dim)
    elif model_name == 'cnn':
        return CNN1DBaseline(input_dim=input_dim, num_classes=num_classes)
    elif model_name == 'lstm':
        return BiLSTMBaseline(input_dim=input_dim, num_classes=num_classes)
    elif model_name == 'transformer':
        return TransformerBaseline(input_dim=input_dim, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    scheduler: Optional[object] = None,
    clip_norm: float = GRAD_CLIP_NORM,
) -> float:
    """训练一个 epoch"""
    model.train()
    tracker = MetricTracker()

    for batch_idx, (data, targets) in enumerate(loader):
        data, targets = data.to(DEVICE), targets.to(DEVICE)

        with autocast('cuda'):
            logits = model(data)
            loss = criterion(logits.squeeze(-1), targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            probs = torch.sigmoid(logits).squeeze(-1)
            preds = (probs >= 0.5).float()

        tracker.update(loss.item(), preds, targets, probs)

    return tracker.compute()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> Dict[str, float]:
    """评估"""
    model.eval()
    tracker = MetricTracker()

    for data, targets in loader:
        data, targets = data.to(DEVICE), targets.to(DEVICE)
        logits = model(data)
        loss = criterion(logits.squeeze(-1), targets)

        probs = torch.sigmoid(logits).squeeze(-1)
        preds = (probs >= 0.5).float()

        tracker.update(loss.item(), preds, targets, probs)

    return tracker.compute()


def train(
    model_name: str = 'lnn-medium',
    num_classes: int = 1,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    epochs: int = EPOCHS,
    num_chunks: int = 32,
    k: int = 4,
    output_dir: str = None,
):
    """
    完整训练流程

    Args:
        model_name: 模型名称
        num_classes: 类别数 (1=二分类)
        batch_size: 批量大小
        lr: 学习率
        epochs: 最大训练轮数
        num_chunks: 每个基因组的 chunk 数
        k: k-mer 长度
        output_dir: 输出目录
    """
    print(f"{'='*60}")
    print(f"训练配置: model={model_name}, classes={num_classes}, k={k}")
    print(f"batch={batch_size}, lr={lr}, epochs={epochs}, chunks={num_chunks}")
    print(f"设备: {DEVICE}")
    print(f"{'='*60}")

    # === 数据加载 ===
    print("\n[1/4] 加载数据...")
    pos_df, neg_df, genome_groups = load_all_data()

    # 二分类时合并所有阳性/阴性；多分类时只用阳性数据
    if num_classes <= 2:
        train_df, val_df, test_df = split_by_genome(pos_df, neg_df)
    else:
        # 多分类：只用阳性数据
        train_df, val_df, test_df = split_by_genome(pos_df, pos_df)
        # 修正：对多分类需要分开处理
        # 这里简化处理，实际使用时会通过 dataset 传入 serotype 信息

    print(f"\n[2/4] 构建序列样本...")
    train_samples = build_sequence_samples(train_df, genome_groups, num_chunks=num_chunks)
    val_samples = build_sequence_samples(val_df, genome_groups, num_chunks=num_chunks)
    test_samples = build_sequence_samples(test_df, genome_groups, num_chunks=num_chunks)

    print(f"  Train: {len(train_samples)} 样本")
    print(f"  Val:   {len(val_samples)} 样本")
    print(f"  Test:  {len(test_samples)} 样本")

    # 如果没有 val/test 样本（基因组不足），从 train 中分
    if len(val_samples) == 0:
        n_val = max(1, int(len(train_samples) * 0.15))
        val_samples = train_samples[-n_val:]
        train_samples = train_samples[:-n_val]
    if len(test_samples) == 0:
        n_test = max(1, int(len(train_samples) * 0.15))
        test_samples = train_samples[-n_test:]
        train_samples = train_samples[:-n_test]

    # === DataLoader ===
    train_ds = KmerSequenceDataset(train_samples, k=k, num_chunks=num_chunks, cache_sequences=True)
    val_ds = KmerSequenceDataset(val_samples, k=k, num_chunks=num_chunks, cache_sequences=True)
    test_ds = KmerSequenceDataset(test_samples, k=k, num_chunks=num_chunks, cache_sequences=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_sequence_batch, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_sequence_batch, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_sequence_batch, num_workers=4, pin_memory=True)

    # === 模型 ===
    print(f"\n[3/4] 创建模型: {model_name}")
    model = create_model(model_name, num_classes=num_classes).to(DEVICE)
    n_params = count_parameters(model)
    print(f"  参数量: {n_params:,}")

    # 损失函数
    if num_classes <= 2:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler('cuda')

    # 学习率调度
    total_steps = epochs * len(train_loader)
    scheduler = get_lr_scheduler(optimizer, WARMUP_STEPS, total_steps)

    early_stopping = EarlyStopping(patience=EARLY_STOP_PATIENCE, mode='max')

    # === 训练 ===
    print(f"\n[4/4] 开始训练 ({epochs} epochs)...")
    history = {'train': [], 'val': []}
    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, scaler, scheduler)
        history['train'].append(train_metrics)

        # Val
        val_metrics = evaluate(model, val_loader, criterion)
        history['val'].append(val_metrics)

        elapsed = time.time() - start_time
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} | "
              f"Time: {format_time(elapsed)}")

        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']

        if early_stopping(val_metrics['accuracy'], model):
            print(f"Early stopping at epoch {epoch}")
            break

    # 加载最佳模型
    early_stopping.load_best(model)

    # === 测试 ===
    print(f"\n{'='*60}")
    print("测试集评估...")
    test_metrics = evaluate(model, test_loader, criterion)
    print(f"Test Accuracy:  {test_metrics['accuracy']:.4f}")
    if 'f1' in test_metrics:
        print(f"Test F1:        {test_metrics['f1']:.4f}")
    if 'auc' in test_metrics:
        print(f"Test AUC:       {test_metrics['auc']:.4f}")

    # === 保存结果 ===
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存模型
        torch.save(model.state_dict(), output_dir / f"{model_name}_best.pt")

        # 保存结果
        results = {
            'model': model_name,
            'num_parameters': n_params,
            'k': k,
            'num_chunks': num_chunks,
            'test_metrics': test_metrics,
            'best_val_acc': best_val_acc,
            'epochs_trained': epoch,
        }
        save_results(results, output_dir / f"{model_name}_results.json")
        print(f"\n结果已保存到 {output_dir}")

    return model, test_metrics


def train_from_cache(
    model_name: str = 'lnn-medium',
    cache_path: str = 'lnn_salmonella/data/cache/kmer4_chunks32.npz',
    num_classes: int = 1,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    epochs: int = EPOCHS,
    output_dir: str = None,
):
    """
    从预计算缓存直接训练（跳过 FASTA 读取，极快）
    """
    from data.dataset import CachedKmerDataset

    print(f"{'='*60}")
    print(f"训练配置 (缓存模式): model={model_name}, classes={num_classes}")
    print(f"batch={batch_size}, lr={lr}, epochs={epochs}")
    print(f"设备: {DEVICE}")
    print(f"{'='*60}")

    # === 加载缓存 ===
    print(f"\n[1/3] 加载预计算缓存: {cache_path}")
    data = np.load(cache_path)
    X_train, y_train = data['train_X'], data['train_y']
    X_val, y_val = data['val_X'], data['val_y']
    X_test, y_test = data['test_X'], data['test_y']

    print(f"  Train: {X_train.shape}, pos_ratio={y_train.mean():.3f}")
    print(f"  Val:   {X_val.shape}, pos_ratio={y_val.mean():.3f}")
    print(f"  Test:  {X_test.shape}, pos_ratio={y_test.mean():.3f}")

    kmer_dim = X_train.shape[-1]
    num_chunks = X_train.shape[1]

    # === DataLoader ===
    train_ds = CachedKmerDataset(X_train, y_train)
    val_ds = CachedKmerDataset(X_val, y_val)
    test_ds = CachedKmerDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_sequence_batch, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_sequence_batch, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_sequence_batch, num_workers=0, pin_memory=True)

    # === 模型 ===
    print(f"\n[2/3] 创建模型: {model_name}")
    model = create_model(model_name, num_classes=num_classes).to(DEVICE)
    n_params = count_parameters(model)
    print(f"  参数量: {n_params:,}")

    # 类别平衡加权
    if num_classes <= 2:
        pos_count = y_train.sum()
        neg_count = len(y_train) - pos_count
        pos_weight = neg_count / max(pos_count, 1)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(DEVICE))
        print(f"  pos_weight: {pos_weight:.3f}")
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler('cuda')

    total_steps = epochs * len(train_loader)
    scheduler = get_lr_scheduler(optimizer, WARMUP_STEPS, total_steps)
    early_stopping = EarlyStopping(patience=EARLY_STOP_PATIENCE, mode='max')

    # === 训练 ===
    print(f"\n[3/3] 开始训练 ({epochs} epochs)...")
    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, scaler, scheduler)
        val_metrics = evaluate(model, val_loader, criterion)

        elapsed = time.time() - start_time
        f1_str = f"F1: {val_metrics.get('f1', 0):.4f}" if 'f1' in val_metrics else ""
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} {f1_str} | "
              f"Time: {format_time(elapsed)}")

        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']

        if early_stopping(val_metrics['accuracy'], model):
            print(f"Early stopping at epoch {epoch}")
            break

    early_stopping.load_best(model)

    # === 测试 ===
    print(f"\n{'='*60}")
    print("测试集评估...")
    test_metrics = evaluate(model, test_loader, criterion)
    print(f"Test Accuracy:  {test_metrics['accuracy']:.4f}")
    if 'f1' in test_metrics:
        print(f"Test F1:        {test_metrics['f1']:.4f}")
    if 'auc' in test_metrics:
        print(f"Test AUC:       {test_metrics['auc']:.4f}")

    # === 保存 ===
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), output_dir / f"{model_name}_best.pt")
        results = {
            'model': model_name,
            'num_parameters': n_params,
            'kmer_dim': int(kmer_dim),
            'num_chunks': int(num_chunks),
            'test_metrics': test_metrics,
            'best_val_acc': best_val_acc,
            'epochs_trained': epoch,
        }
        save_results(results, output_dir / f"{model_name}_results.json")
        print(f"\n结果已保存到 {output_dir}")

    return model, test_metrics


def train_multiclass(
    model_name: str = 'lnn-small',
    cache_path: str = 'lnn_salmonella/data/cache/serotype_kmer4_chunks32.npz',
    num_classes: int = 6,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    epochs: int = EPOCHS,
    output_dir: str = None,
):
    """6 类血清型多分类训练"""
    from data.dataset import CachedKmerDataset
    from sklearn.metrics import classification_report, confusion_matrix

    print(f"{'='*60}")
    print(f"多分类训练: model={model_name}, classes={num_classes}")
    print(f"batch={batch_size}, lr={lr}, epochs={epochs}, device={DEVICE}")
    print(f"{'='*60}")

    # === 加载缓存 ===
    print(f"\n[1/3] 加载缓存: {cache_path}")
    data = np.load(cache_path)
    X_train, y_train = data['train_X'], data['train_y'].astype(np.int64)
    X_val, y_val = data['val_X'], data['val_y'].astype(np.int64)
    X_test, y_test = data['test_X'], data['test_y'].astype(np.int64)

    for name, x, y in [('Train', X_train, y_train), ('Val', X_val, y_val), ('Test', X_test, y_test)]:
        unique, counts = np.unique(y, return_counts=True)
        dist = dict(zip(unique.astype(int), counts))
        print(f"  {name}: {x.shape}, classes={dist}")

    # === DataLoader ===
    train_ds = CachedKmerDataset(X_train, y_train)
    val_ds = CachedKmerDataset(X_val, y_val)
    test_ds = CachedKmerDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_sequence_batch, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_sequence_batch, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_sequence_batch, num_workers=0, pin_memory=True)

    # === 模型 ===
    kmer_dim = X_train.shape[-1]  # 自动检测
    print(f"\n[2/3] 创建模型: {model_name} (input_dim={kmer_dim})")
    model = create_model(model_name, num_classes=num_classes, input_dim=kmer_dim).to(DEVICE)
    n_params = count_parameters(model)
    print(f"  参数量: {n_params:,}")

    # 类别权重 (处理不平衡)
    class_counts = np.bincount(y_train, minlength=num_classes)
    class_weights = 1.0 / (class_counts + 1)
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights_t = torch.from_numpy(class_weights.astype(np.float32)).to(DEVICE)
    print(f"  类别权重: {[f'{w:.3f}' for w in class_weights]}")

    criterion = nn.CrossEntropyLoss(weight=class_weights_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler('cuda')

    total_steps = epochs * len(train_loader)
    scheduler = get_lr_scheduler(optimizer, WARMUP_STEPS, total_steps)
    early_stopping = EarlyStopping(patience=EARLY_STOP_PATIENCE, mode='max')

    # === 训练 ===
    print(f"\n[3/3] 开始训练 ({epochs} epochs)...")

    def train_multiclass_epoch():
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for data, targets in train_loader:
            data, targets = data.to(DEVICE), targets.to(DEVICE, dtype=torch.long)
            with autocast('cuda'):
                logits = model(data)
                loss = criterion(logits, targets)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
        return total_loss / len(train_loader), correct / total

    @torch.no_grad()
    def eval_multiclass(loader):
        model.eval()
        total_loss = 0
        all_preds, all_targets = [], []
        for data, targets in loader:
            data, targets = data.to(DEVICE), targets.to(DEVICE, dtype=torch.long)
            logits = model(data)
            loss = criterion(logits, targets)
            total_loss += loss.item()
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_targets.append(targets.cpu().numpy())
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        acc = (all_preds == all_targets).mean()
        return total_loss / len(loader), acc, all_preds, all_targets

    best_val_acc = 0.0
    start_time = time.time()
    from config import SEROTYPE_CLASSES

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_multiclass_epoch()
        val_loss, val_acc, val_preds, val_targets = eval_multiclass(val_loader)
        elapsed = time.time() - start_time

        # 计算 macro F1
        from sklearn.metrics import f1_score
        val_f1 = f1_score(val_targets, val_preds, average='macro', zero_division=0)

        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1(macro): {val_f1:.4f} | "
              f"Time: {format_time(elapsed)}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if early_stopping(val_acc, model):
            print(f"Early stopping at epoch {epoch}")
            break

    early_stopping.load_best(model)

    # === 测试 ===
    print(f"\n{'='*60}")
    print("测试集评估...")
    _, test_acc, test_preds, test_targets = eval_multiclass(test_loader)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"\n{classification_report(test_targets, test_preds, target_names=SEROTYPE_CLASSES, zero_division=0)}")

    # 混淆矩阵
    cm = confusion_matrix(test_targets, test_preds)
    print("混淆矩阵:")
    header = f"{'':>14}" + ''.join(f'{n:>8}' for n in SEROTYPE_CLASSES)
    print(header)
    for i, name in enumerate(SEROTYPE_CLASSES):
        print(f'{name:>14}' + ''.join(f'{cm[i,j]:>8}' for j in range(6)))

    # 保存
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), output_dir / f"{model_name}_serotype_best.pt")

        results = {
            'model': model_name, 'num_parameters': n_params,
            'test_accuracy': float(test_acc),
            'test_f1_macro': float(f1_score(test_targets, test_preds, average='macro', zero_division=0)),
            'confusion_matrix': cm.tolist(),
            'classes': SEROTYPE_CLASSES,
        }
        save_results(results, output_dir / f"{model_name}_serotype_results.json")
        print(f"\n结果已保存到 {output_dir}")

    return model, {'accuracy': test_acc, 'preds': test_preds, 'targets': test_targets}


def train_embedding(
    model_type: str = 'emb-lnn',
    cache_path: str = 'lnn_salmonella/data/cache/serotype_tokens_seq200_chunks32.npz',
    num_classes: int = 6,
    batch_size: int = 32,
    lr: float = 5e-4,
    epochs: int = 80,
    output_dir: str = None,
):
    """Embedding 模型训练（字符级 token + 可学习 Embedding + LNN/CNN）"""
    from data.dataset import CachedKmerDataset
    from models.embedding_models import create_embedding_model
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    from config import SEROTYPE_CLASSES

    print(f"{'='*60}")
    print(f"Embedding训练: {model_type}, classes={num_classes}")
    print(f"lr={lr}, batch={batch_size}, epochs={epochs}, device={DEVICE}")
    print(f"{'='*60}")

    # 加载
    print(f"\n[1/3] 加载缓存: {cache_path}")
    data = np.load(cache_path)
    X_train, y_train = data['train_X'].astype(np.int64), data['train_y'].astype(np.int64)
    X_val, y_val = data['val_X'].astype(np.int64), data['val_y'].astype(np.int64)
    X_test, y_test = data['test_X'].astype(np.int64), data['test_y'].astype(np.int64)
    for name, x, y in [('Train', X_train, y_train), ('Val', X_val, y_val), ('Test', X_test, y_test)]:
        uniq, cnts = np.unique(y, return_counts=True)
        dist = dict(zip(uniq.astype(int), cnts))
        print(f"  {name}: {x.shape}, classes={dist}")

    vocab_size = 5  # A/T/C/G + pad
    seq_len = X_train.shape[-1]

    # DataLoader
    train_ds = CachedKmerDataset(X_train, y_train)
    val_ds = CachedKmerDataset(X_val, y_val)
    test_ds = CachedKmerDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_sequence_batch, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_sequence_batch, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_sequence_batch, num_workers=0, pin_memory=True)

    # 模型
    print(f"\n[2/3] 创建模型: {model_type}")
    model = create_embedding_model(model_type, vocab_size=vocab_size, num_classes=num_classes).to(DEVICE)
    n_params = count_parameters(model)
    print(f"  参数量: {n_params:,}")

    # 类别权重
    class_counts = np.bincount(y_train, minlength=num_classes)
    class_weights = 1.0 / (class_counts + 1)
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights_t = torch.from_numpy(class_weights.astype(np.float32)).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler('cuda')
    total_steps = epochs * len(train_loader)
    scheduler = get_lr_scheduler(optimizer, WARMUP_STEPS, total_steps)
    early_stopping = EarlyStopping(patience=EARLY_STOP_PATIENCE, mode='max')

    # 训练循环 (同 train_multiclass)
    def train_epoch():
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for data, targets in train_loader:
            data, targets = data.to(DEVICE), targets.to(DEVICE, dtype=torch.long)
            with autocast('cuda'):
                logits = model(data)
                loss = criterion(logits, targets)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
        return total_loss / len(train_loader), correct / total

    @torch.no_grad()
    def eval_epoch(loader):
        model.eval()
        total_loss = 0
        all_preds, all_targets = [], []
        for data, targets in loader:
            data, targets = data.to(DEVICE), targets.to(DEVICE, dtype=torch.long)
            logits = model(data)
            loss = criterion(logits, targets)
            total_loss += loss.item()
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_targets.append(targets.cpu().numpy())
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        acc = (all_preds == all_targets).mean()
        return total_loss / len(loader), acc, all_preds, all_targets

    print(f"\n[3/3] 开始训练 ({epochs} epochs)...")
    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch()
        val_loss, val_acc, val_preds, val_targets = eval_epoch(val_loader)
        elapsed = time.time() - start_time
        val_f1 = f1_score(val_targets, val_preds, average='macro', zero_division=0)
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1(macro): {val_f1:.4f} | "
              f"Time: {format_time(elapsed)}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        if early_stopping(val_acc, model):
            print(f"Early stopping at epoch {epoch}")
            break

    early_stopping.load_best(model)

    _, test_acc, test_preds, test_targets = eval_epoch(test_loader)
    print(f"\n{'='*60}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"\n{classification_report(test_targets, test_preds, target_names=SEROTYPE_CLASSES, zero_division=0)}")

    cm = confusion_matrix(test_targets, test_preds)
    print("混淆矩阵:")
    header = f"{'':>14}" + ''.join(f'{n:>8}' for n in SEROTYPE_CLASSES)
    print(header)
    for i, name in enumerate(SEROTYPE_CLASSES):
        print(f'{name:>14}' + ''.join(f'{cm[i,j]:>8}' for j in range(6)))

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), output_dir / f"{model_type}_serotype_best.pt")
        results = {
            'model': model_type, 'num_parameters': n_params,
            'test_accuracy': float(test_acc),
            'test_f1_macro': float(f1_score(test_targets, test_preds, average='macro', zero_division=0)),
        }
        save_results(results, output_dir / f"{model_type}_serotype_results.json")
        print(f"\n结果已保存到 {output_dir}")

    return model, {'accuracy': test_acc}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练 LNN 沙门氏菌分类器')
    parser.add_argument('--model', type=str, default='lnn-medium',
                        choices=['lnn-small', 'lnn-medium', 'lnn-large',
                                 'cnn', 'lstm', 'transformer'])
    parser.add_argument('--classes', type=int, default=1, help='类别数 (1=二分类)')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--chunks', type=int, default=32, help='每基因组 chunk 数')
    parser.add_argument('--k', type=int, default=4, help='k-mer 长度')
    parser.add_argument('--output', type=str, default='lnn_salmonella/results')
    parser.add_argument('--cache', type=str, default=None,
                        help='预计算缓存路径 (跳过 FASTA 读取)')
    parser.add_argument('--from_cache', action='store_true',
                        help='使用默认缓存训练')
    parser.add_argument('--multiclass', action='store_true',
                        help='血清型多分类模式')

    args = parser.parse_args()

    if args.from_cache or args.cache:
        cache_path = args.cache or 'lnn_salmonella/data/cache/kmer4_chunks32.npz'
        train_from_cache(
            model_name=args.model,
            cache_path=cache_path,
            num_classes=args.classes,
            batch_size=args.batch_size,
            lr=args.lr,
            epochs=args.epochs,
            output_dir=args.output,
        )
    else:
        train(
            model_name=args.model,
            num_classes=args.classes,
            batch_size=args.batch_size,
            lr=args.lr,
            epochs=args.epochs,
            num_chunks=args.chunks,
            k=args.k,
            output_dir=args.output,
        )
