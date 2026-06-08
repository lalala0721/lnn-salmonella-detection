"""
OOD (Out-of-Distribution) 泛化测试
- 留一阴性物种: 训练用 5 种阴性菌 + 全部沙门氏菌，测试被留出的阴性菌
- 留一血清型:   训练用 5 种血清型 + 全部阴性菌，测试被留出的血清型
"""
import numpy as np
from pathlib import Path
import sys
import pickle
from collections import defaultdict, Counter
import random
import json
import time

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    SEROTYPE_DIR, NEGATIVE_DIR,
    KMER_K, NUM_CHUNKS_PER_GENOME,
    RANDOM_SEED, DEVICE,
    SEROTYPE_CLASSES, NEGATIVE_CLASSES,
    BATCH_SIZE, LEARNING_RATE, EPOCHS, WEIGHT_DECAY, DROPOUT,
    EARLY_STOP_PATIENCE, GRAD_CLIP_NORM, WARMUP_STEPS,
)
from data.encoding import KmerEncoder
from data.preprocessing import read_labels, extract_genome_id
from data.dataset import read_fasta, CachedKmerDataset, collate_sequence_batch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.lnn_classifier import create_lnn_model, count_parameters
from utils import EarlyStopping, get_lr_scheduler, format_time


def build_ood_cache(k: int = KMER_K, num_chunks: int = NUM_CHUNKS_PER_GENOME,
                    output_dir: str = "lnn_salmonella/data/cache"):
    """构建所有 OOD 实验所需的缓存数据"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder = KmerEncoder(k=k)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("[1/3] 加载所有数据...")
    # 收集数据: genome_id -> {chunks: [(kmer_vec, label)], serotype/species}
    all_genomes = {}  # genome_id -> {chunks, label(0/1), metadata}

    # 阳性
    for serotype in SEROTYPE_CLASSES:
        csv_path = SEROTYPE_DIR / serotype / f"{serotype}_labels.csv"
        df = read_labels(csv_path)
        for _, row in df.iterrows():
            gid = extract_genome_id(row['file_path'])
            if gid not in all_genomes:
                all_genomes[gid] = {'chunks': [], 'label': 1, 'serotype': serotype}

    # 阴性
    for species in NEGATIVE_CLASSES:
        csv_path = NEGATIVE_DIR / species / f"{species}_labels.csv"
        df = read_labels(csv_path)
        for _, row in df.iterrows():
            gid = extract_genome_id(row['file_path'])
            if gid not in all_genomes:
                all_genomes[gid] = {'chunks': [], 'label': 0, 'species': species}

    # 收集 chunk 路径并编码
    print("[2/3] 编码所有 chunks...")
    all_paths = set()
    for serotype in SEROTYPE_CLASSES:
        csv_path = SEROTYPE_DIR / serotype / f"{serotype}_labels.csv"
        df = read_labels(csv_path)
        for _, row in df.iterrows():
            abs_path = str((SEROTYPE_DIR.parent / row['file_path']).resolve())
            all_paths.add((abs_path, extract_genome_id(row['file_path'])))

    for species in NEGATIVE_CLASSES:
        csv_path = NEGATIVE_DIR / species / f"{species}_labels.csv"
        df = read_labels(csv_path)
        for _, row in df.iterrows():
            abs_path = str((NEGATIVE_DIR.parent / row['file_path']).resolve())
            all_paths.add((abs_path, extract_genome_id(row['file_path'])))

    path_to_kmer = {}
    total = len(all_paths)
    for i, (path, _) in enumerate(all_paths):
        if (i + 1) % 50000 == 0:
            print(f"  编码: {i+1}/{total}")
        try:
            path_to_kmer[path] = encoder.encode(read_fasta(path), normalize=True)
        except Exception:
            path_to_kmer[path] = np.zeros(encoder.dim, dtype=np.float32)
    print(f"  编码完成: {len(path_to_kmer)}")

    # 填充 genome chunks
    for serotype in SEROTYPE_CLASSES:
        csv_path = SEROTYPE_DIR / serotype / f"{serotype}_labels.csv"
        df = read_labels(csv_path)
        for _, row in df.iterrows():
            gid = extract_genome_id(row['file_path'])
            abs_path = str((SEROTYPE_DIR.parent / row['file_path']).resolve())
            if abs_path in path_to_kmer:
                all_genomes[gid]['chunks'].append(path_to_kmer[abs_path])

    for species in NEGATIVE_CLASSES:
        csv_path = NEGATIVE_DIR / species / f"{species}_labels.csv"
        df = read_labels(csv_path)
        for _, row in df.iterrows():
            gid = extract_genome_id(row['file_path'])
            abs_path = str((NEGATIVE_DIR.parent / row['file_path']).resolve())
            if abs_path in path_to_kmer:
                all_genomes[gid]['chunks'].append(path_to_kmer[abs_path])

    print(f"  总基因组: {len(all_genomes)}")

    # 构建序列样本
    def build_samples(genome_list):
        X, y = [], []
        for gid in genome_list:
            chunks = all_genomes[gid]['chunks']
            if len(chunks) < num_chunks:
                continue
            selected = random.sample(chunks, num_chunks)
            X.append(np.stack(selected, axis=0))
            y.append(all_genomes[gid]['label'])
        if not X:
            return (np.zeros((0, num_chunks, encoder.dim), dtype=np.float32),
                    np.array([], dtype=np.int64))
        return np.stack(X, axis=0).astype(np.float32), np.array(y, dtype=np.int64)

    print("[3/3] 构建 OOD splits...")
    ood_data = {}

    # === 留一阴性物种 ===
    for holdout_species in NEGATIVE_CLASSES:
        train_genomes = []
        test_genomes = []
        for gid, info in all_genomes.items():
            if info['label'] == 1:  # 所有沙门氏菌
                train_genomes.append(gid)
            elif info.get('species') == holdout_species:
                test_genomes.append(gid)
            else:
                train_genomes.append(gid)

        # 从训练集中分出 val
        random.shuffle(train_genomes)
        n_val = max(1, int(len(train_genomes) * 0.15))
        val_genomes = train_genomes[-n_val:]
        train_genomes = train_genomes[:-n_val]

        X_train, y_train = build_samples(train_genomes)
        X_val, y_val = build_samples(val_genomes)
        X_test, y_test = build_samples(test_genomes)

        key = f'neg_leave_{holdout_species}'
        ood_data[key] = {
            'train_X': X_train, 'train_y': y_train,
            'val_X': X_val, 'val_y': y_val,
            'test_X': X_test, 'test_y': y_test,
        }
        pos_test = y_test.sum() if len(y_test) > 0 else 0
        print(f"  {key}: train={X_train.shape[0]}, val={X_val.shape[0]}, test={X_test.shape[0]} (pos={int(pos_test)})")

    # === 留一血清型 ===
    for holdout_serotype in SEROTYPE_CLASSES:
        train_genomes = []
        test_genomes = []
        for gid, info in all_genomes.items():
            if info['label'] == 0:  # 所有阴性菌
                train_genomes.append(gid)
            elif info.get('serotype') == holdout_serotype:
                test_genomes.append(gid)
            else:
                train_genomes.append(gid)

        random.shuffle(train_genomes)
        n_val = max(1, int(len(train_genomes) * 0.15))
        val_genomes = train_genomes[-n_val:]
        train_genomes = train_genomes[:-n_val]

        X_train, y_train = build_samples(train_genomes)
        X_val, y_val = build_samples(val_genomes)
        X_test, y_test = build_samples(test_genomes)

        key = f'sero_leave_{holdout_serotype}'
        ood_data[key] = {
            'train_X': X_train, 'train_y': y_train,
            'val_X': X_val, 'val_y': y_val,
            'test_X': X_test, 'test_y': y_test,
        }
        pos_test = y_test.sum() if len(y_test) > 0 else 0
        print(f"  {key}: train={X_train.shape[0]}, val={X_val.shape[0]}, test={X_test.shape[0]} (pos={int(pos_test)})")

    # 保存
    out_path = output_dir / f"ood_kmer{k}_chunks{num_chunks}.npz"
    flat_data = {}
    for key, d in ood_data.items():
        for subkey, arr in d.items():
            flat_data[f'{key}_{subkey}'] = arr
    np.savez_compressed(out_path, **flat_data)

    meta = {'k': k, 'num_chunks': num_chunks, 'kmer_dim': encoder.dim,
            'splits': list(ood_data.keys())}
    with open(output_dir / "meta_ood.pkl", 'wb') as f:
        pickle.dump(meta, f)

    print(f"\nOOD 缓存已保存到 {out_path}")
    return ood_data


def run_ood_experiments(cache_path: str = "lnn_salmonella/data/cache/ood_kmer4_chunks32.npz",
                        output_dir: str = "lnn_salmonella/results"):
    """运行所有 OOD 实验"""
    data = np.load(cache_path)

    # 解析所有 split
    splits = set()
    for k in data.keys():
        prefix = '_'.join(k.split('_')[:3]) if 'neg_leave' in k else '_'.join(k.split('_')[:3])
        if 'neg_leave_' in k:
            parts = k.split('_')
            prefix = '_'.join(parts[:3])  # neg_leave_species
        elif 'sero_leave_' in k:
            parts = k.split('_')
            prefix = '_'.join(parts[:3])  # sero_leave_serotype
        splits.add(prefix)

    # 整理成更清晰的结构
    split_names = set()
    for k in data.keys():
        if k.endswith('_train_X'):
            name = k[:-len('_train_X')]
            split_names.add(name)

    print(f"找到 {len(split_names)} 个 OOD splits")

    all_results = {}

    for split_name in sorted(split_names):
        print(f"\n{'='*60}")
        print(f">>> OOD: {split_name}")
        print(f"{'='*60}")

        X_train = data[f'{split_name}_train_X']
        y_train = data[f'{split_name}_train_y']
        X_val = data[f'{split_name}_val_X']
        y_val = data[f'{split_name}_val_y']
        X_test = data[f'{split_name}_test_X']
        y_test = data[f'{split_name}_test_y']

        print(f"  Train: {X_train.shape}, pos_ratio={y_train.mean():.3f}")
        print(f"  Val:   {X_val.shape}, pos_ratio={y_val.mean():.3f}")
        print(f"  Test:  {X_test.shape}, pos_ratio={y_test.mean():.3f}")

        if X_test.shape[0] == 0:
            print("  Test为空，跳过")
            all_results[split_name] = {'accuracy': 0, 'test_samples': 0}
            continue

        kmer_dim = X_train.shape[-1]

        # DataLoader
        train_ds = CachedKmerDataset(X_train, y_train)
        val_ds = CachedKmerDataset(X_val, y_val)
        test_ds = CachedKmerDataset(X_test, y_test)

        train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,
                                  collate_fn=collate_sequence_batch, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=128, shuffle=False,
                                collate_fn=collate_sequence_batch, num_workers=0, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=128, shuffle=False,
                                 collate_fn=collate_sequence_batch, num_workers=0, pin_memory=True)

        # 模型: LNN-Small
        model = create_lnn_model('small', num_classes=1, input_dim=kmer_dim).to(DEVICE)

        # 类别权重
        pos_count = y_train.sum()
        neg_count = len(y_train) - pos_count
        pos_weight = neg_count / max(pos_count, 1)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(DEVICE))

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY)
        scaler = torch.amp.GradScaler('cuda')
        total_steps = 50 * len(train_loader)
        scheduler = get_lr_scheduler(optimizer, WARMUP_STEPS, total_steps)
        early_stopping = EarlyStopping(patience=15, mode='max')

        # 训练 (简化版，50 epochs)
        from torch.amp import autocast
        for epoch in range(1, 51):
            model.train()
            for data_batch, targets in train_loader:
                data_batch, targets = data_batch.to(DEVICE), targets.to(DEVICE)
                with autocast('cuda'):
                    logits = model(data_batch)
                    loss = criterion(logits.squeeze(-1), targets)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()

            # Val
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for data_batch, targets in val_loader:
                    data_batch, targets = data_batch.to(DEVICE), targets.to(DEVICE)
                    logits = model(data_batch)
                    probs = torch.sigmoid(logits).squeeze(-1)
                    preds = (probs >= 0.5).float()
                    val_correct += (preds == targets).sum().item()
                    val_total += targets.size(0)
            val_acc = val_correct / val_total

            if early_stopping(val_acc, model):
                break

        early_stopping.load_best(model)

        # Test
        model.eval()
        test_correct = 0
        test_total = 0
        all_probs = []
        all_targets = []
        with torch.no_grad():
            for data_batch, targets in test_loader:
                data_batch, targets = data_batch.to(DEVICE), targets.to(DEVICE)
                logits = model(data_batch)
                probs = torch.sigmoid(logits).squeeze(-1)
                preds = (probs >= 0.5).float()
                test_correct += (preds == targets).sum().item()
                test_total += targets.size(0)
                all_probs.extend(probs.cpu().numpy().tolist())
                all_targets.extend(targets.cpu().numpy().tolist())

        test_acc = test_correct / test_total if test_total > 0 else 0

        # 分别统计正负样本准确率
        all_targets = np.array(all_targets)
        all_preds = (np.array(all_probs) >= 0.5).astype(int)
        pos_mask = all_targets == 1
        neg_mask = all_targets == 0
        pos_acc = (all_preds[pos_mask] == 1).mean() if pos_mask.sum() > 0 else 0
        neg_acc = (all_preds[neg_mask] == 0).mean() if neg_mask.sum() > 0 else 0

        print(f"  Test Acc: {test_acc:.4f} (pos={pos_acc:.4f}, neg={neg_acc:.4f}), "
              f"samples={test_total}")

        all_results[split_name] = {
            'accuracy': float(test_acc),
            'pos_accuracy': float(pos_acc),
            'neg_accuracy': float(neg_acc),
            'test_samples': int(test_total),
        }

    # 汇总
    print(f"\n{'='*70}")
    print("=== OOD 泛化测试结果汇总 ===")
    print(f"{'='*70}")

    print(f"\n--- 留一阴性物种 ---")
    print(f"{'Held-out Species':<28} {'Test Acc':>10} {'Pos Acc':>10} {'Neg Acc':>10} {'Samples':>8}")
    print('-'*68)
    for species in NEGATIVE_CLASSES:
        key = f'neg_leave_{species}'
        r = all_results.get(key, {})
        print(f"{species:<28} {r.get('accuracy',0):>10.4f} {r.get('pos_accuracy',0):>10.4f} "
              f"{r.get('neg_accuracy',0):>10.4f} {r.get('test_samples',0):>8}")

    print(f"\n--- 留一血清型 ---")
    print(f"{'Held-out Serotype':<28} {'Test Acc':>10} {'Pos Acc':>10} {'Neg Acc':>10} {'Samples':>8}")
    print('-'*68)
    for serotype in SEROTYPE_CLASSES:
        key = f'sero_leave_{serotype}'
        r = all_results.get(key, {})
        print(f"{serotype:<28} {r.get('accuracy',0):>10.4f} {r.get('pos_accuracy',0):>10.4f} "
              f"{r.get('neg_accuracy',0):>10.4f} {r.get('test_samples',0):>8}")

    # 保存
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "ood_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n结果已保存到 {output_dir / 'ood_results.json'}")

    return all_results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--build_cache', action='store_true', help='构建 OOD 缓存')
    parser.add_argument('--run', action='store_true', help='运行 OOD 实验')
    args = parser.parse_args()

    if args.build_cache or (not args.run):
        build_ood_cache()
    if args.run:
        run_ood_experiments()
