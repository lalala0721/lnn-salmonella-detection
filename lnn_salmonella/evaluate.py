"""
评估脚本
- 加载训练好的模型
- 多维度评估（标准分类、OOD 泛化、序列长度鲁棒性）
- 生成混淆矩阵和分类报告
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import argparse
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))
from config import DEVICE, KMER_DIM, NUM_CHUNKS_PER_GENOME, SEROTYPE_CLASSES
from data.dataset import KmerSequenceDataset, collate_sequence_batch
from data.preprocessing import load_all_data, split_by_genome, build_sequence_samples
from models.lnn_classifier import LNNClassifier, create_lnn_model
from models.baselines import CNN1DBaseline, BiLSTMBaseline, TransformerBaseline
from utils import MetricTracker


def load_model(model_name: str, checkpoint_path: str, num_classes: int = 1) -> nn.Module:
    """加载训练好的模型"""
    mapping = {
        'lnn-small': lambda: create_lnn_model('small', num_classes),
        'lnn-medium': lambda: create_lnn_model('medium', num_classes),
        'lnn-large': lambda: create_lnn_model('large', num_classes),
        'cnn': lambda: CNN1DBaseline(input_dim=KMER_DIM, num_classes=num_classes),
        'lstm': lambda: BiLSTMBaseline(input_dim=KMER_DIM, num_classes=num_classes),
        'transformer': lambda: TransformerBaseline(input_dim=KMER_DIM, num_classes=num_classes),
    }
    model = mapping[model_name]().to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    return model


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader) -> tuple:
    """批量预测"""
    model.eval()
    all_preds, all_targets, all_probs = [], [], []

    for data, targets in loader:
        data = data.to(DEVICE)
        logits = model(data)
        probs = torch.sigmoid(logits).squeeze(-1)
        preds = (probs >= 0.5).float()

        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.numpy())
        all_probs.append(probs.cpu().numpy())

    return (
        np.concatenate(all_preds),
        np.concatenate(all_targets),
        np.concatenate(all_probs),
    )


def plot_confusion_matrix(y_true, y_pred, labels, save_path: str = None):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def evaluate_ood(
    model: nn.Module,
    train_genome_ids: set,
    test_df: pd.DataFrame,
    genome_groups: dict,
    k: int = 4,
    num_chunks: int = 32,
):
    """
    分布外 (OOD) 泛化评估
    测试模型在训练时未见的基因组上的表现
    """
    # 筛选出 train 中不存在的基因组
    ood_samples = []
    for genome_id, chunks in genome_groups.items():
        if genome_id not in train_genome_ids and len(chunks) >= num_chunks:
            # 找到该基因组的标签
            label = None
            for _, row in test_df.iterrows():
                from data.preprocessing import extract_genome_id
                if extract_genome_id(row['file_path']) == genome_id:
                    label = row['label']
                    break
            if label is not None:
                ood_samples.append({
                    'genome_id': genome_id,
                    'chunk_paths': list(chunks)[:num_chunks],
                    'label': label,
                    'metadata': {},
                })

    if not ood_samples:
        return {'num_samples': 0}

    ds = KmerSequenceDataset(ood_samples, k=k, num_chunks=num_chunks, cache_sequences=True)
    loader = DataLoader(ds, batch_size=64, collate_fn=collate_sequence_batch)

    preds, targets, probs = predict(model, loader)
    tracker = MetricTracker()
    for i in range(len(preds)):
        tracker.update(0.0, preds[i:i+1], targets[i:i+1], probs[i:i+1])

    metrics = tracker.compute()
    metrics['num_samples'] = len(ood_samples)
    return metrics


def main():
    parser = argparse.ArgumentParser(description='评估 LNN 沙门氏菌分类器')
    parser.add_argument('--model', type=str, required=True, help='模型名称')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型权重路径')
    parser.add_argument('--classes', type=int, default=1)
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--chunks', type=int, default=32)
    parser.add_argument('--output', type=str, default='lnn_salmonella/results')
    args = parser.parse_args()

    print("加载数据...")
    pos_df, neg_df, genome_groups = load_all_data()
    train_df, val_df, test_df = split_by_genome(pos_df, neg_df)

    print("构建测试集...")
    test_samples = build_sequence_samples(test_df, genome_groups, num_chunks=args.chunks)
    test_ds = KmerSequenceDataset(test_samples, k=args.k, num_chunks=args.chunks, cache_sequences=True)
    test_loader = DataLoader(test_ds, batch_size=64, collate_fn=collate_sequence_batch)

    print(f"加载模型: {args.model}")
    model = load_model(args.model, args.checkpoint, num_classes=args.classes)

    print("\n=== 标准评估 ===")
    preds, targets, probs = predict(model, test_loader)
    pred_labels = (preds.flatten() >= 0.5).astype(int)
    targets = targets.astype(int)

    print(classification_report(targets, pred_labels, target_names=['Negative', 'Salmonella']))

    # 混淆矩阵
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(targets, pred_labels,
                          labels=['Negative', 'Salmonella'],
                          save_path=str(output_dir / f"{args.model}_confusion.png"))

    print("评估完成")


if __name__ == '__main__':
    main()
