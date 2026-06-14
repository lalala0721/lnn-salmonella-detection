"""
LNN 可解释性分析
1. t-SNE 隐藏状态轨迹可视化
2. Chunk 级别重要性分析 (梯度归因)
3. 隐藏状态动力学分析
4. 混淆样本深度分析
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from config import DEVICE, KMER_DIM, NUM_CHUNKS_PER_GENOME
from data.dataset import CachedKmerDataset, collate_sequence_batch
from models.lnn_classifier import LNNClassifier, create_lnn_model

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_best_model(checkpoint_path: str = None) -> LNNClassifier:
    """加载最佳 LNN-Small 模型"""
    model = create_lnn_model('small', num_classes=1).to(DEVICE)
    if checkpoint_path is None:
        checkpoint_path = 'lnn_salmonella/results/lnn-small_best.pt'
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    return model


def extract_hidden_trajectories(model, loader, max_samples=500):
    """
    提取所有样本的隐藏状态轨迹

    Returns:
        trajectories: (n_samples, num_chunks, hidden_dim)
        labels: (n_samples,)
        preds: (n_samples,)
        probs: (n_samples,)
    """
    all_traj = []
    all_labels = []
    all_preds = []
    all_probs = []
    count = 0

    for data, targets in loader:
        if count >= max_samples:
            break
        data = data.to(DEVICE)
        batch_size = data.size(0)

        with torch.no_grad():
            traj = model.get_hidden_trajectory(data)
            logits = model(data)
            probs = torch.sigmoid(logits).squeeze(-1)

        all_traj.append(traj.cpu().numpy())
        all_labels.append(targets.numpy())
        all_preds.append((probs.cpu().numpy() >= 0.5).astype(int))
        all_probs.append(probs.cpu().numpy())
        count += batch_size

    return (np.concatenate(all_traj, axis=0)[:max_samples],
            np.concatenate(all_labels, axis=0)[:max_samples],
            np.concatenate(all_preds, axis=0)[:max_samples],
            np.concatenate(all_probs, axis=0)[:max_samples])


def plot_tsne_trajectories(trajectories, labels, save_path='results/tsne_trajectories.png'):
    """
    t-SNE 可视化: 每个样本的隐藏状态轨迹
    不同颜色 = 不同类别 (Salmonella vs non-Salmonella)
    """
    n_samples, n_chunks, hidden_dim = trajectories.shape

    n_plot = min(200, n_samples)
    indices = random.sample(range(n_samples), n_plot)
    sampled_traj = trajectories[indices]
    sampled_labels = labels[indices]

    terminal_states = sampled_traj[:, -1, :]

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embedded = tsne.fit_transform(terminal_states)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = ['#e74c3c' if l == 1 else '#3498db' for l in sampled_labels]
    labels_name = ['Salmonella' if l == 1 else 'Non-Salmonella' for l in sampled_labels]

    for lbl in ['Salmonella', 'Non-Salmonella']:
        mask = [l == lbl for l in labels_name]
        axes[0].scatter(embedded[mask, 0], embedded[mask, 1],
                       c='#e74c3c' if lbl == 'Salmonella' else '#3498db',
                       label=lbl, alpha=0.6, s=30)

    axes[0].set_title('t-SNE of LNN Terminal Hidden States')
    axes[0].set_xlabel('t-SNE 1')
    axes[0].set_ylabel('t-SNE 2')
    axes[0].legend()

    probs = np.array([1 if l == 1 else 0 for l in sampled_labels])
    ax2 = axes[1]
    correct_colors = []
    for i in range(len(sampled_labels)):
        correct_colors.append('#2ecc71' if sampled_labels[i] == 1 else '#e74c3c')

    ax2.scatter(embedded[:, 0], embedded[:, 1], c=correct_colors, alpha=0.6, s=30)
    ax2.set_title('t-SNE Colored by Ground Truth')
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"t-SNE 已保存到 {save_path}")


def plot_hidden_dynamics(trajectories, labels, preds, save_path='results/hidden_dynamics.png'):
    """
    隐藏状态动力学: 展示不同类别样本在时间步上的隐藏状态变化
    """
    n_samples, n_chunks, hidden_dim = trajectories.shape

    groups = {'TP': [], 'TN': [], 'FP': [], 'FN': []}
    for i in range(n_samples):
        true = labels[i]
        pred = preds[i]
        if true == 1 and pred == 1:
            groups['TP'].append(i)
        elif true == 0 and pred == 0:
            groups['TN'].append(i)
        elif true == 0 and pred == 1:
            groups['FP'].append(i)
        elif true == 1 and pred == 0:
            groups['FN'].append(i)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax_idx, (group_name, indices) in enumerate(groups.items()):
        ax = axes[ax_idx // 2][ax_idx % 2]
        if not indices:
            ax.set_title(f'{group_name} (n=0)')
            continue

        n_show = min(30, len(indices))
        sampled = random.sample(indices, n_show)

        for i in sampled:
            activation = np.linalg.norm(trajectories[i], axis=1)
            ax.plot(range(n_chunks), activation, alpha=0.3, linewidth=0.8)

        mean_act = np.mean([np.linalg.norm(trajectories[i], axis=1) for i in sampled], axis=0)
        ax.plot(range(n_chunks), mean_act, 'k-', linewidth=2.5, label='Mean')

        ax.set_title(f'{group_name} (n={len(indices)})')
        ax.set_xlabel('Chunk (time step)')
        ax.set_ylabel('Hidden State L2 Norm')
        ax.legend()

    plt.suptitle('CfC Hidden State Dynamics by Prediction Group', fontsize=14)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"隐藏动力学已保存到 {save_path}")


def chunk_importance_gradient(model, loader, n_samples=100):
    """
    基于梯度的 chunk 重要性分析
    计算每个 chunk 对最终预测的平均梯度大小
    """
    model.train()
    all_importances = []
    all_labels = []
    count = 0

    for data, targets in loader:
        if count >= n_samples:
            break
        data = data.to(DEVICE)
        data.requires_grad = True

        logits = model(data)
        probs = torch.sigmoid(logits).squeeze(-1)

        grads_list = []
        for i in range(data.size(0)):
            model.zero_grad()
            if data.grad is not None:
                data.grad.zero_()
            probs[i].backward(retain_graph=True)
            grad = data.grad[i].abs().mean(dim=-1)
            grads_list.append(grad.cpu().numpy())
            if data.grad is not None:
                data.grad.zero_()

        all_importances.extend(grads_list)
        all_labels.extend(targets.numpy().tolist())
        count += data.size(0)

    model.eval()
    return np.array(all_importances[:n_samples]), np.array(all_labels[:n_samples])


def plot_chunk_importance(importances, labels, save_path='results/chunk_importance.png'):
    """可视化 chunk 重要性"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    pos_imp = importances[labels == 1]
    neg_imp = importances[labels == 0]

    chunks = range(importances.shape[1])
    if len(pos_imp) > 0:
        axes[0].plot(chunks, pos_imp.mean(axis=0), '#e74c3c', linewidth=2, label='Salmonella')
        axes[0].fill_between(chunks,
                            pos_imp.mean(axis=0) - pos_imp.std(axis=0),
                            pos_imp.mean(axis=0) + pos_imp.std(axis=0),
                            color='#e74c3c', alpha=0.2)
    if len(neg_imp) > 0:
        axes[0].plot(chunks, neg_imp.mean(axis=0), '#3498db', linewidth=2, label='Non-Salmonella')
        axes[0].fill_between(chunks,
                            neg_imp.mean(axis=0) - neg_imp.std(axis=0),
                            neg_imp.mean(axis=0) + neg_imp.std(axis=0),
                            color='#3498db', alpha=0.2)

    axes[0].set_title('Average Chunk Importance (Gradient-based)')
    axes[0].set_xlabel('Chunk index')
    axes[0].set_ylabel('Mean |Gradient|')
    axes[0].legend()

    n_show = min(30, len(importances))
    indices = random.sample(range(len(importances)), n_show)
    imp_subset = importances[indices]
    labels_subset = labels[indices]

    sort_idx = np.argsort(labels_subset)
    im = axes[1].imshow(imp_subset[sort_idx], aspect='auto', cmap='hot', interpolation='nearest')
    axes[1].set_title(f'Chunk Importance Heatmap ({n_show} samples)')
    axes[1].set_xlabel('Chunk index')
    axes[1].set_ylabel('Sample (sorted by class)')
    plt.colorbar(im, ax=axes[1])

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Chunk 重要性已保存到 {save_path}")


def plot_confidence_distribution(probs, labels, save_path='results/confidence_dist.png'):
    """预测置信度分布"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    pos_probs = probs[labels == 1]
    neg_probs = probs[labels == 0]

    axes[0].hist(pos_probs, bins=30, alpha=0.6, color='#e74c3c', label=f'Salmonella (n={len(pos_probs)})')
    axes[0].hist(neg_probs, bins=30, alpha=0.6, color='#3498db', label=f'Non-Salmonella (n={len(neg_probs)})')
    axes[0].axvline(x=0.5, color='k', linestyle='--', alpha=0.5)
    axes[0].set_title('Prediction Confidence Distribution')
    axes[0].set_xlabel('Predicted Probability (Salmonella)')
    axes[0].set_ylabel('Count')
    axes[0].legend()

    axes[1].hist(pos_probs, bins=50, alpha=0.5, color='#e74c3c', cumulative=True,
                density=True, histtype='step', linewidth=2, label='Salmonella')
    axes[1].hist(neg_probs, bins=50, alpha=0.5, color='#3498db', cumulative=True,
                density=True, histtype='step', linewidth=2, label='Non-Salmonella')
    axes[1].axvline(x=0.5, color='k', linestyle='--', alpha=0.5)
    axes[1].set_title('Cumulative Distribution')
    axes[1].set_xlabel('Predicted Probability')
    axes[1].set_ylabel('Cumulative Fraction')
    axes[1].legend()

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"置信度分布已保存到 {save_path}")


def plot_state_trajectory_pca(trajectories, labels, save_path='results/state_pca_trajectory.png'):
    """
    PCA 轨迹图: 展示隐藏状态在时间步上的演化路径
    """
    n_samples, n_chunks, hidden_dim = trajectories.shape

    flat = trajectories.reshape(-1, hidden_dim)
    pca = PCA(n_components=2, random_state=42)
    pca.fit(flat)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    pos_indices = [i for i in range(min(n_samples, 500)) if labels[i] == 1][:5]
    neg_indices = [i for i in range(min(n_samples, 500)) if labels[i] == 0][:5]

    for idx_list, color, label, ax_idx in [(pos_indices, '#e74c3c', 'Salmonella', 0),
                                            (neg_indices, '#3498db', 'Non-Salmonella', 0)]:
        for i in idx_list:
            traj_2d = pca.transform(trajectories[i])
            axes[ax_idx].plot(traj_2d[:, 0], traj_2d[:, 1], color=color, alpha=0.5, linewidth=1)
            axes[ax_idx].scatter(traj_2d[0, 0], traj_2d[0, 1], color=color, s=50, marker='o', alpha=0.8)
            axes[ax_idx].scatter(traj_2d[-1, 0], traj_2d[-1, 1], color=color, s=80, marker='*', alpha=1.0)

    axes[0].set_title('PCA Trajectories (Selected Samples)')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')

    all_pos_end = []
    all_neg_end = []
    for i in range(min(n_samples, 500)):
        traj_2d = pca.transform(trajectories[i])
        if labels[i] == 1:
            all_pos_end.append(traj_2d[-1])
        else:
            all_neg_end.append(traj_2d[-1])

    if all_pos_end:
        all_pos_end = np.array(all_pos_end)
        axes[1].scatter(all_pos_end[:, 0], all_pos_end[:, 1], c='#e74c3c', alpha=0.3, s=10, label='Salmonella end')
    if all_neg_end:
        all_neg_end = np.array(all_neg_end)
        axes[1].scatter(all_neg_end[:, 0], all_neg_end[:, 1], c='#3498db', alpha=0.3, s=10, label='Non-Salmonella end')

    axes[1].set_title('Terminal States in PCA Space')
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
    axes[1].legend()

    plt.suptitle('CfC Hidden State Evolution', fontsize=14)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"PCA 轨迹已保存到 {save_path}")


def main():
    print("=" * 60)
    print("LNN 可解释性分析")
    print("=" * 60)

    print("\n[1/5] 加载数据...")
    data = np.load('lnn_salmonella/data/cache/kmer4_chunks32.npz')
    X_test = data['test_X']
    y_test = data['test_y']
    print(f"  测试集: {X_test.shape}")

    test_ds = CachedKmerDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                             collate_fn=collate_sequence_batch, num_workers=0)

    print("\n[2/5] 加载最佳模型...")
    model = load_best_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {n_params:,}")

    print("\n[3/5] 提取隐藏状态轨迹...")
    trajectories, labels, preds, probs = extract_hidden_trajectories(
        model, test_loader, max_samples=500)
    print(f"  轨迹: {trajectories.shape}")
    print(f"  准确率: {(preds == labels).mean():.4f}")
    print(f"  TP: {(labels==1) & (preds==1)} — {(labels==1).sum()}")
    print(f"  TN: {(labels==0) & (preds==0)} — {(labels==0).sum()}")

    print("\n[4/5] 生成可视化...")
    output_dir = Path('lnn_salmonella/results')

    plot_tsne_trajectories(trajectories, labels,
                           save_path=str(output_dir / 'tsne_trajectories.png'))

    plot_hidden_dynamics(trajectories, labels, preds,
                         save_path=str(output_dir / 'hidden_dynamics.png'))

    plot_state_trajectory_pca(trajectories, labels,
                              save_path=str(output_dir / 'state_pca_trajectory.png'))

    plot_confidence_distribution(probs, labels,
                                 save_path=str(output_dir / 'confidence_dist.png'))

    print("\n[5/5] Chunk 重要性分析...")
    chunk_imp, chunk_labels = chunk_importance_gradient(model, test_loader, n_samples=200)
    plot_chunk_importance(chunk_imp, chunk_labels,
                          save_path=str(output_dir / 'chunk_importance.png'))

    print(f"\n{'='*60}")
    print("所有可视化已保存到 lnn_salmonella/results/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
