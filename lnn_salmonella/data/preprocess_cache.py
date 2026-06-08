"""
预计算 k-mer 缓存：一次性编码所有 chunk，保存为 .npz
避免每次训练重复读取数十万个 FASTA 文件
"""
import numpy as np
from pathlib import Path
import sys
import pickle
from collections import defaultdict
import random

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    SEROTYPE_DIR, NEGATIVE_DIR, KMER_K, NUM_CHUNKS_PER_GENOME,
    RANDOM_SEED, TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    SEROTYPE_CLASSES, NEGATIVE_CLASSES,
)
from data.encoding import KmerEncoder
from data.preprocessing import read_labels, extract_genome_id
from data.dataset import read_fasta


def build_cache(output_dir: str = "lnn_salmonella/data/cache", k: int = KMER_K,
                num_chunks: int = NUM_CHUNKS_PER_GENOME):
    """
    构建完整的预计算缓存：
    1. 读取所有标签 CSV
    2. 编码所有 chunk 为 k-mer 频率向量
    3. 按基因组分组，构建序列样本
    4. 按基因组划分 train/val/test
    5. 保存为 .npz 文件
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder = KmerEncoder(k=k)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # === Step 1: 收集所有 chunk 路径和标签 ===
    print("[1/5] 收集所有 chunk 路径和标签...")
    all_chunks = []  # [(abs_path, label, genome_id)]

    # 阳性样本 (serotype_data)
    for serotype in SEROTYPE_CLASSES:
        csv_path = SEROTYPE_DIR / serotype / f"{serotype}_labels.csv"
        if not csv_path.exists():
            continue
        df = read_labels(csv_path)
        for _, row in df.iterrows():
            abs_path = str(Path(row['file_path']).resolve() if Path(row['file_path']).is_absolute()
                           else (SEROTYPE_DIR.parent / row['file_path']).resolve())
            genome_id = extract_genome_id(row['file_path'])
            all_chunks.append((abs_path, 1, genome_id, serotype, ''))

    # 阴性样本 (negative_species)
    for species in NEGATIVE_CLASSES:
        csv_path = NEGATIVE_DIR / species / f"{species}_labels.csv"
        if not csv_path.exists():
            continue
        df = read_labels(csv_path)
        for _, row in df.iterrows():
            abs_path = str(Path(row['file_path']).resolve() if Path(row['file_path']).is_absolute()
                           else (NEGATIVE_DIR.parent / row['file_path']).resolve())
            genome_id = extract_genome_id(row['file_path'])
            group = row.get('group', '')
            all_chunks.append((abs_path, 0, genome_id, species, group))

    print(f"  共 {len(all_chunks)} 个 chunks")

    # === Step 2: 按基因组分组 ===
    print("[2/5] 按基因组分组...")
    genome_chunks = defaultdict(list)
    for chunk_info in all_chunks:
        genome_chunks[chunk_info[2]].append(chunk_info)

    genome_ids = list(genome_chunks.keys())
    print(f"  共 {len(genome_ids)} 个基因组")

    # === Step 3: 划分基因组 ===
    print("[3/5] 划分 train/val/test...")
    random.shuffle(genome_ids)

    n = len(genome_ids)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    splits = {
        'train': set(genome_ids[:n_train]),
        'val': set(genome_ids[n_train:n_train + n_val]),
        'test': set(genome_ids[n_train + n_val:]),
    }
    for name, s in splits.items():
        print(f"  {name}: {len(s)} 基因组")

    # === Step 4: 编码所有 chunk 并构建序列样本 ===
    print("[4/5] 编码 chunk 并构建序列样本...")
    # 先编码所有 chunk
    path_to_kmer = {}
    unique_paths = set(c[0] for c in all_chunks)
    total = len(unique_paths)

    for i, path in enumerate(unique_paths):
        if (i + 1) % 20000 == 0:
            print(f"  编码进度: {i+1}/{total}")
        try:
            seq = read_fasta(path)
            kmer_vec = encoder.encode(seq, normalize=True)
            path_to_kmer[path] = kmer_vec
        except Exception:
            path_to_kmer[path] = np.zeros(encoder.dim, dtype=np.float32)

    print(f"  编码完成: {len(path_to_kmer)} 个唯一 chunk")

    # 构建每个 split 的序列样本
    def build_split_samples(genome_set):
        X, y = [], []
        for gid in genome_set:
            chunks = genome_chunks[gid]
            if len(chunks) < num_chunks:
                continue

            # 随机选择 num_chunks 个 chunk
            selected = random.sample(chunks, num_chunks)

            # 编码为 (num_chunks, kmer_dim)
            kmer_matrix = np.stack([path_to_kmer[sel[0]] for sel in selected], axis=0)
            label = selected[0][1]  # 使用第一个 chunk 的标签

            X.append(kmer_matrix)
            y.append(label)

        return np.stack(X, axis=0).astype(np.float32), np.array(y, dtype=np.int64)

    cache = {}
    for split_name, genome_set in splits.items():
        print(f"  构建 {split_name} 样本...")
        X, y = build_split_samples(genome_set)
        cache[f'{split_name}_X'] = X
        cache[f'{split_name}_y'] = y
        pos_ratio = y.mean()
        print(f"    {split_name}: {X.shape[0]} 样本, shape={X.shape}, pos_ratio={pos_ratio:.3f}")

    # === Step 5: 保存 ===
    print("[5/5] 保存缓存...")
    np.savez_compressed(output_dir / f"kmer{k}_chunks{num_chunks}.npz", **cache)

    # 保存元数据
    meta = {
        'k': k,
        'num_chunks': num_chunks,
        'kmer_dim': encoder.dim,
        'splits': {k: len(v) for k, v in splits.items()},
        'shapes': {k: cache[k].shape for k in sorted(cache.keys())},
    }
    with open(output_dir / "meta.pkl", 'wb') as f:
        pickle.dump(meta, f)

    print(f"\n缓存已保存到 {output_dir}")
    for k, v in meta['shapes'].items():
        print(f"  {k}: {v}")

    return cache, meta


if __name__ == '__main__':
    build_cache()
