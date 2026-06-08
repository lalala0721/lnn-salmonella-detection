"""
全量数据缓存构建器
整合 4 类数据源:
  1. serotype_data/    — 6 种血清型，预切 chunks (阳性)
  2. negative_species/ — 6 种阴性菌，预切 chunks (阴性)
  3. zheng-yangpin/    — 354 个沙门氏菌完整基因组 (阳性，需切块)
  4. fu-yangpin/       — 161 个阴性菌完整基因组 (阴性，需切块)
"""
import numpy as np
from pathlib import Path
import sys
import pickle
from collections import defaultdict
import random
import re

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    SEROTYPE_DIR, NEGATIVE_DIR, ZHENG_DIR, FU_DIR,
    KMER_K, NUM_CHUNKS_PER_GENOME,
    RANDOM_SEED, TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    SEROTYPE_CLASSES, NEGATIVE_CLASSES,
)
from data.encoding import KmerEncoder
from data.preprocessing import read_labels, extract_genome_id
from data.dataset import read_fasta

CHUNK_SIZE = 1000           # 每段 DNA 长度 (bp)
WINDOW_STRIDE = 500          # 滑动窗口步长
RAW_CHUNKS_PER_GENOME = 100  # 每个原始基因组最终保留的 chunk 数


def is_low_complexity(seq: str, max_single_base: float = 0.5,
                      max_n_ratio: float = 0.05) -> bool:
    """过滤低复杂度序列：单碱基过多 或 N 比例过高"""
    seq = seq.upper()
    if len(seq) == 0:
        return True
    n_ratio = seq.count('N') / len(seq)
    if n_ratio > max_n_ratio:
        return True
    # 检查单一碱基占比
    for base in 'ATCG':
        if seq.count(base) / len(seq) > max_single_base:
            return True
    return False


def gc_content(seq: str) -> float:
    """计算 GC 含量"""
    seq = seq.upper()
    gc = seq.count('G') + seq.count('C')
    return gc / max(len(seq), 1)


def chunk_sequence_smart(seq: str, chunk_size: int = CHUNK_SIZE,
                         stride: int = WINDOW_STRIDE,
                         num_chunks: int = RAW_CHUNKS_PER_GENOME) -> list:
    """
    智能切块：滑动窗口扫描全基因组 → 质量过滤 → 均匀采样

    1. 用 chunk_size 窗口、stride 步长扫描
    2. 过滤低复杂度和极端 GC 的窗口
    3. 从合格窗口中均匀采样 num_chunks 个
    """
    seq = seq.upper().strip()
    seq_len = len(seq)

    if seq_len < chunk_size:
        padded = seq + 'A' * (chunk_size - seq_len)
        return [padded] * num_chunks

    # Step 1: 滑动窗口扫描
    candidates = []
    for start in range(0, seq_len - chunk_size + 1, stride):
        window = seq[start:start + chunk_size]
        candidates.append(window)

    if not candidates:
        # fallback
        return [seq[:chunk_size]] * num_chunks

    # Step 2: 质量过滤
    filtered = []
    for w in candidates:
        if not is_low_complexity(w):
            gc = gc_content(w)
            if 0.25 <= gc <= 0.75:  # 细菌 GC 通常在 30-70%
                filtered.append(w)

    # 如果过滤后太少，放宽条件
    if len(filtered) < num_chunks:
        filtered = [w for w in candidates if not is_low_complexity(w, max_single_base=0.7)]
    if len(filtered) < num_chunks:
        filtered = candidates

    # Step 3: 均匀采样（按基因组位置等间距选择）
    if len(filtered) <= num_chunks:
        # 不够则重复采样
        result = filtered.copy()
        while len(result) < num_chunks:
            result.append(random.choice(filtered))
        return result[:num_chunks]

    # 等间距采样
    indices = np.linspace(0, len(filtered) - 1, num_chunks, dtype=int)
    return [filtered[i] for i in indices]


def build_full_cache(output_dir: str = "lnn_salmonella/data/cache",
                     k: int = KMER_K, num_chunks: int = NUM_CHUNKS_PER_GENOME):
    """构建包含全部 4 类数据源的完整缓存"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder = KmerEncoder(k=k)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # =====================================================
    # Step 1: 收集所有预切 chunks (serotype_data + negative_species)
    # =====================================================
    print("[1/6] 收集预切 chunks...")
    all_chunks = []  # [(abs_path, label, genome_id, metadata)]

    for serotype in SEROTYPE_CLASSES:
        csv_path = SEROTYPE_DIR / serotype / f"{serotype}_labels.csv"
        if not csv_path.exists():
            continue
        df = read_labels(csv_path)
        for _, row in df.iterrows():
            abs_path = str((SEROTYPE_DIR.parent / row['file_path']).resolve())
            genome_id = extract_genome_id(row['file_path'])
            all_chunks.append((abs_path, 1, genome_id, f"serotype:{serotype}"))

    for species in NEGATIVE_CLASSES:
        csv_path = NEGATIVE_DIR / species / f"{species}_labels.csv"
        if not csv_path.exists():
            continue
        df = read_labels(csv_path)
        for _, row in df.iterrows():
            abs_path = str((NEGATIVE_DIR.parent / row['file_path']).resolve())
            genome_id = extract_genome_id(row['file_path'])
            all_chunks.append((abs_path, 0, genome_id, f"negative:{species}"))

    print(f"  预切 chunks: {len(all_chunks)}")

    # =====================================================
    # Step 2: 处理原始完整基因组 (zheng-yangpin + fu-yangpin)
    # =====================================================
    print("[2/6] 处理原始完整基因组...")

    raw_genomes = []  # [(genome_id, label, list_of_chunk_sequences)]

    # zheng-yangpin (阳性)
    zheng_files = list(ZHENG_DIR.glob("*.fna"))
    print(f"  zheng-yangpin: {len(zheng_files)} 个基因组")
    for fna_path in zheng_files:
        try:
            seq = read_fasta(str(fna_path))
            genome_id = fna_path.stem
            chunks = chunk_sequence_smart(seq, CHUNK_SIZE, WINDOW_STRIDE, RAW_CHUNKS_PER_GENOME)
            raw_genomes.append((genome_id, 1, chunks))
        except Exception as e:
            print(f"    跳过 {fna_path.name}: {e}")

    # fu-yangpin (阴性)
    fu_files = list(FU_DIR.glob("*.fna"))
    print(f"  fu-yangpin: {len(fu_files)} 个基因组")
    for fna_path in fu_files:
        try:
            seq = read_fasta(str(fna_path))
            genome_id = fna_path.stem
            chunks = chunk_sequence_smart(seq, CHUNK_SIZE, WINDOW_STRIDE, RAW_CHUNKS_PER_GENOME)
            raw_genomes.append((genome_id, 0, chunks))
        except Exception as e:
            print(f"    跳过 {fna_path.name}: {e}")

    print(f"  原始基因组样本: {len(raw_genomes)}")
    n_pos = sum(1 for g in raw_genomes if g[1] == 1)
    n_neg = sum(1 for g in raw_genomes if g[1] == 0)
    print(f"    阳性: {n_pos}, 阴性: {n_neg}")

    # =====================================================
    # Step 3: 编码所有 chunk（预切 + 原始基因组切片）
    # =====================================================
    print("[3/6] 编码所有 chunk 为 k-mer 向量...")

    genome_to_chunks = defaultdict(list)  # genome_id -> [(kmer_vec, label)]

    # 3a. 编码预切 chunks 并分组
    unique_precut = set(c[0] for c in all_chunks)
    path_to_kmer = {}
    total = len(unique_precut)
    for i, path in enumerate(unique_precut):
        if (i + 1) % 30000 == 0:
            print(f"  预切编码: {i+1}/{total}")
        try:
            seq = read_fasta(path)
            path_to_kmer[path] = encoder.encode(seq, normalize=True)
        except Exception:
            path_to_kmer[path] = np.zeros(encoder.dim, dtype=np.float32)
    print(f"  预切编码完成: {len(path_to_kmer)}")

    for abs_path, label, genome_id, meta in all_chunks:
        if abs_path in path_to_kmer:
            genome_to_chunks[genome_id].append((path_to_kmer[abs_path], label))

    # 3b. 编码原始基因组的随机片段
    for i, (genome_id, label, chunk_seqs) in enumerate(raw_genomes):
        if (i + 1) % 100 == 0:
            print(f"  原始基因组编码: {i+1}/{len(raw_genomes)}")
        for seq in chunk_seqs:
            vec = encoder.encode(seq, normalize=True)
            genome_to_chunks[genome_id].append((vec, label))

    print(f"  总基因组数: {len(genome_to_chunks)}")

    # =====================================================
    # Step 4: 按基因组划分 train/val/test
    # =====================================================
    print("[4/6] 划分 train/val/test...")

    genome_ids = list(genome_to_chunks.keys())
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
        pos_cnt = sum(1 for gid in s if genome_to_chunks[gid][0][1] == 1)
        print(f"  {name}: {len(s)} 基因组 (阳性: {pos_cnt})")

    # =====================================================
    # Step 5: 构建序列样本
    # =====================================================
    print("[5/6] 构建序列样本...")

    def build_samples(genome_set):
        X, y = [], []
        for gid in genome_set:
            chunks = genome_to_chunks[gid]
            if len(chunks) < num_chunks:
                continue

            selected = random.sample(chunks, num_chunks)
            kmer_matrix = np.stack([s[0] for s in selected], axis=0)
            label = selected[0][1]

            X.append(kmer_matrix)
            y.append(label)

        if not X:
            return np.zeros((0, num_chunks, encoder.dim), dtype=np.float32), np.array([], dtype=np.int64)

        return np.stack(X, axis=0).astype(np.float32), np.array(y, dtype=np.int64)

    cache = {}
    for split_name, genome_set in splits.items():
        X, y = build_samples(genome_set)
        cache[f'{split_name}_X'] = X
        cache[f'{split_name}_y'] = y
        pos_ratio = y.mean() if len(y) > 0 else 0
        neg_count = len(y) - y.sum()
        pos_count = int(y.sum())
        print(f"    {split_name}: {X.shape[0]} 样本, shape={X.shape}, "
              f"pos={pos_count}, neg={neg_count}, pos_ratio={pos_ratio:.3f}")

    # =====================================================
    # Step 6: 保存
    # =====================================================
    print("[6/6] 保存缓存...")
    out_path = output_dir / f"kmer{k}_chunks{num_chunks}_full.npz"
    np.savez_compressed(out_path, **cache)

    meta = {
        'k': k, 'num_chunks': num_chunks, 'kmer_dim': encoder.dim,
        'total_genomes': len(genome_ids),
        'sources': 'serotype_data + negative_species + zheng-yangpin + fu-yangpin',
        'shapes': {k: cache[k].shape for k in sorted(cache.keys())},
    }
    with open(output_dir / "meta_full.pkl", 'wb') as f:
        pickle.dump(meta, f)

    print(f"\n缓存已保存到 {out_path}")
    for k, v in meta['shapes'].items():
        print(f"  {k}: {v}")
    total_samples = sum(v[0] for k, v in meta['shapes'].items() if k.endswith('_X'))
    print(f"  总样本数: {total_samples}")

    return cache, meta


if __name__ == '__main__':
    build_full_cache()
