"""
血清型多分类缓存构建器
6 类: dublin(0), enteritidis(1), heidelberg(2), infantis(3), newport(4), typhimurium(5)
"""
import numpy as np
from pathlib import Path
import sys
import pickle
from collections import defaultdict
import random

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    SEROTYPE_DIR, KMER_K, NUM_CHUNKS_PER_GENOME,
    RANDOM_SEED, TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    SEROTYPE_CLASSES,
)
from data.encoding import KmerEncoder
from data.preprocessing import read_labels, extract_genome_id
from data.dataset import read_fasta


def build_serotype_cache(output_dir: str = "lnn_salmonella/data/cache",
                         k: int = KMER_K, num_chunks: int = NUM_CHUNKS_PER_GENOME):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder = KmerEncoder(k=k)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("[1/5] 收集血清型 chunks...")
    all_chunks = []
    serotype_to_idx = {s: i for i, s in enumerate(SEROTYPE_CLASSES)}

    for serotype in SEROTYPE_CLASSES:
        csv_path = SEROTYPE_DIR / serotype / f"{serotype}_labels.csv"
        df = read_labels(csv_path)
        for _, row in df.iterrows():
            abs_path = str((SEROTYPE_DIR.parent / row['file_path']).resolve())
            genome_id = extract_genome_id(row['file_path'])
            all_chunks.append((abs_path, serotype_to_idx[serotype], genome_id))

    print(f"  总 chunks: {len(all_chunks)}")

    from collections import Counter
    class_counts = Counter(c[1] for c in all_chunks)
    genome_class_counts = defaultdict(set)
    for c in all_chunks:
        genome_class_counts[c[1]].add(c[2])
    for i, name in enumerate(SEROTYPE_CLASSES):
        print(f"  {name}: {class_counts[i]} chunks, {len(genome_class_counts[i])} genomes")

    print("[2/5] 按基因组分组...")
    genome_chunks = defaultdict(list)
    for chunk_info in all_chunks:
        genome_chunks[chunk_info[2]].append(chunk_info)

    genome_ids = list(genome_chunks.keys())
    print(f"  共 {len(genome_ids)} 个基因组")

    print("[3/5] 分层划分 train/val/test...")
    genome_by_class = defaultdict(list)
    for gid in genome_ids:
        label = genome_chunks[gid][0][1]
        genome_by_class[label].append(gid)

    splits = {'train': set(), 'val': set(), 'test': set()}
    for label, gids in genome_by_class.items():
        random.shuffle(gids)
        n = len(gids)
        n_train = max(1, int(n * TRAIN_RATIO))
        n_val = max(1, int(n * VAL_RATIO))
        splits['train'].update(gids[:n_train])
        splits['val'].update(gids[n_train:n_train + n_val])
        splits['test'].update(gids[n_train + n_val:])

    for name, s in splits.items():
        dist = Counter(genome_chunks[gid][0][1] for gid in s)
        dist_str = ', '.join(f'{SEROTYPE_CLASSES[i]}:{dist[i]}' for i in range(6))
        print(f"  {name}: {len(s)} genomes ({dist_str})")

    print("[4/5] 编码并构建序列样本...")
    unique_paths = set(c[0] for c in all_chunks)
    path_to_kmer = {}
    total = len(unique_paths)
    for i, path in enumerate(unique_paths):
        if (i + 1) % 30000 == 0:
            print(f"  编码: {i+1}/{total}")
        try:
            path_to_kmer[path] = encoder.encode(read_fasta(path), normalize=True)
        except Exception:
            path_to_kmer[path] = np.zeros(encoder.dim, dtype=np.float32)
    print(f"  编码完成: {len(path_to_kmer)}")

    def build_samples(genome_set):
        X, y = [], []
        for gid in genome_set:
            chunks = genome_chunks[gid]
            if len(chunks) < num_chunks:
                continue
            selected = random.sample(chunks, num_chunks)
            kmer_matrix = np.stack([path_to_kmer[s[0]] for s in selected], axis=0)
            X.append(kmer_matrix)
            y.append(selected[0][1])
        if not X:
            return (np.zeros((0, num_chunks, encoder.dim), dtype=np.float32),
                    np.array([], dtype=np.int64))
        return np.stack(X, axis=0).astype(np.float32), np.array(y, dtype=np.int64)

    cache = {}
    for split_name, genome_set in splits.items():
        X, y = build_samples(genome_set)
        cache[f'{split_name}_X'] = X
        cache[f'{split_name}_y'] = y
        dist = Counter(y.tolist()) if len(y) > 0 else {}
        dist_str = ', '.join(f'{SEROTYPE_CLASSES[i]}:{dist.get(i, 0)}' for i in range(6))
        print(f"  {split_name}: {X.shape[0]} samples, {dist_str}")

    print("[5/5] 保存...")
    out_path = output_dir / f"serotype_kmer{k}_chunks{num_chunks}.npz"
    np.savez_compressed(out_path, **cache)

    meta = {
        'k': k, 'num_chunks': num_chunks, 'kmer_dim': encoder.dim,
        'num_classes': 6, 'classes': SEROTYPE_CLASSES,
        'class_map': serotype_to_idx,
        'shapes': {k: cache[k].shape for k in sorted(cache.keys())},
    }
    with open(output_dir / "meta_serotype.pkl", 'wb') as f:
        pickle.dump(meta, f)

    print(f"\n缓存已保存到 {out_path}")
    for k, v in meta['shapes'].items():
        print(f"  {k}: {v}")

    return cache, meta


if __name__ == '__main__':
    build_serotype_cache()
