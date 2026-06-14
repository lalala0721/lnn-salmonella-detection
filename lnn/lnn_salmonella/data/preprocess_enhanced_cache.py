"""
增强负样本缓存：包含 fu-yangpin (63 E. coli + 其他近缘种) 作为难负样本
正样本: serotype_data (预切) + zheng-yangpin (智能切块)
负样本: negative_species (预切) + fu-yangpin (智能切块)
"""
import numpy as np
from pathlib import Path
import sys
import pickle
from collections import defaultdict, Counter
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

CHUNK_SIZE = 1000
WINDOW_STRIDE = 500
RAW_CHUNKS_PER_GENOME = 150


def is_low_complexity(seq, max_single=0.5, max_n=0.05):
    seq = seq.upper()
    if len(seq) == 0 or seq.count('N') / len(seq) > max_n:
        return True
    for b in 'ATCG':
        if seq.count(b) / len(seq) > max_single:
            return True
    return False


def gc_content(seq):
    seq = seq.upper()
    return (seq.count('G') + seq.count('C')) / max(len(seq), 1)


def smart_chunk(seq, chunk_size=CHUNK_SIZE, stride=WINDOW_STRIDE,
                num_chunks=RAW_CHUNKS_PER_GENOME):
    """智能切块: sliding window + quality filter + uniform sampling"""
    seq = seq.upper().strip()
    if len(seq) < chunk_size:
        padded = seq + 'A' * (chunk_size - len(seq))
        return [padded] * num_chunks

    candidates = []
    for start in range(0, len(seq) - chunk_size + 1, stride):
        candidates.append(seq[start:start + chunk_size])
    if not candidates:
        return [seq[:chunk_size]] * num_chunks

    filtered = [w for w in candidates
                if not is_low_complexity(w) and 0.25 <= gc_content(w) <= 0.75]
    if len(filtered) < num_chunks:
        filtered = [w for w in candidates if not is_low_complexity(w, max_single=0.7)]
    if len(filtered) < num_chunks:
        filtered = candidates

    if len(filtered) <= num_chunks:
        result = filtered.copy()
        while len(result) < num_chunks:
            result.append(random.choice(filtered))
        return result[:num_chunks]

    indices = np.linspace(0, len(filtered) - 1, num_chunks, dtype=int)
    return [filtered[i] for i in indices]


def build_enhanced_cache(output_dir="lnn_salmonella/data/cache",
                         k=KMER_K, num_chunks=NUM_CHUNKS_PER_GENOME):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder = KmerEncoder(k=k)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    genome_to_chunks = defaultdict(list)

    print("[1/4] 加载预切数据...")
    for serotype in SEROTYPE_CLASSES:
        csv_path = SEROTYPE_DIR / serotype / f"{serotype}_labels.csv"
        if not csv_path.exists(): continue
        df = read_labels(csv_path)
        for _, row in df.iterrows():
            abs_path = str((SEROTYPE_DIR.parent / row['file_path']).resolve())
            gid = extract_genome_id(row['file_path'])
            genome_to_chunks[gid].append(('precut', abs_path, 1))

    for species in NEGATIVE_CLASSES:
        csv_path = NEGATIVE_DIR / species / f"{species}_labels.csv"
        if not csv_path.exists(): continue
        df = read_labels(csv_path)
        for _, row in df.iterrows():
            abs_path = str((NEGATIVE_DIR.parent / row['file_path']).resolve())
            gid = extract_genome_id(row['file_path'])
            genome_to_chunks[gid].append(('precut', abs_path, 0))

    print(f"  预切基因组: {len(genome_to_chunks)}")

    print("[2/4] 智能切块原始基因组...")
    for label, dir_path, name in [(1, ZHENG_DIR, 'zheng'), (0, FU_DIR, 'fu')]:
        files = list(dir_path.glob("*.fna"))
        print(f"  {name}: {len(files)} 个基因组")
        for fna in files:
            try:
                seq = read_fasta(str(fna))
                chunks = smart_chunk(seq, num_chunks=RAW_CHUNKS_PER_GENOME)
                gid = fna.stem
                for chunk_seq in chunks:
                    genome_to_chunks[gid].append(('raw_chunk', chunk_seq, label))
            except Exception as e:
                pass

    print(f"  总基因组: {len(genome_to_chunks)}")
    pos_g = sum(1 for g in genome_to_chunks.values() if g[0][2] == 1)
    neg_g = len(genome_to_chunks) - pos_g
    print(f"  阳性基因组: {pos_g}, 阴性基因组: {neg_g}")

    print("[3/4] 编码 + 构建样本...")
    precut_paths = set(c[1] for g in genome_to_chunks.values()
                       for c in g if c[0] == 'precut')
    path_to_kmer = {}
    total = len(precut_paths)
    for i, path in enumerate(precut_paths):
        if (i + 1) % 50000 == 0:
            print(f"  预切编码: {i+1}/{total}")
        try:
            path_to_kmer[path] = encoder.encode(read_fasta(path), normalize=True)
        except Exception:
            path_to_kmer[path] = np.zeros(encoder.dim, dtype=np.float32)
    print(f"  预切编码完成: {len(path_to_kmer)}")

    raw_count = sum(1 for g in genome_to_chunks.values()
                    for c in g if c[0] == 'raw_chunk')
    print(f"  原始基因组 chunks: {raw_count}")
    raw_done = 0
    for gid in list(genome_to_chunks.keys()):
        for i, c in enumerate(genome_to_chunks[gid]):
            if c[0] == 'raw_chunk':
                raw_done += 1
                if raw_done % 20000 == 0:
                    print(f"    raw编码: {raw_done}/{raw_count}")
                vec = encoder.encode(c[1], normalize=True)
                genome_to_chunks[gid][i] = (vec, c[2])

    for gid in genome_to_chunks:
        new_chunks = []
        for c in genome_to_chunks[gid]:
            if isinstance(c[0], str) and c[0] == 'precut':
                vec = path_to_kmer.get(c[1], np.zeros(encoder.dim, dtype=np.float32))
                new_chunks.append((vec, c[2]))
            elif isinstance(c[0], str) and c[0] == 'raw_chunk':
                pass
            else:
                new_chunks.append(c)
        genome_to_chunks[gid] = new_chunks

    print("[4/4] 划分并保存...")

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

    def build_split(genome_set):
        X, y = [], []
        for gid in genome_set:
            chunks = genome_to_chunks[gid]
            if len(chunks) < num_chunks: continue
            selected = random.sample(chunks, num_chunks)
            X.append(np.stack([s[0] for s in selected], axis=0))
            y.append(selected[0][1])
        if not X:
            return (np.zeros((0, num_chunks, encoder.dim), dtype=np.float32),
                    np.array([], dtype=np.int64))
        return np.stack(X, axis=0).astype(np.float32), np.array(y, dtype=np.int64)

    cache = {}
    for sn, gs in splits.items():
        X, y = build_split(gs)
        cache[f'{sn}_X'] = X
        cache[f'{sn}_y'] = y
        pc = int(y.sum()) if len(y) > 0 else 0
        nc = len(y) - pc
        print(f"  {sn}: {X.shape}, pos={pc}, neg={nc}")

    cache_path = output_dir / f"enhanced_kmer{k}_chunks{num_chunks}.npz"
    np.savez_compressed(cache_path, **cache)
    print(f"\n增强缓存已保存到 {cache_path}")
    for sn in ['train', 'val', 'test']:
        print(f"  {sn}: {cache[f'{sn}_X'].shape}")

    return cache


if __name__ == '__main__':
    build_enhanced_cache()
