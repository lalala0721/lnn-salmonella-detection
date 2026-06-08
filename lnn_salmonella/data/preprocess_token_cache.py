"""
Token 序列缓存：直接存储 DNA 的 token 索引，供 Embedding 模型使用
"""
import numpy as np
from pathlib import Path
import sys
import pickle
from collections import defaultdict, Counter
import random

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    SEROTYPE_DIR, KMER_K, NUM_CHUNKS_PER_GENOME,
    RANDOM_SEED, TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    SEROTYPE_CLASSES,
)
from data.tokenizer import DNATokenizer
from data.preprocessing import read_labels, extract_genome_id
from data.dataset import read_fasta

SEQ_LEN = 200  # 每个 chunk 保留的碱基数


def build_token_cache(output_dir: str = "lnn_salmonella/data/cache",
                      seq_len: int = SEQ_LEN, num_chunks: int = NUM_CHUNKS_PER_GENOME):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = DNATokenizer()
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    serotype_to_idx = {s: i for i, s in enumerate(SEROTYPE_CLASSES)}

    # === Step 1: 收集 ===
    print("[1/4] 收集血清型 chunks...")
    all_chunks = []
    for serotype in SEROTYPE_CLASSES:
        csv_path = SEROTYPE_DIR / serotype / f"{serotype}_labels.csv"
        df = read_labels(csv_path)
        for _, row in df.iterrows():
            abs_path = str((SEROTYPE_DIR.parent / row['file_path']).resolve())
            genome_id = extract_genome_id(row['file_path'])
            all_chunks.append((abs_path, serotype_to_idx[serotype], genome_id))

    # 统计
    genome_class = defaultdict(set)
    for c in all_chunks:
        genome_class[c[2]].add(c[1])
    print(f"  {len(all_chunks)} chunks, {len(genome_class)} genomes")

    # === Step 2: 按基因组分组 + 分层划分 ===
    print("[2/4] 按基因组分组 + 分层划分...")
    genome_chunks = defaultdict(list)
    for c in all_chunks:
        genome_chunks[c[2]].append(c)

    genome_ids = list(genome_chunks.keys())
    genome_by_class = defaultdict(list)
    for gid in genome_ids:
        genome_by_class[genome_chunks[gid][0][1]].append(gid)

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
        dstr = ', '.join(f'{SEROTYPE_CLASSES[i]}:{dist[i]}' for i in range(6))
        print(f"  {name}: {len(s)} genomes ({dstr})")

    # === Step 3: Tokenize ===
    print("[3/4] Tokenize chunks...")
    # 收集每个基因组的所有 chunk 路径
    genome_chunk_paths = defaultdict(list)
    for gid in genome_chunks:
        genome_chunk_paths[gid] = [c[0] for c in genome_chunks[gid]]

    def build_split(genome_set):
        X, y = [], []
        for i, gid in enumerate(genome_set):
            if (i + 1) % 500 == 0:
                print(f"  Tokenize进度: {i+1}/{len(genome_set)}")
            paths = genome_chunk_paths[gid]
            if len(paths) < num_chunks:
                continue

            selected = random.sample(paths, num_chunks)
            label = genome_chunks[gid][0][1]

            # Tokenize 每个 chunk
            token_chunks = []
            for p in selected:
                try:
                    seq = read_fasta(p)
                    tokens = tokenizer.encode(seq, max_len=seq_len)
                    token_chunks.append(tokens)
                except Exception:
                    token_chunks.append(np.full(seq_len, tokenizer.PAD_IDX, dtype=np.int64))

            X.append(np.stack(token_chunks, axis=0))  # (num_chunks, seq_len)
            y.append(label)

        if not X:
            return (np.zeros((0, num_chunks, seq_len), dtype=np.int64),
                    np.array([], dtype=np.int64))
        return np.stack(X, axis=0).astype(np.int64), np.array(y, dtype=np.int64)

    cache = {}
    for split_name, genome_set in splits.items():
        X, y = build_split(genome_set)
        cache[f'{split_name}_X'] = X
        cache[f'{split_name}_y'] = y
        dist = Counter(y.tolist()) if len(y) > 0 else {}
        dstr = ', '.join(f'{SEROTYPE_CLASSES[i]}:{dist.get(i,0)}' for i in range(6))
        print(f"  {split_name}: {X.shape} tokens, {dstr}")

    # === Step 4: 保存 ===
    print("[4/4] 保存...")
    out_path = output_dir / f"serotype_tokens_seq{seq_len}_chunks{num_chunks}.npz"
    np.savez_compressed(out_path, **cache)

    meta = {
        'seq_len': seq_len, 'num_chunks': num_chunks,
        'vocab_size': tokenizer.VOCAB_SIZE, 'pad_idx': tokenizer.PAD_IDX,
        'num_classes': 6, 'classes': SEROTYPE_CLASSES,
        'shapes': {k: cache[k].shape for k in sorted(cache.keys())},
    }
    with open(output_dir / "meta_token.pkl", 'wb') as f:
        pickle.dump(meta, f)

    print(f"\n缓存已保存到 {out_path}")
    for k, v in meta['shapes'].items():
        print(f"  {k}: {v}")

    return cache, meta


if __name__ == '__main__':
    build_token_cache()
