"""
数据预处理模块
- 读取标签 CSV 文件
- 按基因组来源划分 train/val/test
- 创建序列化样本（每个基因组取 N 个 chunk 组成序列）
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import random
import re
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    SEROTYPE_DIR, NEGATIVE_DIR,
    NUM_CHUNKS_PER_GENOME,
    SEROTYPE_CLASSES, NEGATIVE_CLASSES,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED,
)


def read_labels(csv_path: Path) -> pd.DataFrame:
    """读取标签 CSV 文件"""
    df = pd.read_csv(csv_path)
    base_dir = csv_path.parent.parent
    df['abs_path'] = df['file_path'].apply(
        lambda p: str(Path(p).resolve() if Path(p).is_absolute() else (base_dir.parent / p).resolve())
    )
    return df


def extract_genome_id(file_path: str) -> str:
    """
    从 chunk 文件名中提取基因组 ID

    命名格式: typhimurium_NZ_CP149380.1_c0012.fasta
    → 基因组 ID: typhimurium_NZ_CP149380.1

    也支持: staphylococcus_aureus_NZ_CP166516.1_c0012.fasta
    """
    basename = Path(file_path).stem
    match = re.match(r'(.+)_c\d+$', basename)
    if match:
        return match.group(1)
    return basename


def load_all_data() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[str]]]:
    """
    加载所有数据

    Returns:
        positive_df: 所有沙门氏菌 chunk 的 DataFrame (含 serotype 信息)
        negative_df: 所有阴性菌 chunk 的 DataFrame (含 species 信息)
        genome_groups: {genome_id: [chunk_paths]}
    """
    positive_dfs = []
    negative_dfs = []

    for serotype in SEROTYPE_CLASSES:
        csv_path = SEROTYPE_DIR / serotype / f"{serotype}_labels.csv"
        if csv_path.exists():
            df = read_labels(csv_path)
            df['serotype'] = serotype
            df['label'] = 1
            positive_dfs.append(df)

    for species in NEGATIVE_CLASSES:
        csv_path = NEGATIVE_DIR / species / f"{species}_labels.csv"
        if csv_path.exists():
            df = read_labels(csv_path)
            df['species'] = species
            df['label'] = 0
            negative_dfs.append(df)

    pos_df = pd.concat(positive_dfs, ignore_index=True) if positive_dfs else pd.DataFrame()
    neg_df = pd.concat(negative_dfs, ignore_index=True) if negative_dfs else pd.DataFrame()

    genome_groups = defaultdict(list)
    for _, row in pos_df.iterrows():
        gid = extract_genome_id(row['file_path'])
        genome_groups[gid].append(row['abs_path'])
    for _, row in neg_df.iterrows():
        gid = extract_genome_id(row['file_path'])
        genome_groups[gid].append(row['abs_path'])

    print(f"加载完成: {len(pos_df)} 个阳性 chunks, {len(neg_df)} 个阴性 chunks")
    print(f"阳性基因组: {len(set(extract_genome_id(r['file_path']) for _, r in pos_df.iterrows()))}")
    print(f"阴性基因组: {len(set(extract_genome_id(r['file_path']) for _, r in neg_df.iterrows()))}")

    return pos_df, neg_df, dict(genome_groups)


def split_by_genome(
    pos_df: pd.DataFrame,
    neg_df: pd.DataFrame,
    random_seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    按基因组来源划分 train/val/test
    确保同一基因组的 chunks 不会同时出现在 train 和 test 中
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    def split_df(df: pd.DataFrame, genome_col_fn) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        genomes = list(set(genome_col_fn(r) for _, r in df.iterrows()))
        random.shuffle(genomes)

        n = len(genomes)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)

        train_genomes = set(genomes[:n_train])
        val_genomes = set(genomes[n_train:n_train + n_val])
        test_genomes = set(genomes[n_train + n_val:])

        train_df = df[df['file_path'].apply(lambda p: extract_genome_id(p) in train_genomes)]
        val_df = df[df['file_path'].apply(lambda p: extract_genome_id(p) in val_genomes)]
        test_df = df[df['file_path'].apply(lambda p: extract_genome_id(p) in test_genomes)]

        return train_df, val_df, test_df

    pos_train, pos_val, pos_test = split_df(pos_df, lambda r: extract_genome_id(r['file_path']))
    neg_train, neg_val, neg_test = split_df(neg_df, lambda r: extract_genome_id(r['file_path']))

    train_df = pd.concat([pos_train, neg_train], ignore_index=True)
    val_df = pd.concat([pos_val, neg_val], ignore_index=True)
    test_df = pd.concat([pos_test, neg_test], ignore_index=True)

    print(f"Train: {len(train_df)} ({len(pos_train)} pos / {len(neg_train)} neg)")
    print(f"Val:   {len(val_df)} ({len(pos_val)} pos / {len(neg_val)} neg)")
    print(f"Test:  {len(test_df)} ({len(pos_test)} pos / {len(neg_test)} neg)")

    return train_df, val_df, test_df


def build_sequence_samples(
    df: pd.DataFrame,
    genome_groups: Dict[str, List[str]],
    num_chunks: int = NUM_CHUNKS_PER_GENOME,
    shuffle_chunks: bool = True,
) -> List[Dict]:
    """
    将同一基因组的 chunks 聚合成序列样本

    每个样本 = 同一基因组的 num_chunks 个 chunk 组成的序列

    Returns:
        List of dicts: [{genome_id, chunk_paths, label, metadata}]
    """
    samples = []

    chunk_labels = {}
    chunk_meta = {}
    for _, row in df.iterrows():
        chunk_labels[row['abs_path']] = row['label']
        meta = {}
        if 'serotype' in row:
            meta['serotype'] = row['serotype']
        if 'species' in row:
            meta['species'] = row['species']
        if 'group' in row:
            meta['group'] = row['group']
        if 'difficulty' in row:
            meta['difficulty'] = row['difficulty']
        chunk_meta[row['abs_path']] = meta

    df_genome_ids = set(extract_genome_id(p) for p in df['file_path'])

    for genome_id in df_genome_ids:
        chunks = genome_groups.get(genome_id, [])
        if len(chunks) < num_chunks:
            continue

        if shuffle_chunks:
            chunks_copy = list(chunks)
            random.shuffle(chunks_copy)
        else:
            chunks_copy = chunks

        selected = chunks_copy[:num_chunks]

        label = chunk_labels.get(selected[0], -1)

        if label == -1:
            continue

        samples.append({
            'genome_id': genome_id,
            'chunk_paths': selected,
            'label': label,
            'metadata': chunk_meta.get(selected[0], {}),
        })

    return samples
