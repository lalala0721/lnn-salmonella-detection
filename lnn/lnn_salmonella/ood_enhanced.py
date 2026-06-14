"""增强 OOD 测试：训练中保留 E. coli 等近缘种"""
import sys, numpy as np, random, json
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent))
from config import *
from data.encoding import KmerEncoder
from data.preprocessing import read_labels, extract_genome_id
from data.dataset import read_fasta, CachedKmerDataset, collate_sequence_batch
from models.lnn_classifier import create_lnn_model
from utils import EarlyStopping, get_lr_scheduler
import torch, torch.nn as nn
from torch.utils.data import DataLoader

random.seed(42); np.random.seed(42)
encoder = KmerEncoder(k=4)

print("Loading genome-level data...")
genome_label = {}
genome_species = {}

for serotype in SEROTYPE_CLASSES:
    csv = SEROTYPE_DIR / serotype / f"{serotype}_labels.csv"
    df = read_labels(csv)
    for _, row in df.iterrows():
        genome_label[extract_genome_id(row['file_path'])] = 1

for species in NEGATIVE_CLASSES:
    csv = NEGATIVE_DIR / species / f"{species}_labels.csv"
    df = read_labels(csv)
    for _, row in df.iterrows():
        gid = extract_genome_id(row['file_path'])
        genome_label[gid] = 0
        genome_species[gid] = species

fu_gids = [f.stem for f in FU_DIR.glob("*.fna")]
zheng_gids = [f.stem for f in ZHENG_DIR.glob("*.fna")]
for gid in fu_gids:
    genome_label[gid] = 0
    genome_species[gid] = 'fu_mixed'
for gid in zheng_gids:
    genome_label[gid] = 1

print(f"Genomes: {len(genome_label)} (pos={sum(1 for v in genome_label.values() if v==1)}, neg={sum(1 for v in genome_label.values() if v==0)})")

print("Encoding chunks...")
path_kmer = {}
precut_set = set()
for serotype in SEROTYPE_CLASSES:
    csv = SEROTYPE_DIR / serotype / f"{serotype}_labels.csv"
    for _, row in read_labels(csv).iterrows():
        path = str((SEROTYPE_DIR.parent / row['file_path']).resolve())
        precut_set.add((path, extract_genome_id(row['file_path'])))
for species in NEGATIVE_CLASSES:
    csv = NEGATIVE_DIR / species / f"{species}_labels.csv"
    for _, row in read_labels(csv).iterrows():
        path = str((NEGATIVE_DIR.parent / row['file_path']).resolve())
        precut_set.add((path, extract_genome_id(row['file_path'])))

total_p = len(precut_set)
for i, (path, gid) in enumerate(precut_set):
    if (i+1) % 50000 == 0:
        print(f"  precut: {i+1}/{total_p}")
    try:
        path_kmer[(path, gid)] = encoder.encode(read_fasta(path), normalize=True)
    except Exception:
        path_kmer[(path, gid)] = np.zeros(256, dtype=np.float32)
print(f"Precut: {len(path_kmer)}")

def smart_chunk(seq, n=150):
    seq = seq.upper().strip()
    if len(seq) < 1000:
        return [seq + 'A'*(1000-len(seq))] * n
    cand = [seq[i:i+1000] for i in range(0, len(seq)-1000, 500)]
    filt = [w for w in cand if w.count('N')/len(w) < 0.05
            and max(w.count(b) for b in 'ATCG')/len(w) < 0.5
            and 0.25 <= (w.count('G')+w.count('C'))/len(w) <= 0.75]
    if len(filt) < n:
        filt = [w for w in cand if w.count('N')/len(w) < 0.05]
    if len(filt) < n:
        filt = cand
    if len(filt) <= n:
        r = filt.copy()
        while len(r) < n:
            r.append(random.choice(filt))
        return r[:n]
    idx = np.linspace(0, len(filt)-1, n, dtype=int)
    return [filt[i] for i in idx]

raw_kmers = {}
for label, dr, nm in [(0, FU_DIR, 'fu'), (1, ZHENG_DIR, 'zheng')]:
    files = list(dr.glob("*.fna"))
    print(f"{nm}: {len(files)} genomes")
    for fi in files:
        try:
            seq = read_fasta(str(fi))
            chunks = smart_chunk(seq)
            raw_kmers[fi.stem] = [encoder.encode(c, normalize=True) for c in chunks]
        except Exception:
            pass
print(f"Raw: {len(raw_kmers)}")

genome_kmers = defaultdict(list)
for (path, gid), vec in path_kmer.items():
    genome_kmers[gid].append(vec)
for gid, vecs in raw_kmers.items():
    genome_kmers[gid].extend(vecs)
print(f"Genomes with kmers: {len(genome_kmers)}")

num_chunks = 32

def build_samples(gids):
    X, y = [], []
    for gid in gids:
        chunks = genome_kmers.get(gid, [])
        if len(chunks) < num_chunks:
            continue
        sel = random.sample(chunks, num_chunks)
        X.append(np.stack(sel, axis=0))
        y.append(genome_label[gid])
    if not X:
        return np.zeros((0,num_chunks,256), dtype=np.float32), np.array([], dtype=np.int64)
    return np.stack(X).astype(np.float32), np.array(y, dtype=np.int64)

def train_eval(train_gids, val_gids, test_gids):
    Xt, yt = build_samples(train_gids)
    Xv, yv = build_samples(val_gids)
    Xe, ye = build_samples(test_gids)
    if len(Xe) == 0:
        return {'accuracy': 0, 'samples': 0}

    tr_ds = CachedKmerDataset(Xt, yt)
    va_ds = CachedKmerDataset(Xv, yv)
    te_ds = CachedKmerDataset(Xe, ye)
    tr_ld = DataLoader(tr_ds, 128, shuffle=True, collate_fn=collate_sequence_batch, num_workers=0, pin_memory=True)
    va_ld = DataLoader(va_ds, 128, shuffle=False, collate_fn=collate_sequence_batch, num_workers=0, pin_memory=True)
    te_ld = DataLoader(te_ds, 128, shuffle=False, collate_fn=collate_sequence_batch, num_workers=0, pin_memory=True)

    model = create_lnn_model('small', num_classes=1, input_dim=256).to(DEVICE)
    pw = (len(yt)-yt.sum()) / max(yt.sum(), 1)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw]).to(DEVICE))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    sch = get_lr_scheduler(opt, 200, 50*len(tr_ld))
    es = EarlyStopping(patience=12, mode='max')

    for ep in range(50):
        model.train()
        for dx, dy in tr_ld:
            dx, dy = dx.to(DEVICE), dy.to(DEVICE)
            with torch.amp.autocast('cuda'):
                lo = model(dx)
                loss = crit(lo.squeeze(-1), dy)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sch.step()
        model.eval()
        vc, vt = 0, 0
        with torch.no_grad():
            for dx, dy in va_ld:
                dx, dy = dx.to(DEVICE), dy.to(DEVICE)
                lo = model(dx)
                pr = (torch.sigmoid(lo).squeeze(-1) >= 0.5).float()
                vc += (pr == dy).sum().item()
                vt += dy.size(0)
        if es(vc/vt, model):
            break
    es.load_best(model)

    model.eval()
    tc, tt = 0, 0
    with torch.no_grad():
        for dx, dy in te_ld:
            dx, dy = dx.to(DEVICE), dy.to(DEVICE)
            lo = model(dx)
            pr = (torch.sigmoid(lo).squeeze(-1) >= 0.5).float()
            tc += (pr == dy).sum().item()
            tt += dy.size(0)
    return {'accuracy': tc/tt if tt > 0 else 0, 'samples': tt}

results = {}

print("\n--- Leave-one-negative-species-out (E. coli 保留) ---")
for species in NEGATIVE_CLASSES:
    test_gids = [g for g in genome_label if genome_species.get(g) == species]
    train_gids = [g for g in genome_label if g not in test_gids]
    random.shuffle(train_gids)
    nv = max(1, int(len(train_gids)*0.12))
    val_gids = train_gids[-nv:]
    train_gids = train_gids[:-nv]
    r = train_eval(train_gids, val_gids, test_gids)
    results[f"neg_leave_{species}"] = r
    print(f"  {species:<30} acc={r['accuracy']:.4f}  n={r['samples']}")

print("\n--- Leave-one-serotype-out ---")
for serotype in SEROTYPE_CLASSES:
    test_gids = [g for g in genome_label if g.startswith(f"{serotype}_")]
    train_gids = [g for g in genome_label if g not in test_gids]
    random.shuffle(train_gids)
    nv = max(1, int(len(train_gids)*0.12))
    val_gids = train_gids[-nv:]
    train_gids = train_gids[:-nv]
    r = train_eval(train_gids, val_gids, test_gids)
    results[f"sero_leave_{serotype}"] = r
    print(f"  {serotype:<30} acc={r['accuracy']:.4f}  n={r['samples']}")

print(f"\n{'='*75}")
print("OOD 对比: 原始 vs 增强 (含 E. coli)")
print(f"{'='*75}")
print(f"{'Held-out':<28} {'原始 Acc':>10} {'增强 Acc':>10} {'变化':>10}")
print('-'*60)

old_results = {
    'neg_leave_enterococcus_faecalis': 1.0000,
    'neg_leave_klebsiella_pneumoniae': 0.1031,
    'neg_leave_listeria_monocytogenes': 1.0000,
    'neg_leave_pseudomonas_aeruginosa': 0.0000,
    'neg_leave_shigella_flexneri': 0.0029,
    'neg_leave_staphylococcus_aureus': 1.0000,
    'sero_leave_dublin': 0.9946,
    'sero_leave_enteritidis': 1.0000,
    'sero_leave_heidelberg': 0.9956,
    'sero_leave_infantis': 1.0000,
    'sero_leave_newport': 1.0000,
    'sero_leave_typhimurium': 0.7576,
}

for k in sorted(results.keys()):
    old = old_results.get(k, 0)
    new = results[k]['accuracy']
    diff = new - old
    marker = '⬆' if diff > 0.05 else ('⬇' if diff < -0.05 else '→')
    print(f"{k:<28} {old:>10.4f} {new:>10.4f} {diff:>+9.4f} {marker}")

with open('lnn_salmonella/results/ood_enhanced.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nDone.")
