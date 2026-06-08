"""3次重复实验 + 统计显著性"""
import numpy as np, torch, torch.nn as nn, time, random
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
import sys
sys.path.insert(0, str(__file__).rsplit('\\', 1)[0])
from config import DEVICE
from data.dataset import CachedKmerDataset, collate_sequence_batch
from models.lnn_classifier import LNNClassifier, count_parameters
from models.baselines import CNN1DBaseline, BiLSTMBaseline
from utils import EarlyStopping, get_lr_scheduler

data = np.load('lnn_salmonella/data/cache/kmer4_chunks32.npz')
te_ds = CachedKmerDataset(data['test_X'], data['test_y'])
te_ld = DataLoader(te_ds, 128, shuffle=False, collate_fn=collate_sequence_batch)

def train_model(model, tr_ld, va_ld, lr=1e-3, epochs=50):
    yt = np.concatenate([dy.numpy() for _, dy in tr_ld])
    pw = (len(yt) - yt.sum()) / max(yt.sum(), 1)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw]).to(DEVICE))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    sch = get_lr_scheduler(opt, 50, epochs * len(tr_ld))
    es = EarlyStopping(patience=15, mode='max')
    for ep in range(epochs):
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
                pr = (torch.sigmoid(model(dx)).squeeze(-1) >= 0.5).float()
                vc += (pr == dy).sum().item()
                vt += dy.size(0)
        if es(vc / vt, model):
            break
    es.load_best(model)
    return model

def eval_model(model):
    model.eval()
    ap, at = [], []
    with torch.no_grad():
        for dx, dy in te_ld:
            dx = dx.to(DEVICE)
            pr = torch.sigmoid(model(dx)).squeeze(-1).cpu().numpy()
            ap.extend(pr)
            at.extend(dy.numpy())
    ap = np.array(ap); at = np.array(at); apr = (ap >= 0.5).astype(int)
    return {'acc': accuracy_score(at, apr), 'f1': f1_score(at, apr), 'auc': roc_auc_score(at, ap)}

models_to_run = ['LNN-Small', 'CNN', 'LSTM', 'XGBoost']
all_runs = {m: {'acc': [], 'f1': [], 'auc': [], 'time': []} for m in models_to_run}

for run in range(3):
    seed = 42 + run * 100
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    print(f'--- Run {run+1}/3 (seed={seed}) ---')

    tr_ds = CachedKmerDataset(data['train_X'], data['train_y'])
    va_ds = CachedKmerDataset(data['val_X'], data['val_y'])
    tr_ld = DataLoader(tr_ds, 128, shuffle=True, collate_fn=collate_sequence_batch)
    va_ld = DataLoader(va_ds, 128, shuffle=False, collate_fn=collate_sequence_batch)

    # LNN-Small
    t0 = time.time()
    lnn = LNNClassifier(256, [128, 64, 32], pool_mode='last').to(DEVICE)
    lnn = train_model(lnn, tr_ld, va_ld)
    r = eval_model(lnn); r['time'] = time.time() - t0
    for k in ['acc', 'f1', 'auc', 'time']: all_runs['LNN-Small'][k].append(r[k])
    print(f'  LNN-Small: acc={r["acc"]:.4f} f1={r["f1"]:.4f} auc={r["auc"]:.4f} time={r["time"]:.0f}s')

    # CNN
    t0 = time.time()
    cnn = CNN1DBaseline(256, num_classes=1).to(DEVICE)
    cnn = train_model(cnn, tr_ld, va_ld)
    r = eval_model(cnn); r['time'] = time.time() - t0
    for k in ['acc', 'f1', 'auc', 'time']: all_runs['CNN'][k].append(r[k])
    print(f'  CNN:       acc={r["acc"]:.4f} f1={r["f1"]:.4f} auc={r["auc"]:.4f} time={r["time"]:.0f}s')

    # LSTM
    t0 = time.time()
    lstm = BiLSTMBaseline(input_dim=256, num_classes=1).to(DEVICE)
    lstm = train_model(lstm, tr_ld, va_ld)
    r = eval_model(lstm); r['time'] = time.time() - t0
    for k in ['acc', 'f1', 'auc', 'time']: all_runs['LSTM'][k].append(r[k])
    print(f'  LSTM:      acc={r["acc"]:.4f} f1={r["f1"]:.4f} auc={r["auc"]:.4f} time={r["time"]:.0f}s')

    # XGBoost
    t0 = time.time()
    Xtr = data['train_X'].reshape(len(data['train_X']), -1)
    Xte = data['test_X'].reshape(len(data['test_X']), -1)
    ytr = data['train_y'].astype(int)
    yte = data['test_y'].astype(int)
    sw = (len(ytr) - ytr.sum()) / max(ytr.sum(), 1)
    xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                        scale_pos_weight=sw, tree_method='hist', device='cuda',
                        random_state=seed, verbosity=0)
    xgb.fit(Xtr, ytr)
    xp = xgb.predict(Xte); xprob = xgb.predict_proba(Xte)[:, 1]
    r = {'acc': accuracy_score(yte, xp), 'f1': f1_score(yte, xp),
         'auc': roc_auc_score(yte, xprob), 'time': time.time() - t0}
    for k in ['acc', 'f1', 'auc', 'time']: all_runs['XGBoost'][k].append(r[k])
    print(f'  XGBoost:   acc={r["acc"]:.4f} f1={r["f1"]:.4f} auc={r["auc"]:.4f} time={r["time"]:.0f}s')

# 汇总
print()
print('=' * 75)
print('Final Results (mean +/- std, n=3)')
print('=' * 75)
header = f'{"Model":<14} {"Accuracy":>20} {"F1":>20} {"AUC":>20} {"Time":>10}'
print(header)
print('-' * 75)
for m in models_to_run:
    accs = all_runs[m]['acc']; f1s = all_runs[m]['f1']
    aucs = all_runs[m]['auc']; times = all_runs[m]['time']
    print(f'{m:<14} {np.mean(accs):.4f} +/- {np.std(accs):.4f}  '
          f'{np.mean(f1s):.4f} +/- {np.std(f1s):.4f}  '
          f'{np.mean(aucs):.4f} +/- {np.std(aucs):.4f}  '
          f'{np.mean(times):.0f}s')

# 保存
import json
out = {}
for m in models_to_run:
    out[m] = {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))}
              for k, v in all_runs[m].items()}
with open('lnn_salmonella/results/repeat3_results.json', 'w') as f:
    json.dump(out, f, indent=2)
print('\nSaved.')
