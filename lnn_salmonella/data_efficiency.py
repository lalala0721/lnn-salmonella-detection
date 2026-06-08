"""数据效率实验: LNN vs CNN vs XGBoost @ 10/25/50/100%"""
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
import time, random, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import DEVICE
from data.dataset import CachedKmerDataset, collate_sequence_batch
from models.lnn_classifier import create_lnn_model
from models.baselines import CNN1DBaseline
from utils import EarlyStopping, get_lr_scheduler

random.seed(42); np.random.seed(42); torch.manual_seed(42)

# 加载数据
data = np.load('lnn_salmonella/data/cache/kmer4_chunks32.npz')
X_train_all = data['train_X']; y_train_all = data['train_y'].astype(int)
X_test = data['test_X']; y_test = data['test_y'].astype(int)
X_val = data['val_X']; y_val = data['val_y'].astype(int)
N = len(X_train_all)
fractions = [0.10, 0.25, 0.50, 1.00]
n_repeats = 3

results = {m: {f: {'acc': [], 'f1': [], 'auc': [], 'time': []} for f in fractions}
           for m in ['LNN-Small', 'CNN', 'XGBoost']}

for repeat in range(n_repeats):
    seed = 42 + repeat * 100
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    for frac in fractions:
        n_samples = max(10, int(N * frac))
        indices = random.sample(range(N), n_samples)
        X_sub = X_train_all[indices]; y_sub = y_train_all[indices]
        print(f'Repeat {repeat+1}, Data {frac:.0%} (n={n_samples})')
        print('-'*45)

        train_ds = CachedKmerDataset(X_sub, y_sub)
        val_ds = CachedKmerDataset(X_val, y_val)
        test_ds = CachedKmerDataset(X_test, y_test)
        train_ld = DataLoader(train_ds, 128, shuffle=True, collate_fn=collate_sequence_batch)
        val_ld = DataLoader(val_ds, 128, shuffle=False, collate_fn=collate_sequence_batch)
        test_ld = DataLoader(test_ds, 128, shuffle=False, collate_fn=collate_sequence_batch)
        pw = (len(y_sub)-y_sub.sum())/max(y_sub.sum(),1)

        # === LNN-Small ===
        t0 = time.time()
        model = create_lnn_model('small', num_classes=1, input_dim=256).to(DEVICE)
        crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw]).to(DEVICE))
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scaler = torch.amp.GradScaler('cuda')
        sch = get_lr_scheduler(opt, 50, 50*len(train_ld))
        es = EarlyStopping(patience=15, mode='max')
        for ep in range(50):
            model.train()
            for dx, dy in train_ld:
                dx, dy = dx.to(DEVICE), dy.to(DEVICE)
                with torch.amp.autocast('cuda'):
                    lo = model(dx); loss = crit(lo.squeeze(-1), dy)
                opt.zero_grad(); scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); sch.step()
            model.eval()
            vc, vt = 0, 0
            with torch.no_grad():
                for dx, dy in val_ld:
                    dx, dy = dx.to(DEVICE), dy.to(DEVICE)
                    pr = (torch.sigmoid(model(dx)).squeeze(-1)>=0.5).float()
                    vc+=(pr==dy).sum().item(); vt+=dy.size(0)
            if es(vc/vt, model): break
        es.load_best(model)
        model.eval()
        all_p, all_t = [], []
        with torch.no_grad():
            for dx, dy in test_ld:
                dx = dx.to(DEVICE)
                pr = torch.sigmoid(model(dx)).squeeze(-1).cpu().numpy()
                all_p.extend(pr); all_t.extend(dy.numpy())
        all_p=np.array(all_p); all_t=np.array(all_t); all_pr=(all_p>=0.5).astype(int)
        t_lnn = time.time()-t0
        results['LNN-Small'][frac]['acc'].append(accuracy_score(all_t, all_pr))
        results['LNN-Small'][frac]['f1'].append(f1_score(all_t, all_pr))
        results['LNN-Small'][frac]['auc'].append(roc_auc_score(all_t, all_p))
        results['LNN-Small'][frac]['time'].append(t_lnn)
        print(f'  LNN: acc={accuracy_score(all_t,all_pr):.4f}, time={t_lnn:.0f}s')

        # === CNN ===
        t0 = time.time()
        cnn = CNN1DBaseline(input_dim=256, num_classes=1).to(DEVICE)
        opt_c = torch.optim.AdamW(cnn.parameters(), lr=1e-3, weight_decay=1e-4)
        scaler_c = torch.amp.GradScaler('cuda')
        sch_c = get_lr_scheduler(opt_c, 50, 50*len(train_ld))
        es_c = EarlyStopping(patience=15, mode='max')
        for ep in range(50):
            cnn.train()
            for dx, dy in train_ld:
                dx, dy = dx.to(DEVICE), dy.to(DEVICE)
                with torch.amp.autocast('cuda'):
                    lo = cnn(dx); loss = crit(lo.squeeze(-1), dy)
                opt_c.zero_grad(); scaler_c.scale(loss).backward()
                scaler_c.unscale_(opt_c)
                torch.nn.utils.clip_grad_norm_(cnn.parameters(), 1.0)
                scaler_c.step(opt_c); scaler_c.update(); sch_c.step()
            cnn.eval()
            vc, vt = 0, 0
            with torch.no_grad():
                for dx, dy in val_ld:
                    dx, dy = dx.to(DEVICE), dy.to(DEVICE)
                    pr = (torch.sigmoid(cnn(dx)).squeeze(-1)>=0.5).float()
                    vc+=(pr==dy).sum().item(); vt+=dy.size(0)
            if es_c(vc/vt, cnn): break
        es_c.load_best(cnn)
        cnn.eval()
        all_p, all_t = [], []
        with torch.no_grad():
            for dx, dy in test_ld:
                dx = dx.to(DEVICE)
                pr = torch.sigmoid(cnn(dx)).squeeze(-1).cpu().numpy()
                all_p.extend(pr); all_t.extend(dy.numpy())
        all_p=np.array(all_p); all_t=np.array(all_t); all_pr=(all_p>=0.5).astype(int)
        t_cnn = time.time()-t0
        results['CNN'][frac]['acc'].append(accuracy_score(all_t, all_pr))
        results['CNN'][frac]['f1'].append(f1_score(all_t, all_pr))
        results['CNN'][frac]['auc'].append(roc_auc_score(all_t, all_p))
        results['CNN'][frac]['time'].append(t_cnn)
        print(f'  CNN: acc={accuracy_score(all_t,all_pr):.4f}, time={t_cnn:.0f}s')

        # === XGBoost ===
        t0 = time.time()
        X_sub_flat = X_sub.reshape(len(X_sub), -1)
        X_test_flat = X_test.reshape(len(X_test), -1)
        sw = (len(y_sub)-y_sub.sum())/max(y_sub.sum(),1)
        xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                            scale_pos_weight=sw, tree_method='hist', device='cuda',
                            random_state=seed, verbosity=0)
        xgb.fit(X_sub_flat, y_sub)
        xgb_pred = xgb.predict(X_test_flat)
        xgb_prob = xgb.predict_proba(X_test_flat)[:,1]
        t_xgb = time.time()-t0
        results['XGBoost'][frac]['acc'].append(accuracy_score(y_test, xgb_pred))
        results['XGBoost'][frac]['f1'].append(f1_score(y_test, xgb_pred))
        results['XGBoost'][frac]['auc'].append(roc_auc_score(y_test, xgb_prob))
        results['XGBoost'][frac]['time'].append(t_xgb)
        print(f'  XGB: acc={accuracy_score(y_test,xgb_pred):.4f}, time={t_xgb:.0f}s')

# 汇总
print(f'\n{"="*65}')
print('Data Efficiency Results (mean +/- std, n=3)')
print(f'{"="*65}')
print(f'{"Model":<12} {"10%":>14} {"25%":>14} {"50%":>14} {"100%":>14}')
print('-'*60)
for model in ['LNN-Small', 'CNN', 'XGBoost']:
    means = []
    for f in fractions:
        accs = results[model][f]['acc']
        means.append(f'{np.mean(accs):.4f}+/-{np.std(accs):.3f}')
    print(f'{model:<12} {means[0]:>14} {means[1]:>14} {means[2]:>14} {means[3]:>14}')

# 相对衰减
print(f'\n{"Model":<12} {"10% vs 100%":>14}')
print('-'*28)
for model in ['LNN-Small', 'CNN', 'XGBoost']:
    drop = np.mean(results[model][0.10]['acc']) - np.mean(results[model][1.00]['acc'])
    print(f'{model:<12} {drop:>+13.4f}')

# 保存
out = {}
for m in results:
    out[m] = {}
    for f in fractions:
        out[m][str(f)] = {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))}
                          for k, v in results[m][f].items()}
Path('lnn_salmonella/results').mkdir(parents=True, exist_ok=True)
with open('lnn_salmonella/results/data_efficiency.json', 'w') as f:
    json.dump(out, f, indent=2)
print('\nDone.')
