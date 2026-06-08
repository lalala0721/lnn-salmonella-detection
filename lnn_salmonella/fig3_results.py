"""Fig 3: Core results comparison — publication-quality figure"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
    'legend.fontsize': 10, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

models = ['CNN', 'GRU', 'LNN\n+Attn', 'XGBoost', 'LSTM', 'RF', 'LNN\n(last)', 'Trans-\nformer']
colors = ['#2196F3', '#4CAF50', '#00BCD4', '#FF9800', '#9C27B0', '#795548', '#607D8B', '#F44336']

# Data: Accuracy, F1, AUC
accuracy = [0.9964, 0.9827, 0.9784, 0.9416, 0.8882, 0.8723, 0.8095, 0.7987]
f1       = [0.9977, 0.9891, 0.9864, 0.9643, 0.9347, 0.9260, 0.8922, 0.8881]
auc      = [1.000,  0.9987, 0.9944, 0.9888, 0.8431, 0.9751, 0.6710, 0.5000]
params   = [174,    196,    413,    500,    824,    5000,   101,    851]   # in K
train_t  = [2,      4,      130,    2,      3,      1,      44,     3]     # seconds

x = np.arange(len(models))
width = 0.22

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# (a) Accuracy & F1
ax = axes[0, 0]
bars1 = ax.bar(x - width/2, accuracy, width, label='Accuracy', color='#2196F3', edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x + width/2, f1, width, label='F1 Score', color='#4CAF50', edgecolor='white', linewidth=0.5)
for bar, val in zip(bars1, accuracy):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008, f'{val:.3f}',
            ha='center', va='bottom', fontsize=7, rotation=90)
for bar, val in zip(bars2, f1):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008, f'{val:.3f}',
            ha='center', va='bottom', fontsize=7, rotation=90)
ax.set_xticks(x); ax.set_xticklabels(models)
ax.set_ylabel('Score'); ax.set_title('(a) Accuracy and F1 Score', fontweight='bold')
ax.legend(loc='lower right'); ax.set_ylim(0.70, 1.06)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# (b) AUC
ax = axes[0, 1]
bars = ax.bar(x, auc, width*1.2, color=colors, edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, auc):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}',
            ha='center', va='bottom', fontsize=8, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(models)
ax.set_ylabel('AUC-ROC'); ax.set_title('(b) Area Under ROC Curve', fontweight='bold')
ax.set_ylim(0.40, 1.08); ax.grid(axis='y', alpha=0.3, linestyle='--')

# (c) Parameters
ax = axes[1, 0]
bars = ax.bar(x, params, width*1.2, color=colors, edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, params):
    label = f'{val}K' if val < 1000 else f'{val/1000:.1f}M'
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15, label,
            ha='center', va='bottom', fontsize=8, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(models)
ax.set_ylabel('Parameters'); ax.set_title('(c) Model Size', fontweight='bold')
ax.set_yscale('log'); ax.grid(axis='y', alpha=0.3, linestyle='--')

# (d) Train time
ax = axes[1, 1]
bars = ax.bar(x[:4], train_t[:4], width*1.2, color=colors[:4], edgecolor='white', linewidth=0.5)  # first 4 are fast
# LNN+Attn is tall
for bar, val in zip(bars, train_t[:4]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'{val}s',
            ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_xticks(x[:4]); ax.set_xticklabels(models[:4])
ax.set_ylabel('Training Time (seconds)'); ax.set_title('(d) Training Efficiency', fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--')
# Add note about LNN+Attn
ax.annotate('LNN+Attn: 130s\n(off-scale)', xy=(0.98, 0.95), xycoords='axes fraction',
            ha='right', va='top', fontsize=9, fontstyle='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Comprehensive Benchmark for WGS-based Salmonella Detection',
             fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('lnn_salmonella/results/fig3_results.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved: lnn_salmonella/results/fig3_results.png')
