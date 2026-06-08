"""Fig 1: Overall pipeline schematic"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams.update({
    'font.size': 11, 'font.family': 'sans-serif',
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

fig, ax = plt.subplots(1, 1, figsize=(16, 8))
ax.set_xlim(0, 16); ax.set_ylim(0, 8)
ax.axis('off')
ax.set_facecolor('#FAFAFA')

def draw_box(x, y, w, h, text, color, text_color='white', fontsize=10, bold=True):
    """Draw a rounded box with text"""
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                          facecolor=color, edgecolor='#333333', linewidth=1.5, alpha=0.92)
    ax.add_patch(rect)
    lines = text.split('\n')
    for i, line in enumerate(lines):
        yy = y + h/2 + (len(lines)-1)*0.22 - i*0.45
        ax.text(x + w/2, yy, line, ha='center', va='center',
                fontsize=fontsize, fontweight='bold' if bold and i==0 else 'normal',
                color=text_color)

def draw_arrow(x1, y1, x2, y2, color='#555555', lw=2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                               connectionstyle='arc3,rad=0'))

def draw_label(x, y, text, fontsize=9, color='#333333'):
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=color, fontweight='bold')

# ===== ROW 1: Input =====
draw_box(0.3, 5.8, 3.0, 1.5,
         'Bacterial Genome\n(FASTA)',
         '#2C3E50', fontsize=11)
draw_label(0.3+3.0/2, 7.5, 'Step 1: Input', color='#2C3E50')

# DNA sequence visual
seq = 'ATCGATCGGCTATAGCTAGCTAGCTAG...'
ax.text(0.5, 6.2, seq, fontsize=9, fontfamily='monospace', color='#E74C3C',
        fontweight='bold', alpha=0.8)

# ===== ROW 2: Chunking =====
draw_box(4.2, 5.8, 3.0, 1.5,
         'Genomic Chunking\n32 fragments x 1000 bp',
         '#34495E', fontsize=10)
draw_label(4.2+3.0/2, 7.5, 'Step 2: Preprocessing', color='#34495E')

# Mini chunks
for i in range(5):
    yy = 6.4 - i*0.2
    ax.add_patch(plt.Rectangle((4.5, yy), 2.2, 0.15, facecolor='#3498DB', alpha=0.3+0.15*i, edgecolor='none'))

# Arrow 1->2
draw_arrow(3.35, 6.55, 4.15, 6.55)

# ===== ROW 3: k-mer Encoding =====
draw_box(8.1, 5.8, 3.0, 1.5,
         'k-mer Frequency Encoding\nk=4, 256-dim vector',
         '#1ABC9C', text_color='white', fontsize=10)
draw_label(8.1+3.0/2, 7.5, 'Step 3: Feature Extraction', color='#1ABC9C')

# k-mer visual
kmer_text = 'f = [0.012, 0.003, 0.021, ...]_{256}'
ax.text(8.3, 6.2, kmer_text, fontsize=9, fontfamily='monospace', color='white',
        fontweight='bold', alpha=0.9)

# Arrow 2->3
draw_arrow(7.25, 6.55, 8.05, 6.55)

# ===== ROW 4: Model =====
# CNN
draw_box(1.0, 2.5, 2.8, 2.5,
         '1D-CNN\n\nConv1D(3 layers)\n+ MaxPool\n+ FC',
         '#2196F3', fontsize=10)

# LNN
draw_box(4.5, 2.5, 2.8, 2.5,
         'LNN (CfC)\n\ndx/dt = -x/τ + f(x,I,θ)\n+ Attention Pool',
         '#00BCD4', fontsize=10)

# GRU
draw_box(8.0, 2.5, 2.8, 2.5,
         'GRU\n\nGated Recurrent\n+ Attention Pool',
         '#4CAF50', fontsize=10)

# XGBoost
draw_box(11.5, 2.5, 2.8, 2.5,
         'XGBoost\n\nGradient Boosted\nTrees',
         '#FF9800', fontsize=10)

draw_label(8.0, 5.3, 'Step 4: Model Architectures', color='#555555')

# Arrows from k-mer to each model
for mx in [2.4, 5.9, 9.4, 12.9]:
    draw_arrow(9.6, 5.75, mx, 5.05, color='#888888', lw=1.5)

# ===== ROW 5: Output =====
draw_box(5.5, 0.5, 3.8, 1.3,
         'Salmonella ✓\nNon-Salmonella ✗',
         '#E74C3C' if False else '#27AE60', text_color='white', fontsize=11)
draw_label(5.5+3.8/2, 2.1, 'Step 5: Classification', color='#555555')

# Accuracy badge
ax.add_patch(plt.Circle((13.2, 1.15), 0.8, facecolor='#F39C12', edgecolor='#333', linewidth=2, alpha=0.9))
ax.text(13.2, 1.35, '99.6%', ha='center', va='center', fontsize=16, fontweight='bold', color='white')
ax.text(13.2, 0.88, 'Accuracy', ha='center', va='center', fontsize=8, color='white')

# Arrows from models to output
for mx in [2.4, 5.9, 9.4, 12.9]:
    draw_arrow(mx, 2.45, 7.4, 1.85, color='#888888', lw=1.2)

# ===== Performance comparison mini-table =====
table_data = [
    ['CNN', '99.6%', '174K', '2s'],
    ['GRU', '98.3%', '196K', '4s'],
    ['LNN+Attn', '97.8%', '413K', '130s'],
    ['XGBoost', '94.2%', '500K', '2s'],
]
table_x, table_y = 0.3, 0.3
ax.text(0.5, 1.7, 'Performance Summary:', fontsize=8, fontweight='bold', color='#333')
for i, row in enumerate(table_data):
    for j, val in enumerate(row):
        ax.text(0.5 + j*1.4, 1.3 - i*0.3, val, fontsize=7,
                fontweight='bold' if j==1 else 'normal',
                color='#2196F3' if i==0 else '#4CAF50' if i==1 else '#00BCD4' if i==2 else '#FF9800')

# ===== Title =====
ax.text(8.0, 8.1, 'WGS-based Salmonella Detection Pipeline',
        ha='center', va='center', fontsize=18, fontweight='bold', color='#2C3E50')

# ===== Annotations =====
ax.annotate('32 randomly sampled\n1000bp regions',
            xy=(4.5, 5.5), xytext=(2.5, 5.0),
            fontsize=8, color='#555', ha='center',
            arrowprops=dict(arrowstyle='->', color='#999', lw=1))

ax.annotate('Normalized\nk-mer counts',
            xy=(8.5, 5.5), xytext=(8.5, 4.5),
            fontsize=8, color='#555', ha='center',
            arrowprops=dict(arrowstyle='->', color='#999', lw=1))

plt.tight_layout()
plt.savefig('lnn_salmonella/results/fig1_pipeline.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print('Saved: lnn_salmonella/results/fig1_pipeline.png')
