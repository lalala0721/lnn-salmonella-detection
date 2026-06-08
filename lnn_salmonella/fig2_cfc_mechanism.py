"""Fig 2: CfC ODE mechanism vs GRU vs RNN — conceptual comparison"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc, Circle, Wedge
import numpy as np

plt.rcParams.update({'font.size': 10, 'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})

fig, axes = plt.subplots(1, 3, figsize=(18, 7))
fig.patch.set_facecolor('white')

def draw_rounded_box(ax, x, y, w, h, text, color, tc='white', fs=9, bold=False):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor='#333', linewidth=1.2, alpha=0.9)
    ax.add_patch(rect)
    lines = text.split('\n')
    for i, line in enumerate(lines):
        yy = y + h/2 + (len(lines)-1)*0.18 - i*0.36
        ax.text(x+w/2, yy, line, ha='center', va='center', fontsize=fs,
                fontweight='bold' if (bold and i==0) else 'normal', color=tc)

def draw_arrow(ax, x1, y1, x2, y2, color='#555', lw=1.5, style='->'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))

# ===== PANEL A: Simple RNN =====
ax = axes[0]
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
ax.set_title('(a) Simple RNN', fontsize=14, fontweight='bold', color='#E74C3C', pad=10)

# Formula
ax.text(5, 9.3, r'$\mathbf{h}_t = \tanh(\mathbf{W}_h \mathbf{h}_{t-1} + \mathbf{W}_x \mathbf{x}_t + \mathbf{b})$',
        ha='center', fontsize=12, fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#FADBD8', alpha=0.5))

# Unrolled diagram
for t, tx in [(0, 1.5), (1, 4.0), (2, 6.5), (3, 9.0)]:
    color = '#E74C3C' if t < 3 else '#E67E22'
    c = Circle((tx, 6.5), 0.7, facecolor=color, edgecolor='#333', linewidth=1.5, alpha=0.85)
    ax.add_patch(c)
    ax.text(tx, 6.5, f'$h_{t}$', ha='center', va='center', fontsize=11,
            fontweight='bold', color='white')

# Input arrows
for t, tx in enumerate([1.5, 4.0, 6.5, 9.0]):
    c = Circle((tx, 4.5), 0.5, facecolor='#3498DB', edgecolor='#333', linewidth=1, alpha=0.7)
    ax.add_patch(c)
    ax.text(tx, 4.5, f'$x_{t}$', ha='center', va='center', fontsize=9, color='white', fontweight='bold')
    draw_arrow(ax, tx, 5.0, tx, 5.7, '#555', 1.2)

# Horizontal connections
for t in range(3):
    draw_arrow(ax, 1.5+t*2.5+0.7, 6.5, 4.0+t*2.5-0.7, 6.5, '#E74C3C', 1.5)

# Output
draw_arrow(ax, 9.0, 7.2, 9.0, 8.2, '#555', 1.5)
draw_rounded_box(ax, 8.0, 8.3, 2.0, 0.8, 'Output\ny', '#2C3E50', fs=9)

# Limitations
ax.text(5, 2.8, 'Limitation: Vanishing gradients\nNo gating mechanism',
        ha='center', fontsize=10, color='#E74C3C', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#FADBD8', alpha=0.4))

# Key parameters
ax.text(5, 1.5, f'Parameters: 66K', ha='center', fontsize=10, fontweight='bold', color='#555')
ax.text(5, 0.8, f'Accuracy: 84.2%', ha='center', fontsize=11, fontweight='bold', color='#E74C3C')

# ===== PANEL B: GRU =====
ax = axes[1]
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
ax.set_title('(b) GRU (Gated Recurrent Unit)', fontsize=14, fontweight='bold', color='#4CAF50', pad=10)

# Formulas
ax.text(5, 9.3, r'$\mathbf{z}_t = \sigma(\mathbf{W}_z[\mathbf{h}_{t-1}, \mathbf{x}_t])$   Update gate',
        ha='center', fontsize=10, fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#D5F5E3', alpha=0.5))
ax.text(5, 8.5, r'$\mathbf{r}_t = \sigma(\mathbf{W}_r[\mathbf{h}_{t-1}, \mathbf{x}_t])$   Reset gate',
        ha='center', fontsize=10, fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#D5F5E3', alpha=0.5))
ax.text(5, 7.7, r'$\mathbf{\tilde{h}}_t = \tanh(\mathbf{W}[\mathbf{r}_t \odot \mathbf{h}_{t-1}, \mathbf{x}_t])$',
        ha='center', fontsize=10, fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#D5F5E3', alpha=0.5))

# GRU cell diagram
cx, cy = 5, 5.2
c = Circle((cx, cy), 1.5, facecolor='#4CAF50', edgecolor='#333', linewidth=2, alpha=0.15)
ax.add_patch(c)

# Gates inside
draw_rounded_box(ax, cx-1.0, cy+0.4, 1.8, 0.6, 'z (Update)', '#27AE60', fs=8)
draw_rounded_box(ax, cx-1.0, cy-0.7, 1.8, 0.6, 'r (Reset)', '#2ECC71', fs=8)
draw_rounded_box(ax, cx+1.0, cy-0.15, 1.5, 0.8, 'h̃ (Candidate)', '#1ABC9C', fs=8)

# Input
c_in = Circle((cx, 3.0), 0.4, facecolor='#3498DB', edgecolor='#333', linewidth=1)
ax.add_patch(c_in)
ax.text(cx, 3.0, '$x_t$', ha='center', va='center', fontsize=9, color='white', fontweight='bold')
draw_arrow(ax, cx, 3.4, cx, 3.6, '#555', 1.2)

# Output
draw_arrow(ax, cx, 6.7, cx, 7.3, '#555', 1.2)
draw_rounded_box(ax, cx-1.0, 7.4, 2.0, 0.7, '$h_t$ (Output)', '#2C3E50', fs=9)

# Advantages
ax.text(5, 1.8, 'Advantage: Gating controls information flow\nLearns what to keep vs. forget',
        ha='center', fontsize=10, color='#4CAF50', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#D5F5E3', alpha=0.4))
ax.text(5, 0.8, f'Parameters: 196K  |  Accuracy: 98.3%', ha='center', fontsize=11, fontweight='bold', color='#4CAF50')

# ===== PANEL C: CfC (LNN) =====
ax = axes[2]
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
ax.set_title('(c) CfC — Closed-form Continuous-time (LNN)', fontsize=14, fontweight='bold', color='#00BCD4', pad=10)

# ODE formula
ax.text(5, 9.3, r'$\mathbf{\dot{x}}(t) = -\frac{\mathbf{x}(t)}{\tau} + f(\mathbf{x}(t), \mathbf{I}(t), \theta)$',
        ha='center', fontsize=12, fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#D6EAF8', alpha=0.5))

# Key features
features = [
    (5, 8.2, 'Continuous-time ODE dynamics', '#2980B9'),
    (5, 7.5, 'Closed-form solution — no numerical solver', '#1ABC9C'),
    (5, 6.8, 'Liquid time-constant τ adapts per neuron', '#16A085'),
]
for fx, fy, ft, fc in features:
    ax.text(fx, fy, f'• {ft}', ha='center', fontsize=10, color=fc, fontweight='bold')

# Continuous trajectory visualization
t = np.linspace(0, 4*np.pi, 200)
# Salmonella-like smooth trajectory
x_s = 3 + 2*np.sin(t)*np.exp(-0.3*t)
y_s = 5 + 2*np.cos(t)*np.exp(-0.3*t)
# Non-Salmonella-like erratic trajectory
x_n = 3 + 2.5*np.sin(1.5*t)*np.exp(-0.15*t) + 0.3*np.random.randn(200)
y_n = 5 + 2.5*np.cos(1.5*t)*np.exp(-0.15*t) + 0.3*np.random.randn(200)

ax.plot(x_s, y_s, color='#E74C3C', linewidth=2.5, alpha=0.8, label='Salmonella trajectory')
ax.plot(x_n, y_n, color='#3498DB', linewidth=2.0, alpha=0.6, label='Non-Salmonella trajectory')

# Start/end markers
ax.scatter([x_s[0]], [y_s[0]], s=80, color='#E74C3C', marker='o', edgecolors='white', linewidth=1, zorder=5)
ax.scatter([x_s[-1]], [y_s[-1]], s=120, color='#E74C3C', marker='*', edgecolors='white', linewidth=1, zorder=5)
ax.scatter([x_n[0]], [y_n[0]], s=80, color='#3498DB', marker='o', edgecolors='white', linewidth=1, zorder=5)
ax.scatter([x_n[-1]], [y_n[-1]], s=120, color='#3498DB', marker='*', edgecolors='white', linewidth=1, zorder=5)

ax.legend(fontsize=8, loc='upper left')
ax.set_xlim(0, 7); ax.set_ylim(1.5, 8.5)

# Attractor annotation
ax.annotate('Attractor\nbasin', xy=(x_s[-1], y_s[-1]), xytext=(5, 5.5),
            fontsize=8, ha='center', color='#E74C3C',
            arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=1))

# Advantages
ax.text(5, 2.0, 'Advantage: Interpretable dynamics\nSmooth trajectories reveal decision process',
        ha='center', fontsize=10, color='#00BCD4', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#D6EAF8', alpha=0.4))
ax.text(5, 1.0, f'Parameters: 101-413K  |  Accuracy: 90-98%', ha='center', fontsize=11, fontweight='bold', color='#00BCD4')
ax.text(5, 0.4, '6x more stable than CNN at low data', ha='center', fontsize=9, fontstyle='italic', color='#555')

# ===== GLOBAL =====
fig.suptitle('Recurrent Architectures for Genomic Sequence Classification',
             fontsize=16, fontweight='bold', y=1.02)

# Performance comparison strip at bottom
fig.text(0.5, -0.01,
         'Simple RNN: 84.2% Acc, 66K params  |  GRU: 98.3% Acc, 196K params  |  CfC/LNN: 97.8% Acc (+Attn), 101-413K params, 6x more data-efficient',
         ha='center', fontsize=11, fontweight='bold', color='#333',
         bbox=dict(boxstyle='round', facecolor='#F8F9F9', edgecolor='#CCC', alpha=0.8))

plt.tight_layout()
plt.savefig('lnn_salmonella/results/fig2_cfc_mechanism.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print('Saved: lnn_salmonella/results/fig2_cfc_mechanism.png')
