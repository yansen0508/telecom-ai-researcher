"""
Generate all paper figures for the ComplexUNet OFDM channel estimation paper.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as FancyArrowPatch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D
import os

# Paths
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))          # .../state/04_experiments/code
EXPERIMENTS_DIR = os.path.dirname(SCRIPT_DIR)                         # .../state/04_experiments
STATE_DIR      = os.path.dirname(EXPERIMENTS_DIR)                     # .../state

RESULTS_JSON  = os.path.join(SCRIPT_DIR, "results.json")
TRAINING_JSON = os.path.join(STATE_DIR, "05_analysis/training_data.json")
FIGURES_DIR   = os.path.join(STATE_DIR, "06_manuscript/figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# IEEE style settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'legend.fontsize': 7,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'lines.linewidth': 1.4,
    'lines.markersize': 4.5,
    'grid.alpha': 0.35,
    'grid.linestyle': '--',
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
})

# Load data
with open(RESULTS_JSON) as f:
    results = json.load(f)

with open(TRAINING_JSON) as f:
    training = json.load(f)

snr = results["snr_values"]  # [5,10,15,20,25]

# Colors and markers for the 5 methods
METHODS = ['LS', 'MMSE', 'DNN', 'RealUNet', 'ComplexUNet']
COLORS  = ['#999999', '#2166ac', '#d6604d', '#4dac26', '#762a83']
MARKERS = ['s',       '*',       '^',       'o',       'D']
LSTYLES = ['--',      '--',      '--',      '-',       '-']
LABELS  = ['LS', 'MMSE', 'DNN', 'RealUNet (ablation)', 'ComplexUNet (proposed)']

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: NMSE vs SNR
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(3.3, 1.95))
for method, color, marker, ls, label in zip(METHODS, COLORS, MARKERS, LSTYLES, LABELS):
    nmse = results["nmse_db"][method]
    ax.plot(snr, nmse, linestyle=ls, color=color, marker=marker,
            label=label, markerfacecolor='white' if ls == '--' else color,
            markeredgecolor=color, linewidth=1.5)

ax.set_xlabel('SNR (dB)')
ax.set_ylabel('NMSE (dB)')
ax.set_xlim(4, 26)
ax.set_xticks([5, 10, 15, 20, 25])
ax.set_ylim(-32, -4)
ax.set_yticks([-30, -25, -20, -15, -10, -5])
ax.grid(True)
ax.legend(loc='lower left', framealpha=0.9, ncol=2, fontsize=6,
          handlelength=1.5, columnspacing=0.8, handletextpad=0.4)
ax.set_title('')
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "fig1_nmse.pdf"))
plt.close(fig)
print("Saved fig1_nmse.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: BER vs SNR (log scale)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(3.3, 1.95))
for method, color, marker, ls, label in zip(METHODS, COLORS, MARKERS, LSTYLES, LABELS):
    ber = results["ber"][method]
    ax.semilogy(snr, ber, linestyle=ls, color=color, marker=marker,
                label=label, markerfacecolor='white' if ls == '--' else color,
                markeredgecolor=color, linewidth=1.5)

ax.set_xlabel('SNR (dB)')
ax.set_ylabel('BER')
ax.set_xlim(4, 26)
ax.set_xticks([5, 10, 15, 20, 25])
ax.set_ylim(1e-3, 3e-1)
ax.grid(True, which='both')
ax.legend(loc='lower left', framealpha=0.9, ncol=2, fontsize=6,
          handlelength=1.5, columnspacing=0.8, handletextpad=0.4)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "fig2_ber.pdf"))
plt.close(fig)
print("Saved fig2_ber.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Training convergence
# ─────────────────────────────────────────────────────────────────────────────
cu = training["ComplexUNet"]
ru = training["RealUNet"]

cu_epochs = cu["epochs"]
cu_val    = cu["val_nmses"]
ru_epochs = ru["epochs"]
ru_val    = ru["val_nmses"]

# Best epoch info from training_results.json
cu_best_epoch = cu_epochs[cu_val.index(min(cu_val))]
ru_best_epoch = ru_epochs[ru_val.index(min(ru_val))]
cu_best_nmse  = min(cu_val)
ru_best_nmse  = min(ru_val)

fig, ax = plt.subplots(figsize=(3.3, 1.95))
ax.plot(cu_epochs, cu_val, color='#762a83', linewidth=1.4, label='ComplexUNet')
ax.plot(ru_epochs, ru_val, color='#4dac26', linewidth=1.4, linestyle='--', label='RealUNet')

# Mark best epochs with vertical dashed lines
ax.axvline(x=cu_best_epoch, color='#762a83', linestyle=':', linewidth=1.0, alpha=0.8)
ax.axvline(x=ru_best_epoch, color='#4dac26',  linestyle=':', linewidth=1.0, alpha=0.8)

# Annotate best points
ax.scatter([cu_best_epoch], [cu_best_nmse], color='#762a83', zorder=5, s=30)
ax.scatter([ru_best_epoch], [ru_best_nmse], color='#4dac26',  zorder=5, s=30)
ax.annotate(f'ep {cu_best_epoch}\n{cu_best_nmse:.2f} dB',
            xy=(cu_best_epoch, cu_best_nmse), xytext=(cu_best_epoch - 58, cu_best_nmse + 2.0),
            fontsize=6.0, color='#762a83',
            arrowprops=dict(arrowstyle='->', color='#762a83', lw=0.8))
ax.annotate(f'ep {ru_best_epoch}\n{ru_best_nmse:.2f} dB',
            xy=(ru_best_epoch, ru_best_nmse), xytext=(ru_best_epoch + 8, ru_best_nmse + 2.0),
            fontsize=6.0, color='#4dac26',
            arrowprops=dict(arrowstyle='->', color='#4dac26', lw=0.8))

ax.set_xlabel('Epoch')
ax.set_ylabel('Validation NMSE (dB)')
ax.set_xlim(1, 170)
ax.set_xticks([1, 40, 80, 120, 166])
ax.set_ylim(-17, -7)
ax.grid(True)
ax.legend(loc='upper right', framealpha=0.9, fontsize=6)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "fig3_training.pdf"))
plt.close(fig)
print("Saved fig3_training.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 0: Methodology System Figure (fig0_system.pdf)
# Left panel: clean system flow (bold labels only, no sub-text)
# Right panel: architecture details as a table
# ─────────────────────────────────────────────────────────────────────────────

fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(7.0, 2.2),
                                         gridspec_kw={'width_ratios': [1, 1.15]})

for ax in (ax_left, ax_right):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

# ── Helper functions ──────────────────────────────────────────────────────────
def draw_box(ax, x, y, w, h, label, color='#cfe2f3', fontsize=7.0, multiline=False):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.08", linewidth=0.8,
                         edgecolor='#555555', facecolor=color, zorder=3)
    ax.add_patch(box)
    ax.text(x, y, label, ha='center', va='center', fontsize=fontsize,
            fontweight='bold', zorder=4,
            multialignment='center' if multiline else 'center')

def draw_arrow(ax, x0, y0, x1, y1, label=None, color='#333333'):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.0),
                zorder=2)
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx + 0.12, my, label, ha='left', va='center', fontsize=5.5,
                color='#555555', zorder=5)

# ── LEFT PANEL: System Flow (bold labels only) ───────────────────────────────
# Row 1: TX → Channel → RX
draw_box(ax_left, 1.5, 8.5, 2.2, 0.85, 'OFDM TX', color='#d9ead3')
draw_box(ax_left, 5.0, 8.5, 2.6, 0.85, 'Multipath\nChannel', color='#fff2cc', multiline=True)
draw_box(ax_left, 8.5, 8.5, 2.2, 0.85, 'OFDM RX', color='#d9ead3')
draw_arrow(ax_left, 2.6, 8.5, 3.7, 8.5)
draw_arrow(ax_left, 6.3, 8.5, 7.4, 8.5)

# Row 2: Pilot extraction + LS estimate
draw_box(ax_left, 5.0, 6.8, 2.8, 0.80, 'Pilot Extraction', color='#fff2cc')
draw_arrow(ax_left, 8.5, 8.08, 8.5, 7.8)
ax_left.annotate('', xy=(5.0, 7.2), xytext=(8.5, 7.8),
                arrowprops=dict(arrowstyle='->', color='#333333', lw=1.0), zorder=2)
draw_arrow(ax_left, 5.0, 6.4, 5.0, 5.9, label='$Y[k_p]$')

draw_box(ax_left, 5.0, 5.4, 2.8, 0.80,
         'LS Estimate $\\hat{\\mathbf{h}}_{\\mathrm{LS}}$', color='#fce5cd', fontsize=6.5)

# Row 3: 5-channel input
draw_box(ax_left, 5.0, 3.9, 5.8, 0.90, '5-Channel Input', color='#d0e4f7', fontsize=7.0)
draw_arrow(ax_left, 5.0, 5.0, 5.0, 4.35)
# noise input
ax_left.text(1.5, 4.3, 'Noise $\\boldsymbol{\\epsilon}_t$', ha='center', va='center',
             fontsize=6.0, color='#666666',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='#eeeeee', edgecolor='#aaaaaa', lw=0.6))
ax_left.annotate('', xy=(2.1, 3.95), xytext=(2.1, 4.15),
                arrowprops=dict(arrowstyle='->', color='#888888', lw=0.9), zorder=2)

# Row 4: ComplexUNet denoiser
draw_box(ax_left, 5.0, 2.5, 4.6, 0.80,
         'ComplexUNet Denoiser $\\boldsymbol{\\epsilon}_\\theta$', color='#cfe2f3', fontsize=6.5)
draw_arrow(ax_left, 5.0, 3.45, 5.0, 2.90)

# Row 5: Output
draw_box(ax_left, 5.0, 1.35, 4.0, 0.80,
         'Channel Estimate $\\hat{\\mathbf{h}}$', color='#d9ead3', fontsize=6.8)
draw_arrow(ax_left, 5.0, 2.10, 5.0, 1.75)

ax_left.set_title('(a) System Pipeline', fontsize=8.0, pad=2)

# ── RIGHT PANEL: Architecture Table ──────────────────────────────────────────
ax_right.axis('off')

col_labels = ['Block', 'Ch.', 'Operation / Details']
table_data = [
    ['Input',         '5',          r'$[\mathrm{Re}(h_t),\,\mathrm{Im}(h_t),\,\mathrm{Re}(\hat{h}_{LS}),\,\mathrm{Im}(\hat{h}_{LS}),\,m_p]$'],
    ['Encoder ×3',    '32→64→128',  'Conv1d, GroupNorm(8), SiLU, AvgPool'],
    ['Bottleneck',    '128',        'SelfAttention1D (4 heads)'],
    ['Decoder ×3',    '128→64→32',  'Upsample + Conv1d, GN, SiLU + skip'],
    ['Output Gate',   '2',          'Quantum Phase Rotation (Rz matrix)'],
    ['Inference',     '—',          'DDIM, 200 steps, η = 0'],
]

# Row colors: header dark, alternating light
row_colors = [['#c9daf8', '#c9daf8', '#c9daf8']] + \
             [['#eaf1fb' if i % 2 == 0 else '#f8f9fa'] * 3 for i in range(len(table_data))]

tbl = ax_right.table(
    cellText=table_data,
    colLabels=col_labels,
    loc='center',
    cellLoc='left',
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(6.0)

# Style header
for j in range(3):
    tbl[0, j].set_facecolor('#3c78d8')
    tbl[0, j].set_text_props(color='white', fontweight='bold')
    tbl[0, j].set_edgecolor('#aaaaaa')

# Style data rows
for i in range(1, len(table_data) + 1):
    bg = '#eaf1fb' if i % 2 == 1 else '#f8f9fa'
    for j in range(3):
        tbl[i, j].set_facecolor(bg)
        tbl[i, j].set_edgecolor('#cccccc')

# Column widths: narrow for Block & Ch, wide for Details
tbl.auto_set_column_width([0, 1, 2])
# Manual width tuning via scale
tbl.scale(1.0, 1.32)

ax_right.set_title('(b) ComplexUNet Architecture', fontsize=8.0, pad=2)

plt.tight_layout(w_pad=0.5)
fig.savefig(os.path.join(FIGURES_DIR, "fig0_system.pdf"))
plt.close(fig)
print("Saved fig0_system.pdf")

print("\nAll figures generated successfully.")
