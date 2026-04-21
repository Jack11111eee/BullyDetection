"""plot_confusion_r10_r11.py — Round 10 & Round 11 测试集混淆矩阵热力图"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

CLASSES = ['normal', 'fighting', 'bullying', 'falling', 'climbing']

# Round 10 Test (3606 samples) — limb-only, from eval_results.txt
cm_r10 = np.array([
    [1174, 248,  6, 44,  7],   # normal
    [ 226, 961,  5, 15,  2],   # fighting
    [   7,  22, 20,  2,  0],   # bullying
    [   7,  21,  0, 774, 3],   # falling
    [   3,   8,  0,  4, 47],   # climbing
])

# Round 11 Test (3606 samples) — MIL cleaned, from eval_results.txt
cm_r11 = np.array([
    [1380,  81, 0, 18,  0],   # normal
    [  96, 1085, 5, 23,  0],  # fighting
    [   7,   4, 40,  0,  0],  # bullying
    [   2,   3,  0, 798, 2],  # falling
    [   0,   0,  0,  1, 61],  # climbing
])

titles = ['Round 10 (Limb-only)  Test 82.5%',
          'Round 11 (MIL Cleaned) Test 93.3%']
cms = [cm_r10, cm_r11]

# ── color map: white → deep blue ──────────────────────────────────────────────
cmap = LinearSegmentedColormap.from_list(
    'wb', ['#ffffff', '#1565c0'], N=256
)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Test Set Confusion Matrix: Round 10 vs Round 11',
             fontsize=15, fontweight='bold', y=1.01)

for ax, cm, title in zip(axes, cms, titles):
    row_totals = cm.sum(axis=1, keepdims=True)
    cm_pct = cm / row_totals * 100          # row-normalized (%)

    im = ax.imshow(cm_pct, cmap=cmap, vmin=0, vmax=100, aspect='auto')

    n = len(CLASSES)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(CLASSES, fontsize=11)
    ax.set_yticklabels(CLASSES, fontsize=11)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)

    # cell annotations
    for i in range(n):
        for j in range(n):
            count = cm[i, j]
            pct   = cm_pct[i, j]
            text_color = 'white' if pct > 55 else 'black'
            # diagonal: bold percentage + count
            if i == j:
                ax.text(j, i, f'{pct:.1f}%\n({count})',
                        ha='center', va='center',
                        fontsize=10, fontweight='bold',
                        color=text_color)
            elif count > 0:
                ax.text(j, i, f'{pct:.1f}%\n({count})',
                        ha='center', va='center',
                        fontsize=8.5,
                        color=text_color)

    # colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Row %', fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # per-class accuracy along diagonal
    per_class_acc = np.diag(cm_pct)
    overall = cm.trace() / cm.sum() * 100
    mean_cls = per_class_acc.mean()
    info = (f'Overall: {overall:.1f}%   Mean-cls: {mean_cls:.1f}%\n'
            + '  '.join(f'{c}:{a:.0f}%' for c, a in zip(CLASSES, per_class_acc)))
    ax.set_xlabel(f'Predicted\n\n{info}', fontsize=10)

plt.tight_layout()
out = 'training_docs/confusion_r10_r11.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved → {out}')
plt.show()
