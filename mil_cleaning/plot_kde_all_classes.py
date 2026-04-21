"""
plot_kde_all_classes.py — 5 类 P(true_label) KDE 曲线叠放总览
"""
import os, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

def gaussian_kde_manual(data, xs, bw=0.08):
    """手动 KDE，避免 scipy 依赖"""
    data = np.asarray(data)
    result = np.zeros_like(xs, dtype=float)
    h = bw * data.std()
    for xi in data:
        result += np.exp(-0.5 * ((xs - xi) / h) ** 2)
    return result / (len(data) * h * np.sqrt(2 * np.pi))

plt.rcParams['font.family'] = ['Heiti TC', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.spines.top']   = False
plt.rcParams['axes.spines.right'] = False

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
SCORES_PATH = os.path.join(SCRIPT_DIR, 'scores.pkl')
OUTPUT_PATH = os.path.join(SCRIPT_DIR, 'kde_all_classes.png')

CLASSES = ['normal', 'fighting', 'bullying', 'falling', 'climbing']
COLORS  = ['#4C72B0', '#DD8452', '#C44E52', '#55A868', '#8172B2']
# 噪声率 P<0.3（来自 training_docs/03_mil_cleaning.md）
NOISE_RATE = [14.7, 20.0, 53.8, 4.1, 23.4]

with open(SCORES_PATH, 'rb') as f:
    scores = pickle.load(f)

by_class = defaultdict(list)
for r in scores:
    by_class[r['label']].append(r)

xs = np.linspace(0, 1, 500)

fig, ax = plt.subplots(figsize=(12, 5.5))

# 阈值区域
ax.axvspan(0, 0.3, alpha=0.07, color='red', zorder=0)
ax.axvline(0.3, color='#CC0000', lw=2, ls='--', alpha=0.8, zorder=2,
           label='threshold = 0.3')

for i, (cls_name, color, nr) in enumerate(zip(CLASSES, COLORS, NOISE_RATE)):
    p_true = np.array([r['probs'][i] for r in by_class[i]])
    ys  = gaussian_kde_manual(p_true, xs, bw=0.08)
    ys_norm = ys / ys.max()   # 归一化到 [0,1] 便于叠放对比

    ax.plot(xs, ys_norm, color=color, lw=2.2, label=f'{cls_name}  (噪声 {nr}%)',
            zorder=3)
    ax.fill_between(xs, 0, ys_norm, color=color, alpha=0.08)

    # 右端标签
    right_y = ys_norm[-1]
    ax.text(1.005, right_y, cls_name, color=color, fontsize=9,
            va='center', fontweight='bold')

# 谷底标注
ax.annotate('谷底区域\n(valley)',
            xy=(0.22, 0.08), xytext=(0.38, 0.22),
            fontsize=9, color='#CC0000', fontstyle='italic',
            arrowprops=dict(arrowstyle='->', color='#CC0000', lw=1.2))

ax.text(0.02, 1.05, '← 噪声峰', fontsize=9, color='#888', fontstyle='italic',
        transform=ax.transAxes)
ax.text(0.82, 1.05, '干净峰 →', fontsize=9, color='#888', fontstyle='italic',
        transform=ax.transAxes)

ax.set_xlabel('P(true_label)', fontsize=11)
ax.set_ylabel('归一化密度', fontsize=11)
ax.set_xlim(-0.02, 1.0)
ax.set_ylim(-0.02, 1.18)
ax.set_title('Cross-Fold P(true_label) KDE：5 类噪声分布总览',
             fontsize=13, fontweight='bold', pad=14)
ax.legend(loc='upper center', fontsize=9.5, ncol=3,
          framealpha=0.9, bbox_to_anchor=(0.5, -0.12))

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
print(f'Saved: {OUTPUT_PATH}')
