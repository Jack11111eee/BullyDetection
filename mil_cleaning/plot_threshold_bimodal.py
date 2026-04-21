"""
plot_threshold_bimodal.py — normal vs fighting P(true_label) 双峰分布直方图
标出 threshold=0.3 位于谷底的依据
"""
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

plt.rcParams['font.family'] = ['Heiti TC', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCORES_PATH = os.path.join(SCRIPT_DIR, 'scores.pkl')
OUTPUT_PATH = os.path.join(SCRIPT_DIR, 'threshold_bimodal.png')

CLASSES = ['normal', 'fighting', 'bullying', 'falling', 'climbing']
COLORS = {'normal': '#4C72B0', 'fighting': '#DD8452'}

with open(SCORES_PATH, 'rb') as f:
    scores = pickle.load(f)

by_class = defaultdict(list)
for r in scores:
    by_class[r['label']].append(r)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

for ax, (cls_idx, cls_name) in zip(axes, [(0, 'normal'), (1, 'fighting')]):
    cls_data = by_class[cls_idx]
    p_true = np.array([r['probs'][cls_idx] for r in cls_data])

    counts, bin_edges, patches = ax.hist(
        p_true, bins=50, range=(0, 1),
        color=COLORS[cls_name], alpha=0.8,
        edgecolor='white', linewidth=0.5,
    )

    ymax = counts.max()

    # 谷底区域着色 [0.2, 0.4]
    for i, patch in enumerate(patches):
        bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2
        if 0.2 <= bin_center <= 0.4:
            patch.set_facecolor('#FF9999')
            patch.set_alpha(0.6)

    # threshold=0.3 竖线
    ax.axvline(0.3, color='#CC0000', linewidth=2.5, linestyle='--', zorder=5)

    # 统计
    n_left = int((p_true < 0.3).sum())
    pct_left = n_left / len(p_true) * 100

    # 左侧填充背景标识噪声区
    ax.axvspan(0, 0.3, alpha=0.06, color='red', zorder=0)

    # 阈值标签
    ax.text(0.33, ymax * 1.05, 'threshold = 0.3',
            fontsize=11, color='#CC0000', fontweight='bold', va='bottom')

    # 噪声数量标注
    ax.text(0.14, ymax * 0.55,
            f'← 移除 {n_left} 条\n   ({pct_left:.1f}%)',
            fontsize=10, color='#CC0000', fontweight='bold',
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#CC0000', alpha=0.85))

    # 标注左峰
    left_peak_mask = bin_edges[:-1] < 0.15
    if counts[left_peak_mask].size > 0:
        lp_idx = np.where(left_peak_mask)[0][counts[left_peak_mask].argmax()]
        lp_x = (bin_edges[lp_idx] + bin_edges[lp_idx + 1]) / 2
        lp_y = counts[lp_idx]
        ax.annotate('噪声峰\n(低置信度)',
                    xy=(lp_x, lp_y), xytext=(lp_x + 0.08, lp_y * 0.95),
                    fontsize=9, color='#555', fontstyle='italic', ha='center',
                    arrowprops=dict(arrowstyle='->', color='#888', lw=1.2))

    # 标注右峰
    right_peak_mask = bin_edges[:-1] > 0.7
    if counts[right_peak_mask].size > 0:
        rp_idx = np.where(right_peak_mask)[0][counts[right_peak_mask].argmax()]
        rp_x = (bin_edges[rp_idx] + bin_edges[rp_idx + 1]) / 2
        rp_y = counts[rp_idx]
        ax.annotate('干净峰\n(高置信度)',
                    xy=(rp_x, rp_y), xytext=(rp_x - 0.15, rp_y * 0.75),
                    fontsize=9, color='#555', fontstyle='italic', ha='center',
                    arrowprops=dict(arrowstyle='->', color='#888', lw=1.2))

    # 谷底标注
    valley_mask = (bin_edges[:-1] >= 0.2) & (bin_edges[:-1] <= 0.4)
    if counts[valley_mask].size > 0:
        valley_min_idx = np.where(valley_mask)[0][counts[valley_mask].argmin()]
        vm_x = (bin_edges[valley_min_idx] + bin_edges[valley_min_idx + 1]) / 2
        vm_y = counts[valley_min_idx]
        ax.annotate('谷底 (valley)',
                    xy=(vm_x, vm_y), xytext=(0.50, ymax * 0.35),
                    fontsize=9, color='#CC0000', fontstyle='italic', ha='center',
                    arrowprops=dict(arrowstyle='->', color='#CC0000', lw=1.2))

    ax.set_title(f'{cls_name.capitalize()} 样本  (n = {len(cls_data):,})',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('P(true_label)', fontsize=11)
    ax.set_ylabel('样本数', fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, ymax * 1.18)

fig.suptitle(
    'Cross-Fold P(true_label) 双峰分布 — threshold=0.3 位于谷底',
    fontsize=14, fontweight='bold', y=1.01,
)
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
print(f'Saved: {OUTPUT_PATH}')
