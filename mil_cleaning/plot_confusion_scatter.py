"""
plot_confusion_scatter.py — fighting 样本 P(fighting) vs P(normal) 散点图
展示 normal↔fighting 混淆的空间分布与 threshold=0.3 的切割效果
"""
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from collections import defaultdict

plt.rcParams['font.family'] = ['Heiti TC', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
SCORES_PATH = os.path.join(SCRIPT_DIR, 'scores.pkl')
OUTPUT_PATH = os.path.join(SCRIPT_DIR, 'confusion_scatter.png')

with open(SCORES_PATH, 'rb') as f:
    scores = pickle.load(f)

by_class = defaultdict(list)
for r in scores:
    by_class[r['label']].append(r)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, (cls_idx, cls_name, opp_idx, opp_name) in zip(
    axes,
    [(1, 'fighting', 0, 'normal'),
     (0, 'normal',   1, 'fighting')]
):
    cls_data = by_class[cls_idx]
    p_self = np.array([r['probs'][cls_idx] for r in cls_data])
    p_opp  = np.array([r['probs'][opp_idx] for r in cls_data])

    # 按 P(true_label) 染色：噪声(<0.3) / 可疑(0.3-0.6) / 干净(>0.6)
    noise_mask    = p_self < 0.3
    suspect_mask  = (p_self >= 0.3) & (p_self < 0.6)
    clean_mask    = p_self >= 0.6

    ax.scatter(p_self[clean_mask],   p_opp[clean_mask],
               s=4, alpha=0.25, c='#2196F3', label=f'干净 P≥0.6  ({clean_mask.sum():,})')
    ax.scatter(p_self[suspect_mask], p_opp[suspect_mask],
               s=5, alpha=0.4,  c='#FF9800', label=f'可疑 0.3≤P<0.6  ({suspect_mask.sum():,})')
    ax.scatter(p_self[noise_mask],   p_opp[noise_mask],
               s=7, alpha=0.55, c='#F44336', label=f'噪声 P<0.3  ({noise_mask.sum():,})')

    # 对角线（等概率）
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.3)

    # threshold=0.3 竖线
    ax.axvline(0.3, color='#CC0000', lw=1.8, ls='--', alpha=0.7)
    ax.text(0.32, 0.92, 'threshold=0.3', fontsize=9, color='#CC0000',
            fontweight='bold', transform=ax.transAxes,
            verticalalignment='top')

    # 噪声区域着色
    ax.axvspan(0, 0.3, alpha=0.06, color='red')

    # 区域标注
    ax.text(0.05, 0.85, f'← 被错判为\n   {opp_name}',
            fontsize=9, color='#F44336', fontweight='bold',
            transform=ax.transAxes)
    ax.text(0.65, 0.08, f'正确判为\n{cls_name} →',
            fontsize=9, color='#2196F3', fontweight='bold',
            transform=ax.transAxes)

    ax.set_xlabel(f'P({cls_name})', fontsize=11)
    ax.set_ylabel(f'P({opp_name})', fontsize=11)
    ax.set_title(f'{cls_name.capitalize()} 样本 (n={len(cls_data):,})\n'
                 f'P({cls_name}) vs P({opp_name})',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8.5, loc='upper right', markerscale=3,
              framealpha=0.9)
    ax.set_aspect('equal')

fig.suptitle('Normal ↔ Fighting 混淆散点图：threshold=0.3 切除噪声聚类',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
print(f'Saved: {OUTPUT_PATH}')
