"""
plot_cleaning_impact.py — MIL 清洗前后各类别样本量对比
说明为何只清洗 normal/fighting，bullying 样本太少不敢动
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
plt.rcParams['axes.spines.top']   = False
plt.rcParams['axes.spines.right'] = False

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
SCORES_PATH = os.path.join(SCRIPT_DIR, 'scores.pkl')
OUTPUT_PATH = os.path.join(SCRIPT_DIR, 'cleaning_impact.png')

CLASSES = ['normal', 'fighting', 'bullying', 'falling', 'climbing']
CLEANED = {0, 1}   # 只清洗 normal + fighting

with open(SCORES_PATH, 'rb') as f:
    scores = pickle.load(f)

by_class = defaultdict(list)
for r in scores:
    by_class[r['label']].append(r)

totals   = [len(by_class[i]) for i in range(5)]
removed  = []
kept     = []
for i in range(5):
    p_true = np.array([r['probs'][i] for r in by_class[i]])
    if i in CLEANED:
        n_rm = int((p_true < 0.3).sum())
    else:
        n_rm = 0
    removed.append(n_rm)
    kept.append(totals[i] - n_rm)

x     = np.arange(len(CLASSES))
width = 0.48

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5),
                          gridspec_kw={'width_ratios': [3, 2]})

# ── 左：堆叠柱（保留 + 移除）────────────────────────────────────────────────
ax = axes[0]
BAR_KEEP  = ['#4C72B0', '#DD8452', '#C0C0C0', '#55A868', '#8172B2']
BAR_NOISE = '#E74C3C'

for i, cls in enumerate(CLASSES):
    color_kept = BAR_KEEP[i] if i in CLEANED else '#AAAAAA'
    ax.bar(i, kept[i],    width, color=color_kept, alpha=0.85,
           edgecolor='white', linewidth=0.7)
    ax.bar(i, removed[i], width, bottom=kept[i],
           color=BAR_NOISE, alpha=0.75 if i in CLEANED else 0.0,
           edgecolor='white', linewidth=0.7,
           label='移除噪声 (P<0.3)' if i == 0 else '')

    # 总量标签
    ax.text(i, totals[i] + 150, f'{totals[i]:,}', ha='center', fontsize=9,
            color='#333', fontweight='bold')

    # 移除量标签
    if removed[i] > 0:
        ax.text(i, kept[i] + removed[i] / 2,
                f'-{removed[i]:,}\n({removed[i]/totals[i]*100:.1f}%)',
                ha='center', va='center', fontsize=8.5,
                color='white', fontweight='bold')

    # bullying 特别说明
    if cls == 'bullying':
        ax.text(i, totals[i] + 600,
                '样本极少\n不敢清洗',
                ha='center', fontsize=8, color='#999',
                fontstyle='italic')

# 合计移除标注
total_rm = sum(removed)
ax.text(len(CLASSES) - 1, max(totals) * 0.6,
        f'合计移除\n{total_rm:,} 条\n(-{total_rm/sum(totals)*100:.1f}%)',
        ha='right', fontsize=10, color=BAR_NOISE, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  edgecolor=BAR_NOISE, alpha=0.9))

ax.set_xticks(x)
ax.set_xticklabels(CLASSES, fontsize=11)
ax.set_ylabel('样本数', fontsize=11)
ax.set_ylim(0, max(totals) * 1.22)
ax.set_title('MIL 清洗前后各类别样本量', fontsize=12, fontweight='bold')

# 图例
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(facecolor=BAR_KEEP[0],  label='normal（保留）'),
    Patch(facecolor=BAR_KEEP[1],  label='fighting（保留）'),
    Patch(facecolor='#AAAAAA',    label='其他类（未清洗）'),
    Patch(facecolor=BAR_NOISE,    label='移除噪声 P<0.3'),
], fontsize=9, loc='upper right')
ax.grid(axis='y', alpha=0.25, lw=0.8)

# ── 右：清洗前后总量对比（饼图风格的条形）───────────────────────────────────
ax2 = axes[1]
before_total = sum(totals)
after_total  = sum(kept)

categories = ['清洗前\n全量', '移除噪声\n(normal+fighting)', '清洗后\n有效']
values     = [before_total, total_rm, after_total]
colors_bar = ['#4C72B0', '#E74C3C', '#2ca02c']

bars = ax2.bar(range(3), values, 0.55, color=colors_bar, alpha=0.85,
               edgecolor='white', linewidth=0.8)
for bar, v in zip(bars, values):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 200,
             f'{v:,}', ha='center', va='bottom',
             fontsize=11, fontweight='bold', color=bar.get_facecolor())

ax2.set_xticks(range(3))
ax2.set_xticklabels(categories, fontsize=10)
ax2.set_ylabel('样本总数', fontsize=11)
ax2.set_ylim(0, before_total * 1.18)
ax2.set_title('清洗总体效果', fontsize=12, fontweight='bold')
ax2.grid(axis='y', alpha=0.25, lw=0.8)

fig.suptitle(f'MIL 交叉折清洗：移除 {total_rm:,} 噪声样本（-{total_rm/sum(totals)*100:.1f}%），'
             f'仅针对 normal + fighting',
             fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
print(f'Saved: {OUTPUT_PATH}')
