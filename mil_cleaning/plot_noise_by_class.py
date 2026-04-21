"""
plot_noise_by_class.py — 各类别噪声比例对比（来自 scores.pkl 交叉折打分）
展示为何 threshold=0.3 只清洗 normal/fighting，bullying 另行处理
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCORES_PATH = os.path.join(SCRIPT_DIR, 'scores.pkl')
OUTPUT_PATH = os.path.join(SCRIPT_DIR, 'noise_by_class.png')

CLASSES    = ['normal', 'fighting', 'bullying', 'falling', 'climbing']
BAR_COLORS = ['#4C72B0', '#DD8452', '#C44E52', '#55A868', '#8172B2']
THRESHOLDS = [0.3, 0.5]

with open(SCORES_PATH, 'rb') as f:
    scores = pickle.load(f)

by_class = defaultdict(list)
for r in scores:
    by_class[r['label']].append(r)

totals   = [len(by_class[i]) for i in range(5)]
noise_03 = [int((np.array([r['probs'][i] for r in by_class[i]]) < 0.3).sum())
            for i in range(5)]
noise_05 = [int((np.array([r['probs'][i] for r in by_class[i]]) < 0.5).sum())
            for i in range(5)]

pct_03 = [n / t * 100 for n, t in zip(noise_03, totals)]
pct_05 = [n / t * 100 for n, t in zip(noise_05, totals)]
# 0.3~0.5 区间（可疑但未删除）
pct_mid = [b - a for a, b in zip(pct_03, pct_05)]
# 干净 (>=0.5)
pct_clean = [100 - b for b in pct_05]

x = np.arange(len(CLASSES))
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5),
                         gridspec_kw={'width_ratios': [2, 1]})

# ── 左：堆叠柱状图 ─────────────────────────────────────────────────────────
ax = axes[0]
bars_noise  = ax.bar(x, pct_03,  0.5, label='明确噪声 P<0.3',  color='#E74C3C', alpha=0.85)
bars_mid    = ax.bar(x, pct_mid, 0.5, bottom=pct_03,
                     label='可疑 0.3≤P<0.5', color='#F39C12', alpha=0.75)
bars_clean  = ax.bar(x, pct_clean, 0.5, bottom=pct_05,
                     label='干净 P≥0.5',    color='#27AE60', alpha=0.75)

# 数值标签（噪声比例）
for i, (p3, p5, tot) in enumerate(zip(pct_03, pct_05, totals)):
    ax.text(i, p3 / 2, f'{p3:.1f}%', ha='center', va='center',
            fontsize=9, color='white', fontweight='bold')
    ax.text(i, 101, f'n={tot:,}', ha='center', va='bottom',
            fontsize=8, color='#555')

# 清洗区域标注（只清洗 normal+fighting）
for i in [0, 1]:
    ax.annotate('← 已清洗', xy=(i, pct_03[i]),
                xytext=(i + 0.28, pct_03[i] + 4),
                fontsize=8, color='#CC0000',
                arrowprops=dict(arrowstyle='->', color='#CC0000', lw=1.0))

ax.set_xticks(x)
ax.set_xticklabels(CLASSES, fontsize=11)
ax.set_yticks(range(0, 101, 10))
ax.set_ylim(0, 112)
ax.set_ylabel('样本占比 (%)', fontsize=11)
ax.set_title('各类别样本质量分布（交叉折打分）', fontsize=12, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.grid(axis='y', alpha=0.25, lw=0.8)

# threshold=0.3 参考线
ax.axhline(0, color='#E74C3C', lw=0.8, ls=':')

# ── 右：噪声绝对数量横向条形图 ───────────────────────────────────────────────
ax2 = axes[1]
y = np.arange(len(CLASSES))
hbars = ax2.barh(y, noise_03, 0.5, color=BAR_COLORS, alpha=0.82,
                 edgecolor='white', linewidth=0.6)

for bar, n, pct in zip(hbars, noise_03, pct_03):
    ax2.text(bar.get_width() + 30, bar.get_y() + bar.get_height() / 2,
             f'{n:,}  ({pct:.1f}%)', va='center', fontsize=9,
             color='#333', fontweight='bold')

ax2.set_yticks(y)
ax2.set_yticklabels(CLASSES, fontsize=11)
ax2.set_xlabel('噪声样本数 (P < 0.3)', fontsize=10)
ax2.set_title('噪声绝对数量', fontsize=12, fontweight='bold')
ax2.set_xlim(0, max(noise_03) * 1.65)
ax2.grid(axis='x', alpha=0.25, lw=0.8)

fig.suptitle('MIL 交叉折打分：各类别噪声比例与数量（threshold = 0.3）',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
print(f'Saved: {OUTPUT_PATH}')
