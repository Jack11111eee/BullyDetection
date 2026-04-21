"""
plot_accuracy_trend.py — R1~R11 Val/Test 准确率趋势，简洁版（适合报告插图）
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

plt.rcParams['font.family'] = ['Heiti TC', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.spines.top']   = False
plt.rcParams['axes.spines.right'] = False

rounds = list(range(1, 12))
val    = [41.5, 31.8, 65.1, 33.1, 56.2, 53.8, 17.8, 81.3, 82.4, 80.7, 89.9]
test   = [None, None, None, None, None, None, None, 83.5, 84.2, 82.5, 93.3]

test_r = [r for r, t in zip(rounds, test) if t is not None]
test_v = [t for t in test if t is not None]

fig, ax = plt.subplots(figsize=(12, 5.5))

# ── 90% 以下灰色阴影 ─────────────────────────────────────────────────────────
ax.axhspan(0, 90, color='#f5f5f5', zorder=0)
ax.axhline(90, color='#2ca02c', linewidth=1.4, linestyle=':', alpha=0.8, zorder=1)
ax.text(11.05, 90.5, '90%', fontsize=9, color='#2ca02c', va='bottom')

# ── 折线 ─────────────────────────────────────────────────────────────────────
ax.plot(rounds, val,  'o-', color='#4C72B0', lw=2.2, ms=7,
        label='Val top-1',  zorder=3)
ax.plot(test_r, test_v, 's--', color='#DD8452', lw=2.2, ms=7,
        label='Test top-1', zorder=3)

# 所有 val 数值标签
for r, v in zip(rounds, val):
    offset = 2.5 if r not in (2, 4, 7) else -5
    ha = 'center'
    ax.text(r, v + offset, f'{v}%', fontsize=8, ha=ha,
            color='#4C72B0', fontweight='bold' if v > 80 else 'normal')

# test 数值标签
for r, t in zip(test_r, test_v):
    ax.text(r, t - 4.5, f'{t}%', fontsize=8, ha='center',
            color='#DD8452', fontweight='bold' if t > 90 else 'normal')

# ── 四个关键节点垂直带 ────────────────────────────────────────────────────────
milestone_cfg = [
    (7,  8,  '#17BECF', '移除 class_weight\n引入 5-fold CV'),
    (10, 11, '#CC0000', 'MIL 交叉折清洗'),
]
for x0, x1, color, label in milestone_cfg:
    ax.axvspan(x0, x1, alpha=0.10, color=color, zorder=0)
    ax.text((x0 + x1) / 2, 10, label, ha='center', fontsize=8.5,
            color=color, fontweight='bold', va='bottom',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor=color, alpha=0.85, lw=0.8))

# 其他节点仅加虚线
for r, lbl in [(5, '修复 kp 格式'), (7, '过滤\n零关键点')]:
    ax.axvline(r, color='#999', lw=1, ls=':', alpha=0.7)
    ax.text(r, 72, lbl, ha='center', fontsize=7.5, color='#666',
            fontstyle='italic')

# ── 起终点强调 ───────────────────────────────────────────────────────────────
ax.annotate('', xy=(11, 93.3), xytext=(1, 41.5),
            arrowprops=dict(arrowstyle='->', color='#888', lw=1.2,
                            connectionstyle='arc3,rad=-0.25'))
ax.text(6, 62, '+51.8 pp\n(val)', fontsize=9, ha='center',
        color='#888', fontstyle='italic')

# ── 轴 ───────────────────────────────────────────────────────────────────────
ax.set_xticks(rounds)
ax.set_xticklabels([f'R{r}' for r in rounds], fontsize=10)
ax.set_yticks(range(10, 101, 10))
ax.set_ylim(5, 104)
ax.set_xlim(0.6, 11.7)
ax.set_xlabel('Training Round', fontsize=11)
ax.set_ylabel('Top-1 Accuracy (%)', fontsize=11)
ax.set_title('PoseC3D Campus — 多轮训练准确率变化趋势 (R1~R11)',
             fontsize=13, fontweight='bold', pad=14)
ax.legend(fontsize=10, loc='upper left', framealpha=0.9)
ax.grid(axis='y', alpha=0.25, lw=0.8)

plt.tight_layout()
out = 'training_docs/accuracy_trend.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved: {out}')
