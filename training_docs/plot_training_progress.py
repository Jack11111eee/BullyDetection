"""
plot_training_progress.py — R1~R11 训练准确率折线图，标注关键节点
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams['font.family'] = ['Heiti TC', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ── 数据 ──────────────────────────────────────────────────────────────────────
rounds = list(range(1, 12))

val  = [41.5, 31.8, 65.1, 33.1, 56.2, 53.8, 17.8, 81.3, 82.4, 80.7, 89.9]
test = [None, None, None, None, None, None, None, 83.5, 84.2, 82.5, 93.3]

# ── 关键节点 (round, label, y_offset_for_text) ───────────────────────────────
milestones = [
    (3,  "引入平衡采样\n(6-class balanced)",      +6),
    (5,  "修复 keypoint_score 格式",              +6),
    (7,  "过滤全零关键点\n(v7 dataset)",          -10),
    (8,  "移除 class_weight\n+ 引入 5-fold CV",  +6),
    (11, "MIL 交叉折清洗\n(threshold=0.3)",       +6),
]

fig, ax = plt.subplots(figsize=(14, 6.5))

# ── 折线 ─────────────────────────────────────────────────────────────────────
ax.plot(rounds, val, 'o-', color='#4C72B0', linewidth=2.2, markersize=7,
        label='Val top-1', zorder=3)

test_rounds = [r for r, t in zip(rounds, test) if t is not None]
test_vals   = [t for t in test if t is not None]
ax.plot(test_rounds, test_vals, 's--', color='#DD8452', linewidth=2.2,
        markersize=7, label='Test top-1', zorder=3)

# ── 起点 / 终点 强调 ──────────────────────────────────────────────────────────
ax.annotate('41.5%', xy=(1, 41.5), xytext=(1.35, 36),
            fontsize=10, fontweight='bold', color='#4C72B0',
            arrowprops=dict(arrowstyle='->', color='#4C72B0', lw=1.4))
ax.annotate('Test 93.3%', xy=(11, 93.3), xytext=(9.8, 96.5),
            fontsize=11, fontweight='bold', color='#CC0000',
            arrowprops=dict(arrowstyle='->', color='#CC0000', lw=1.6))
ax.annotate('Val 89.9%', xy=(11, 89.9), xytext=(9.5, 85),
            fontsize=10, color='#4C72B0',
            arrowprops=dict(arrowstyle='->', color='#4C72B0', lw=1.2))

# ── 90% 目标线 ────────────────────────────────────────────────────────────────
ax.axhline(90, color='green', linewidth=1.2, linestyle=':', alpha=0.7)
ax.text(1.1, 91.2, '目标线 90%', fontsize=9, color='green', alpha=0.85)

# ── 关键节点竖线 + 标注 ───────────────────────────────────────────────────────
COLORS_MS = {3: '#9467BD', 5: '#8C564B', 7: '#E377C2', 8: '#17BECF', 11: '#CC0000'}

for r, label, yoff in milestones:
    ax.axvline(r, color=COLORS_MS[r], linewidth=1.3, linestyle='--', alpha=0.55, zorder=1)
    # val 点的 y 值
    yval = val[r - 1]
    ax.annotate(
        label,
        xy=(r, yval),
        xytext=(r + 0.08, yval + yoff),
        fontsize=8.5,
        color=COLORS_MS[r],
        fontweight='bold',
        va='center',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                  edgecolor=COLORS_MS[r], alpha=0.85, linewidth=0.8),
        arrowprops=dict(arrowstyle='->', color=COLORS_MS[r], lw=1.0),
    )

# ── 最大跃升区间着色 R7→R8 ───────────────────────────────────────────────────
ax.axvspan(7, 8, alpha=0.07, color='#17BECF')
ax.text(7.5, 53, '+63.5pp\n最大跃升', fontsize=8.5, ha='center',
        color='#17BECF', fontweight='bold', va='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='#17BECF', alpha=0.8))

# ── 数值标签（测试集 R8-R11）────────────────────────────────────────────────
for r, t in zip(test_rounds, test_vals):
    ax.text(r, t + 1.5, f'{t}%', fontsize=8.5, ha='center',
            color='#DD8452', fontweight='bold')

# ── 轴设置 ───────────────────────────────────────────────────────────────────
ax.set_xticks(rounds)
ax.set_xticklabels([f'R{r}' for r in rounds], fontsize=10)
ax.set_yticks(range(10, 101, 10))
ax.set_ylim(5, 102)
ax.set_xlim(0.5, 11.8)
ax.set_xlabel('Training Round', fontsize=11)
ax.set_ylabel('Top-1 Accuracy (%)', fontsize=11)
ax.set_title('PoseC3D Campus Safety — R1~R11 训练进度与关键节点',
             fontsize=13, fontweight='bold', pad=14)
ax.legend(fontsize=10, loc='upper left')
ax.grid(axis='y', alpha=0.3, linewidth=0.8)
ax.grid(axis='x', alpha=0.15, linewidth=0.6)

plt.tight_layout()
out = 'training_docs/training_progress.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved: {out}')
