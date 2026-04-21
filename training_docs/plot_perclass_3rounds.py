"""
plot_perclass_3rounds.py — R8 / R9 / R11 各类别 Test top-1 三轮对比
完整展示"5-fold 突破 → MIL 清洗"的逐类进展
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = ['Heiti TC', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.spines.top']   = False
plt.rcParams['axes.spines.right'] = False

classes = ['normal', 'fighting', 'bullying', 'falling', 'climbing']

# Test top-1 per class (from training_docs/02_training_rounds.md)
r8  = [81.8, 79.0, 51.0, 96.3, 74.2]
r9  = [82.2, 80.8, 45.1, 96.1, 77.4]
r11 = [93.3, 89.7, 78.4, 99.1, 98.4]

rounds_data = [('R8\n(5-fold + 移除 class_weight)', r8,  '#9ECAE1'),
               ('R9\n(clip_len=64)',                 r9,  '#4292C6'),
               ('R11\n(MIL 交叉折清洗)',              r11, '#084594')]

x     = np.arange(len(classes))
width = 0.24
fig, ax = plt.subplots(figsize=(13, 6))

offsets = [-1, 0, 1]
for (label, vals, color), offset in zip(rounds_data, offsets):
    bars = ax.bar(x + offset * width, vals, width,
                  label=label, color=color,
                  edgecolor='white', linewidth=0.7, alpha=0.9)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.6,
                f'{v:.1f}', ha='center', va='bottom',
                fontsize=8, color=color,
                fontweight='bold' if color == '#084594' else 'normal')

# ── R9→R11 delta 标注（只标显著提升 >10pp）──────────────────────────────────
for i, (v9, v11) in enumerate(zip(r9, r11)):
    delta = v11 - v9
    if delta >= 10:
        x_pos = x[i] + 1 * width
        ax.annotate(f'+{delta:.1f}pp',
                    xy=(x_pos, v11 + 1.5),
                    fontsize=9, ha='center', va='bottom',
                    color='#CC0000', fontweight='bold')

# ── 90% 目标线 ────────────────────────────────────────────────────────────────
ax.axhline(90, color='#2ca02c', lw=1.3, ls=':', alpha=0.8)
ax.text(len(classes) - 0.15, 91, '目标线 90%', fontsize=9, color='#2ca02c')

ax.set_xticks(x)
ax.set_xticklabels(classes, fontsize=11)
ax.set_yticks(range(0, 105, 10))
ax.set_ylim(0, 108)
ax.set_ylabel('Test Top-1 Accuracy (%)', fontsize=11)
ax.set_title('各类别 Test Top-1：R8 → R9 → R11 三轮进展对比',
             fontsize=13, fontweight='bold', pad=14)
ax.legend(fontsize=10, loc='lower right', framealpha=0.9)
ax.grid(axis='y', alpha=0.25, lw=0.8)

# bullying / climbing 最大受益标注
for cls_name, idx in [('bullying\n最大受益\n+33.3pp', 2),
                       ('climbing\n+21.0pp', 4)]:
    ax.text(x[idx], 2, cls_name, ha='center', fontsize=7.5,
            color='#CC0000', fontstyle='italic', va='bottom')

plt.tight_layout()
out = 'training_docs/perclass_3rounds.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved: {out}')
