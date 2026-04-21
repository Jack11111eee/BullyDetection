"""
plot_radar_r8_r9_r11.py — R8/R9/R11 各类别 Test top-1 雷达图
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams['font.family'] = ['Heiti TC', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

classes = ['normal', 'fighting', 'bullying', 'falling', 'climbing']
N = len(classes)

r8  = [81.8, 79.0, 51.0, 96.3, 74.2]
r9  = [82.2, 80.8, 45.1, 96.1, 77.4]
r11 = [93.3, 89.7, 78.4, 99.1, 98.4]

# 角度（首尾闭合）
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

def close(vals):
    return vals + vals[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# ── 网格圆 ──────────────────────────────────────────────────────────────────
levels = [50, 60, 70, 80, 90, 100]
for lv in levels:
    ax.plot(angles, [lv] * (N + 1), color='#ccc', lw=0.7, ls='-', zorder=0)
    ax.text(angles[0], lv + 1.5, f'{lv}%', fontsize=7.5, ha='center',
            color='#999', va='bottom')

# ── 90% 目标圆加粗 ───────────────────────────────────────────────────────────
ax.plot(angles, [90] * (N + 1), color='#2ca02c', lw=1.5, ls=':', zorder=1)

# ── 三轮数据 ─────────────────────────────────────────────────────────────────
styles = [
    ('R8 (5-fold + 移除class_weight)', r8,  '#9ECAE1', '-.', 0.55),
    ('R9 (clip_len=64)',               r9,  '#4292C6', '--', 0.65),
    ('R11 (MIL 交叉折清洗)',            r11, '#084594', '-',  0.85),
]
for label, vals, color, ls, alpha in styles:
    v = close(vals)
    ax.plot(angles, v, color=color, lw=2.2, ls=ls, alpha=alpha, zorder=3)
    ax.fill(angles, v, color=color, alpha=0.07)

# ── 轴标签 ────────────────────────────────────────────────────────────────────
ax.set_xticks(angles[:-1])
ax.set_xticklabels(classes, fontsize=12, fontweight='bold')
ax.set_yticks([])
ax.set_ylim(40, 105)
ax.spines['polar'].set_visible(False)

# R11 数值标注
for angle, v, cls in zip(angles[:-1], r11, classes):
    offset = 4
    ax.text(angle, v + offset, f'{v:.1f}%', ha='center', va='center',
            fontsize=9, color='#084594', fontweight='bold')

# ── 图例 ─────────────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(facecolor='#9ECAE1', alpha=0.7, label='R8  (5-fold + 移除 class_weight)  '),
    mpatches.Patch(facecolor='#4292C6', alpha=0.7, label='R9  (clip_len=64)'),
    mpatches.Patch(facecolor='#084594', alpha=0.9, label='R11 (MIL 交叉折清洗)'),
]
ax.legend(handles=legend_handles, loc='lower center',
          bbox_to_anchor=(0.5, -0.13), fontsize=10, ncol=1,
          framealpha=0.9, edgecolor='#ccc')

ax.set_title('R8 → R9 → R11\n各类别 Test Top-1 雷达图',
             fontsize=14, fontweight='bold', pad=28)

plt.tight_layout()
out = 'training_docs/radar_r8_r9_r11.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved: {out}')
