"""
plot_perclass_r9_r11.py — R9 vs R11 各类别 Test top-1 对比（MIL 清洗效果）
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = ['Heiti TC', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

classes  = ['normal', 'fighting', 'bullying', 'falling', 'climbing']
r9_test  = [82.2, 80.8, 45.1, 96.1, 77.4]
r11_test = [93.3, 89.7, 78.4, 99.1, 98.4]
deltas   = [r11 - r9 for r9, r11 in zip(r9_test, r11_test)]

x = np.arange(len(classes))
width = 0.32

fig, ax = plt.subplots(figsize=(11, 6))

bars_r9  = ax.bar(x - width/2, r9_test,  width, label='R9  (无清洗)',
                  color='#9ABBE0', edgecolor='white', linewidth=0.8)
bars_r11 = ax.bar(x + width/2, r11_test, width, label='R11 (MIL 清洗后)',
                  color='#4C72B0', edgecolor='white', linewidth=0.8)

# 数值标签
for bar in bars_r9:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f'{bar.get_height():.1f}%', ha='center', va='bottom',
            fontsize=9, color='#555')

for bar in bars_r11:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f'{bar.get_height():.1f}%', ha='center', va='bottom',
            fontsize=9, color='#1a3f6f', fontweight='bold')

# delta 标注（箭头 + 涨幅）
for i, (delta, cls) in enumerate(zip(deltas, classes)):
    y_bot = r9_test[i]
    y_top = r11_test[i]
    x_pos = x[i] + width / 2
    # 只画增幅箭头
    ax.annotate('', xy=(x_pos, y_top + 1.5), xytext=(x_pos, y_bot + 1.5),
                arrowprops=dict(arrowstyle='->', color='#CC0000', lw=1.4))
    ax.text(x_pos + 0.18, (y_bot + y_top) / 2 + 1,
            f'+{delta:.1f}pp', fontsize=9.5, color='#CC0000', fontweight='bold',
            va='center')

# 90% 目标线
ax.axhline(90, color='green', linewidth=1.2, linestyle=':', alpha=0.7)
ax.text(len(classes) - 0.4, 91, '目标线 90%', fontsize=9, color='green', alpha=0.9)

ax.set_xticks(x)
ax.set_xticklabels(classes, fontsize=11)
ax.set_yticks(range(0, 105, 10))
ax.set_ylim(0, 108)
ax.set_ylabel('Test Top-1 Accuracy (%)', fontsize=11)
ax.set_title('MIL 交叉折清洗效果：R9 vs R11 各类别 Test Top-1',
             fontsize=13, fontweight='bold', pad=14)
ax.legend(fontsize=10, loc='lower right')
ax.grid(axis='y', alpha=0.3, linewidth=0.8)

# bullying 和 climbing 特别说明（提升最大）
for cls_name, idx in [('bullying', 2), ('climbing', 4)]:
    ax.text(x[idx], 4, '最大受益', ha='center', fontsize=8.5,
            color='#CC0000', fontstyle='italic')

plt.tight_layout()
out = 'training_docs/perclass_r9_r11.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved: {out}')
