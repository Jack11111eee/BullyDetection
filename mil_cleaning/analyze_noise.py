"""
analyze_noise.py — 分析 scores.pkl 中的噪声分布

输出:
  mil_cleaning/noise_distribution.png  — 每个类别 P(true_label) 直方图
  mil_cleaning/noise_report.txt        — 详细统计报告

用法:
  cd /home/hzcu/BullyDetection && python mil_cleaning/analyze_noise.py
  cd /home/hzcu/BullyDetection && python mil_cleaning/analyze_noise.py --scores mil_cleaning/scores.pkl
"""

import os
import argparse
import pickle
import numpy as np
from collections import defaultdict

CLASSES = ['normal', 'fighting', 'bullying', 'falling', 'climbing']
THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scores', default=os.path.join(SCRIPT_DIR, 'scores.pkl'))
    parser.add_argument('--output-img', default=os.path.join(SCRIPT_DIR, 'noise_distribution.png'))
    parser.add_argument('--output-txt', default=os.path.join(SCRIPT_DIR, 'noise_report.txt'))
    return parser.parse_args()


def analyze(scores):
    """返回按类别分组的分析结果"""
    by_class = defaultdict(list)
    for r in scores:
        by_class[r['label']].append(r)
    return by_class


def gen_report(by_class):
    """生成文本报告"""
    lines = []
    lines.append("=" * 70)
    lines.append("MIL Noise Analysis Report")
    lines.append("=" * 70)

    total_samples = sum(len(v) for v in by_class.values())
    lines.append(f"\nTotal samples: {total_samples}\n")

    # 每个类别的 P(true_label) 统计
    for cls_idx, cls_name in enumerate(CLASSES):
        cls_data = by_class.get(cls_idx, [])
        if not cls_data:
            continue

        p_true = np.array([r['probs'][cls_idx] for r in cls_data])
        preds = np.array([np.argmax(r['probs']) for r in cls_data])

        lines.append(f"\n{'─' * 60}")
        lines.append(f"Class: {cls_name} (label={cls_idx}), N={len(cls_data)}")
        lines.append(f"{'─' * 60}")
        lines.append(f"  P(true_label) stats:")
        lines.append(f"    mean   = {p_true.mean():.4f}")
        lines.append(f"    median = {np.median(p_true):.4f}")
        lines.append(f"    std    = {p_true.std():.4f}")
        lines.append(f"    min    = {p_true.min():.4f}")
        lines.append(f"    max    = {p_true.max():.4f}")

        lines.append(f"\n  Threshold analysis (samples to REMOVE if P(true) < threshold):")
        for t in THRESHOLDS:
            n_below = (p_true < t).sum()
            pct = n_below / len(cls_data) * 100
            lines.append(f"    P(true) < {t:.1f}: {n_below:>6} / {len(cls_data)} ({pct:>5.1f}%)")

        # 被误判为哪些类
        lines.append(f"\n  Prediction breakdown (model thinks these are):")
        for j, other_name in enumerate(CLASSES):
            n = (preds == j).sum()
            pct = n / len(cls_data) * 100
            marker = " ✓" if j == cls_idx else (" ← NOISE" if pct > 10 else "")
            lines.append(f"    → {other_name:<12} {n:>6} ({pct:>5.1f}%){marker}")

    # 重点：normal ↔ fighting 交叉分析
    lines.append(f"\n{'=' * 70}")
    lines.append("FOCUS: normal ↔ fighting Cross-Confusion")
    lines.append("=" * 70)

    for src_idx, src_name, dst_idx, dst_name in [
        (0, 'normal', 1, 'fighting'),
        (1, 'fighting', 0, 'normal'),
    ]:
        cls_data = by_class.get(src_idx, [])
        if not cls_data:
            continue

        # 被模型判为 dst 的样本
        confused = [r for r in cls_data if np.argmax(r['probs']) == dst_idx]
        p_dst = np.array([r['probs'][dst_idx] for r in confused]) if confused else np.array([])

        lines.append(f"\n  {src_name} samples predicted as {dst_name}: "
                     f"{len(confused)}/{len(cls_data)} ({len(confused)/len(cls_data)*100:.1f}%)")
        if len(confused) > 0:
            lines.append(f"    P({dst_name}) in these: mean={p_dst.mean():.3f}, "
                         f"max={p_dst.max():.3f}")
            # 高置信度误判（P(dst) > 0.7）
            high_conf = (p_dst > 0.7).sum()
            lines.append(f"    High confidence (P>{0.7}): {high_conf} samples")

    # 建议阈值
    lines.append(f"\n{'=' * 70}")
    lines.append("RECOMMENDATION")
    lines.append("=" * 70)
    lines.append("\n  Based on the analysis above, consider these cleaning strategies:")
    lines.append("  1. Conservative: --threshold 0.3 (remove only obvious noise)")
    lines.append("  2. Moderate:     --threshold 0.4")
    lines.append("  3. Aggressive:   --threshold 0.5 (remove all uncertain samples)")
    lines.append("\n  Recommended: start with --threshold 0.4 on fighting+normal,")
    lines.append("  then compare training results.")

    return lines


def plot_distributions(by_class, output_path):
    """画每个类别的 P(true_label) 直方图"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not available, skipping plot")
        return

    n_classes = len([c for c in range(5) if by_class.get(c)])
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    plot_idx = 0
    for cls_idx, cls_name in enumerate(CLASSES):
        cls_data = by_class.get(cls_idx, [])
        if not cls_data:
            continue

        ax = axes[plot_idx]
        p_true = np.array([r['probs'][cls_idx] for r in cls_data])

        ax.hist(p_true, bins=50, range=(0, 1), color='steelblue', alpha=0.7,
                edgecolor='white', linewidth=0.5)

        # 标记阈值线
        for t in [0.3, 0.5]:
            n_below = (p_true < t).sum()
            ax.axvline(t, color='red', linestyle='--', alpha=0.7)
            ax.text(t + 0.02, ax.get_ylim()[1] * 0.9,
                    f'<{t}: {n_below}\n({n_below/len(cls_data)*100:.0f}%)',
                    fontsize=8, color='red')

        ax.set_title(f'{cls_name} (n={len(cls_data)})', fontsize=12, fontweight='bold')
        ax.set_xlabel('P(true_label)')
        ax.set_ylabel('Count')
        ax.set_xlim(0, 1)
        plot_idx += 1

    # 第 6 个子图：fighting 样本的 P(normal) vs P(fighting) 散点图
    fighting_data = by_class.get(1, [])
    if fighting_data and plot_idx < len(axes):
        ax = axes[plot_idx]
        p_fight = np.array([r['probs'][1] for r in fighting_data])
        p_normal = np.array([r['probs'][0] for r in fighting_data])

        ax.scatter(p_fight, p_normal, s=2, alpha=0.3, c='steelblue')
        ax.plot([0, 1], [0, 1], 'r--', alpha=0.5)
        ax.set_xlabel('P(fighting)')
        ax.set_ylabel('P(normal)')
        ax.set_title('Fighting samples:\nP(fighting) vs P(normal)', fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plot_idx += 1

    # 隐藏多余的子图
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('MIL Noise Analysis: P(true_label) Distribution per Class',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved: {output_path}")


def main():
    args = parse_args()

    print(f"Loading {args.scores} ...")
    with open(args.scores, 'rb') as f:
        scores = pickle.load(f)
    print(f"Loaded {len(scores)} scored samples.\n")

    by_class = analyze(scores)

    # 文本报告
    report_lines = gen_report(by_class)
    report_text = '\n'.join(report_lines)
    print(report_text)

    with open(args.output_txt, 'w') as f:
        f.write(report_text + '\n')
    print(f"\nReport saved: {args.output_txt}")

    # 直方图
    plot_distributions(by_class, args.output_img)


if __name__ == '__main__':
    main()
