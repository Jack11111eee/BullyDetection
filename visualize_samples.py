"""
visualize_samples.py — 从 pkl 中抽取样本，可视化骨骼关键点
每个样本选 6 帧（均匀间隔），画出 COCO-17 骨骼连线
输出: skeleton_samples.png
"""

import pickle
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

# ===== 配置 =====
PKL_FILE = '/home/hzcu/BullyDetection/data/campus/campus_balanced_v7.pkl'
CLASSES = ['normal', 'fighting', 'bullying', 'falling', 'climbing']
OUTPUT = 'skeleton_samples.png'
N_SAMPLES = 10
FRAMES_PER_SAMPLE = 6

# COCO-17 骨骼连线
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),       # 头部
    (5, 6),                                 # 肩膀
    (5, 7), (7, 9),                         # 左臂
    (6, 8), (8, 10),                        # 右臂
    (5, 11), (6, 12), (11, 12),             # 躯干
    (11, 13), (13, 15),                     # 左腿
    (12, 14), (14, 16),                     # 右腿
]

# 关键点颜色 (按身体部位)
KP_COLORS = {
    0: 'red',       # nose
    1: 'orange', 2: 'orange',   # eyes
    3: 'yellow', 4: 'yellow',   # ears
    5: 'cyan', 6: 'cyan',       # shoulders
    7: 'blue', 8: 'blue',       # elbows
    9: 'purple', 10: 'purple',  # wrists
    11: 'green', 12: 'green',   # hips
    13: 'lime', 14: 'lime',     # knees
    15: 'pink', 16: 'pink',     # ankles
}

PERSON_COLORS = ['#2196F3', '#F44336']  # 蓝=P1, 红=P2


def draw_skeleton(ax, kp_frame, score_frame=None, color='#2196F3', alpha=1.0):
    """画一帧的骨骼
    kp_frame: (17, 2) — x, y 坐标
    score_frame: (17,) — 置信度（可选）
    """
    # 画骨骼连线
    for (i, j) in SKELETON:
        if kp_frame[i, 0] == 0 and kp_frame[i, 1] == 0:
            continue
        if kp_frame[j, 0] == 0 and kp_frame[j, 1] == 0:
            continue
        ax.plot([kp_frame[i, 0], kp_frame[j, 0]],
                [kp_frame[i, 1], kp_frame[j, 1]],
                color=color, linewidth=1.5, alpha=alpha * 0.8)

    # 画关键点
    for k in range(17):
        x, y = kp_frame[k]
        if x == 0 and y == 0:
            continue
        s = 20
        if score_frame is not None and score_frame[k] < 0.3:
            s = 8  # 低置信度画小点
        ax.scatter(x, y, c=color, s=s, zorder=5, alpha=alpha)


def main():
    print(f"Loading {PKL_FILE} ...")
    with open(PKL_FILE, 'rb') as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        annotations = data.get('annotations', [])
        split = data.get('split', {})
    elif isinstance(data, list):
        annotations = data
        split = {}
    else:
        print(f"Unknown data type: {type(data)}")
        return

    print(f"Total annotations: {len(annotations)}")

    # 按类别分组
    by_class = defaultdict(list)
    for ann in annotations:
        label = ann['label']
        kp = ann.get('keypoint')
        if kp is not None and not np.all(kp == 0):
            by_class[label].append(ann)

    print("Non-zero samples per class:")
    for label in sorted(by_class):
        name = CLASSES[label] if label < len(CLASSES) else f"unknown({label})"
        print(f"  {label} {name}: {len(by_class[label])}")

    # 每类抽 2 个样本 (5类 × 2 = 10)
    random.seed(42)
    selected = []
    for label in sorted(by_class):
        pool = by_class[label]
        n_pick = min(2, len(pool))
        picks = random.sample(pool, n_pick)
        selected.extend(picks)

    # 如果不够 10 个，从剩余中补
    if len(selected) < N_SAMPLES:
        remaining = [a for a in annotations
                     if a.get('keypoint') is not None
                     and not np.all(a['keypoint'] == 0)
                     and a not in selected]
        extra = random.sample(remaining, min(N_SAMPLES - len(selected), len(remaining)))
        selected.extend(extra)

    selected = selected[:N_SAMPLES]
    print(f"\nSelected {len(selected)} samples for visualization")

    # ===== 画图 =====
    fig, axes = plt.subplots(N_SAMPLES, FRAMES_PER_SAMPLE,
                             figsize=(FRAMES_PER_SAMPLE * 3, N_SAMPLES * 3))
    fig.suptitle('Skeleton Samples from campus_balanced_v7.pkl',
                 fontsize=16, fontweight='bold', y=1.01)

    for row, ann in enumerate(selected):
        kp = ann['keypoint']          # (M, T, 17, 2)
        kp_score = ann.get('keypoint_score')  # (M, T, 17)
        label = ann['label']
        fd = ann['frame_dir']
        class_name = CLASSES[label] if label < len(CLASSES) else f"unknown({label})"
        M, T, _, _ = kp.shape

        # 均匀选 FRAMES_PER_SAMPLE 帧
        frame_indices = np.linspace(0, T - 1, FRAMES_PER_SAMPLE, dtype=int)

        for col, t in enumerate(frame_indices):
            ax = axes[row, col]
            ax.set_aspect('equal')

            # 画每个人的骨骼
            for m in range(M):
                if np.all(kp[m, t] == 0):
                    continue
                score_t = kp_score[m, t] if kp_score is not None else None
                color = PERSON_COLORS[m] if m < len(PERSON_COLORS) else '#888888'
                draw_skeleton(ax, kp[m, t], score_t, color=color)

            # 自动调整坐标范围
            all_pts = kp[:, t].reshape(-1, 2)
            valid = all_pts[np.any(all_pts != 0, axis=1)]
            if len(valid) > 0:
                xmin, ymin = valid.min(axis=0)
                xmax, ymax = valid.max(axis=0)
                pad_x = max((xmax - xmin) * 0.2, 30)
                pad_y = max((ymax - ymin) * 0.2, 30)
                ax.set_xlim(xmin - pad_x, xmax + pad_x)
                ax.set_ylim(ymax + pad_y, ymin - pad_y)  # y 轴翻转（图像坐标）
            else:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                        ha='center', va='center', color='gray')

            ax.set_xticks([])
            ax.set_yticks([])

            if col == 0:
                ax.set_ylabel(f"[{label}] {class_name}\n{fd[:35]}...",
                              fontsize=7, rotation=0, labelpad=80,
                              ha='right', va='center')
            if row == 0:
                ax.set_title(f"Frame {t}/{T}", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {OUTPUT}")
    print("Done!")


if __name__ == '__main__':
    main()
