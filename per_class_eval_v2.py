"""per_class_eval_v2.py — 查看各类准确率"""
import pickle
import numpy as np
from collections import Counter

CLASSES = ['normal', 'fighting', 'bullying', 'falling', 'climbing']

results_file = '/home/hzcu/BullyDetection/pyskl/work_dirs/posec3d_campus_v8/results.pkl'
ann_file = '/home/hzcu/BullyDetection/data/campus/campus_balanced_v7.pkl'

with open(results_file, 'rb') as f:
    results = pickle.load(f)

with open(ann_file, 'rb') as f:
    dataset = pickle.load(f)

val_ids = set(dataset['split']['val'])
labels = [a['label'] for a in dataset['annotations'] if a['frame_dir'] in val_ids]
labels = np.array(labels)
preds = np.array([r.argmax() for r in results])
print(f"labels: {len(labels)}, preds: {len(preds)}")
assert len(labels) == len(preds)

print("=" * 50)
print(f"总样本数: {len(labels)}\n")

print(f"{'类别':<12} {'正确':>6} {'总数':>6} {'准确率':>8}")
print("-" * 40)
for i, name in enumerate(CLASSES):
    mask = labels == i
    total = mask.sum()
    if total == 0:
        print(f"{name:<12} {'—':>6} {0:>6} {'N/A':>8}")
        continue
    correct = (preds[mask] == i).sum()
    acc = correct / total * 100
    print(f"{name:<12} {correct:>6} {total:>6} {acc:>7.1f}%")

overall = (preds == labels).mean() * 100
print(f"\nOverall top1: {overall:.1f}%")
print("=" * 50)

# 混淆矩阵（看谁被误判成谁）
print("\n混淆矩阵（行=真实, 列=预测）：\n")
header = f"{'':>12}" + "".join(f"{c:>10}" for c in CLASSES)
print(header)
print("-" * len(header))

for i, name in enumerate(CLASSES):
    row = labels == i
    counts = []
    for j in range(len(CLASSES)):
        n = (preds[row] == j).sum()
        counts.append(n)
    total = sum(counts)
    row_str = f"{name:>12}"
    for j, n in enumerate(counts):
        pct = n / total * 100 if total > 0 else 0
        if j == i:
            row_str += f"  [{n:>5}]"
        elif pct > 10:
            row_str += f"  *{n:>5}*"
        else:
            row_str += f"   {n:>5} "
    row_str += f"  | {total}"
    print(row_str)