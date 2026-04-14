"""confusion_matrix.py — 看清楚谁被误判成了谁"""
import pickle
import numpy as np

CLASSES = ['normal', 'fighting', 'bullying', 'falling', 'climbing', 'vandalism']

with open('/home/hzcu/BullyDetection/pyskl/work_dirs/posec3d_campus_v3/results.pkl', 'rb') as f:
    results = pickle.load(f)
with open('/home/hzcu/BullyDetection/data/campus/campus_balanced.pkl', 'rb') as f:
    dataset = pickle.load(f)

val_ids = set(dataset['split']['val'])
labels = np.array([a['label'] for a in dataset['annotations'] if a['frame_dir'] in val_ids])
preds = np.array([r.argmax() for r in results])

print("混淆矩阵（行=真实, 列=预测）:\n")
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
            row_str += f"  [{n:>5}]"   # 对角线（正确）
        elif pct > 10:
            row_str += f"  *{n:>5}*"   # 严重误判
        else:
            row_str += f"   {n:>5} "
    row_str += f"  | {total}"
    print(row_str)