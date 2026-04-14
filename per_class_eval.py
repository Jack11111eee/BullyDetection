"""评估脚本：适配 6 类（去掉 self_harm）"""
import pickle
import numpy as np
from collections import Counter

CLASSES = ['normal', 'fighting', 'bullying', 'falling', 'climbing', 'vandalism']

results_file = '/home/hzcu/BullyDetection/pyskl/work_dirs/posec3d_campus_v3/results.pkl'
ann_file = '/home/hzcu/BullyDetection/data/campus/campus_balanced.pkl'

with open(results_file, 'rb') as f:
    results = pickle.load(f)

with open(ann_file, 'rb') as f:
    dataset = pickle.load(f)

val_ids = set(dataset['split']['val'])
labels = []
for ann in dataset['annotations']:
    if ann['frame_dir'] in val_ids:
        labels.append(ann['label'])

labels = np.array(labels)
preds = np.array([r.argmax() for r in results])

print(f"总样本数: {len(labels)}, 预测数: {len(preds)}")
assert len(labels) == len(preds), "数量不匹配！"

print(f"\n{'类别':<12} {'正确':>6} {'总数':>6} {'准确率':>8}")
print("-" * 36)
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

# 预测分布（检查是否坍塌）
print(f"\n预测分布（是否有类别坍塌）：")
pred_counts = Counter(preds.tolist())
for label in sorted(pred_counts):
    name = CLASSES[label] if label < len(CLASSES) else f'?{label}'
    print(f"  {name:<12} 被预测了 {pred_counts[label]:>6} 次 ({pred_counts[label]/len(preds)*100:.1f}%)")