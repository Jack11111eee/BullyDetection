"""check_distribution.py — 查看 pkl 中 train/val 的类别分布"""
import pickle
import numpy as np
from collections import Counter

PKL = '/home/hzcu/BullyDetection/data/campus/campus_balanced_v7.pkl'
CLASSES = ['normal', 'fighting', 'bullying', 'falling', 'climbing']

with open(PKL, 'rb') as f:
    data = pickle.load(f)

train_ids = set(data['split']['train'])
val_ids = set(data['split']['val'])

train_labels = [a['label'] for a in data['annotations'] if a['frame_dir'] in train_ids]
val_labels = [a['label'] for a in data['annotations'] if a['frame_dir'] in val_ids]

print(f"Train: {len(train_labels)},  Val: {len(val_labels)},  Total annotations: {len(data['annotations'])}")
print()

print(f"{'Class':<12} {'Train':>7} {'Train%':>7} {'Val':>7} {'Val%':>7} {'Ratio':>7}")
print("-" * 55)
for i, name in enumerate(CLASSES):
    t = sum(1 for l in train_labels if l == i)
    v = sum(1 for l in val_labels if l == i)
    tp = t / len(train_labels) * 100 if train_labels else 0
    vp = v / len(val_labels) * 100 if val_labels else 0
    ratio = t / v if v > 0 else float('inf')
    print(f"{name:<12} {t:>7} {tp:>6.1f}% {v:>7} {vp:>6.1f}% {ratio:>6.1f}x")

print(f"{'TOTAL':<12} {len(train_labels):>7} {'100%':>7} {len(val_labels):>7} {'100%':>7}")

# 检查是否有 dup 样本
n_dup_train = sum(1 for l in train_ids if '_dup' in l)
print(f"\nTrain 中过采样(dup)样本数: {n_dup_train}")
print(f"Train 中原始样本数: {len(train_labels) - n_dup_train}")
