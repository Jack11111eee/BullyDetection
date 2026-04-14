"""check_keypoint_quality.py"""
import pickle
import numpy as np
from collections import Counter

pkl_file = '/home/hzcu/BullyDetection/data/campus/campus_balanced_v2.pkl'

with open(pkl_file, 'rb') as f:
    data = pickle.load(f)

# 统计所有关键点的置信度
all_scores = []
for ann in data['annotations']:
    keypoint = ann['keypoint']  # shape: (T, K, 2) 或 (T, K, 3)?
    if isinstance(keypoint, np.ndarray):
        all_scores.extend(keypoint.flatten())

all_scores = np.array(all_scores)
print(f"关键点置信度统计：")
print(f"  min: {all_scores.min():.3f}")
print(f"  max: {all_scores.max():.3f}")
print(f"  mean: {all_scores.mean():.3f}")
print(f"  median: {np.median(all_scores):.3f}")
print(f"  std: {all_scores.std():.3f}")
print(f"  < 0.3 的比例: {(all_scores < 0.3).mean():.1%}")
print(f"  < 0.5 的比例: {(all_scores < 0.5).mean():.1%}")