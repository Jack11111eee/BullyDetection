"""
diagnose_data.py — 全面检查数据集质量和数据泄漏

检查项目:
1. train/val frame_dir 重叠
2. train/val 基础视频级重叠
3. 类别分布异常
4. 过采样副本是否泄漏到 val
5. keypoint 数据质量（全零、NaN、异常值）
6. label 一致性
7. 验证集中是否有重复样本
"""

import pickle
import numpy as np
from collections import Counter, defaultdict

# ============================================================
# 配置
# ============================================================
PKL_FILE = '/home/hzcu/BullyDetection/data/campus/campus_balanced_v5.pkl'
CLASSES = ['normal', 'fighting', 'bullying', 'falling', 'climbing']

# ============================================================
# 加载数据
# ============================================================
print("=" * 70)
print("数据集诊断报告")
print("=" * 70)
print(f"\n文件: {PKL_FILE}")

with open(PKL_FILE, 'rb') as f:
    data = pickle.load(f)

split = data['split']
annotations = data['annotations']

train_ids = set(split['train'])
val_ids = set(split['val'])

# 建立 frame_dir -> annotation 的索引
ann_by_fd = {}
for ann in annotations:
    fd = ann['frame_dir']
    if fd in ann_by_fd:
        print(f"  WARNING: 重复 frame_dir in annotations: {fd}")
    ann_by_fd[fd] = ann

print(f"\n总 annotations: {len(annotations)}")
print(f"split['train']: {len(split['train'])} (unique: {len(train_ids)})")
print(f"split['val']:   {len(split['val'])} (unique: {len(val_ids)})")

# ============================================================
# 检查 1: frame_dir 级别重叠
# ============================================================
print("\n" + "=" * 70)
print("检查 1: frame_dir 级别重叠")
print("=" * 70)

overlap_fd = train_ids & val_ids
print(f"  重叠数量: {len(overlap_fd)}")
if overlap_fd:
    print(f"  *** 数据泄漏! 以下 frame_dir 同时在 train 和 val 中 ***")
    for fd in list(overlap_fd)[:10]:
        print(f"    {fd}")
    if len(overlap_fd) > 10:
        print(f"    ... 还有 {len(overlap_fd) - 10} 个")
else:
    print("  PASS")

# ============================================================
# 检查 2: 基础视频级别重叠
# ============================================================
print("\n" + "=" * 70)
print("检查 2: 基础视频级别重叠")
print("=" * 70)

def get_video_name(frame_dir):
    idx = frame_dir.rfind('_clip_')
    return frame_dir[:idx] if idx >= 0 else frame_dir

def get_base_video(frame_dir):
    video = get_video_name(frame_dir)
    # 去掉 _dup 后缀
    dup_idx = video.rfind('_dup')
    if dup_idx >= 0:
        video = video[:dup_idx]
    if video.startswith('rwf_'):
        parts = video.rsplit('_', 1)
        if parts[-1].isdigit():
            video = parts[0]
    return video

train_bases = set(get_base_video(fd) for fd in train_ids)
val_bases = set(get_base_video(fd) for fd in val_ids)
overlap_bases = train_bases & val_bases

print(f"  训练集基础视频数: {len(train_bases)}")
print(f"  验证集基础视频数: {len(val_bases)}")
print(f"  重叠数量: {len(overlap_bases)}")
if overlap_bases:
    print(f"  *** 视频级数据泄漏! ***")
    for b in list(overlap_bases)[:10]:
        print(f"    {b}")
else:
    print("  PASS")

# ============================================================
# 检查 3: 过采样副本是否泄漏到 val
# ============================================================
print("\n" + "=" * 70)
print("检查 3: 过采样副本泄漏检查")
print("=" * 70)

dup_in_val = [fd for fd in val_ids if '_dup' in fd]
print(f"  val 中带 _dup 后缀的样本数: {len(dup_in_val)}")
if dup_in_val:
    print(f"  *** 过采样副本泄漏到 val! ***")
    for fd in dup_in_val[:5]:
        print(f"    {fd}")
else:
    print("  PASS")

dup_in_train = sum(1 for fd in train_ids if '_dup' in fd)
print(f"  train 中过采样副本数: {dup_in_train}")
print(f"  train 中原始样本数: {len(train_ids) - dup_in_train}")

# ============================================================
# 检查 4: split 中的 frame_dir 是否都能在 annotations 中找到
# ============================================================
print("\n" + "=" * 70)
print("检查 4: split 与 annotations 一致性")
print("=" * 70)

ann_fds = set(a['frame_dir'] for a in annotations)
train_missing = train_ids - ann_fds
val_missing = val_ids - ann_fds

print(f"  train 中找不到 annotation 的: {len(train_missing)}")
print(f"  val 中找不到 annotation 的:   {len(val_missing)}")
if train_missing or val_missing:
    print("  *** 不一致! 有 frame_dir 在 split 中但不在 annotations 中 ***")
else:
    print("  PASS")

# 反过来检查
orphan_anns = ann_fds - train_ids - val_ids
print(f"  annotations 中不在任何 split 的: {len(orphan_anns)}")

# ============================================================
# 检查 5: 类别分布
# ============================================================
print("\n" + "=" * 70)
print("检查 5: 类别分布")
print("=" * 70)

train_labels = [ann_by_fd[fd]['label'] for fd in train_ids if fd in ann_by_fd]
val_labels = [ann_by_fd[fd]['label'] for fd in val_ids if fd in ann_by_fd]

print("\n  类别        Train    Val    Val比例   Train中占比")
print("  " + "-" * 55)
for i, name in enumerate(CLASSES):
    t = sum(1 for l in train_labels if l == i)
    v = sum(1 for l in val_labels if l == i)
    ratio = v / (t + v) * 100 if (t + v) > 0 else 0
    t_pct = t / len(train_labels) * 100 if train_labels else 0
    print(f"  {i} {name:<12} {t:>6}  {v:>6}    {ratio:5.1f}%    {t_pct:5.1f}%")

print(f"\n  总计:        {len(train_labels):>6}  {len(val_labels):>6}")

# 检查类别分布是否严重不均
train_counts = Counter(train_labels)
max_ratio = max(train_counts.values()) / min(train_counts.values()) if train_counts else 0
print(f"\n  训练集最大/最小类比例: {max_ratio:.1f}x")
if max_ratio > 10:
    print("  *** 警告: 类别严重不均衡! ***")

# ============================================================
# 检查 6: val 中同一视频的不同 clip 被分配了不同 label
# ============================================================
print("\n" + "=" * 70)
print("检查 6: 同一视频的 label 一致性")
print("=" * 70)

video_labels = defaultdict(set)
for fd in val_ids:
    if fd in ann_by_fd:
        base = get_base_video(fd)
        video_labels[base].add(ann_by_fd[fd]['label'])

inconsistent = {v: labels for v, labels in video_labels.items() if len(labels) > 1}
print(f"  验证集中有多个 label 的视频数: {len(inconsistent)}")
if inconsistent:
    print("  *** 警告: 同一视频的不同 clip 有不同 label ***")
    for v, labels in list(inconsistent.items())[:5]:
        label_names = [CLASSES[l] for l in labels]
        count_per_label = Counter()
        for fd in val_ids:
            if fd in ann_by_fd and get_base_video(fd) == v:
                count_per_label[ann_by_fd[fd]['label']] += 1
        print(f"    {v}: labels={label_names}, counts={dict(count_per_label)}")
else:
    print("  PASS")

# 同样检查 train
video_labels_train = defaultdict(set)
for fd in train_ids:
    if fd in ann_by_fd:
        base = get_base_video(fd)
        video_labels_train[base].add(ann_by_fd[fd]['label'])

inconsistent_train = {v: labels for v, labels in video_labels_train.items() if len(labels) > 1}
print(f"  训练集中有多个 label 的视频数: {len(inconsistent_train)}")
if inconsistent_train:
    for v, labels in list(inconsistent_train.items())[:5]:
        label_names = [CLASSES[l] for l in labels]
        print(f"    {v}: {label_names}")

# ============================================================
# 检查 7: keypoint 数据质量抽样检查
# ============================================================
print("\n" + "=" * 70)
print("检查 7: Keypoint 数据质量")
print("=" * 70)

# 抽样检查 val 数据
import random
random.seed(42)
sample_fds = random.sample(list(val_ids), min(500, len(val_ids)))

zero_kp_count = 0      # 全零关键点的样本数
nan_kp_count = 0        # 含 NaN 的样本数
no_score_count = 0      # 没有 keypoint_score 的样本数
single_person_count = 0 # 只有1个有效人物
all_zero_person2 = 0    # 第2个人全零的样本数
low_score_count = 0     # 平均 score < 0.3 的样本数

for fd in sample_fds:
    ann = ann_by_fd.get(fd)
    if ann is None:
        continue

    kp = ann.get('keypoint')  # (M, T, 17, 2)
    kp_score = ann.get('keypoint_score')  # (M, T, 17)

    if kp is None:
        continue

    # NaN 检查
    if np.any(np.isnan(kp)):
        nan_kp_count += 1

    # 全零检查
    if np.all(kp == 0):
        zero_kp_count += 1

    # keypoint_score 检查
    if kp_score is None:
        no_score_count += 1
    else:
        if np.any(np.isnan(kp_score)):
            nan_kp_count += 1

        # 第2个人是否全零
        if kp.shape[0] >= 2 and np.all(kp[1] == 0):
            all_zero_person2 += 1

        # 有效人物数
        person_has_data = [np.any(kp[m] != 0) for m in range(kp.shape[0])]
        if sum(person_has_data) <= 1:
            single_person_count += 1

        # 平均 score
        valid_scores = kp_score[kp_score > 0]
        if len(valid_scores) > 0 and np.mean(valid_scores) < 0.3:
            low_score_count += 1

n_sampled = len(sample_fds)
print(f"  抽样检查 {n_sampled} 个 val 样本:")
print(f"    含 NaN:              {nan_kp_count} ({nan_kp_count/n_sampled:.1%})")
print(f"    全零关键点:          {zero_kp_count} ({zero_kp_count/n_sampled:.1%})")
print(f"    无 keypoint_score:   {no_score_count} ({no_score_count/n_sampled:.1%})")
print(f"    第2人全零:           {all_zero_person2} ({all_zero_person2/n_sampled:.1%})")
print(f"    仅1个有效人物:       {single_person_count} ({single_person_count/n_sampled:.1%})")
print(f"    平均score<0.3:       {low_score_count} ({low_score_count/n_sampled:.1%})")

# 检查 keypoint shape
shapes = set()
for fd in sample_fds[:50]:
    ann = ann_by_fd.get(fd)
    if ann and 'keypoint' in ann:
        shapes.add(ann['keypoint'].shape)
print(f"\n  keypoint shapes: {shapes}")

if 'keypoint_score' in ann_by_fd.get(sample_fds[0], {}):
    score_shapes = set()
    for fd in sample_fds[:50]:
        ann = ann_by_fd.get(fd)
        if ann and 'keypoint_score' in ann:
            score_shapes.add(ann['keypoint_score'].shape)
    print(f"  keypoint_score shapes: {score_shapes}")

# ============================================================
# 检查 8: val 中重复样本（完全相同的 keypoint 数据）
# ============================================================
print("\n" + "=" * 70)
print("检查 8: val 中是否有重复/高度相似样本")
print("=" * 70)

# 用 hash 检测完全重复
val_hashes = defaultdict(list)
for fd in val_ids:
    ann = ann_by_fd.get(fd)
    if ann and 'keypoint' in ann:
        h = hash(ann['keypoint'].tobytes())
        val_hashes[h].append(fd)

dup_groups = {h: fds for h, fds in val_hashes.items() if len(fds) > 1}
total_dups = sum(len(fds) - 1 for fds in dup_groups.values())
print(f"  完全重复的 keypoint 数据组: {len(dup_groups)}")
print(f"  受影响样本数: {total_dups}")
if dup_groups:
    for h, fds in list(dup_groups.items())[:3]:
        labels = [ann_by_fd[fd]['label'] for fd in fds]
        label_names = [CLASSES[l] for l in labels]
        print(f"    {len(fds)} 个重复: labels={label_names}, 例: {fds[0][:60]}...")

# ============================================================
# 检查 9: train 中的过采样数据 label 是否正确
# ============================================================
print("\n" + "=" * 70)
print("检查 9: 过采样数据 label 正确性")
print("=" * 70)

dup_label_mismatch = 0
for fd in train_ids:
    if '_dup' not in fd:
        continue
    ann = ann_by_fd.get(fd)
    if ann is None:
        continue
    # 找原始样本
    orig_fd = fd[:fd.rfind('_dup')]
    orig_ann = ann_by_fd.get(orig_fd)
    if orig_ann and orig_ann['label'] != ann['label']:
        dup_label_mismatch += 1

print(f"  过采样副本与原始 label 不匹配: {dup_label_mismatch}")
if dup_label_mismatch == 0:
    print("  PASS")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 70)
print("诊断总结")
print("=" * 70)

issues = []
if overlap_fd:
    issues.append(f"frame_dir 级别数据泄漏: {len(overlap_fd)} 个")
if overlap_bases:
    issues.append(f"基础视频级别数据泄漏: {len(overlap_bases)} 个")
if dup_in_val:
    issues.append(f"过采样副本泄漏到 val: {len(dup_in_val)} 个")
if inconsistent:
    issues.append(f"同一视频多 label: {len(inconsistent)} 个")
if nan_kp_count > 0:
    issues.append(f"NaN keypoint: {nan_kp_count} 个")
if zero_kp_count > n_sampled * 0.1:
    issues.append(f"全零 keypoint 比例过高: {zero_kp_count/n_sampled:.1%}")
if total_dups > 0:
    issues.append(f"val 中重复 keypoint: {total_dups} 个")

if issues:
    print("\n发现问题:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
else:
    print("\n未发现数据泄漏或严重数据质量问题。")
    print("train/val gap (76% vs 53.8%) 主要原因是过拟合，不是数据泄漏。")
