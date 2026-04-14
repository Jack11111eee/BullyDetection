"""
clean_and_rebuild.py — 根据 MIL 打分过滤噪声样本，重建 kfold 数据集

输出:
  data/campus/campus_mil_kfold_0~4.pkl  (清洗后的 5-fold)
  data/campus/campus_mil_test.pkl       (清洗后的 test set)

用法:
  cd /home/hzcu/BullyDetection && python mil_cleaning/clean_and_rebuild.py
  cd /home/hzcu/BullyDetection && python mil_cleaning/clean_and_rebuild.py --threshold 0.4
  cd /home/hzcu/BullyDetection && python mil_cleaning/clean_and_rebuild.py --threshold 0.5 --classes fighting normal
"""

import os
import argparse
import pickle
import random
import copy
import numpy as np
from collections import Counter, defaultdict

random.seed(42)
np.random.seed(42)

CLASSES = ['normal', 'fighting', 'bullying', 'falling', 'climbing']
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

N_FOLDS = 5
TEST_RATIO = 0.1
CAP = 6000
MAX_OVERSAMPLE = 3


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scores', default=os.path.join(SCRIPT_DIR, 'scores.pkl'))
    parser.add_argument('--data-pkl',
                        default='/home/hzcu/BullyDetection/data/campus/campus.pkl')
    parser.add_argument('--out-dir',
                        default='/home/hzcu/BullyDetection/data/campus')
    parser.add_argument('--threshold', type=float, default=0.4,
                        help='P(true_label) 低于此值的样本被移除')
    parser.add_argument('--classes', nargs='+', default=['fighting', 'normal'],
                        help='只清洗哪些类别 (default: fighting normal)')
    parser.add_argument('--prefix', default='campus_mil',
                        help='输出文件前缀')
    return parser.parse_args()


def get_video_name(frame_dir):
    idx = frame_dir.rfind('_clip_')
    return frame_dir[:idx] if idx >= 0 else frame_dir


def get_base_video(frame_dir):
    video = get_video_name(frame_dir)
    if video.startswith('rwf_'):
        parts = video.rsplit('_', 1)
        if parts[-1].isdigit():
            video = parts[0]
    return video


def balance_train(train_anns):
    """对训练集做 undersample + oversample 平衡（同 build_kfold_data.py）"""
    by_class = defaultdict(list)
    for ann in train_anns:
        by_class[ann['label']].append(ann)

    balanced = []
    for label in sorted(by_class):
        samples = by_class[label]
        n = len(samples)

        if n >= CAP:
            selected = random.sample(samples, CAP)
            balanced.extend(selected)
        elif n * MAX_OVERSAMPLE <= CAP:
            target = n * MAX_OVERSAMPLE
            balanced.extend(samples)
            for i in range(target - n):
                s = copy.deepcopy(random.choice(samples))
                s['frame_dir'] = s['frame_dir'] + f'_dup{i}'
                balanced.append(s)
        else:
            balanced.extend(samples)
            for i in range(CAP - n):
                s = copy.deepcopy(random.choice(samples))
                s['frame_dir'] = s['frame_dir'] + f'_dup{i}'
                balanced.append(s)

    random.shuffle(balanced)
    return balanced


def print_distribution(anns, label_str):
    counts = Counter(a['label'] for a in anns)
    print(f"  {label_str}: {len(anns)} samples")
    for i, name in enumerate(CLASSES):
        print(f"    {i} {name:<12} {counts.get(i, 0):>6}")


def main():
    args = parse_args()

    clean_class_indices = set()
    for name in args.classes:
        if name in CLASSES:
            clean_class_indices.add(CLASSES.index(name))
        else:
            print(f"WARNING: unknown class '{name}', skipping")

    print(f"=== MIL Data Cleaning ===")
    print(f"  Threshold: {args.threshold}")
    print(f"  Classes to clean: {[CLASSES[i] for i in sorted(clean_class_indices)]}")
    print()

    # ===== 1. 加载打分结果 =====
    print(f"Loading scores from {args.scores} ...")
    with open(args.scores, 'rb') as f:
        scores = pickle.load(f)
    print(f"  {len(scores)} scored samples")

    # 建立 frame_dir → score 的映射
    score_map = {r['frame_dir']: r for r in scores}

    # ===== 2. 加载原始数据并清洗 =====
    print(f"\nLoading {args.data_pkl} ...")
    with open(args.data_pkl, 'rb') as f:
        data = pickle.load(f)

    raw_anns = [a for a in data['annotations'] if a['label'] < 5]
    valid_anns = [a for a in raw_anns if not np.all(a['keypoint'] == 0)]

    # frame_dir 去重
    seen = {}
    for ann in valid_anns:
        fd = ann['frame_dir']
        if fd not in seen:
            seen[fd] = ann
    all_anns = list(seen.values())

    print(f"  Valid annotations (after basic cleaning): {len(all_anns)}")
    print("\nBEFORE MIL cleaning:")
    print_distribution(all_anns, "ALL")

    # ===== 3. MIL 过滤 =====
    kept = []
    removed_counts = Counter()

    for ann in all_anns:
        fd = ann['frame_dir']
        label = ann['label']

        if label not in clean_class_indices:
            # 不在清洗范围内的类，直接保留
            kept.append(ann)
            continue

        if fd not in score_map:
            # 没有打分（不应该发生），保留
            kept.append(ann)
            continue

        p_true = score_map[fd]['probs'][label]
        if p_true < args.threshold:
            removed_counts[label] += 1
        else:
            kept.append(ann)

    total_removed = sum(removed_counts.values())
    print(f"\n--- MIL Filtering (threshold={args.threshold}) ---")
    print(f"  Removed: {total_removed} samples")
    for cls_idx in sorted(removed_counts):
        orig = sum(1 for a in all_anns if a['label'] == cls_idx)
        n_rm = removed_counts[cls_idx]
        print(f"    {CLASSES[cls_idx]:<12}: -{n_rm} ({n_rm/orig*100:.1f}% of class)")
    print(f"  Remaining: {len(kept)}")

    print("\nAFTER MIL cleaning:")
    print_distribution(kept, "ALL")

    # ===== 4. 重建 kfold =====
    print(f"\n{'=' * 60}")
    print("Rebuilding K-Fold splits ...")

    # 按 base_video 分组
    base_to_anns = defaultdict(list)
    for ann in kept:
        base = get_base_video(ann['frame_dir'])
        base_to_anns[base].append(ann)

    all_bases = list(base_to_anns.keys())
    random.shuffle(all_bases)
    print(f"Total base videos: {len(all_bases)}")

    # 10% test
    test_split = int(len(all_bases) * TEST_RATIO)
    test_bases = set(all_bases[:test_split])
    cv_bases = all_bases[test_split:]

    test_anns = []
    for base in test_bases:
        test_anns.extend(base_to_anns[base])

    print(f"\nTEST SET: {len(test_bases)} base videos, {len(test_anns)} samples")
    print_distribution(test_anns, "TEST")

    # 保存 test
    test_out = {
        'split': {'test': [a['frame_dir'] for a in test_anns]},
        'annotations': test_anns,
    }
    test_path = os.path.join(args.out_dir, f'{args.prefix}_test.pkl')
    with open(test_path, 'wb') as f:
        pickle.dump(test_out, f)
    print(f"Saved: {test_path}")

    # 5-Fold
    fold_size = len(cv_bases) // N_FOLDS
    folds = []
    for i in range(N_FOLDS):
        start = i * fold_size
        end = len(cv_bases) if i == N_FOLDS - 1 else start + fold_size
        folds.append(set(cv_bases[start:end]))

    for fold_idx in range(N_FOLDS):
        print(f"\n{'─' * 40}")
        print(f"FOLD {fold_idx}")

        val_bases_fold = folds[fold_idx]
        train_bases_fold = set()
        for j in range(N_FOLDS):
            if j != fold_idx:
                train_bases_fold |= folds[j]

        fold_train = []
        fold_val = []
        for base in train_bases_fold:
            fold_train.extend(base_to_anns[base])
        for base in val_bases_fold:
            fold_val.extend(base_to_anns[base])

        # 验证无重叠
        train_check = set(get_base_video(a['frame_dir']) for a in fold_train)
        val_check = set(get_base_video(a['frame_dir']) for a in fold_val)
        test_check = set(get_base_video(a['frame_dir']) for a in test_anns)
        assert len(train_check & val_check) == 0
        assert len(train_check & test_check) == 0
        assert len(val_check & test_check) == 0

        print_distribution(fold_train, f"Fold {fold_idx} TRAIN (raw)")
        print_distribution(fold_val, f"Fold {fold_idx} VAL")

        balanced_train = balance_train(fold_train)
        print_distribution(balanced_train, f"Fold {fold_idx} TRAIN (balanced)")

        fold_data = {
            'split': {
                'train': [a['frame_dir'] for a in balanced_train],
                'val': [a['frame_dir'] for a in fold_val],
            },
            'annotations': balanced_train + fold_val,
        }
        fold_path = os.path.join(args.out_dir, f'{args.prefix}_kfold_{fold_idx}.pkl')
        with open(fold_path, 'wb') as f:
            pickle.dump(fold_data, f)
        print(f"Saved: {fold_path}")

    # ===== 5. 总结 =====
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Original samples:  {len(all_anns)}")
    print(f"  Removed (noise):   {total_removed}")
    print(f"  Remaining:         {len(kept)}")
    print(f"  Threshold:         {args.threshold}")
    print(f"  Classes cleaned:   {[CLASSES[i] for i in sorted(clean_class_indices)]}")
    print(f"\n  Output files:")
    print(f"    {args.out_dir}/{args.prefix}_test.pkl")
    for i in range(N_FOLDS):
        print(f"    {args.out_dir}/{args.prefix}_kfold_{i}.pkl")
    print(f"\n  To train with cleaned data, update config ann_file to:")
    print(f"    {args.out_dir}/{args.prefix}_kfold_0.pkl")


if __name__ == '__main__':
    main()
