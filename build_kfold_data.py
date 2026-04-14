"""
build_kfold_data.py — 5-Fold Cross Validation + 10% Held-out Test
输出:
  data/campus/campus_kfold_0.pkl ~ campus_kfold_4.pkl  (每个含 train/val split)
  data/campus/campus_test.pkl                          (held-out test set)

用法:
  cd /home/hzcu/BullyDetection && python build_kfold_data.py
"""

import pickle
import random
import copy
from collections import Counter, defaultdict
import numpy as np

random.seed(42)
np.random.seed(42)

ann_file = '/home/hzcu/BullyDetection/data/campus/campus.pkl'
out_dir = '/home/hzcu/BullyDetection/data/campus'

CLASSES = ['normal', 'fighting', 'bullying', 'falling', 'climbing']
N_FOLDS = 5
TEST_RATIO = 0.1
CAP = 6000
MAX_OVERSAMPLE = 3


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
    """对训练集做 undersample + oversample 平衡"""
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
    # ===== 1. 读取 + 清洗 =====
    print("Loading data ...")
    with open(ann_file, 'rb') as f:
        data = pickle.load(f)

    raw_anns = [a for a in data['annotations'] if a['label'] < 5]
    print(f"Filter label>=5: {len(raw_anns)}")

    # 过滤全零 keypoint
    valid_anns = [a for a in raw_anns if not np.all(a['keypoint'] == 0)]
    n_zero = len(raw_anns) - len(valid_anns)
    print(f"Filter zero keypoint: removed {n_zero} ({n_zero/len(raw_anns):.1%}), remaining {len(valid_anns)}")

    # 去重 frame_dir
    seen = {}
    for ann in valid_anns:
        fd = ann['frame_dir']
        if fd not in seen:
            seen[fd] = ann
    all_anns = list(seen.values())
    n_dup = len(valid_anns) - len(all_anns)
    print(f"Dedup frame_dir: removed {n_dup}, final valid {len(all_anns)}")

    print("\nClean data distribution:")
    print_distribution(all_anns, "ALL")

    # ===== 2. 按 base_video 分组 =====
    base_to_anns = defaultdict(list)
    for ann in all_anns:
        base = get_base_video(ann['frame_dir'])
        base_to_anns[base].append(ann)

    all_bases = list(base_to_anns.keys())
    random.shuffle(all_bases)
    print(f"\nTotal base videos: {len(all_bases)}")

    # ===== 3. 划出 10% test =====
    test_split = int(len(all_bases) * TEST_RATIO)
    test_bases = set(all_bases[:test_split])
    cv_bases = all_bases[test_split:]

    test_anns = []
    for base in test_bases:
        test_anns.extend(base_to_anns[base])

    print(f"\n{'='*60}")
    print(f"TEST SET: {len(test_bases)} base videos, {len(test_anns)} samples")
    print_distribution(test_anns, "TEST")

    # 保存 test set
    test_out = {
        'split': {'test': [a['frame_dir'] for a in test_anns]},
        'annotations': test_anns,
    }
    test_path = f'{out_dir}/campus_test.pkl'
    with open(test_path, 'wb') as f:
        pickle.dump(test_out, f)
    print(f"Saved: {test_path}")

    # ===== 4. 5-Fold CV on remaining 90% =====
    print(f"\n{'='*60}")
    print(f"CV POOL: {len(cv_bases)} base videos")

    # 分成 5 份
    fold_size = len(cv_bases) // N_FOLDS
    folds = []
    for i in range(N_FOLDS):
        start = i * fold_size
        if i == N_FOLDS - 1:
            end = len(cv_bases)  # 最后一个 fold 包含余数
        else:
            end = start + fold_size
        folds.append(set(cv_bases[start:end]))

    print(f"Fold sizes (base videos): {[len(f) for f in folds]}")

    # ===== 5. 生成每个 fold 的 train/val pkl =====
    for fold_idx in range(N_FOLDS):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx}")

        val_bases_fold = folds[fold_idx]
        train_bases_fold = set()
        for j in range(N_FOLDS):
            if j != fold_idx:
                train_bases_fold |= folds[j]

        # 收集 annotations
        fold_train = []
        fold_val = []
        for base in train_bases_fold:
            fold_train.extend(base_to_anns[base])
        for base in val_bases_fold:
            fold_val.extend(base_to_anns[base])

        # 检查无重叠
        train_check = set(get_base_video(a['frame_dir']) for a in fold_train)
        val_check = set(get_base_video(a['frame_dir']) for a in fold_val)
        test_check = set(get_base_video(a['frame_dir']) for a in test_anns)
        assert len(train_check & val_check) == 0, f"Fold {fold_idx}: train/val overlap!"
        assert len(train_check & test_check) == 0, f"Fold {fold_idx}: train/test overlap!"
        assert len(val_check & test_check) == 0, f"Fold {fold_idx}: val/test overlap!"

        print_distribution(fold_train, f"Fold {fold_idx} TRAIN (raw)")
        print_distribution(fold_val, f"Fold {fold_idx} VAL")

        # 平衡训练集
        balanced_train = balance_train(fold_train)
        print_distribution(balanced_train, f"Fold {fold_idx} TRAIN (balanced)")

        # 保存
        fold_data = {
            'split': {
                'train': [a['frame_dir'] for a in balanced_train],
                'val': [a['frame_dir'] for a in fold_val],
            },
            'annotations': balanced_train + fold_val,
        }
        fold_path = f'{out_dir}/campus_kfold_{fold_idx}.pkl'
        with open(fold_path, 'wb') as f:
            pickle.dump(fold_data, f)
        print(f"Saved: {fold_path}")

    print(f"\n{'='*60}")
    print("DONE! Generated files:")
    print(f"  {out_dir}/campus_test.pkl")
    for i in range(N_FOLDS):
        print(f"  {out_dir}/campus_kfold_{i}.pkl")


if __name__ == '__main__':
    main()
