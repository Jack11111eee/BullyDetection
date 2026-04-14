"""
diagnose_all_data.py — 全面检查整条数据 pipeline 的所有 pkl 文件

检查: train.pkl → val.pkl → campus.pkl → campus_balanced_v5.pkl
追踪问题的来源（全零样本、重复 frame_dir、标签冲突）
"""

import pickle
import numpy as np
from collections import Counter, defaultdict

DATA_DIR = '/home/hzcu/BullyDetection/data/campus'
CLASSES_6 = ['normal', 'fighting', 'bullying', 'falling', 'climbing', 'vandalism']
CLASSES_5 = ['normal', 'fighting', 'bullying', 'falling', 'climbing']

FILES = [
    ('train.pkl', '原始训练集 (step4_build_pkl 输出)'),
    ('val.pkl', '原始验证集 (step4_build_pkl 输出)'),
    ('campus.pkl', '合并数据集 (reformat_pkl 输出)'),
    ('campus_balanced_v5.pkl', '最终平衡数据集 (fix_and_balance 输出)'),
]


def check_pkl(filepath, description, classes):
    """全面检查单个 pkl 文件"""
    print("\n" + "=" * 70)
    print(f"文件: {filepath}")
    print(f"描述: {description}")
    print("=" * 70)

    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print("  文件不存在，跳过")
        return None

    # ===== 基本结构 =====
    if isinstance(data, list):
        print(f"\n  Data is a list (length: {len(data)})")
        if len(data) > 0 and isinstance(data[0], dict):
            print(f"  First element keys: {list(data[0].keys())}")
        annotations = data
        split = {}
    elif isinstance(data, dict):
        print(f"\n  Keys: {list(data.keys())}")
        annotations = data.get('annotations', [])
        split = data.get('split', {})
    else:
        print(f"\n  Data type: {type(data)}")
        annotations = []
        split = {}

    print(f"  Annotations 总数: {len(annotations)}")
    if split:
        train_ids = split.get('train', [])
        val_ids = split.get('val', [])
        print(f"  split['train']: {len(train_ids)} (unique: {len(set(train_ids))})")
        print(f"  split['val']:   {len(val_ids)} (unique: {len(set(val_ids))})")

        # 重复 frame_dir 在 split 中
        train_dups = len(train_ids) - len(set(train_ids))
        val_dups = len(val_ids) - len(set(val_ids))
        if train_dups > 0 or val_dups > 0:
            print(f"  *** split 中重复: train={train_dups}, val={val_dups} ***")

    # ===== 重复 frame_dir 在 annotations 中 =====
    fd_counts = Counter(a['frame_dir'] for a in annotations)
    dup_fds = {fd: cnt for fd, cnt in fd_counts.items() if cnt > 1}
    print(f"\n  重复 frame_dir 的 annotation 数: {len(dup_fds)}")
    if dup_fds:
        # 检查重复的是否有不同 label
        conflicting = 0
        for fd, cnt in dup_fds.items():
            labels = set(a['label'] for a in annotations if a['frame_dir'] == fd)
            if len(labels) > 1:
                conflicting += 1
        print(f"  其中 label 冲突的: {conflicting}")
        if conflicting > 0:
            # 显示几个例子
            shown = 0
            for fd, cnt in dup_fds.items():
                labels = [a['label'] for a in annotations if a['frame_dir'] == fd]
                if len(set(labels)) > 1:
                    label_names = [classes[l] if l < len(classes) else f"unknown({l})" for l in labels]
                    print(f"    {fd}: labels={label_names}")
                    shown += 1
                    if shown >= 5:
                        break

    # ===== 类别分布 =====
    label_counts = Counter(a['label'] for a in annotations)
    print(f"\n  类别分布:")
    for label in sorted(label_counts):
        name = classes[label] if label < len(classes) else f"unknown({label})"
        print(f"    {label} {name:<12} {label_counts[label]:>6}")
    print(f"    总计: {len(annotations)}")

    # ===== Keypoint 质量全量检查 =====
    total = len(annotations)
    zero_kp = 0          # keypoint 全零
    nan_kp = 0           # 含 NaN
    no_score = 0         # 无 keypoint_score
    p2_zero = 0          # 第2人全零
    single_person = 0    # 仅1个有效人物
    low_valid_frames = 0 # 有效帧数 < 50%

    shapes = set()
    score_shapes = set()

    for ann in annotations:
        kp = ann.get('keypoint')
        kp_score = ann.get('keypoint_score')

        if kp is None:
            zero_kp += 1
            continue

        shapes.add(kp.shape)

        if np.any(np.isnan(kp)):
            nan_kp += 1

        if np.all(kp == 0):
            zero_kp += 1
            continue

        if kp_score is None:
            no_score += 1
        else:
            score_shapes.add(kp_score.shape)

        # 第2人检查
        if kp.shape[0] >= 2 and np.all(kp[1] == 0):
            p2_zero += 1

        # 有效人物数
        person_has_data = [np.any(kp[m] != 0) for m in range(kp.shape[0])]
        if sum(person_has_data) <= 1:
            single_person += 1

        # 有效帧数（至少第一个人有数据的帧数）
        if kp.shape[0] >= 1:
            frames_with_data = sum(1 for t in range(kp.shape[1]) if np.any(kp[0, t] != 0))
            if frames_with_data < kp.shape[1] * 0.5:
                low_valid_frames += 1

    print(f"\n  Keypoint 质量（全量 {total} 个样本）:")
    print(f"    全零 keypoint:     {zero_kp:>6} ({zero_kp/total:.1%})")
    print(f"    含 NaN:            {nan_kp:>6} ({nan_kp/total:.1%})")
    print(f"    无 keypoint_score: {no_score:>6} ({no_score/total:.1%})")
    print(f"    第2人全零:         {p2_zero:>6} ({p2_zero/total:.1%})")
    print(f"    仅1个有效人物:     {single_person:>6} ({single_person/total:.1%})")
    print(f"    有效帧<50%:        {low_valid_frames:>6} ({low_valid_frames/total:.1%})")
    print(f"    shapes: {shapes}")
    if score_shapes:
        print(f"    score shapes: {score_shapes}")

    # 全零样本的类别分布
    if zero_kp > 0:
        zero_labels = Counter()
        for ann in annotations:
            kp = ann.get('keypoint')
            if kp is None or np.all(kp == 0):
                zero_labels[ann['label']] += 1
        print(f"\n  全零样本的类别分布:")
        for label in sorted(zero_labels):
            name = classes[label] if label < len(classes) else f"unknown({label})"
            pct = zero_labels[label] / label_counts[label] * 100 if label_counts[label] > 0 else 0
            print(f"    {label} {name:<12} {zero_labels[label]:>6} (该类的 {pct:.1f}%)")

    # ===== 如果有 split，分别统计 train/val 的质量 =====
    if split and 'train' in split and 'val' in split:
        ann_by_fd = {}
        for ann in annotations:
            ann_by_fd[ann['frame_dir']] = ann  # 后出现的覆盖先出现的

        for split_name, split_ids in [('train', set(split['train'])), ('val', set(split['val']))]:
            split_total = len(split_ids)
            split_zero = 0
            split_found = 0
            for fd in split_ids:
                ann = ann_by_fd.get(fd)
                if ann is None:
                    continue
                split_found += 1
                kp = ann.get('keypoint')
                if kp is None or np.all(kp == 0):
                    split_zero += 1

            print(f"\n  {split_name} split 质量:")
            print(f"    总数: {split_total}, 匹配annotation: {split_found}")
            print(f"    全零 keypoint: {split_zero} ({split_zero/split_total:.1%})" if split_total > 0 else "")

    return {
        'total': total,
        'zero_kp': zero_kp,
        'dup_fds': len(dup_fds),
        'nan_kp': nan_kp,
    }


def main():
    print("=" * 70)
    print("全数据 Pipeline 诊断报告")
    print("=" * 70)

    results = {}
    for filename, desc in FILES:
        filepath = f"{DATA_DIR}/{filename}"
        # 前几个文件用6类，最后一个用5类
        cls = CLASSES_5 if 'v5' in filename else CLASSES_6
        results[filename] = check_pkl(filepath, desc, cls)

    # ===== 跨文件对比 =====
    print("\n\n" + "=" * 70)
    print("跨文件对比总结")
    print("=" * 70)

    print("\n  文件                      总数     全零    全零%    重复fd")
    print("  " + "-" * 65)
    for filename, desc in FILES:
        r = results.get(filename)
        if r:
            print(f"  {filename:<28} {r['total']:>6}  {r['zero_kp']:>6}  {r['zero_kp']/r['total']:.1%}   {r['dup_fds']:>6}")

    # ===== 追踪全零样本来源 =====
    print("\n\n" + "=" * 70)
    print("追踪全零样本来源")
    print("=" * 70)

    for filename in ['train.pkl', 'val.pkl']:
        filepath = f"{DATA_DIR}/{filename}"
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
        except FileNotFoundError:
            continue

        if isinstance(data, list):
            annotations = data
        elif isinstance(data, dict):
            annotations = data.get('annotations', [])
        else:
            continue
        # 按数据源前缀分组统计全零比例
        source_stats = defaultdict(lambda: {'total': 0, 'zero': 0})
        for ann in annotations:
            fd = ann['frame_dir']
            # 提取数据源前缀
            if fd.startswith('rwf_'):
                source = 'rwf'
            elif fd.startswith('ucf_'):
                source = 'ucf'
            elif fd.startswith('punch_'):
                source = 'punch'
            elif fd.startswith('sht_'):
                source = 'shanghaitech'
            elif fd.startswith('rlvs_'):
                source = 'rlvs'
            elif fd.startswith('climb_'):
                source = 'climbing'
            elif fd.startswith('bully_'):
                source = 'bullying'
            else:
                prefix = fd.split('_')[0]
                source = prefix

            source_stats[source]['total'] += 1
            kp = ann.get('keypoint')
            if kp is None or np.all(kp == 0):
                source_stats[source]['zero'] += 1

        print(f"\n  {filename} — 按数据源的全零比例:")
        print(f"    {'数据源':<20} {'总数':>6} {'全零':>6} {'全零%':>8}")
        print(f"    {'-'*45}")
        for source in sorted(source_stats, key=lambda s: source_stats[s]['zero'], reverse=True):
            s = source_stats[source]
            pct = s['zero'] / s['total'] * 100 if s['total'] > 0 else 0
            flag = " ***" if pct > 20 else ""
            print(f"    {source:<20} {s['total']:>6} {s['zero']:>6} {pct:>7.1f}%{flag}")

    print("\n\n" + "=" * 70)
    print("修复建议")
    print("=" * 70)
    print("""
  1. 在 fix_and_balance.py 中过滤全零样本:
     all_anns = [a for a in data['annotations']
                 if a['label'] < 5
                 and not np.all(a['keypoint'] == 0)]

  2. 去重 frame_dir（保留 keypoint 非零的那个）:
     seen = {}
     for ann in all_anns:
         fd = ann['frame_dir']
         if fd not in seen or np.all(seen[fd]['keypoint'] == 0):
             seen[fd] = ann
     all_anns = list(seen.values())

  3. 重新运行 fix_and_balance.py → 生成 campus_balanced_v6.pkl
  4. 用新数据集重新训练
""")


if __name__ == '__main__':
    main()
