"""
resolve_video_paths.py — 根据 noisy_samples.csv 在数据集目录中查找源视频的真实路径

1. 递归扫描数据集目录，建立 文件名→完整路径 的索引
2. 根据 CSV 中的 video_name 反查匹配
3. 输出带真实路径的 CSV + 按视频去重的审核清单

用法:
  cd /home/hzcu/BullyDetection && python mil_cleaning/resolve_video_paths.py --dataset-dir /path/to/mounted/dataset
  cd /home/hzcu/BullyDetection && python mil_cleaning/resolve_video_paths.py --dataset-dir /home/hzcu/zjc/dataset
"""

import os
import csv
import argparse
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_EXTS = {'.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv', '.webm'}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', required=True,
                        help='数据集根目录（包含 RWF-2000, RLVS 等子目录）')
    parser.add_argument('--csv', default=os.path.join(SCRIPT_DIR, 'noisy_samples.csv'),
                        help='noisy_samples.csv 路径')
    parser.add_argument('--output', default=os.path.join(SCRIPT_DIR, 'noisy_with_paths.csv'),
                        help='输出 CSV（添加 resolved_path 列）')
    parser.add_argument('--review-list', default=os.path.join(SCRIPT_DIR, 'review_list.txt'),
                        help='按视频去重的人工审核清单')
    return parser.parse_args()


def build_video_index(dataset_dir):
    """递归扫描目录，建立 filename(不含后缀) → [full_path, ...] 的索引"""
    print(f"Scanning {dataset_dir} ...")
    index = defaultdict(list)  # stem → [path1, path2, ...]
    file_count = 0

    for root, dirs, files in os.walk(dataset_dir):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in VIDEO_EXTS:
                stem = os.path.splitext(fname)[0]
                full_path = os.path.join(root, fname)
                index[stem].append(full_path)
                file_count += 1

    print(f"  Indexed {file_count} video files, {len(index)} unique stems")
    return index


def strip_prefix(video_name):
    """
    从 video_name 剥离数据集前缀，返回多个候选搜索名
    例: rwf_train_xxx → ['xxx', 'rwf_train_xxx']
        rlvs_V_123   → ['V_123', 'rlvs_V_123']
        chute01_cam1  → ['chute01_cam1']
    """
    candidates = [video_name]  # 原始名称总是作为候选

    prefixes = [
        ('rlvs_', 5),
        ('rwf_train_', 10),
        ('rwf_val_', 8),
        ('sht_train_', 10),
        ('sht_test_', 9),
        ('ucf_train_', 10),
        ('ucf_test_', 9),
        ('ucf_', 4),
        ('urfall_', 7),
        ('chute_', 6),
        ('fallfloor_', 10),
        ('vandalism2_', 11),
        ('multicam_', 9),
        ('punch_', 6),
        ('climb_', 6),
    ]

    for prefix, length in prefixes:
        if video_name.startswith(prefix):
            stripped = video_name[length:]
            if stripped:
                candidates.insert(0, stripped)  # 优先搜索去前缀版本
            break

    # RWF 特殊处理: rwf_train_xxx_0 → xxx (去掉末尾的 _数字 track id)
    for c in list(candidates):
        parts = c.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) <= 2:
            candidates.append(parts[0])

    return candidates


def resolve_one(video_name, index):
    """尝试在索引中找到视频，返回 (path, match_method) 或 (None, None)"""
    candidates = strip_prefix(video_name)

    for candidate in candidates:
        if candidate in index:
            paths = index[candidate]
            return paths[0], f"exact:{candidate}"

    # 模糊匹配：candidate 作为子串搜索
    for candidate in candidates:
        if len(candidate) < 4:
            continue  # 太短的不做模糊匹配
        for stem, paths in index.items():
            if candidate in stem or stem in candidate:
                return paths[0], f"fuzzy:{candidate}~{stem}"

    return None, None


def main():
    args = parse_args()

    # 1. 建索引
    index = build_video_index(args.dataset_dir)

    # 2. 读 CSV
    print(f"\nReading {args.csv} ...")
    rows = []
    with open(args.csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    print(f"  {len(rows)} noisy samples")

    # 3. 逐行解析
    resolved = 0
    not_found = []

    for row in rows:
        video_name = row['video_name']
        path, method = resolve_one(video_name, index)
        row['resolved_path'] = path or ''
        row['match_method'] = method or ''
        if path:
            resolved += 1
        else:
            not_found.append(video_name)

    print(f"\n  Resolved: {resolved}/{len(rows)} ({resolved/len(rows)*100:.1f}%)")
    if not_found:
        unique_missing = sorted(set(not_found))
        print(f"  Not found: {len(unique_missing)} unique videos")
        for v in unique_missing[:20]:
            print(f"    {v}")
        if len(unique_missing) > 20:
            print(f"    ... and {len(unique_missing) - 20} more")

    # 4. 输出带路径的 CSV
    fieldnames = list(rows[0].keys())
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved: {args.output}")

    # 5. 按视频去重的审核清单
    video_groups = defaultdict(list)
    for row in rows:
        video_groups[row['video_name']].append(row)

    with open(args.review_list, 'w') as f:
        f.write(f"{'=' * 80}\n")
        f.write(f"人工审核清单: {len(video_groups)} 个视频, {len(rows)} 个噪声片段\n")
        f.write(f"{'=' * 80}\n\n")

        # 按噪声片段数量降序
        sorted_videos = sorted(video_groups.items(), key=lambda x: -len(x[1]))

        for video_name, clips in sorted_videos:
            label = clips[0]['label']
            dataset = clips[0]['dataset']
            path = clips[0].get('resolved_path', '')
            total_in_scores = clips[0].get('video_name', '')

            f.write(f"{'─' * 70}\n")
            f.write(f"视频: {video_name}\n")
            f.write(f"标注: {label}  |  数据集: {dataset}  |  噪声片段: {len(clips)}\n")
            if path:
                f.write(f"路径: {path}\n")
            else:
                f.write(f"路径: [未找到]\n")
            f.write(f"\n")
            f.write(f"  {'片段':<30} {'帧范围':<16} {'模型预测':<12} {'P(true)':<10} {'P(pred)':<10}\n")
            f.write(f"  {'-' * 78}\n")

            for clip in sorted(clips, key=lambda c: int(c['clip_start'])):
                frame_range = f"{clip['clip_start']}-{clip['clip_end']}"
                f.write(f"  {clip['frame_dir']:<30} {frame_range:<16} "
                        f"{clip['predicted']:<12} {clip['p_true']:<10} {clip['p_predicted']:<10}\n")
            f.write(f"\n")

    print(f"Review list saved: {args.review_list}")
    print(f"  {len(video_groups)} unique videos to review")


if __name__ == '__main__':
    main()
