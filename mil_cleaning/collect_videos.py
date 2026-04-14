"""
collect_videos.py — 根据 noisy_with_paths.csv 复制噪声视频到汇总文件夹

排列规则：
  1. 优先按同 source_video 聚合（同一视频的片段放在一起）
  2. 视频组之间按组内最小 P(true) 排序（最可疑的视频排最前）
  3. 组内按 clip_start 排序

命名: {起始行号}-{结束行号}_{label}_pred{predicted}_{原文件名}
  例: 002-004_fighting_prednormal_xxx.avi  → CSV 第 2~4 条都描述这个视频

用法:
  cd /home/hzcu/BullyDetection && python mil_cleaning/collect_videos.py
  cd /home/hzcu/BullyDetection && python mil_cleaning/collect_videos.py --top-n 200 --out-dir mil_cleaning/review_v2
"""

import os
import csv
import shutil
import random
import argparse
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default=os.path.join(SCRIPT_DIR, 'noisy_with_paths.csv'))
    parser.add_argument('--out-dir', default=os.path.join(SCRIPT_DIR, 'review_videos'))
    parser.add_argument('--top-n', type=int, default=100,
                        help='取前 N 条记录（按重排后的顺序）')
    parser.add_argument('--sample', type=int, default=0,
                        help='随机抽取 N 个视频（0=不随机，取 top-n）')
    parser.add_argument('--out-csv', default=None,
                        help='输出重排后的 CSV（默认 out-dir 同级 _sorted.csv）')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.csv, 'r') as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows)} rows from CSV")

    # 1. 按 resolved_path 聚合（同一视频的片段分到同一组）
    groups = defaultdict(list)
    for row in rows:
        src = row.get('resolved_path', '').strip()
        if not src:
            continue
        groups[src].append(row)

    # 2. 每组内按 clip_start 排序；组间按最小 P(true) 排序
    sorted_groups = []
    for src, clips in groups.items():
        clips.sort(key=lambda r: int(r['clip_start']))
        min_p = min(float(r['p_true']) for r in clips)
        sorted_groups.append((src, clips, min_p))
    sorted_groups.sort(key=lambda x: x[2])

    # 2.5 随机抽样模式：从所有视频组中随机选 N 个
    if args.sample > 0:
        n = min(args.sample, len(sorted_groups))
        sorted_groups = sorted(random.sample(sorted_groups, n), key=lambda x: x[2])
        print(f"Random sample: {n} videos from {len(groups)} total")
        args.top_n = sum(len(clips) for _, clips, _ in sorted_groups)

    # 3. 展平，取前 top-n 条，同时记录每个视频覆盖的行号范围
    flat_rows = []
    video_plan = []  # (src, start_idx, end_idx, clips)

    for src, clips, min_p in sorted_groups:
        start_idx = len(flat_rows)
        for clip in clips:
            if len(flat_rows) >= args.top_n:
                break
            flat_rows.append((src, clip))
        end_idx = len(flat_rows) - 1
        if end_idx >= start_idx:
            video_plan.append((src, start_idx, end_idx, clips[:end_idx - start_idx + 1]))
        if len(flat_rows) >= args.top_n:
            break

    print(f"Selected {len(flat_rows)} rows, {len(video_plan)} unique videos\n")

    # 4. 输出重排后的 CSV
    out_csv = args.out_csv or os.path.join(SCRIPT_DIR, 'noisy_sorted.csv')
    original_fields = list(rows[0].keys())
    csv_fields = ['row_no', 'video_file'] + original_fields
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for src, start_idx, end_idx, clips in video_plan:
            s = start_idx + 1
            e = end_idx + 1
            range_str = f"{s:03d}-{e:03d}" if s != e else f"{s:03d}"
            orig_name = os.path.basename(src) if src else ''
            for clip in clips:
                row_out = {k: clip.get(k, '') for k in original_fields}
                row_out['row_no'] = range_str
                row_out['video_file'] = orig_name
                writer.writerow(row_out)
    print(f"Sorted CSV saved: {out_csv}\n")

    # 5. 复制视频
    success = 0
    skipped_not_exist = 0

    for src, start_idx, end_idx, clips in video_plan:
        if not os.path.isfile(src):
            skipped_not_exist += 1
            print(f"  [MISS] {src}")
            continue

        # 行号用 1-based 显示
        s = start_idx + 1
        e = end_idx + 1
        range_str = f"{s:03d}-{e:03d}" if s != e else f"{s:03d}"

        label = clips[0]['label']
        # 取组内出现最多的预测类作为代表
        pred_counts = defaultdict(int)
        for c in clips:
            pred_counts[c['predicted']] += 1
        main_pred = max(pred_counts, key=pred_counts.get)

        orig_name = os.path.basename(src)
        new_name = f"{range_str}_{label}_pred{main_pred}_{orig_name}"

        dst = os.path.join(args.out_dir, new_name)
        shutil.copy2(src, dst)
        success += 1

        # 打印详情
        frame_ranges = [f"f{c['clip_start']}-{c['clip_end']}" for c in clips]
        print(f"  {new_name}")
        print(f"    噪声片段 {len(clips)} 个: {', '.join(frame_ranges)}")

    print(f"\nDone!")
    print(f"  Copied:  {success} videos")
    print(f"  Missing: {skipped_not_exist} videos")
    print(f"  Output:  {args.out_dir}/")


if __name__ == '__main__':
    main()
