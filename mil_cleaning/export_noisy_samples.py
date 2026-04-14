"""
export_noisy_samples.py — 导出噪声样本清单，追溯到源视频 + 帧范围，供人工审核

输出:
  mil_cleaning/noisy_samples.csv  — 可用 Excel 打开
  终端打印摘要

用法:
  cd /home/hzcu/BullyDetection && python mil_cleaning/export_noisy_samples.py
  cd /home/hzcu/BullyDetection && python mil_cleaning/export_noisy_samples.py --threshold 0.3
  cd /home/hzcu/BullyDetection && python mil_cleaning/export_noisy_samples.py --threshold 0.5 --classes fighting normal
"""

import os
import csv
import argparse
import pickle
import numpy as np
from collections import Counter

CLASSES = ['normal', 'fighting', 'bullying', 'falling', 'climbing']
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CLIP_LEN = 48

# ===== 数据集前缀 → 源视频路径映射 =====
# frame_dir 格式: {video_name}_clip_{start}
# video_name 即 JSON 文件名（不含 .json）

DATASET_ROOTS = {
    'rlvs': {
        'Violence':    '/home/hzcu/zjc/dataset/NTU/RLVS/Real Life Violence Dataset/Violence',
        'NonViolence': '/home/hzcu/zjc/dataset/NTU/RLVS/Real Life Violence Dataset/NonViolence',
        'ext': '.mp4',
    },
    'rwf_train': {
        'Fight':    '/home/hzcu/zjc/dataset/RWF-2000/RWF-2000/train/Fight',
        'NonFight': '/home/hzcu/zjc/dataset/RWF-2000/RWF-2000/train/NonFight',
        'ext': '.avi',
    },
    'rwf_val': {
        'Fight':    '/home/hzcu/zjc/dataset/RWF-2000/RWF-2000/val/Fight',
        'NonFight': '/home/hzcu/zjc/dataset/RWF-2000/RWF-2000/val/NonFight',
        'ext': '.avi',
    },
    'sht_train': {
        'videos': '/home/hzcu/zjc/dataset/SHT/SHT/shanghaitech/training/videos',
        'ext': '.avi',
    },
    'sht_test': {
        'frames': '/home/hzcu/zjc/dataset/SHT/SHT/shanghaitech/testing/frames',
        'ext': '[frames]',
    },
    'chute': {
        'root': '/home/hzcu/zjc/dataset/Fall Multiple Cameras Fall Dataset/dataset_extracted/dataset',
        'ext': '.avi',
    },
    'urfall': {
        'Fall':   '/home/hzcu/zjc/dataset/Fall UR Fall Detection/Fall',
        'Normal': '/home/hzcu/zjc/dataset/Fall UR Fall Detection/Normal',
        'ext': '.mp4',
    },
    'punch': {
        'root': '/home/hzcu/zjc/dataset/punch/punch_extracted/punch',
        'ext': '.avi',
    },
    'climb': {
        'root': '/home/hzcu/zjc/dataset/climb/climb_extracted/climb',
        'ext': '.avi',
    },
}


def parse_frame_dir(frame_dir):
    """从 frame_dir 解析出 video_name 和 clip_start"""
    idx = frame_dir.rfind('_clip_')
    if idx < 0:
        return frame_dir, 0
    video_name = frame_dir[:idx]
    clip_start = int(frame_dir[idx + 6:])
    return video_name, clip_start


def detect_dataset(video_name):
    """根据 video_name 前缀判断数据集来源"""
    if video_name.startswith('rlvs_V_'):
        return 'rlvs', video_name[5:]  # V_123
    if video_name.startswith('rwf_train_'):
        return 'rwf_train', video_name[10:]
    if video_name.startswith('rwf_val_'):
        return 'rwf_val', video_name[8:]
    if video_name.startswith('sht_train_'):
        return 'sht_train', video_name[10:]
    if video_name.startswith('sht_test_'):
        return 'sht_test', video_name[9:]
    if video_name.startswith('ucf_'):
        return 'ucf', video_name[4:]
    if video_name.startswith('chute'):
        return 'chute', video_name
    if video_name.startswith('urfall_'):
        return 'urfall', video_name[7:]
    if video_name.startswith('punch_'):
        return 'punch', video_name[6:]
    if video_name.startswith('climb_'):
        return 'climb', video_name[6:]
    if video_name.startswith('fallfloor_'):
        return 'fallfloor', video_name[10:]
    return 'unknown', video_name


def guess_video_path(video_name):
    """尝试推断源视频路径（best-effort，不保证存在）"""
    dataset, local_name = detect_dataset(video_name)

    if dataset == 'rlvs':
        # rlvs_V_123 → V_123.mp4，在 Violence 或 NonViolence 子目录下
        return f'RLVS:/{local_name}.mp4'
    if dataset in ('rwf_train', 'rwf_val'):
        # rwf_train_xxx → xxx.avi，在 Fight 或 NonFight 子目录下
        return f'RWF:/{local_name}.avi'
    if dataset == 'sht_train':
        return f'SHT:/training/videos/{local_name}.avi'
    if dataset == 'sht_test':
        return f'SHT:/testing/frames/{local_name}/'
    if dataset == 'ucf':
        return f'UCF:/{local_name}'
    if dataset == 'chute':
        return f'CHUTE:/{video_name}.avi'
    if dataset == 'urfall':
        return f'URFALL:/{local_name}.mp4'
    if dataset == 'punch':
        return f'PUNCH:/{local_name}.avi'
    if dataset == 'climb':
        return f'CLIMB:/{local_name}.avi'

    return f'UNKNOWN:/{video_name}'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scores', default=os.path.join(SCRIPT_DIR, 'scores.pkl'))
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='P(true_label) < threshold 的样本视为噪声')
    parser.add_argument('--classes', nargs='+', default=['fighting', 'normal'],
                        help='导出哪些类别的噪声（默认 fighting normal）')
    parser.add_argument('--output', default=os.path.join(SCRIPT_DIR, 'noisy_samples.csv'))
    parser.add_argument('--top-n', type=int, default=0,
                        help='每类只导出最"吵"的 N 个样本（0=全部）')
    return parser.parse_args()


def main():
    args = parse_args()

    clean_classes = set()
    for name in args.classes:
        if name in CLASSES:
            clean_classes.add(CLASSES.index(name))

    print(f"Loading {args.scores} ...")
    with open(args.scores, 'rb') as f:
        scores = pickle.load(f)
    print(f"  {len(scores)} scored samples")

    # 筛选噪声样本
    noisy = []
    for r in scores:
        if r['label'] not in clean_classes:
            continue
        p_true = r['probs'][r['label']]
        if p_true < args.threshold:
            pred = int(np.argmax(r['probs']))
            video_name, clip_start = parse_frame_dir(r['frame_dir'])
            noisy.append({
                'frame_dir': r['frame_dir'],
                'label': CLASSES[r['label']],
                'label_idx': r['label'],
                'predicted': CLASSES[pred],
                'p_true': float(p_true),
                'p_predicted': float(r['probs'][pred]),
                'video_name': video_name,
                'clip_start': clip_start,
                'clip_end': clip_start + CLIP_LEN - 1,
                'dataset': detect_dataset(video_name)[0],
                'source_video': guess_video_path(video_name),
                'json_path': f'data/raw_skeletons/{CLASSES[r["label"]]}/{video_name}.json',
                'fold': r.get('fold', -1),
            })

    # 按 P(true) 升序排（最可疑的在最前面）
    noisy.sort(key=lambda x: x['p_true'])

    if args.top_n > 0:
        # 每类取 top-N
        by_class = {}
        for item in noisy:
            by_class.setdefault(item['label'], []).append(item)
        noisy = []
        for cls_name in sorted(by_class):
            noisy.extend(by_class[cls_name][:args.top_n])

    # 打印摘要
    print(f"\n{'=' * 60}")
    print(f"Noisy samples (P(true) < {args.threshold}): {len(noisy)}")
    print(f"{'=' * 60}")

    ds_counts = Counter(item['dataset'] for item in noisy)
    label_counts = Counter(item['label'] for item in noisy)
    pred_counts = Counter((item['label'], item['predicted']) for item in noisy)

    print(f"\nBy label:")
    for label, count in label_counts.most_common():
        print(f"  {label:<12} {count:>6}")

    print(f"\nBy dataset:")
    for ds, count in ds_counts.most_common():
        print(f"  {ds:<12} {count:>6}")

    print(f"\nBy confusion pair (label → predicted):")
    for (label, pred), count in pred_counts.most_common(10):
        print(f"  {label:<12} → {pred:<12} {count:>6}")

    # 按视频聚合：哪些视频贡献最多噪声
    video_noise_count = Counter(item['video_name'] for item in noisy)
    print(f"\nTop 20 noisiest videos:")
    for video_name, count in video_noise_count.most_common(20):
        # 该视频总共有多少 clip
        total_clips = sum(1 for r in scores
                          if parse_frame_dir(r['frame_dir'])[0] == video_name)
        label = next(item['label'] for item in noisy if item['video_name'] == video_name)
        ds = detect_dataset(video_name)[0]
        print(f"  {video_name:<45} {count:>3}/{total_clips} noisy  [{label}] ({ds})")

    # 导出 CSV
    fieldnames = ['frame_dir', 'label', 'predicted', 'p_true', 'p_predicted',
                  'video_name', 'clip_start', 'clip_end', 'dataset',
                  'source_video', 'json_path', 'fold']
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in noisy:
            row = {k: item[k] for k in fieldnames}
            row['p_true'] = f"{item['p_true']:.4f}"
            row['p_predicted'] = f"{item['p_predicted']:.4f}"
            writer.writerow(row)

    print(f"\nCSV saved: {args.output}")
    print(f"  {len(noisy)} rows, sorted by P(true) ascending (most suspicious first)")
    print(f"\n  用法: 用 Excel/WPS 打开 CSV，从上到下审核")
    print(f"  找到对应视频后，跳到 clip_start 帧检查该片段是否真的是标注的类别")


if __name__ == '__main__':
    main()
