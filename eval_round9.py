"""
eval_round9.py — 通用模型评估（val + test，per-class + 混淆矩阵）
结果自动保存到 work_dir/eval_results.txt

用法:
  cd /home/hzcu/BullyDetection && python eval_round9.py
  cd /home/hzcu/BullyDetection && python eval_round9.py --config <config.py> --work-dir <dir> --val-pkl <pkl> --test-pkl <pkl>
"""

import os
import sys
import glob
import argparse
import pickle
import numpy as np
import torch
from datetime import datetime

sys.path.insert(0, '/home/hzcu/BullyDetection/pyskl')
from pyskl.models import build_model
from mmcv import Config
from mmcv.runner import load_checkpoint
from pyskl.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter

CLASSES = ['normal', 'fighting', 'bullying', 'falling', 'climbing']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='/home/hzcu/BullyDetection/pyskl/configs/posec3d/finetune_campus_v4.py')
    parser.add_argument('--work-dir', default='/home/hzcu/BullyDetection/pyskl/work_dirs/posec3d_campus_v9')
    parser.add_argument('--val-pkl', default='/home/hzcu/BullyDetection/data/campus/campus_kfold_0.pkl')
    parser.add_argument('--test-pkl', default='/home/hzcu/BullyDetection/data/campus/campus_test.pkl')
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


class Logger:
    """同时输出到终端和文件"""
    def __init__(self, filepath):
        self.filepath = filepath
        self.lines = []

    def log(self, msg=''):
        print(msg)
        self.lines.append(msg)

    def save(self):
        with open(self.filepath, 'w') as f:
            f.write('\n'.join(self.lines) + '\n')
        print(f"\n结果已保存: {self.filepath}")


def get_best_checkpoint(work_dir):
    best = glob.glob(os.path.join(work_dir, 'best_*.pth'))
    if best:
        return best[0]
    pattern = os.path.join(work_dir, 'epoch_*.pth')
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=lambda f: int(os.path.basename(f).split('_')[1].split('.')[0]))


def load_data(pkl_path, split_name):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        ids = set(data['split'][split_name])
        anns = [a for a in data['annotations'] if a['frame_dir'] in ids]
    else:
        anns = data
    labels = np.array([a['label'] for a in anns])
    return anns, labels


def inference_all(model, anns, pipeline, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for i, ann in enumerate(anns):
            sample = dict(
                frame_dir=ann['frame_dir'],
                label=ann['label'],
                keypoint=ann['keypoint'].copy(),
                total_frames=ann['keypoint'].shape[1],
                img_shape=(1080, 1920),
                start_index=0,
            )
            if 'keypoint_score' in ann:
                sample['keypoint_score'] = ann['keypoint_score'].copy()

            sample = pipeline(sample)
            data = collate([sample], samples_per_gpu=1)
            if 'img_metas' in data:
                data['img_metas'] = data['img_metas'].data[0]
            if device != 'cpu':
                data = scatter(data, [device])[0]

            result = model(return_loss=False, **data)
            prob = result[0] if isinstance(result, list) else result
            preds.append(np.argmax(prob))

            if (i + 1) % 500 == 0:
                print(f"    {i+1}/{len(anns)}", flush=True)
    return np.array(preds)


def format_results(preds, labels, title):
    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(title)
    lines.append('='*60)

    overall = (preds == labels).mean() * 100
    lines.append(f"\nOverall top1: {overall:.1f}%\n")
    lines.append(f"{'Class':<12} {'Correct':>8} {'Total':>8} {'Acc':>8}")
    lines.append("-" * 40)

    accs = []
    for i, name in enumerate(CLASSES):
        mask = labels == i
        total = mask.sum()
        correct = (preds[mask] == i).sum()
        acc = correct / total * 100 if total > 0 else 0
        accs.append(acc)
        lines.append(f"{name:<12} {correct:>8} {total:>8} {acc:>7.1f}%")
    mean_class = np.mean(accs)
    lines.append(f"\nMean class acc: {mean_class:.1f}%")

    # 混淆矩阵
    lines.append(f"\n{'':>12}" + "".join(f"{c:>10}" for c in CLASSES))
    lines.append("-" * 65)
    for i, name in enumerate(CLASSES):
        row_mask = labels == i
        total = row_mask.sum()
        row_str = f"{name:>12}"
        for j in range(len(CLASSES)):
            n = (preds[row_mask] == j).sum()
            pct = n / total * 100 if total > 0 else 0
            if j == i:
                row_str += f"  [{n:>5}]"
            elif pct > 10:
                row_str += f"  *{n:>5}*"
            else:
                row_str += f"   {n:>5} "
        row_str += f"  | {total}"
        lines.append(row_str)

    return lines, overall, mean_class


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    pipeline = Compose(cfg.data.val.pipeline)

    ckpt = get_best_checkpoint(args.work_dir)
    if ckpt is None:
        print(f"No checkpoint found in {args.work_dir}")
        return

    log = Logger(os.path.join(args.work_dir, 'eval_results.txt'))
    log.log(f"Evaluation @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.log(f"Config: {args.config}")
    log.log(f"Checkpoint: {ckpt}")

    model = build_model(cfg.model)
    model = model.to(args.device)
    load_checkpoint(model, ckpt, map_location=args.device, strict=False)

    # === Val set ===
    anns, labels = load_data(args.val_pkl, 'val')
    log.log(f"Val samples: {len(anns)}")
    preds = inference_all(model, anns, pipeline, args.device)
    lines, val_acc, val_mean = format_results(preds, labels, "VAL (Fold 0)")
    for l in lines:
        log.log(l)

    # === Test set ===
    test_acc = None
    if os.path.exists(args.test_pkl):
        anns, labels = load_data(args.test_pkl, 'test')
        log.log(f"\nTest samples: {len(anns)}")
        preds = inference_all(model, anns, pipeline, args.device)
        lines, test_acc, test_mean = format_results(preds, labels, "TEST (Held-out)")
        for l in lines:
            log.log(l)

    # === 对比 Round 8 ===
    log.log(f"\n{'='*60}")
    log.log("COMPARISON: Round 8 vs This Round")
    log.log('='*60)
    log.log(f"  {'':>12} {'R8':>10} {'This':>10} {'Delta':>10}")
    log.log(f"  {'Val top1':>12} {'81.3%':>10} {val_acc:>9.1f}% {val_acc-81.3:>+9.1f}%")
    if test_acc is not None:
        log.log(f"  {'Test top1':>12} {'83.5%':>10} {test_acc:>9.1f}% {test_acc-83.5:>+9.1f}%")

    log.save()


if __name__ == '__main__':
    main()
