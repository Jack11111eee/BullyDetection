"""
eval_kfold.py — 汇总 5-fold CV 结果 + held-out test 最终评估

用法:
  cd /home/hzcu/BullyDetection && python eval_kfold.py
  cd /home/hzcu/BullyDetection && python eval_kfold.py --test  # 用最佳 fold 在 test set 上评估
"""

import os
import sys
import glob
import argparse
import pickle
import numpy as np
import torch
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/hzcu/BullyDetection/pyskl')
from pyskl.models import build_model
from mmcv import Config
from mmcv.runner import load_checkpoint
from pyskl.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter

CLASSES = ['normal', 'fighting', 'bullying', 'falling', 'climbing']
N_FOLDS = 5
DATA_DIR = '/home/hzcu/BullyDetection/data/campus'
WORK_BASE = '/home/hzcu/BullyDetection/pyskl/work_dirs'
BASE_CONFIG = '/home/hzcu/BullyDetection/pyskl/configs/posec3d/finetune_campus_v3.py'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Evaluate best fold on held-out test set')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', default='kfold_results.png')
    return parser.parse_args()


def get_best_checkpoint(work_dir):
    """找 work_dir 中 epoch 最大的 checkpoint"""
    pattern = os.path.join(work_dir, 'epoch_*.pth')
    files = glob.glob(pattern)
    if not files:
        return None
    best = max(files, key=lambda f: int(os.path.basename(f).replace('epoch_', '').replace('.pth', '')))
    return best


def load_val_data(pkl_path, split_name='val'):
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
    probs_list = []
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
            probs_list.append(prob)
            preds.append(np.argmax(prob))

            if (i + 1) % 500 == 0:
                print(f"    {i+1}/{len(anns)}", flush=True)

    return np.array(preds), probs_list


def eval_metrics(preds, labels):
    overall = (preds == labels).mean() * 100
    per_class = {}
    for i, name in enumerate(CLASSES):
        mask = labels == i
        total = mask.sum()
        per_class[name] = (preds[mask] == i).sum() / total * 100 if total > 0 else 0.0
    mean_class = np.mean(list(per_class.values()))
    return overall, mean_class, per_class


def print_confusion_matrix(preds, labels):
    print(f"\n{'':>12}" + "".join(f"{c:>10}" for c in CLASSES))
    print("-" * 65)
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
        print(row_str)


def main():
    args = parse_args()

    cfg = Config.fromfile(BASE_CONFIG)
    pipeline = Compose(cfg.data.val.pipeline)

    # ===== 评估每个 fold 的 val =====
    fold_results = []
    for fold in range(N_FOLDS):
        work_dir = f'{WORK_BASE}/posec3d_campus_fold{fold}'
        ckpt = get_best_checkpoint(work_dir)
        pkl_path = f'{DATA_DIR}/campus_kfold_{fold}.pkl'

        if ckpt is None:
            print(f"\nFold {fold}: No checkpoint found in {work_dir}, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"FOLD {fold}")
        print(f"  Checkpoint: {os.path.basename(ckpt)}")
        print(f"  Data: {os.path.basename(pkl_path)}")

        # 加载模型
        model = build_model(cfg.model)
        model = model.to(args.device)
        load_checkpoint(model, ckpt, map_location=args.device, strict=False)

        # 加载 val 数据
        anns, labels = load_val_data(pkl_path, 'val')
        print(f"  Val samples: {len(anns)}")

        # 推理
        preds, _ = inference_all(model, anns, pipeline, args.device)

        # 评估
        overall, mean_class, per_class = eval_metrics(preds, labels)
        print(f"\n  Overall top1: {overall:.1f}%")
        print(f"  Mean class acc: {mean_class:.1f}%")
        for name, acc in per_class.items():
            print(f"    {name:<12} {acc:.1f}%")
        print_confusion_matrix(preds, labels)

        fold_results.append({
            'fold': fold,
            'ckpt': ckpt,
            'overall': overall,
            'mean_class': mean_class,
            'per_class': per_class,
        })

        del model
        torch.cuda.empty_cache()

    if not fold_results:
        print("No fold results found!")
        return

    # ===== 汇总 =====
    print(f"\n\n{'='*60}")
    print("5-FOLD CV SUMMARY")
    print(f"{'='*60}")

    header = f"{'Fold':>6} {'Overall':>8} {'MeanCls':>8}"
    for name in CLASSES:
        header += f" {name:>10}"
    print(header)
    print("-" * len(header))

    overalls = []
    mean_classes = []
    for r in fold_results:
        line = f"{r['fold']:>6} {r['overall']:>7.1f}% {r['mean_class']:>7.1f}%"
        for name in CLASSES:
            line += f" {r['per_class'][name]:>9.1f}%"
        print(line)
        overalls.append(r['overall'])
        mean_classes.append(r['mean_class'])

    avg_overall = np.mean(overalls)
    std_overall = np.std(overalls)
    avg_mean = np.mean(mean_classes)
    std_mean = np.std(mean_classes)
    print("-" * len(header))
    print(f"{'AVG':>6} {avg_overall:>6.1f}%  {avg_mean:>6.1f}%")
    print(f"{'STD':>6} {std_overall:>6.1f}%  {std_mean:>6.1f}%")

    best_fold = fold_results[np.argmax(overalls)]
    print(f"\nBest fold: {best_fold['fold']} (overall {best_fold['overall']:.1f}%)")

    # ===== 画图 =====
    fig, ax = plt.subplots(figsize=(10, 6))
    folds_x = [r['fold'] for r in fold_results]
    ax.bar(np.array(folds_x) - 0.15, [r['overall'] for r in fold_results],
           width=0.3, label='Overall top1', color='#2196F3')
    ax.bar(np.array(folds_x) + 0.15, [r['mean_class'] for r in fold_results],
           width=0.3, label='Mean class acc', color='#FF9800')
    ax.axhline(y=avg_overall, color='#2196F3', linestyle='--', alpha=0.5, label=f'Avg overall={avg_overall:.1f}%')
    ax.axhline(y=avg_mean, color='#FF9800', linestyle='--', alpha=0.5, label=f'Avg mean={avg_mean:.1f}%')
    ax.set_xlabel('Fold')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('5-Fold Cross Validation Results')
    ax.set_xticks(folds_x)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {args.output}")

    # ===== Test set 评估 =====
    if args.test:
        test_pkl = f'{DATA_DIR}/campus_test.pkl'
        if not os.path.exists(test_pkl):
            print(f"\nTest set not found: {test_pkl}")
            return

        print(f"\n\n{'='*60}")
        print(f"HELD-OUT TEST SET EVALUATION (using best fold {best_fold['fold']})")
        print(f"{'='*60}")

        model = build_model(cfg.model)
        model = model.to(args.device)
        load_checkpoint(model, best_fold['ckpt'], map_location=args.device, strict=False)

        anns, labels = load_val_data(test_pkl, 'test')
        print(f"Test samples: {len(anns)}")

        preds, _ = inference_all(model, anns, pipeline, args.device)
        overall, mean_class, per_class = eval_metrics(preds, labels)

        print(f"\n  Overall top1: {overall:.1f}%")
        print(f"  Mean class acc: {mean_class:.1f}%")
        for name, acc in per_class.items():
            print(f"    {name:<12} {acc:.1f}%")
        print_confusion_matrix(preds, labels)

        print(f"\n  FINAL TEST ACCURACY: {overall:.1f}%")


if __name__ == '__main__':
    main()
