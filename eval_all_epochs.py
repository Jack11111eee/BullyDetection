"""
eval_all_epochs.py — 遍历所有 epoch checkpoint，评估 val 集的 per-class accuracy
输出：表格 + epoch_accuracy_curve.png

用法:
  cd /home/hzcu/BullyDetection && python eval_all_epochs.py [--work-dir WORK_DIR] [--config CONFIG] [--ann-file ANN]

默认评估 v7 的所有 checkpoint
"""

import os
import sys
import glob
import argparse
import pickle
import numpy as np
import torch
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ===== PYSKL imports =====
sys.path.insert(0, '/home/hzcu/BullyDetection/pyskl')
from pyskl.models import build_model
from mmcv import Config
from mmcv.runner import load_checkpoint
from pyskl.datasets import build_dataset
from mmcv.parallel import collate, scatter
from pyskl.datasets.pipelines import Compose

CLASSES = ['normal', 'fighting', 'bullying', 'falling', 'climbing']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work-dir', default='/home/hzcu/BullyDetection/pyskl/work_dirs/posec3d_campus_v7')
    parser.add_argument('--config', default='/home/hzcu/BullyDetection/pyskl/configs/posec3d/finetune_campus_v3.py')
    parser.add_argument('--ann-file', default='/home/hzcu/BullyDetection/data/campus/campus_balanced_v7.pkl')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', default='epoch_accuracy_curve.png')
    return parser.parse_args()


def get_checkpoints(work_dir):
    """找到所有 epoch_*.pth 并按 epoch 排序"""
    pattern = os.path.join(work_dir, 'epoch_*.pth')
    files = glob.glob(pattern)
    epoch_ckpts = []
    for f in files:
        basename = os.path.basename(f)
        epoch_num = int(basename.replace('epoch_', '').replace('.pth', ''))
        epoch_ckpts.append((epoch_num, f))
    epoch_ckpts.sort(key=lambda x: x[0])
    return epoch_ckpts


def build_val_data(cfg, ann_file):
    """构建 val 数据：加载 annotations + labels"""
    with open(ann_file, 'rb') as f:
        dataset = pickle.load(f)

    if isinstance(dataset, dict):
        val_ids = set(dataset['split']['val'])
        val_anns = [a for a in dataset['annotations'] if a['frame_dir'] in val_ids]
    else:
        print("WARNING: dataset is a list, using all as val")
        val_anns = dataset

    labels = np.array([a['label'] for a in val_anns])
    return val_anns, labels


def build_pipeline(cfg):
    """构建 val 的数据预处理 pipeline"""
    # 从 cfg 的 val pipeline 中提取
    val_pipeline_cfg = None
    for item in cfg.data.val.pipeline:
        pass  # just iterate
    return Compose(cfg.data.val.pipeline)


def inference_one_epoch(model, val_anns, pipeline, device):
    """用当前模型对所有 val 样本推理，返回预测结果"""
    model.eval()
    all_preds = []

    with torch.no_grad():
        for i, ann in enumerate(val_anns):
            # 构造样本 dict
            sample = dict(
                frame_dir=ann['frame_dir'],
                label=ann['label'],
                keypoint=ann['keypoint'].copy(),
                total_frames=ann['keypoint'].shape[1],
                img_shape=(1080, 1920),  # 默认尺寸，heatmap 会归一化
                start_index=0,
            )
            if 'keypoint_score' in ann:
                sample['keypoint_score'] = ann['keypoint_score'].copy()

            # apply pipeline
            sample = pipeline(sample)

            # collate + scatter
            data = collate([sample], samples_per_gpu=1)
            if 'img_metas' in data:
                data['img_metas'] = data['img_metas'].data[0]

            # scatter to device
            if device != 'cpu':
                data = scatter(data, [device])[0]
            else:
                # handle DataContainer
                for key in data:
                    if hasattr(data[key], 'data'):
                        data[key] = data[key].data[0]

            # forward
            with torch.no_grad():
                result = model(return_loss=False, **data)

            pred = np.argmax(result[0]) if isinstance(result, list) else np.argmax(result)
            all_preds.append(pred)

            if (i + 1) % 500 == 0:
                print(f"    {i+1}/{len(val_anns)}", flush=True)

    return np.array(all_preds)


def eval_predictions(preds, labels, classes):
    """计算 overall 和 per-class accuracy"""
    overall = (preds == labels).mean() * 100
    per_class = {}
    for i, name in enumerate(classes):
        mask = labels == i
        total = mask.sum()
        if total == 0:
            per_class[name] = 0.0
        else:
            per_class[name] = (preds[mask] == i).sum() / total * 100
    mean_class = np.mean(list(per_class.values()))
    return overall, mean_class, per_class


def main():
    args = parse_args()

    # 找 checkpoints
    epoch_ckpts = get_checkpoints(args.work_dir)
    if not epoch_ckpts:
        print(f"No checkpoints found in {args.work_dir}")
        return

    print(f"Found {len(epoch_ckpts)} checkpoints: {[e for e, _ in epoch_ckpts]}")

    # 加载 config
    cfg = Config.fromfile(args.config)
    # 覆盖 ann_file 确保一致
    cfg.data.test.ann_file = args.ann_file
    cfg.data.val.ann_file = args.ann_file

    # 构建模型（只构建一次，后面只换权重）
    cfg.model.cls_head.num_classes = len(CLASSES)
    model = build_model(cfg.model)
    model = model.to(args.device)

    # 构建 val 数据
    print("Loading val data ...")
    val_anns, labels = build_val_data(cfg, args.ann_file)
    print(f"Val samples: {len(val_anns)}")
    print(f"Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

    # 构建 pipeline
    pipeline = build_pipeline(cfg)

    # ===== 遍历每个 epoch =====
    results = []
    for epoch, ckpt_path in epoch_ckpts:
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}: {os.path.basename(ckpt_path)}")
        print(f"{'='*50}")

        # 加载权重
        load_checkpoint(model, ckpt_path, map_location=args.device, strict=False)

        # 推理
        preds = inference_one_epoch(model, val_anns, pipeline, args.device)

        # 评估
        overall, mean_class, per_class = eval_predictions(preds, labels, CLASSES)

        print(f"\n  Overall top1: {overall:.1f}%")
        print(f"  Mean class acc: {mean_class:.1f}%")
        for name, acc in per_class.items():
            print(f"    {name:<12} {acc:.1f}%")

        results.append({
            'epoch': epoch,
            'overall': overall,
            'mean_class': mean_class,
            'per_class': per_class,
        })

    # ===== 输出汇总表 =====
    print(f"\n\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    header = f"{'Epoch':>6} {'Overall':>8} {'MeanCls':>8}"
    for name in CLASSES:
        header += f" {name:>10}"
    print(header)
    print("-" * len(header))

    best_epoch = None
    best_overall = 0
    for r in results:
        line = f"{r['epoch']:>6} {r['overall']:>7.1f}% {r['mean_class']:>7.1f}%"
        for name in CLASSES:
            line += f" {r['per_class'][name]:>9.1f}%"
        print(line)
        if r['overall'] > best_overall:
            best_overall = r['overall']
            best_epoch = r['epoch']

    print(f"\nBest epoch: {best_epoch} (overall top1 = {best_overall:.1f}%)")

    # ===== 画图 =====
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    epochs = [r['epoch'] for r in results]

    # 上图：Overall + Mean Class
    ax1.plot(epochs, [r['overall'] for r in results], 'b-o', label='Overall top1', linewidth=2)
    ax1.plot(epochs, [r['mean_class'] for r in results], 'r--s', label='Mean class acc', linewidth=2)
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Val Accuracy across Epochs')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=best_epoch, color='green', linestyle=':', alpha=0.5, label=f'Best epoch={best_epoch}')

    # 下图：Per-class
    colors = ['#4CAF50', '#F44336', '#FF9800', '#2196F3', '#9C27B0']
    for i, name in enumerate(CLASSES):
        vals = [r['per_class'][name] for r in results]
        ax2.plot(epochs, vals, '-o', color=colors[i], label=name, linewidth=1.5, markersize=4)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Per-class Val Accuracy across Epochs')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {args.output}")

    # 保存详细结果
    results_path = os.path.join(args.work_dir, 'epoch_eval_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved: {results_path}")


if __name__ == '__main__':
    main()
