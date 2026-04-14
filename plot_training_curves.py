"""
plot_training_curves.py — 解析训练日志 + 评估所有 checkpoint + 画曲线

Usage:
  # 仅画训练曲线（不需要 GPU，本地即可）
  python plot_training_curves.py --log-json work_dirs/posec3d_campus_v6/20260331_201408.log.json

  # 画训练曲线 + 评估所有 checkpoint（需要 GPU）
  python plot_training_curves.py \
    --log-json work_dirs/posec3d_campus_v6/20260331_201408.log.json \
    --eval \
    --config pyskl/configs/posec3d/finetune_campus_v3.py \
    --work-dir pyskl/work_dirs/posec3d_campus_v6

  输出: training_curves.png
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')  # 无 GUI 环境
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def parse_log_json(log_path):
    """解析 mmcv 的 .log.json 文件"""
    train_records = []
    val_records = []

    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            mode = record.get('mode')
            if mode == 'train':
                train_records.append(record)
            elif mode == 'val':
                val_records.append(record)

    return train_records, val_records


def aggregate_by_epoch(records, key):
    """按 epoch 聚合，返回 {epoch: mean_value}"""
    epoch_vals = defaultdict(list)
    for r in records:
        if key in r:
            epoch_vals[r['epoch']].append(r[key])
    return {ep: np.mean(vals) for ep, vals in sorted(epoch_vals.items())}


def evaluate_checkpoints(config_path, work_dir, ann_file=None):
    """评估所有保存的 checkpoint，返回 {epoch: {metric: value}}"""
    import pickle
    import torch
    import mmcv
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import load_checkpoint
    from pyskl.apis import init_recognizer
    from pyskl.datasets import build_dataset, build_dataloader

    cfg = mmcv.Config.fromfile(config_path)
    if ann_file:
        cfg.data.test.ann_file = ann_file

    # 找所有 checkpoint
    checkpoints = {}
    for f in sorted(os.listdir(work_dir)):
        if f.startswith('epoch_') and f.endswith('.pth'):
            ep = int(f.replace('epoch_', '').replace('.pth', ''))
            checkpoints[ep] = os.path.join(work_dir, f)

    print(f"Found {len(checkpoints)} checkpoints: {sorted(checkpoints.keys())}")

    # 构建数据集（只需要一次）
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    dataloader = build_dataloader(
        dataset,
        videos_per_gpu=16,
        workers_per_gpu=4,
        dist=False,
        shuffle=False,
    )

    results = {}
    for ep in sorted(checkpoints.keys()):
        ckpt_path = checkpoints[ep]
        print(f"\nEvaluating epoch {ep}...")

        model = init_recognizer(cfg, ckpt_path, device='cuda:0')
        model = MMDataParallel(model, device_ids=[0])
        model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for data in dataloader:
                result = model(return_loss=False, **data)
                preds = np.argmax(result, axis=1)
                labels = data['label'].numpy()
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.tolist())

        # 1. 强制转为一维 numpy 数组 (解决所有维度不匹配和类型转换问题)
        all_labels = np.array(all_labels).flatten()
        all_preds = np.array(all_preds).flatten()
        top1 = np.mean(all_preds == all_labels)

        # 2. 获取去重后的类别列表 (现在可以安全地使用 np.unique)
        classes = sorted(np.unique(all_labels))

        # 3. 随后的循环代码不需要改动，但确保它看起来是这样的：
        class_accs = {}
        for c in classes:
            mask = (all_labels == c)
            if np.any(mask):
                class_accs[c] = np.mean(all_preds[mask] == c)
            else:
                class_accs[c] = 0.0

        mean_class_acc = np.mean(list(class_accs.values()))

        results[ep] = {
            'top1': top1,
            'mean_class_acc': mean_class_acc,
            'class_accs': class_accs,
        }
        print(f"  top1={top1:.4f}, mean_class_acc={mean_class_acc:.4f}, per_class={class_accs}")

    return results


def plot_curves(train_records, val_records=None, eval_results=None,
                class_names=None, output='training_curves.png'):
    """画训练曲线"""

    # 聚合训练数据
    epoch_loss = aggregate_by_epoch(train_records, 'loss')
    epoch_train_acc = aggregate_by_epoch(train_records, 'top1_acc')
    epoch_lr = aggregate_by_epoch(train_records, 'lr')
    epoch_grad_norm = aggregate_by_epoch(train_records, 'grad_norm')

    epochs_train = sorted(epoch_loss.keys())
    has_eval = eval_results is not None and len(eval_results) > 0

    # 决定子图数量
    n_plots = 3  # loss, accuracy, lr+grad_norm
    if has_eval and class_names:
        n_plots = 4  # 加上 per-class accuracy

    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots))
    fig.suptitle('Round 6 Training Curves (5 classes, no vandalism)', fontsize=14, fontweight='bold')

    # ===== Plot 1: Loss =====
    ax = axes[0]
    ax.plot(epochs_train, [epoch_loss[e] for e in epochs_train],
            'b-o', markersize=3, label='Train Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))

    # ===== Plot 2: Accuracy =====
    ax = axes[1]
    ax.plot(epochs_train, [epoch_train_acc[e] for e in epochs_train],
            'b-o', markersize=3, label='Train top1_acc')

    if has_eval:
        eval_epochs = sorted(eval_results.keys())
        eval_top1 = [eval_results[e]['top1'] for e in eval_epochs]
        eval_mean = [eval_results[e]['mean_class_acc'] for e in eval_epochs]
        ax.plot(eval_epochs, eval_top1, 'r-s', markersize=6, linewidth=2, label='Val top1_acc')
        ax.plot(eval_epochs, eval_mean, 'g-^', markersize=6, linewidth=2, label='Val mean_class_acc')

        # 标注最佳 epoch
        best_ep = eval_epochs[np.argmax(eval_top1)]
        best_val = max(eval_top1)
        ax.annotate(f'Best: {best_val:.1%} (ep{best_ep})',
                    xy=(best_ep, best_val),
                    xytext=(best_ep + 3, best_val + 0.03),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, color='red', fontweight='bold')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy (Train vs Val)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))

    # ===== Plot 3: LR + Grad Norm =====
    ax = axes[2]
    ax2 = ax.twinx()
    ln1 = ax.plot(epochs_train, [epoch_lr[e] for e in epochs_train],
                  'c-', linewidth=1.5, label='Learning Rate')
    ln2 = ax2.plot(epochs_train, [epoch_grad_norm[e] for e in epochs_train],
                   'm-', linewidth=1.5, alpha=0.7, label='Grad Norm')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate', color='c')
    ax2.set_ylabel('Grad Norm', color='m')
    ax.set_title('Learning Rate & Gradient Norm')
    lns = ln1 + ln2
    labels = [l.get_label() for l in lns]
    ax.legend(lns, labels, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))

    # ===== Plot 4: Per-class Val Accuracy =====
    if has_eval and class_names:
        ax = axes[3]
        eval_epochs = sorted(eval_results.keys())
        n_classes = len(class_names)
        colors = plt.cm.Set1(np.linspace(0, 1, n_classes))

        for c_idx, c_name in enumerate(class_names):
            class_vals = []
            for e in eval_epochs:
                ca = eval_results[e].get('class_accs', {})
                class_vals.append(ca.get(c_idx, 0))
            ax.plot(eval_epochs, class_vals, '-o', color=colors[c_idx],
                    markersize=5, linewidth=2, label=c_name)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Per-Class Validation Accuracy')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output}")

    # 打印摘要
    print("\n========== Training Summary ==========")
    print(f"Epochs: {min(epochs_train)} - {max(epochs_train)}")
    print(f"Final train loss: {epoch_loss[max(epochs_train)]:.4f}")
    print(f"Final train acc:  {epoch_train_acc[max(epochs_train)]:.1%}")

    if has_eval:
        eval_epochs = sorted(eval_results.keys())
        for ep in eval_epochs:
            r = eval_results[ep]
            print(f"  Epoch {ep:3d}: val_top1={r['top1']:.1%}, mean_class={r['mean_class_acc']:.1%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-json', required=True, help='Path to .log.json file')
    parser.add_argument('--eval', action='store_true', help='Evaluate all checkpoints (needs GPU)')
    parser.add_argument('--config', default=None, help='Config path (required if --eval)')
    parser.add_argument('--work-dir', default=None, help='Work dir with checkpoints (required if --eval)')
    parser.add_argument('--ann-file', default=None, help='Override ann_file path')
    parser.add_argument('--class-names', nargs='+',
                        default=['normal', 'violence', 'falling', 'climbing'],
                        help='Class names for per-class plot')
    parser.add_argument('--output', default='training_curves.png', help='Output image path')
    args = parser.parse_args()

    print("Parsing training log...")
    train_records, val_records = parse_log_json(args.log_json)
    print(f"  Train records: {len(train_records)}")
    print(f"  Val records:   {len(val_records)}")

    eval_results = None
    if args.eval:
        if not args.config or not args.work_dir:
            print("ERROR: --config and --work-dir required when using --eval")
            sys.exit(1)
        eval_results = evaluate_checkpoints(args.config, args.work_dir, args.ann_file)

    plot_curves(train_records, val_records, eval_results,
                class_names=args.class_names, output=args.output)


if __name__ == '__main__':
    main()
