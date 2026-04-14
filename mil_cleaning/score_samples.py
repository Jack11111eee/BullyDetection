"""
score_samples.py — 交叉折打分：每个样本只被没见过它的模型打分

原理：
  Fold 0 模型 → 打分 Fold 0 的 val 数据（模型训练时没见过）
  Fold 1 模型 → 打分 Fold 1 的 val 数据（模型训练时没见过）
  ...
  合并 = 每个样本都有无偏的 P(true_label)

输出: mil_cleaning/scores.pkl
  列表，每个元素: {'frame_dir': str, 'label': int, 'probs': np.array(5,), 'fold': int}

用法:
  cd /home/hzcu/BullyDetection && python mil_cleaning/score_samples.py
  cd /home/hzcu/BullyDetection && python mil_cleaning/score_samples.py --folds 0 1
"""

import os
import sys
import glob
import argparse
import pickle
import numpy as np
import torch
from collections import Counter

sys.path.insert(0, '/home/hzcu/BullyDetection/pyskl')
from pyskl.models import build_model
from mmcv import Config
from mmcv.runner import load_checkpoint
from pyskl.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter

CLASSES = ['normal', 'fighting', 'bullying', 'falling', 'climbing']
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 每个 fold 的 config 和 work_dir
FOLD_CONFIGS = {
    0: {
        'config': '/home/hzcu/BullyDetection/pyskl/work_dirs/posec3d_campus_fold0/finetune_campus_fold0.py',
        'work_dir': '/home/hzcu/BullyDetection/pyskl/work_dirs/posec3d_campus_fold0',
        'data_pkl': '/home/hzcu/BullyDetection/data/campus/campus_kfold_0.pkl',
    },
    1: {
        'config': '/home/hzcu/BullyDetection/pyskl/work_dirs/posec3d_campus_fold1/finetune_campus_fold1.py',
        'work_dir': '/home/hzcu/BullyDetection/pyskl/work_dirs/posec3d_campus_fold1',
        'data_pkl': '/home/hzcu/BullyDetection/data/campus/campus_kfold_1.pkl',
    },
    2: {
        'config': '/home/hzcu/BullyDetection/pyskl/work_dirs/posec3d_campus_fold2/finetune_campus_fold2.py',
        'work_dir': '/home/hzcu/BullyDetection/pyskl/work_dirs/posec3d_campus_fold2',
        'data_pkl': '/home/hzcu/BullyDetection/data/campus/campus_kfold_2.pkl',
    },
    3: {
        'config': '/home/hzcu/BullyDetection/pyskl/work_dirs/posec3d_campus_fold3/finetune_campus_fold3.py',
        'work_dir': '/home/hzcu/BullyDetection/pyskl/work_dirs/posec3d_campus_fold3',
        'data_pkl': '/home/hzcu/BullyDetection/data/campus/campus_kfold_3.pkl',
    },
    4: {
        'config': '/home/hzcu/BullyDetection/pyskl/work_dirs/posec3d_campus_fold4/finetune_campus_fold4.py',
        'work_dir': '/home/hzcu/BullyDetection/pyskl/work_dirs/posec3d_campus_fold4',
        'data_pkl': '/home/hzcu/BullyDetection/data/campus/campus_kfold_4.pkl',
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', nargs='+', type=int, default=[0, 1],
                        help='要打分的 fold 编号（默认 0 1）')
    parser.add_argument('--output', default=os.path.join(SCRIPT_DIR, 'scores.pkl'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-log', type=int, default=500)
    return parser.parse_args()


def get_best_checkpoint(work_dir):
    best = glob.glob(os.path.join(work_dir, 'best_*.pth'))
    if best:
        return best[0]
    pattern = os.path.join(work_dir, 'epoch_*.pth')
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=lambda f: int(os.path.basename(f).split('_')[1].split('.')[0]))


def load_val_data(pkl_path):
    """加载 kfold pkl 中的 val split 样本"""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    val_ids = set(data['split']['val'])
    anns = [a for a in data['annotations'] if a['frame_dir'] in val_ids]
    return anns


def score_samples(model, anns, pipeline, device, batch_log=500):
    """对每个样本推理，返回完整概率向量"""
    model.eval()
    results = []
    n = len(anns)

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
            probs = result[0] if isinstance(result, list) else result
            probs = np.atleast_1d(probs.squeeze())

            results.append({
                'frame_dir': ann['frame_dir'],
                'label': ann['label'],
                'probs': probs.astype(np.float32),
            })

            if (i + 1) % batch_log == 0:
                pred = np.argmax(probs)
                p_true = probs[ann['label']]
                print(f"    [{i+1}/{n}] label={CLASSES[ann['label']]}, "
                      f"pred={CLASSES[pred]}, P(true)={p_true:.3f}")

    return results


def print_quick_stats(results, title):
    """打印每个类别的 P(true_label) 统计"""
    print(f"\n--- {title} ---")
    for cls_idx, cls_name in enumerate(CLASSES):
        cls_results = [r for r in results if r['label'] == cls_idx]
        if not cls_results:
            continue
        p_true = np.array([r['probs'][cls_idx] for r in cls_results])
        preds = np.array([np.argmax(r['probs']) for r in cls_results])
        correct = (preds == cls_idx).sum()
        print(f"\n  {cls_name} (n={len(cls_results)}):")
        print(f"    Accuracy: {correct}/{len(cls_results)} ({correct/len(cls_results)*100:.1f}%)")
        print(f"    P(true): mean={p_true.mean():.3f}, median={np.median(p_true):.3f}, min={p_true.min():.3f}")
        print(f"    P(true) < 0.3: {(p_true < 0.3).sum()} ({(p_true < 0.3).mean()*100:.1f}%)")
        print(f"    P(true) < 0.5: {(p_true < 0.5).sum()} ({(p_true < 0.5).mean()*100:.1f}%)")


def main():
    args = parse_args()
    all_results = []

    print(f"Cross-fold scoring: folds = {args.folds}\n")

    for fold_idx in args.folds:
        if fold_idx not in FOLD_CONFIGS:
            print(f"ERROR: fold {fold_idx} config not defined, skipping")
            continue

        fc = FOLD_CONFIGS[fold_idx]
        print(f"{'=' * 60}")
        print(f"FOLD {fold_idx}: model scores its own val split")
        print(f"{'=' * 60}")

        # 检查 checkpoint
        ckpt = get_best_checkpoint(fc['work_dir'])
        if ckpt is None:
            print(f"  ERROR: No checkpoint in {fc['work_dir']}, skipping fold {fold_idx}")
            continue
        print(f"  Config:     {fc['config']}")
        print(f"  Checkpoint: {ckpt}")
        print(f"  Data:       {fc['data_pkl']}")

        # 加载模型
        cfg = Config.fromfile(fc['config'])
        pipeline = Compose(cfg.data.val.pipeline)

        model = build_model(cfg.model)
        model = model.to(args.device)
        load_checkpoint(model, ckpt, map_location=args.device, strict=False)

        # 加载 val 数据
        val_anns = load_val_data(fc['data_pkl'])
        counts = Counter(a['label'] for a in val_anns)
        print(f"  Val samples: {len(val_anns)}")
        for i, name in enumerate(CLASSES):
            print(f"    {i} {name:<12} {counts.get(i, 0):>6}")

        # 打分
        print(f"\n  Scoring {len(val_anns)} val samples ...")
        fold_results = score_samples(model, val_anns, pipeline, args.device, args.batch_log)

        # 标记 fold 来源
        for r in fold_results:
            r['fold'] = fold_idx

        print_quick_stats(fold_results, f"Fold {fold_idx} Val Stats")
        all_results.extend(fold_results)

        # 释放 GPU 显存，为下一个 fold 腾空间
        del model
        torch.cuda.empty_cache()
        print()

    # 保存合并结果
    with open(args.output, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\n{'=' * 60}")
    print(f"TOTAL: {len(all_results)} samples scored across {len(args.folds)} folds")
    print(f"Saved to {args.output}")

    # 合并统计
    print_quick_stats(all_results, "Combined Stats (all folds)")


if __name__ == '__main__':
    main()
