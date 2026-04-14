"""
eval_multiclip.py — 用 multi-clip 推理评估现有模型（不需要重训）

原理：val 时对同一样本采样 10 次不同的 48 帧片段，取平均概率，减少随机采样的噪声。
这对 normal/fighting 这种容易被单次采样误判的类别特别有效。

用法:
  cd /home/hzcu/BullyDetection && python eval_multiclip.py
  cd /home/hzcu/BullyDetection && python eval_multiclip.py --num-clips 10 --checkpoint .../epoch_30.pth
"""

import os
import sys
import argparse
import pickle
import numpy as np
import torch

sys.path.insert(0, '/home/hzcu/BullyDetection/pyskl')
from pyskl.models import build_model
from mmcv import Config
from mmcv.runner import load_checkpoint
from pyskl.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter

CLASSES = ['normal', 'fighting', 'bullying', 'falling', 'climbing']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',
                        default='/home/hzcu/BullyDetection/pyskl/work_dirs/posec3d_campus_fold0/epoch_30.pth')
    parser.add_argument('--config',
                        default='/home/hzcu/BullyDetection/pyskl/configs/posec3d/finetune_campus_v3.py')
    parser.add_argument('--ann-file',
                        default='/home/hzcu/BullyDetection/data/campus/campus_kfold_0.pkl')
    parser.add_argument('--split', default='val', choices=['val', 'test'])
    parser.add_argument('--test-file',
                        default='/home/hzcu/BullyDetection/data/campus/campus_test.pkl')
    parser.add_argument('--num-clips', type=int, default=10)
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


def build_multiclip_pipeline(cfg, num_clips):
    """构建 multi-clip val pipeline"""
    pipeline_cfg = [
        dict(type='UniformSampleFrames', clip_len=48, num_clips=num_clips, test_mode=True),
        dict(type='PoseDecode'),
        dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
        dict(type='Resize', scale=(64, 64), keep_ratio=False),
        dict(type='GeneratePoseTarget',
             sigma=0.6, use_score=True, with_kp=True, with_limb=False),
        dict(type='FormatShape', input_format='NCTHW_Heatmap'),
        dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs', 'label']),
    ]
    return Compose(pipeline_cfg)


def load_data(pkl_path, split_name):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    ids = set(data['split'][split_name])
    anns = [a for a in data['annotations'] if a['frame_dir'] in ids]
    labels = np.array([a['label'] for a in anns])
    return anns, labels


def inference_multiclip(model, anns, pipeline, device, num_clips):
    """Multi-clip 推理：每个样本采样 num_clips 次，模型分别预测后取平均"""
    model.eval()
    all_preds = []
    all_probs = []

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
            data_batch = collate([sample], samples_per_gpu=1)
            if 'img_metas' in data_batch:
                data_batch['img_metas'] = data_batch['img_metas'].data[0]
            if device != 'cpu':
                data_batch = scatter(data_batch, [device])[0]

            result = model(return_loss=False, **data_batch)
            prob = result[0] if isinstance(result, list) else result

            # result 可能已经是 multi-clip 平均后的结果
            # 如果是 (num_clips, num_classes)，手动取平均
            if isinstance(prob, np.ndarray) and prob.ndim == 2:
                prob = prob.mean(axis=0)

            all_probs.append(prob)
            all_preds.append(np.argmax(prob))

            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(anns)}", flush=True)

    return np.array(all_preds), all_probs


def print_results(preds, labels, title):
    overall = (preds == labels).mean() * 100
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"Overall top1: {overall:.1f}%\n")

    print(f"{'Class':<12} {'Correct':>8} {'Total':>8} {'Acc':>8}")
    print("-" * 40)
    accs = []
    for i, name in enumerate(CLASSES):
        mask = labels == i
        total = mask.sum()
        if total == 0:
            continue
        correct = (preds[mask] == i).sum()
        acc = correct / total * 100
        accs.append(acc)
        print(f"{name:<12} {correct:>8} {total:>8} {acc:>7.1f}%")

    mean_class = np.mean(accs)
    print(f"\nMean class acc: {mean_class:.1f}%")

    # 混淆矩阵
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

    return overall, mean_class


def main():
    args = parse_args()
    print(f"Multi-clip evaluation: num_clips={args.num_clips}")

    cfg = Config.fromfile(args.config)
    pipeline = build_multiclip_pipeline(cfg, args.num_clips)

    # 加载模型
    model = build_model(cfg.model)
    model = model.to(args.device)
    load_checkpoint(model, args.checkpoint, map_location=args.device, strict=False)
    print(f"Loaded: {args.checkpoint}")

    # 评估 val
    print(f"\nLoading val data from {args.ann_file} ...")
    val_anns, val_labels = load_data(args.ann_file, 'val')
    print(f"Val samples: {len(val_anns)}")
    val_preds, _ = inference_multiclip(model, val_anns, pipeline, args.device, args.num_clips)
    val_overall, val_mean = print_results(val_preds, val_labels,
                                          f"VAL (num_clips={args.num_clips})")

    # 评估 test
    if os.path.exists(args.test_file):
        print(f"\nLoading test data from {args.test_file} ...")
        test_anns, test_labels = load_data(args.test_file, 'test')
        print(f"Test samples: {len(test_anns)}")
        test_preds, _ = inference_multiclip(model, test_anns, pipeline, args.device, args.num_clips)
        test_overall, test_mean = print_results(test_preds, test_labels,
                                                f"TEST (num_clips={args.num_clips})")

    # 对比总结
    print(f"\n{'='*60}")
    print(f"COMPARISON: 1-clip vs {args.num_clips}-clip")
    print(f"  Fold 0 val  (1-clip): 81.3%")
    print(f"  Fold 0 val  ({args.num_clips}-clip): {val_overall:.1f}%  ({val_overall - 81.3:+.1f}%)")
    if os.path.exists(args.test_file):
        print(f"  Test        (1-clip): 83.5%")
        print(f"  Test        ({args.num_clips}-clip): {test_overall:.1f}%  ({test_overall - 83.5:+.1f}%)")


if __name__ == '__main__':
    main()
