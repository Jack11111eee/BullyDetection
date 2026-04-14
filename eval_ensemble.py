"""
eval_ensemble.py — KP + Limb 双模型 Ensemble 评估

对每个样本分别用 kp 模型和 limb 模型推理，softmax 概率取平均后 argmax。
结果保存到指定目录。

用法:
  cd /home/hzcu/BullyDetection && python eval_ensemble.py
  cd /home/hzcu/BullyDetection && python eval_ensemble.py --kp-work-dir ... --limb-work-dir ...
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

DEFAULTS = dict(
    kp_config='/home/hzcu/BullyDetection/pyskl/configs/posec3d/finetune_campus_v4.py',
    kp_work_dir='/home/hzcu/BullyDetection/pyskl/work_dirs/posec3d_campus_v9',
    limb_config='/home/hzcu/BullyDetection/pyskl/configs/posec3d/finetune_campus_v5.py',
    limb_work_dir='/home/hzcu/BullyDetection/pyskl/work_dirs/posec3d_campus_v10',
    val_pkl='/home/hzcu/BullyDetection/data/campus/campus_kfold_0.pkl',
    test_pkl='/home/hzcu/BullyDetection/data/campus/campus_test.pkl',
    output_dir='/home/hzcu/BullyDetection/pyskl/work_dirs/posec3d_campus_ensemble',
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--kp-config', default=DEFAULTS['kp_config'])
    p.add_argument('--kp-work-dir', default=DEFAULTS['kp_work_dir'])
    p.add_argument('--limb-config', default=DEFAULTS['limb_config'])
    p.add_argument('--limb-work-dir', default=DEFAULTS['limb_work_dir'])
    p.add_argument('--val-pkl', default=DEFAULTS['val_pkl'])
    p.add_argument('--test-pkl', default=DEFAULTS['test_pkl'])
    p.add_argument('--output-dir', default=DEFAULTS['output_dir'])
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--kp-weight', type=float, default=0.5,
                   help='KP model weight in ensemble (default 0.5 = equal)')
    return p.parse_args()


class Logger:
    def __init__(self, filepath):
        self.filepath = filepath
        self.lines = []

    def log(self, msg=''):
        print(msg)
        self.lines.append(msg)

    def save(self):
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
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


def build_model_and_pipeline(config_path, ckpt_path, device):
    """加载模型和对应的 val pipeline"""
    cfg = Config.fromfile(config_path)
    pipeline = Compose(cfg.data.val.pipeline)

    model = build_model(cfg.model)
    model = model.to(device)
    load_checkpoint(model, ckpt_path, map_location=device, strict=False)
    model.eval()
    return model, pipeline


def inference_probs(model, anns, pipeline, device):
    """推理所有样本，返回 softmax 概率矩阵 (N, num_classes)"""
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
            data = collate([sample], samples_per_gpu=1)
            if 'img_metas' in data:
                data['img_metas'] = data['img_metas'].data[0]
            if device != 'cpu':
                data = scatter(data, [device])[0]

            result = model(return_loss=False, **data)
            prob = result[0] if isinstance(result, list) else result
            prob = np.array(prob).flatten()  # 确保是 1D (num_classes,)
            all_probs.append(prob)

            if (i + 1) % 500 == 0:
                print(f"    {i+1}/{len(anns)}", flush=True)

    return np.array(all_probs)  # (N, num_classes)


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

    kp_ckpt = get_best_checkpoint(args.kp_work_dir)
    limb_ckpt = get_best_checkpoint(args.limb_work_dir)

    if kp_ckpt is None:
        print(f"No KP checkpoint in {args.kp_work_dir}")
        return
    if limb_ckpt is None:
        print(f"No Limb checkpoint in {args.limb_work_dir}")
        return

    log = Logger(os.path.join(args.output_dir, 'eval_results.txt'))
    log.log(f"Ensemble Evaluation @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.log(f"KP config:   {args.kp_config}")
    log.log(f"KP ckpt:     {kp_ckpt}")
    log.log(f"Limb config: {args.limb_config}")
    log.log(f"Limb ckpt:   {limb_ckpt}")
    log.log(f"KP weight:   {args.kp_weight}, Limb weight: {1 - args.kp_weight}")

    # ===== 加载两个模型 =====
    print("\n[1/4] Loading KP model...")
    kp_model, kp_pipeline = build_model_and_pipeline(args.kp_config, kp_ckpt, args.device)

    print("[2/4] Loading Limb model...")
    limb_model, limb_pipeline = build_model_and_pipeline(args.limb_config, limb_ckpt, args.device)

    # ===== 评估函数 =====
    def eval_set(pkl_path, split_name, set_title):
        anns, labels = load_data(pkl_path, split_name)
        log.log(f"\n{set_title} samples: {len(anns)}")

        print(f"\n  KP model inference...")
        kp_probs = inference_probs(kp_model, anns, kp_pipeline, args.device)

        print(f"  Limb model inference...")
        limb_probs = inference_probs(limb_model, anns, limb_pipeline, args.device)

        # Ensemble: 加权平均
        w = args.kp_weight
        ensemble_probs = w * kp_probs + (1 - w) * limb_probs

        # 三组预测
        kp_preds = kp_probs.argmax(axis=1)
        limb_preds = limb_probs.argmax(axis=1)
        ens_preds = ensemble_probs.argmax(axis=1)

        # KP 单模型
        lines, kp_acc, kp_mean = format_results(kp_preds, labels, f"{set_title} — KP Model (R9)")
        for l in lines:
            log.log(l)

        # Limb 单模型
        lines, limb_acc, limb_mean = format_results(limb_preds, labels, f"{set_title} — Limb Model (R10)")
        for l in lines:
            log.log(l)

        # Ensemble
        lines, ens_acc, ens_mean = format_results(ens_preds, labels, f"{set_title} — ENSEMBLE (KP={w:.1f} + Limb={1-w:.1f})")
        for l in lines:
            log.log(l)

        # 对比
        log.log(f"\n{'='*60}")
        log.log(f"{set_title} COMPARISON")
        log.log(f"{'='*60}")
        log.log(f"  {'Model':>20} {'top1':>8} {'mean_cls':>10} {'vs KP':>8}")
        log.log(f"  {'KP (R9)':>20} {kp_acc:>7.1f}% {kp_mean:>9.1f}%      —")
        log.log(f"  {'Limb (R10)':>20} {limb_acc:>7.1f}% {limb_mean:>9.1f}% {limb_acc-kp_acc:>+7.1f}%")
        log.log(f"  {'ENSEMBLE':>20} {ens_acc:>7.1f}% {ens_mean:>9.1f}% {ens_acc-kp_acc:>+7.1f}%")

        return kp_acc, limb_acc, ens_acc

    # ===== Val =====
    print("\n[3/4] Evaluating Val set...")
    val_kp, val_limb, val_ens = eval_set(args.val_pkl, 'val', 'VAL')

    # ===== Test =====
    test_kp, test_limb, test_ens = None, None, None
    if os.path.exists(args.test_pkl):
        print("\n[4/4] Evaluating Test set...")
        test_kp, test_limb, test_ens = eval_set(args.test_pkl, 'test', 'TEST')

    # ===== 最终汇总 =====
    log.log(f"\n\n{'='*60}")
    log.log("FINAL SUMMARY")
    log.log('='*60)
    log.log(f"  {'':>12} {'KP(R9)':>10} {'Limb(R10)':>10} {'Ensemble':>10} {'Ens vs KP':>10}")
    log.log(f"  {'Val top1':>12} {val_kp:>9.1f}% {val_limb:>9.1f}% {val_ens:>9.1f}% {val_ens-val_kp:>+9.1f}%")
    if test_ens is not None:
        log.log(f"  {'Test top1':>12} {test_kp:>9.1f}% {test_limb:>9.1f}% {test_ens:>9.1f}% {test_ens-test_kp:>+9.1f}%")

    log.save()


if __name__ == '__main__':
    main()
