"""
probe_wall_impact.py — 撞墙 / 头撞墙 行为的特征探查脚本

目的：在写规则之前先看清楚样本的真实物理信号，避免凭感觉拍阈值。

用法:
  python probe_wall_impact.py \
    --videos-dir /path/to/probe_samples \
    --output-dir probe_output \
    --yolo-pose yolo11m-pose.pt \
    [--with-posec3d --posec3d-config ... --posec3d-ckpt ...]

videos-dir 结构（每个子目录一个类别）:
  probe_samples/
    impact/     <- 快速撞击式样本
    headbang/   <- 扶墙撞头样本
    normal/     <- 可选负样本（走路、蹲下、拥抱等易混淆）

每个视频产出:
  <output>/per_video/<class>_<video_stem>/
    features.npz      所有时序数据（numpy 数组）
    trace.png         12 行共享 x 轴的特征时序图
    summary.json      峰值、事件窗、统计摘要

跨视频产出:
  <output>/aggregate/
    overlay_<class>.png       该类所有视频关键时序峰对齐叠加
    comparison_<feature>.png  关键特征跨类直方图 + KDE
    distribution.json         各类特征分布统计（mean/std/p5/p50/p95）

设计要点：
  - 复用 pipeline.py 的 EMA 平滑（α=0.5, score<=0.3 沿用前值衰减）
  - 归一化以 bbox 高度为基准（消除摄像头距离影响）
  - 自相关窗口 90 帧（~3s@30fps）覆盖撞头节奏 0.5~2Hz
  - 主 track = 帧数最多的 track；不强制单人
  - PoseC3D 推理可选（--with-posec3d），默认关闭以便快速迭代
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print('[FATAL] matplotlib 未安装，请 pip install matplotlib', file=sys.stderr)
    sys.exit(1)

from ultralytics import YOLO


EMA_ALPHA = 0.5  # 与 pipeline.py SkeletonBuffer.smooth_alpha 对齐
SCORE_THRESHOLD = 0.3

# COCO 17 关键点索引
KP_NOSE = 0
KP_L_SHOULDER, KP_R_SHOULDER = 5, 6
KP_L_ELBOW, KP_R_ELBOW = 7, 8
KP_L_WRIST, KP_R_WRIST = 9, 10
KP_L_HIP, KP_R_HIP = 11, 12
KP_L_KNEE, KP_R_KNEE = 13, 14
KP_L_ANKLE, KP_R_ANKLE = 15, 16

UPPER_KPS = list(range(0, 11))
LOWER_KPS = [13, 14, 15, 16]


# ============================================================
# 视频读取 + YOLO-Pose track
# ============================================================
def extract_tracks(video_path, yolo_model, yolo_conf=0.3):
    """对一个视频跑 YOLO-Pose + ByteTrack,返回每个 track 的原始时序数据。

    Returns:
        dict: {track_id: [{frame_idx, bbox, kps_raw, scores_raw}, ...]}
        meta: {fps, frame_count, width, height}
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'无法打开视频: {video_path}')

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tracks = defaultdict(list)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = yolo_model.track(
            source=frame, persist=True, tracker='bytetrack.yaml',
            conf=yolo_conf, iou=0.5, verbose=False,
        )
        result = results[0]
        if result.boxes is not None and result.keypoints is not None:
            for i, box in enumerate(result.boxes):
                if box.id is None:
                    continue
                tid = int(box.id.item())
                kps_data = result.keypoints.data[i].cpu().numpy()
                tracks[tid].append({
                    'frame_idx': frame_idx,
                    'bbox': box.xyxy[0].cpu().numpy().astype(np.float32),
                    'kps_raw': kps_data[:, :2].astype(np.float32),
                    'scores_raw': kps_data[:, 2].astype(np.float32),
                })
        frame_idx += 1

    cap.release()
    meta = dict(fps=fps, frame_count=frame_idx, width=width, height=height)
    return tracks, meta


# ============================================================
# EMA 平滑（复现 pipeline.py SkeletonBuffer.update 逻辑）
# ============================================================
def apply_ema(kps_raw_seq, scores_raw_seq, alpha=EMA_ALPHA, thr=SCORE_THRESHOLD):
    """按 pipeline.py 的逻辑对时序关键点做 EMA。
    kps_raw_seq: (T, 17, 2), scores_raw_seq: (T, 17)
    返回同 shape 的平滑版本。不存在的帧（score NaN）会保留 NaN。
    """
    T = len(kps_raw_seq)
    kps_ema = np.full_like(kps_raw_seq, np.nan)
    sc_ema = np.full_like(scores_raw_seq, np.nan)
    prev_k, prev_s = None, None
    for t in range(T):
        k = kps_raw_seq[t].copy()
        s = scores_raw_seq[t].copy()
        if np.all(np.isnan(s)):
            # 缺帧：保持 NaN，但前值延续以便下一个实帧能平滑
            kps_ema[t] = np.nan
            sc_ema[t] = np.nan
            continue
        if prev_k is not None and prev_s is not None:
            for j in range(17):
                if s[j] > thr and prev_s[j] > thr:
                    k[j] = alpha * k[j] + (1 - alpha) * prev_k[j]
                    s[j] = alpha * s[j] + (1 - alpha) * prev_s[j]
                elif s[j] <= thr and prev_s[j] > thr:
                    k[j] = prev_k[j]
                    s[j] = prev_s[j] * 0.9
                # 其它: 新帧有效但旧帧无效 → 用新值
        prev_k = k.copy()
        prev_s = s.copy()
        kps_ema[t] = k
        sc_ema[t] = s
    return kps_ema, sc_ema


# ============================================================
# 单 track 时序密集化：填入 NaN 到缺失帧
# ============================================================
def densify_track(entries, global_start, global_end):
    """把稀疏的 track entries 拉成 [global_start, global_end] 每帧一行的密集数组。
    缺失帧填 NaN（bbox 全 NaN，kps 全 NaN，scores 全 NaN）。
    """
    T = global_end - global_start + 1
    frame_ids = np.arange(global_start, global_end + 1)
    bboxes = np.full((T, 4), np.nan, dtype=np.float32)
    kps = np.full((T, 17, 2), np.nan, dtype=np.float32)
    scores = np.full((T, 17), np.nan, dtype=np.float32)
    present = np.zeros(T, dtype=bool)
    for e in entries:
        idx = e['frame_idx'] - global_start
        if 0 <= idx < T:
            bboxes[idx] = e['bbox']
            kps[idx] = e['kps_raw']
            scores[idx] = e['scores_raw']
            present[idx] = True
    return frame_ids, bboxes, kps, scores, present


# ============================================================
# 特征计算
# ============================================================
def safe_mean_kps(kps_t, scores_t, idx_list, thr=SCORE_THRESHOLD):
    """对关键点子集取有效点的均值。返回 (2,) 或 NaN。"""
    if np.all(np.isnan(scores_t)):
        return np.array([np.nan, np.nan])
    valid = np.array([scores_t[i] > thr for i in idx_list])
    if valid.sum() == 0:
        return np.array([np.nan, np.nan])
    selected = np.array([kps_t[i] for i, v in zip(idx_list, valid) if v])
    return selected.mean(axis=0)


def compute_bbox_height(bboxes, kps, scores):
    """优先用 YOLO bbox 高度；缺失时退化为关键点包围框高度。"""
    T = len(bboxes)
    h = np.full(T, np.nan, dtype=np.float32)
    for t in range(T):
        if not np.any(np.isnan(bboxes[t])):
            h[t] = max(1.0, bboxes[t][3] - bboxes[t][1])
        elif not np.all(np.isnan(scores[t])):
            valid = scores[t] > SCORE_THRESHOLD
            if valid.sum() >= 3:
                ys = kps[t][valid, 1]
                h[t] = max(1.0, ys.max() - ys.min())
    return h


def rolling_std(x, window):
    """1D 滑窗标准差。x: (T,). 返回 (T,)，前 window-1 个为 NaN。
    对 NaN 鲁棒：窗口内有效点 >= window/2 才计算。
    """
    T = len(x)
    out = np.full(T, np.nan)
    for t in range(window - 1, T):
        seg = x[t - window + 1:t + 1]
        valid = ~np.isnan(seg)
        if valid.sum() >= window // 2:
            out[t] = np.std(seg[valid])
    return out


def rolling_max(x, window):
    """1D 滑窗最大值（NaN 安全）。前 window-1 为 NaN。"""
    T = len(x)
    out = np.full(T, np.nan)
    for t in range(window - 1, T):
        seg = x[t - window + 1:t + 1]
        valid = ~np.isnan(seg)
        if valid.any():
            out[t] = seg[valid].max()
    return out


def rolling_count_exceed(x, thr, window):
    """1D 滑窗内 > thr 的帧数（NaN 视为不超）。前 window-1 为 NaN。"""
    T = len(x)
    out = np.full(T, np.nan)
    for t in range(window - 1, T):
        seg = x[t - window + 1:t + 1]
        valid = ~np.isnan(seg)
        if valid.sum() >= window // 2:
            out[t] = int(((seg > thr) & valid).sum())
    return out


def count_peaks_exceeding(x, thr, min_gap=5):
    """数序列里 >thr 且与邻居局部最大的"独立"峰数（min_gap 帧内去重）。"""
    peaks = []
    for t in range(1, len(x) - 1):
        if np.isnan(x[t]) or x[t] <= thr:
            continue
        if x[t] >= x[t - 1] and x[t] >= x[t + 1]:
            if not peaks or t - peaks[-1] >= min_gap:
                peaks.append(t)
    return len(peaks), peaks


def rolling_autocorr_peak(x, window, min_lag=10, max_lag=60):
    """对序列滑窗做自相关，返回每个 t 的 (峰值, 峰值对应 lag)。
    min_lag=10 (0.33s@30fps), max_lag=60 (2s@30fps) → 对应 0.5~3Hz 撞头频率。
    """
    T = len(x)
    peak_val = np.full(T, np.nan)
    peak_lag = np.full(T, np.nan)
    for t in range(window - 1, T):
        seg = x[t - window + 1:t + 1]
        valid = ~np.isnan(seg)
        if valid.sum() < window * 0.7:
            continue
        # 填充 NaN 为均值（简单处理，避免破坏 autocorr）
        seg_f = seg.copy()
        seg_f[~valid] = seg[valid].mean()
        seg_f = seg_f - seg_f.mean()
        if np.std(seg_f) < 1e-6:
            continue
        seg_f = seg_f / (np.std(seg_f) * len(seg_f))
        # autocorrelate
        ac = np.correlate(seg_f, seg_f, mode='full')
        ac = ac[len(seg_f) - 1:]  # 取非负 lag 部分
        if len(ac) <= max_lag:
            continue
        segment = ac[min_lag:max_lag + 1]
        if len(segment) == 0:
            continue
        best_i = int(np.argmax(segment))
        peak_val[t] = float(segment[best_i])
        peak_lag[t] = float(min_lag + best_i)
    return peak_val, peak_lag


def compute_features(frame_ids, bboxes, kps_ema, scores_ema, all_tracks_dense,
                     main_tid=None):
    """计算所有衍生特征。all_tracks_dense: {tid: (bboxes, kps_ema, scores_ema)}
    用于场景上下文特征（nearest dist, overlap, active count）。
    main_tid 用于在场景上下文计算里排除 self。
    """
    T = len(frame_ids)

    bbox_h = compute_bbox_height(bboxes, kps_ema, scores_ema)

    # 关键点聚合
    hip_center = np.array([safe_mean_kps(kps_ema[t], scores_ema[t],
                                         [KP_L_HIP, KP_R_HIP]) for t in range(T)])
    shoulder_center = np.array([safe_mean_kps(kps_ema[t], scores_ema[t],
                                              [KP_L_SHOULDER, KP_R_SHOULDER]) for t in range(T)])
    torso_center = (hip_center + shoulder_center) / 2
    head_pt = kps_ema[:, KP_NOSE, :]  # (T, 2)
    l_wrist = kps_ema[:, KP_L_WRIST, :]
    r_wrist = kps_ema[:, KP_R_WRIST, :]

    # 速度（帧间差分）—— 按 bbox_h 归一化
    def vel_mag_norm(pos_seq):
        delta = np.diff(pos_seq, axis=0, prepend=pos_seq[:1])
        mag = np.linalg.norm(delta, axis=1)  # (T,)
        with np.errstate(invalid='ignore', divide='ignore'):
            return mag / bbox_h

    hip_vel = np.diff(hip_center, axis=0, prepend=hip_center[:1])
    head_vel = np.diff(head_pt, axis=0, prepend=head_pt[:1])
    torso_vel = np.diff(torso_center, axis=0, prepend=torso_center[:1])

    with np.errstate(invalid='ignore', divide='ignore'):
        hip_vel_mag_norm = np.linalg.norm(hip_vel, axis=1) / bbox_h
        head_vel_mag_norm = np.linalg.norm(head_vel, axis=1) / bbox_h
        torso_vel_mag_norm = np.linalg.norm(torso_vel, axis=1) / bbox_h

    # 滑窗 position std（归一化到 bbox_h）
    def norm_std(pos_seq, window):
        """对 (T,2) 位置序列分别求 x、y 的滑窗 std，返回合成 std (sqrt(sx^2+sy^2))/bbox_h"""
        sx = rolling_std(pos_seq[:, 0], window)
        sy = rolling_std(pos_seq[:, 1], window)
        combined = np.sqrt(sx ** 2 + sy ** 2)
        return combined / bbox_h

    hip_std_w30 = norm_std(hip_center, 30)
    hip_std_w60 = norm_std(hip_center, 60)
    hip_std_w90 = norm_std(hip_center, 90)
    head_std_w30 = norm_std(head_pt, 30)
    head_std_w60 = norm_std(head_pt, 60)
    head_std_w90 = norm_std(head_pt, 90)

    eps = 1e-6
    head_hip_std_ratio_w60 = head_std_w60 / (hip_std_w60 + eps)
    head_hip_std_ratio_w90 = head_std_w90 / (hip_std_w90 + eps)

    # 头部 Y/X 自相关峰（撞头节奏）
    head_y_autocorr_val, head_y_autocorr_lag = rolling_autocorr_peak(
        head_pt[:, 1], window=90)
    head_x_autocorr_val, head_x_autocorr_lag = rolling_autocorr_peak(
        head_pt[:, 0], window=90)

    # 姿态特征
    head_hip_y_diff = hip_center[:, 1] - head_pt[:, 1]  # 正值=头高于髋
    head_hip_ratio = head_hip_y_diff / bbox_h

    # 骨骼包围框 h/w
    bbox_aspect_hw = np.full(T, np.nan)
    for t in range(T):
        if np.all(np.isnan(scores_ema[t])):
            continue
        valid = scores_ema[t] > SCORE_THRESHOLD
        if valid.sum() < 3:
            continue
        pts = kps_ema[t][valid]
        w_ = max(1.0, pts[:, 0].max() - pts[:, 0].min())
        h_ = max(1.0, pts[:, 1].max() - pts[:, 1].min())
        bbox_aspect_hw[t] = h_ / w_

    # valid kp count
    upper_valid = np.array([int((scores_ema[t][UPPER_KPS] > SCORE_THRESHOLD).sum())
                            if not np.all(np.isnan(scores_ema[t])) else 0
                            for t in range(T)])
    lower_valid = np.array([int((scores_ema[t][LOWER_KPS] > SCORE_THRESHOLD).sum())
                            if not np.all(np.isnan(scores_ema[t])) else 0
                            for t in range(T)])

    # 手-头距离（最小手腕）
    with np.errstate(invalid='ignore'):
        d_lw = np.linalg.norm(l_wrist - head_pt, axis=1)
        d_rw = np.linalg.norm(r_wrist - head_pt, axis=1)
        wrist_head_dist = np.fmin(d_lw, d_rw)
        wrist_head_dist_norm = wrist_head_dist / bbox_h

    # 新指标 A: 多阈值下 head_vel 超阈计数（滑窗 w60 / w90）
    head_vel_thrs = [0.05, 0.08, 0.10, 0.12, 0.15]
    head_vel_exceed_w60 = {}
    head_vel_exceed_w90 = {}
    for thr in head_vel_thrs:
        head_vel_exceed_w60[thr] = rolling_count_exceed(head_vel_mag_norm, thr, 60)
        head_vel_exceed_w90[thr] = rolling_count_exceed(head_vel_mag_norm, thr, 90)

    # 新指标 B: 滑窗内 head_vel / hip_vel 峰值比（>>1 → 扶墙撞头；~1 → 整身撞击；低 → normal）
    head_vel_max_w60 = rolling_max(head_vel_mag_norm, 60)
    hip_vel_max_w60 = rolling_max(hip_vel_mag_norm, 60)
    with np.errstate(invalid='ignore', divide='ignore'):
        head_to_hip_peak_ratio_w60 = head_vel_max_w60 / (hip_vel_max_w60 + eps)

    # 场景上下文：最近他 track 距离 + 最大 bbox overlap（排除 self）
    nearest_dist_norm = np.full(T, np.nan)
    max_overlap = np.full(T, np.nan)
    active_count = np.zeros(T, dtype=int)
    my_center_seq = hip_center
    for t in range(T):
        my_c = my_center_seq[t]
        my_b = bboxes[t]
        n_active = 0
        best_d = np.inf
        best_ov = 0.0
        for other_tid, (other_bboxes, _, _) in all_tracks_dense.items():
            if other_tid == main_tid:
                continue  # 排除 self
            other_b = other_bboxes[t]
            if np.any(np.isnan(other_b)):
                continue
            other_c = np.array([(other_b[0] + other_b[2]) / 2,
                                (other_b[1] + other_b[3]) / 2])
            n_active += 1
            if np.any(np.isnan(my_c)) or np.any(np.isnan(my_b)):
                continue
            d = np.linalg.norm(my_c - other_c)
            if d < best_d:
                best_d = d
            ov = _bbox_overlap(my_b, other_b)
            if ov > best_ov:
                best_ov = ov
        active_count[t] = n_active  # 他 track 数（不含 self）
        if not np.isnan(bbox_h[t]) and best_d < np.inf:
            nearest_dist_norm[t] = best_d / bbox_h[t]
        if n_active > 0:
            max_overlap[t] = best_ov
        else:
            max_overlap[t] = 0.0  # 单人场景显式 0 而非 NaN

    return dict(
        frame_ids=frame_ids,
        bbox_h=bbox_h,
        bbox_xyxy=bboxes,
        kps_ema=kps_ema,
        scores_ema=scores_ema,
        hip_center=hip_center,
        head_pt=head_pt,
        torso_center=torso_center,
        hip_vel_mag_norm=hip_vel_mag_norm,
        head_vel_mag_norm=head_vel_mag_norm,
        torso_vel_mag_norm=torso_vel_mag_norm,
        hip_std_w30=hip_std_w30,
        hip_std_w60=hip_std_w60,
        hip_std_w90=hip_std_w90,
        head_std_w30=head_std_w30,
        head_std_w60=head_std_w60,
        head_std_w90=head_std_w90,
        head_hip_std_ratio_w60=head_hip_std_ratio_w60,
        head_hip_std_ratio_w90=head_hip_std_ratio_w90,
        head_y_autocorr_val=head_y_autocorr_val,
        head_y_autocorr_lag=head_y_autocorr_lag,
        head_x_autocorr_val=head_x_autocorr_val,
        head_x_autocorr_lag=head_x_autocorr_lag,
        head_hip_ratio=head_hip_ratio,
        bbox_aspect_hw=bbox_aspect_hw,
        upper_kp_valid=upper_valid,
        lower_kp_valid=lower_valid,
        wrist_head_dist_norm=wrist_head_dist_norm,
        nearest_dist_norm=nearest_dist_norm,
        max_overlap=max_overlap,
        active_track_count=active_count,
        head_vel_max_w60=head_vel_max_w60,
        hip_vel_max_w60=hip_vel_max_w60,
        head_to_hip_peak_ratio_w60=head_to_hip_peak_ratio_w60,
        **{f'head_vel_exceed_{int(thr*100):03d}_w60': head_vel_exceed_w60[thr]
           for thr in head_vel_thrs},
        **{f'head_vel_exceed_{int(thr*100):03d}_w90': head_vel_exceed_w90[thr]
           for thr in head_vel_thrs},
    )


def _bbox_overlap(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = max(1, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    a2 = max(1, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    return inter / min(a1, a2)


# ============================================================
# PoseC3D 推理（可选）
# ============================================================
def maybe_run_posec3d(kps_ema_main, scores_ema_main, img_shape, config, ckpt,
                     device='cuda:0', stride=16):
    """对主 track 时序每 stride 帧跑一次 PoseC3D。返回 (frame_ids, probs)。
    不做双人配对（探查只关心主 track 语义）。
    """
    sys.path.insert(0, '/home/hzcu/BullyDetection/pyskl')
    import mmcv
    from pyskl.apis import init_recognizer
    from pyskl.datasets.pipelines import Compose
    from mmcv.parallel import collate, scatter
    import torch

    cfg = mmcv.Config.fromfile(config)
    test_pipeline_cfg = cfg.data.test.pipeline
    clip_len = 48
    for step in test_pipeline_cfg:
        if step['type'] == 'UniformSampleFrames':
            clip_len = step['clip_len']
            break
    cfg.data.test.pipeline = [x for x in test_pipeline_cfg if x['type'] != 'DecompressPose']
    model = init_recognizer(cfg, ckpt, device=device)
    model.eval()
    dev = next(model.parameters()).device
    pipeline = Compose(cfg.data.test.pipeline)

    T = len(kps_ema_main)
    probe_frames = list(range(clip_len // 2, T, stride))
    results = []
    with torch.no_grad():
        for end in probe_frames:
            start = max(0, end - clip_len * 4)
            seg_k = kps_ema_main[start:end + 1]
            seg_s = scores_ema_main[start:end + 1]
            valid_mask = ~np.all(np.isnan(seg_s), axis=1)
            if valid_mask.sum() < clip_len // 2:
                continue
            seg_k = seg_k[valid_mask]
            seg_s = seg_s[valid_mask]
            # 均匀采样到 clip_len
            n = len(seg_k)
            if n >= clip_len:
                idxs = np.linspace(0, n - 1, clip_len, dtype=int)
            else:
                idxs = np.concatenate([np.zeros(clip_len - n, dtype=int), np.arange(n)])
            kp_seq = seg_k[idxs]
            sc_seq = seg_s[idxs]
            keypoint = np.zeros((2, clip_len, 17, 2), dtype=np.float32)
            keypoint_score = np.zeros((2, clip_len, 17), dtype=np.float32)
            keypoint[0] = np.nan_to_num(kp_seq)
            keypoint_score[0] = np.nan_to_num(sc_seq)
            fake = dict(
                frame_dir='', label=-1,
                img_shape=img_shape, original_shape=img_shape,
                start_index=0, modality='Pose',
                total_frames=clip_len,
                keypoint=keypoint, keypoint_score=keypoint_score,
            )
            data = pipeline(fake)
            data = collate([data], samples_per_gpu=1)
            data = scatter(data, [dev])[0]
            probs = model(return_loss=False, **data)[0]
            results.append((end, np.asarray(probs, dtype=np.float32)))
    if not results:
        return np.array([], dtype=int), np.zeros((0, 5), dtype=np.float32)
    frames = np.array([r[0] for r in results])
    probs_arr = np.stack([r[1] for r in results])
    return frames, probs_arr


# ============================================================
# 每视频作图
# ============================================================
def plot_trace(feat, out_path, title, posec3d_frames=None, posec3d_probs=None):
    """14 行共享 x 轴的特征时序图（R24 二轮加了 head_vel 超阈计数 + head/hip 比）"""
    T = len(feat['frame_ids'])
    x = feat['frame_ids']
    fig, axes = plt.subplots(14, 1, figsize=(14, 26), sharex=True)

    ax = axes[0]
    ax.plot(x, feat['hip_vel_mag_norm'], label='hip', color='C0', lw=1)
    ax.plot(x, feat['head_vel_mag_norm'], label='head', color='C1', lw=1)
    ax.plot(x, feat['torso_vel_mag_norm'], label='torso', color='C2', lw=1)
    ax.set_ylabel('vel/bbox_h')
    ax.set_title('[1] 速度幅值（归一化）— 撞击看峰形')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(x, feat['hip_std_w60'], label='hip w60', color='C0')
    ax.plot(x, feat['head_std_w60'], label='head w60', color='C1')
    ax.plot(x, feat['hip_std_w90'], label='hip w90', color='C0', ls='--', alpha=0.5)
    ax.plot(x, feat['head_std_w90'], label='head w90', color='C1', ls='--', alpha=0.5)
    ax.set_ylabel('std/bbox_h')
    ax.set_title('[2] 位置滑窗标准差 — 扶墙撞头看 hip 静+head 动')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(x, feat['head_hip_std_ratio_w60'], label='w60', color='C3')
    ax.plot(x, feat['head_hip_std_ratio_w90'], label='w90', color='C4', ls='--')
    ax.axhline(3, color='k', ls=':', alpha=0.4, label='y=3 候选阈值')
    ax.set_ylabel('head_std/hip_std')
    ax.set_title('[3] 头/髋位置变化比 — 扶墙撞头核心指标')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim(0, min(20, np.nanmax(feat['head_hip_std_ratio_w60']) + 2
                       if np.any(~np.isnan(feat['head_hip_std_ratio_w60'])) else 20))
    ax.grid(alpha=0.3)

    ax = axes[3]
    ax.plot(x, feat['head_y_autocorr_val'], label='head Y acf peak', color='C5')
    ax.plot(x, feat['head_x_autocorr_val'], label='head X acf peak', color='C6')
    ax.set_ylabel('acf peak (lag 10-60)')
    ax.set_title('[4] 头部自相关峰值 — 撞头周期性')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[4]
    ax.plot(x, feat['head_y_autocorr_lag'], label='head Y peak lag', color='C5')
    ax.plot(x, feat['head_x_autocorr_lag'], label='head X peak lag', color='C6')
    ax.set_ylabel('lag (frames)')
    ax.set_title('[5] 周期性峰值对应 lag（帧数）')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[5]
    ax.plot(x, feat['bbox_aspect_hw'], label='bbox h/w', color='C7')
    ax.axhline(1.1, color='k', ls=':', alpha=0.4, label='h/w=1.1')
    ax2 = ax.twinx()
    ax2.plot(x, feat['head_hip_ratio'], label='head_hip / bbox_h', color='C8', alpha=0.7)
    ax.set_ylabel('h/w')
    ax2.set_ylabel('head_hip_ratio')
    ax.set_title('[6] 姿态：骨骼 h/w + 头-髋 Y 差')
    lns = ax.get_lines() + ax2.get_lines()
    ax.legend(lns, [l.get_label() for l in lns], loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[6]
    ax.plot(x, feat['upper_kp_valid'], label='upper (0-10)', color='C0')
    ax.plot(x, feat['lower_kp_valid'], label='lower (13-16)', color='C3')
    ax.set_ylabel('# valid kp')
    ax.set_title('[7] 关键点有效数 — 遮挡/裁切诊断')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[7]
    ax.plot(x, feat['wrist_head_dist_norm'], color='C9')
    ax.set_ylabel('dist/bbox_h')
    ax.set_title('[8] 最近手腕到头距离 — 扶墙/捂头线索')
    ax.grid(alpha=0.3)

    ax = axes[8]
    ax.plot(x, feat['nearest_dist_norm'], label='nearest / bbox_h', color='C4')
    ax.axhline(1.5, color='k', ls=':', alpha=0.4, label='y=1.5 proximity')
    ax.set_ylabel('nearest dist/bbox_h')
    ax.set_title('[9] 最近其他 track 距离（单人撞墙 → 大 / ∞）')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[9]
    ax.plot(x, feat['max_overlap'], color='C2')
    ax.axhline(0.1, color='k', ls=':', alpha=0.4, label='y=0.1 overlap')
    ax.set_ylabel('max bbox overlap')
    ax.set_title('[10] 与其他 track 最大 bbox 重叠（单人 → 0）')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[10]
    ax.plot(x, feat['active_track_count'], color='C5')
    ax.set_ylabel('# other tracks')
    ax.set_title('[11] 同帧其他 track 数量（已排除 self）')
    ax.grid(alpha=0.3)

    # [12] 新指标 A — head_vel 多阈值超阈计数（w90 窗口）
    ax = axes[11]
    for thr, color in [(0.05, 'C0'), (0.08, 'C1'), (0.10, 'C3'), (0.12, 'C5'), (0.15, 'C4')]:
        key = f'head_vel_exceed_{int(thr*100):03d}_w90'
        if key in feat:
            ax.plot(x, feat[key], label=f'>{thr:.2f}', color=color, lw=1)
    ax.set_ylabel('# frames > thr in w=90')
    ax.set_title('[12] head_vel 多阈值累计超阈数 — 反复撞头关键')
    ax.legend(loc='upper right', fontsize=8, ncol=5)
    ax.grid(alpha=0.3)

    # [13] 新指标 B — head/hip peak ratio (w60)
    ax = axes[12]
    ax.plot(x, feat['head_to_hip_peak_ratio_w60'], color='C6', lw=1)
    ax.set_ylabel('head_max / hip_max (w60)')
    ax.set_title('[13] 头/髋峰值速度比 — 扶墙撞头 >> 1，整身撞击 ~1，normal 低')
    ax.set_ylim(0, min(40, np.nanmax(feat['head_to_hip_peak_ratio_w60']) + 2
                       if np.any(~np.isnan(feat['head_to_hip_peak_ratio_w60'])) else 40))
    ax.grid(alpha=0.3)

    ax = axes[13]
    if posec3d_frames is not None and len(posec3d_frames) > 0:
        labels = ['normal', 'fighting', 'bullying', 'falling', 'climbing']
        for i, lbl in enumerate(labels):
            ax.plot(posec3d_frames, posec3d_probs[:, i], marker='.', label=lbl, lw=1)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', fontsize=8, ncol=5)
    ax.set_ylabel('PoseC3D prob')
    ax.set_title('[14] PoseC3D 5 类概率（若跑了）')
    ax.set_xlabel('frame index')
    ax.grid(alpha=0.3)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_path, dpi=90)
    plt.close(fig)


# ============================================================
# 单视频摘要
# ============================================================
def compute_summary(feat, meta, video_path, class_name, main_tid, main_frames):
    """提取峰值、候选事件窗、分布关键分位数。"""
    def safe(arr, f):
        a = arr[~np.isnan(arr)]
        return float(f(a)) if len(a) else None

    def peak_info(arr):
        if np.all(np.isnan(arr)):
            return {'peak': None, 'peak_frame': None}
        idx = int(np.nanargmax(arr))
        return {'peak': float(arr[idx]), 'peak_frame': int(feat['frame_ids'][idx])}

    def pct(arr, q):
        a = arr[~np.isnan(arr)]
        return float(np.percentile(a, q)) if len(a) else None

    # 撞击候选：hip_vel 峰 + 其后 3 帧衰减
    impact_events = []
    hv = feat['hip_vel_mag_norm']
    if not np.all(np.isnan(hv)):
        # 找所有高于 90% 分位数的局部峰
        p90 = np.nanpercentile(hv, 90)
        for t in range(2, len(hv) - 3):
            if np.isnan(hv[t]) or hv[t] < p90:
                continue
            if hv[t] >= hv[t - 1] and hv[t] >= hv[t + 1] and hv[t] > 0:
                peak = hv[t]
                after = hv[t + 1:t + 4]
                decay_ratio = np.nanmin(after) / peak if np.any(~np.isnan(after)) else None
                impact_events.append({
                    'frame': int(feat['frame_ids'][t]),
                    'peak_hip_vel_norm': float(peak),
                    'min_in_next3_frames': float(np.nanmin(after)) if decay_ratio else None,
                    'decay_ratio': float(decay_ratio) if decay_ratio else None,
                })
        # 保留前 10 个最高峰事件
        impact_events = sorted(impact_events, key=lambda e: -e['peak_hip_vel_norm'])[:10]

    # 扶墙撞头候选：head_hip_std_ratio w60 高 + head_y_autocorr 高
    ratio = feat['head_hip_std_ratio_w60']
    acf = feat['head_y_autocorr_val']
    headbang_score = np.where(
        (~np.isnan(ratio)) & (~np.isnan(acf)),
        ratio * np.clip(acf, 0, 1),
        np.nan,
    )
    hb_peak = peak_info(headbang_score)

    return dict(
        video=str(video_path),
        class_name=class_name,
        fps=meta['fps'],
        frame_count=meta['frame_count'],
        resolution=[meta['width'], meta['height']],
        main_track_id=int(main_tid),
        main_track_frames=int(main_frames),

        hip_vel_mag_norm=peak_info(feat['hip_vel_mag_norm']),
        head_vel_mag_norm=peak_info(feat['head_vel_mag_norm']),
        torso_vel_mag_norm=peak_info(feat['torso_vel_mag_norm']),

        head_hip_std_ratio_w60=peak_info(feat['head_hip_std_ratio_w60']),
        head_hip_std_ratio_w90=peak_info(feat['head_hip_std_ratio_w90']),
        head_y_autocorr_val=peak_info(feat['head_y_autocorr_val']),
        head_x_autocorr_val=peak_info(feat['head_x_autocorr_val']),

        wrist_head_dist_norm_median=safe(feat['wrist_head_dist_norm'], np.median),
        nearest_dist_norm_median=safe(feat['nearest_dist_norm'], np.median),
        max_overlap_max=safe(feat['max_overlap'], np.max),
        active_track_count_median=safe(feat['active_track_count'], np.median),
        active_track_count_max=safe(feat['active_track_count'], np.max),

        # 新指标：分位数（比 peak 稳健）
        hip_vel_p95=pct(feat['hip_vel_mag_norm'], 95),
        hip_vel_p99=pct(feat['hip_vel_mag_norm'], 99),
        head_vel_p95=pct(feat['head_vel_mag_norm'], 95),
        head_vel_p99=pct(feat['head_vel_mag_norm'], 99),

        # 新指标：多阈值下 head_vel 超阈最大计数（在 w60/w90 滑窗里）
        head_vel_exceed_max={
            f'thr_{int(thr*100):03d}': {
                'w60': safe(feat[f'head_vel_exceed_{int(thr*100):03d}_w60'], np.max),
                'w90': safe(feat[f'head_vel_exceed_{int(thr*100):03d}_w90'], np.max),
            } for thr in [0.05, 0.08, 0.10, 0.12, 0.15]
        },

        # 新指标：head/hip peak 比（滑窗 w60）
        head_to_hip_peak_ratio_w60=peak_info(feat['head_to_hip_peak_ratio_w60']),

        impact_event_candidates=impact_events,
        headbang_score_peak=hb_peak,
    )


# ============================================================
# 跨视频聚合
# ============================================================
def aggregate_class(class_summaries, class_features, class_name, out_dir):
    """单类样本特征叠加 + 分布"""
    if not class_summaries:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # 峰对齐速度叠加（撞击类的核心图）
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(f'{class_name}: hip_vel_mag_norm (peak-aligned)')
    for s, f in zip(class_summaries, class_features):
        peak_f = s['hip_vel_mag_norm']['peak_frame']
        if peak_f is None:
            continue
        x = f['frame_ids']
        y = f['hip_vel_mag_norm']
        # 对齐：x 轴 = frame - peak_frame
        ax.plot(x - peak_f, y, alpha=0.5, lw=1,
                label=Path(s['video']).stem[:20])
    ax.axvline(0, color='k', ls=':', alpha=0.4)
    ax.set_xlim(-60, 60)
    ax.set_xlabel('frame offset from hip_vel peak')
    ax.set_ylabel('hip_vel_mag_norm')
    ax.grid(alpha=0.3)
    if len(class_summaries) <= 10:
        ax.legend(loc='upper right', fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / f'overlay_{class_name}_hip_vel_peak_aligned.png', dpi=90)
    plt.close(fig)

    # head_hip_std_ratio 叠加（扶墙撞头核心图）
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(f'{class_name}: head_hip_std_ratio_w60')
    for s, f in zip(class_summaries, class_features):
        x = f['frame_ids']
        y = f['head_hip_std_ratio_w60']
        ax.plot(x - x[0], y, alpha=0.5, lw=1, label=Path(s['video']).stem[:20])
    ax.axhline(3, color='k', ls=':', alpha=0.4)
    ax.set_xlabel('frame (from video start)')
    ax.set_ylabel('head_std / hip_std (w60)')
    ax.set_ylim(0, 20)
    ax.grid(alpha=0.3)
    if len(class_summaries) <= 10:
        ax.legend(loc='upper right', fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / f'overlay_{class_name}_head_hip_std_ratio.png', dpi=90)
    plt.close(fig)


def plot_cross_class_comparison(all_summaries, out_dir):
    """关键特征跨类直方图"""
    classes = sorted(set(s['class_name'] for s in all_summaries))
    features_to_compare = [
        ('hip_vel_mag_norm', 'peak'),
        ('head_vel_mag_norm', 'peak'),
        ('head_hip_std_ratio_w60', 'peak'),
        ('head_y_autocorr_val', 'peak'),
        ('max_overlap_max', None),
        ('nearest_dist_norm_median', None),
        ('wrist_head_dist_norm_median', None),
        ('active_track_count_max', None),
        # 新指标
        ('hip_vel_p95', None),
        ('hip_vel_p99', None),
        ('head_vel_p95', None),
        ('head_vel_p99', None),
        ('head_to_hip_peak_ratio_w60', 'peak'),
    ]

    def get_val(s, name, sub):
        v = s.get(name)
        if v is None:
            return None
        if sub is None:
            return v
        return v.get(sub) if isinstance(v, dict) else None

    for feat_name, sub in features_to_compare:
        fig, ax = plt.subplots(figsize=(8, 5))
        for cls in classes:
            vals = []
            for s in all_summaries:
                if s['class_name'] != cls:
                    continue
                v = get_val(s, feat_name, sub)
                if v is not None:
                    vals.append(v)
            if vals:
                ax.hist(vals, bins=min(20, max(5, len(vals))),
                        alpha=0.5, label=f'{cls} (n={len(vals)})')
        ax.set_title(feat_name + (f'.{sub}' if sub else ''))
        ax.set_xlabel(feat_name)
        ax.set_ylabel('count')
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f'comparison_{feat_name}.png', dpi=90)
        plt.close(fig)

    # head_vel 多阈值超阈计数对比（每个阈值一张 w90 图）
    for thr_key in ('thr_005', 'thr_008', 'thr_010', 'thr_012', 'thr_015'):
        for win in ('w60', 'w90'):
            fig, ax = plt.subplots(figsize=(8, 5))
            for cls in classes:
                vals = []
                for s in all_summaries:
                    if s['class_name'] != cls:
                        continue
                    em = s.get('head_vel_exceed_max')
                    if em and thr_key in em and em[thr_key].get(win) is not None:
                        vals.append(em[thr_key][win])
                if vals:
                    ax.hist(vals, bins=min(20, max(5, len(vals))),
                            alpha=0.5, label=f'{cls} (n={len(vals)})')
            ax.set_title(f'head_vel_exceed_{thr_key}_{win} (max count)')
            ax.set_xlabel('max frames exceeding in window')
            ax.set_ylabel('count')
            ax.legend()
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_dir / f'comparison_head_vel_exceed_{thr_key}_{win}.png', dpi=90)
            plt.close(fig)


def write_distribution_stats(all_summaries, out_path):
    """把各类特征的 p5/p50/p95 写进 json，方便直接定阈值"""
    classes = sorted(set(s['class_name'] for s in all_summaries))

    def get_val(s, key, sub):
        v = s.get(key)
        if v is None:
            return None
        if sub is None:
            return v
        return v.get(sub) if isinstance(v, dict) else None

    def describe(vals):
        return {
            'n': len(vals),
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals)),
            'p5': float(np.percentile(vals, 5)),
            'p50': float(np.percentile(vals, 50)),
            'p95': float(np.percentile(vals, 95)),
            'min': float(np.min(vals)),
            'max': float(np.max(vals)),
        }

    stats = {}
    base_keys = [
        ('hip_vel_mag_norm', 'peak'),
        ('head_vel_mag_norm', 'peak'),
        ('head_hip_std_ratio_w60', 'peak'),
        ('head_hip_std_ratio_w90', 'peak'),
        ('head_y_autocorr_val', 'peak'),
        ('head_x_autocorr_val', 'peak'),
        ('max_overlap_max', None),
        ('nearest_dist_norm_median', None),
        ('wrist_head_dist_norm_median', None),
        ('active_track_count_median', None),
        ('active_track_count_max', None),
        # R24 二轮新指标
        ('hip_vel_p95', None),
        ('hip_vel_p99', None),
        ('head_vel_p95', None),
        ('head_vel_p99', None),
        ('head_to_hip_peak_ratio_w60', 'peak'),
    ]
    for cls in classes:
        subset = [s for s in all_summaries if s['class_name'] == cls]
        cls_stats = {}
        for key, sub in base_keys:
            vals = [get_val(s, key, sub) for s in subset]
            vals = [v for v in vals if v is not None]
            if vals:
                cls_stats[f'{key}.{sub}' if sub else key] = describe(vals)
        # head_vel 超阈最大计数
        for thr_key in ('thr_005', 'thr_008', 'thr_010', 'thr_012', 'thr_015'):
            for win in ('w60', 'w90'):
                vals = []
                for s in subset:
                    em = s.get('head_vel_exceed_max')
                    if em and thr_key in em and em[thr_key].get(win) is not None:
                        vals.append(em[thr_key][win])
                if vals:
                    cls_stats[f'head_vel_exceed_{thr_key}_{win}_max'] = describe(vals)
        stats[cls] = {'sample_count': len(subset), 'features': cls_stats}
    with open(out_path, 'w') as f:
        json.dump(stats, f, indent=2)


# ============================================================
# 主流程
# ============================================================
def process_one_video(video_path, class_name, yolo_model, args, out_root):
    t_start = time.time()
    print(f'\n[{class_name}] {video_path}')

    tracks, meta = extract_tracks(video_path, yolo_model, yolo_conf=args.yolo_conf)
    if not tracks:
        print(f'  无 track,跳过')
        return None, None

    # 选主 track = 帧数最多
    main_tid = max(tracks.keys(), key=lambda tid: len(tracks[tid]))
    main_entries = tracks[main_tid]
    main_frames = len(main_entries)
    print(f'  主 track T{main_tid}: {main_frames} 帧')

    # 密集化 + EMA
    g_start = min(e['frame_idx'] for e in main_entries)
    g_end = max(e['frame_idx'] for e in main_entries)

    all_tracks_dense = {}
    for tid, entries in tracks.items():
        frame_ids, bboxes, kps, scores, present = densify_track(entries, g_start, g_end)
        kps_ema, scores_ema = apply_ema(kps, scores)
        all_tracks_dense[tid] = (bboxes, kps_ema, scores_ema)

    main_bboxes, main_kps_ema, main_scores_ema = all_tracks_dense[main_tid]
    main_frame_ids = np.arange(g_start, g_end + 1)

    feat = compute_features(main_frame_ids, main_bboxes, main_kps_ema, main_scores_ema,
                            all_tracks_dense, main_tid=main_tid)

    # PoseC3D（可选）
    pc_frames, pc_probs = None, None
    if args.with_posec3d:
        try:
            pc_frames, pc_probs = maybe_run_posec3d(
                main_kps_ema, main_scores_ema,
                (meta['height'], meta['width']),
                args.posec3d_config, args.posec3d_ckpt,
                device=args.device, stride=args.posec3d_stride,
            )
        except Exception as e:
            print(f'  [WARN] PoseC3D 失败: {e}')

    summary = compute_summary(feat, meta, video_path, class_name, main_tid, main_frames)
    if pc_probs is not None and len(pc_probs) > 0:
        summary['posec3d_mean_probs'] = {
            lbl: float(pc_probs[:, i].mean())
            for i, lbl in enumerate(['normal', 'fighting', 'bullying', 'falling', 'climbing'])
        }

    # 输出
    vid_stem = Path(video_path).stem
    vid_out = out_root / 'per_video' / f'{class_name}_{vid_stem}'
    vid_out.mkdir(parents=True, exist_ok=True)

    npz_path = vid_out / 'features.npz'
    np.savez_compressed(
        npz_path,
        **{k: v for k, v in feat.items() if isinstance(v, np.ndarray)},
        posec3d_frames=np.asarray(pc_frames) if pc_frames is not None else np.array([]),
        posec3d_probs=np.asarray(pc_probs) if pc_probs is not None else np.zeros((0, 5)),
    )

    title = f'{class_name} / {vid_stem} / T{main_tid} / {main_frames} frames @ {meta["fps"]:.0f}fps'
    plot_trace(feat, vid_out / 'trace.png', title, pc_frames, pc_probs)

    with open(vid_out / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    dt = time.time() - t_start
    print(f'  完成,耗时 {dt:.1f}s → {vid_out}')
    return summary, feat


def main():
    parser = argparse.ArgumentParser(description='撞墙 / 头撞墙 特征探查脚本')
    parser.add_argument('--videos-dir', required=True,
                        help='视频目录（每个子目录一个类别）')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--yolo-pose', default='yolo11m-pose.pt')
    parser.add_argument('--yolo-conf', type=float, default=0.3)

    parser.add_argument('--with-posec3d', action='store_true',
                        help='同时跑 PoseC3D 推理（慢但能看语义概率）')
    parser.add_argument('--posec3d-config', default=None)
    parser.add_argument('--posec3d-ckpt', default=None)
    parser.add_argument('--posec3d-stride', type=int, default=16)
    parser.add_argument('--device', default='cuda:0')

    parser.add_argument('--limit', type=int, default=0,
                        help='每类最多处理多少个（0=全部）')
    parser.add_argument('--exts', default='mp4,mov,avi,mkv',
                        help='视频扩展名（逗号分隔）')
    args = parser.parse_args()

    if args.with_posec3d and not (args.posec3d_config and args.posec3d_ckpt):
        parser.error('--with-posec3d 需同时指定 --posec3d-config 和 --posec3d-ckpt')

    videos_dir = Path(args.videos_dir)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # 类别 = 第一层子目录名
    exts = {'.' + e.strip().lower().lstrip('.') for e in args.exts.split(',') if e.strip()}
    class_videos = defaultdict(list)
    for sub in sorted(videos_dir.iterdir()):
        if not sub.is_dir():
            continue
        cls = sub.name
        for v in sorted(sub.iterdir()):
            if v.suffix.lower() in exts:
                class_videos[cls].append(v)

    if not class_videos:
        # 允许单层：直接把所有视频标为 "unlabeled"
        for v in sorted(videos_dir.iterdir()):
            if v.suffix.lower() in exts:
                class_videos['unlabeled'].append(v)

    if not class_videos:
        print('[FATAL] 未找到任何视频', file=sys.stderr)
        sys.exit(1)

    print(f'Classes: {dict((c, len(v)) for c, v in class_videos.items())}')

    print(f'Loading YOLO-Pose: {args.yolo_pose}')
    yolo_model = YOLO(args.yolo_pose)

    all_summaries = []
    class_to_features = defaultdict(list)
    class_to_summaries = defaultdict(list)

    for cls, vids in class_videos.items():
        if args.limit > 0:
            vids = vids[:args.limit]
        for vp in vids:
            s, f = process_one_video(vp, cls, yolo_model, args, out_root)
            if s is None:
                continue
            all_summaries.append(s)
            class_to_features[cls].append(f)
            class_to_summaries[cls].append(s)

    if not all_summaries:
        print('[FATAL] 所有视频都没产出摘要')
        sys.exit(1)

    agg_out = out_root / 'aggregate'
    agg_out.mkdir(parents=True, exist_ok=True)

    for cls in class_to_features:
        aggregate_class(class_to_summaries[cls], class_to_features[cls], cls, agg_out)

    plot_cross_class_comparison(all_summaries, agg_out)
    write_distribution_stats(all_summaries, agg_out / 'distribution.json')

    with open(out_root / 'index.json', 'w') as f:
        json.dump({
            'video_count': len(all_summaries),
            'classes': {c: len(v) for c, v in class_to_summaries.items()},
            'summaries': all_summaries,
        }, f, indent=2, ensure_ascii=False)

    print(f'\n全部完成 → {out_root}')
    print(f'  per_video/   每个视频的 features.npz + trace.png + summary.json')
    print(f'  aggregate/   跨视频叠加图 + comparison + distribution.json')
    print(f'  index.json   整体目录')


if __name__ == '__main__':
    main()
