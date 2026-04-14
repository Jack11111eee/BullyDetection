"""
main_inference.py — 校园安防视频行为感知系统 主推理循环

Pipeline:
  视频帧 → YOLO11m-Pose + ByteTrack → 骨骼缓冲(48帧) → PoseC3D(每16帧) → 规则引擎 → 可视化/告警

Usage:
  python main_inference.py --video path/to/video.mp4 \
    --pose-config pyskl/configs/posec3d/finetune_campus_v3.py \
    --pose-checkpoint pyskl/work_dirs/posec3d_campus_v6/best_top1_acc_epoch_50.pth \
    --show --output out.mp4
"""

import argparse
import time
from collections import defaultdict

import cv2
import mmcv
import numpy as np
import torch
from ultralytics import YOLO

from pyskl.apis import init_recognizer
from pyskl.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter

from rule_engine import RuleEngine, POSE_CLASSES, FINAL_CLASSES


# ============================================================
# COCO 17 骨骼连线定义（用于可视化）
# ============================================================
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),       # 头部
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 上肢
    (5, 11), (6, 12), (11, 12),             # 躯干
    (11, 13), (13, 15), (12, 14), (14, 16), # 下肢
]

# 行为 → 颜色 (BGR)
LABEL_COLORS = {
    'normal':     (0, 200, 0),     # 绿
    'fighting':   (0, 0, 255),     # 红
    'bullying':   (0, 0, 200),     # 深红
    'falling':    (0, 140, 255),   # 橙
    'climbing':   (0, 255, 255),   # 黄
    'vandalism':  (255, 0, 255),   # 紫
    'smoking':    (255, 100, 0),   # 蓝紫
    'phone_call': (255, 200, 0),   # 浅蓝
}


# ============================================================
# 骨骼缓冲区
# ============================================================
class SkeletonBuffer:
    """
    按 track_id 缓存最近 CLIP_LEN 帧的骨骼关键点。
    每累积 STRIDE 新帧触发一次 PoseC3D 推理。
    """

    def __init__(self, clip_len=48, stride=16, max_person=2):
        self.clip_len = clip_len
        self.stride = stride
        self.max_person = max_person

        # {track_id: {'kps': list of (17,2), 'scores': list of (17,), 'count': int}}
        self.tracks = defaultdict(lambda: {
            'kps': [],
            'scores': [],
            'new_frames': 0,
        })

    def update(self, track_id, keypoints_17x2, scores_17):
        """添加一帧骨骼数据"""
        buf = self.tracks[track_id]
        buf['kps'].append(keypoints_17x2.copy())
        buf['scores'].append(scores_17.copy())
        buf['new_frames'] += 1

        # 只保留最近 clip_len 帧
        if len(buf['kps']) > self.clip_len:
            buf['kps'] = buf['kps'][-self.clip_len:]
            buf['scores'] = buf['scores'][-self.clip_len:]

    def should_infer(self, track_id):
        """是否该触发推理"""
        buf = self.tracks[track_id]
        return buf['new_frames'] >= self.stride and len(buf['kps']) >= self.clip_len

    def get_clip(self, track_id):
        """
        获取用于 PoseC3D 推理的数据。

        Returns:
            keypoint: (1, clip_len, 17, 2) — 只用 1 个人（当前 track）
            keypoint_score: (1, clip_len, 17)
        """
        buf = self.tracks[track_id]
        buf['new_frames'] = 0  # 重置计数

        kps_list = buf['kps'][-self.clip_len:]
        scores_list = buf['scores'][-self.clip_len:]

        # 补零到 clip_len（track 刚出现时可能不足）
        n = len(kps_list)
        keypoint = np.zeros((1, self.clip_len, 17, 2), dtype=np.float32)
        keypoint_score = np.zeros((1, self.clip_len, 17), dtype=np.float32)

        offset = self.clip_len - n
        for i in range(n):
            keypoint[0, offset + i] = kps_list[i]
            keypoint_score[0, offset + i] = scores_list[i]

        return keypoint, keypoint_score

    def get_active_track_ids(self):
        return set(self.tracks.keys())

    def remove_stale(self, active_ids, max_missing=30):
        """移除已消失的 track"""
        stale = [tid for tid in self.tracks if tid not in active_ids]
        for tid in stale:
            del self.tracks[tid]


# ============================================================
# PoseC3D 推理封装
# ============================================================
class PoseC3DInferencer:
    """封装 PoseC3D 模型加载和推理"""

    def __init__(self, config_path, checkpoint_path, device='cuda:0'):
        config = mmcv.Config.fromfile(config_path)

        # 移除 DecompressPose（只有从压缩 pkl 加载时才需要）
        config.data.test.pipeline = [
            x for x in config.data.test.pipeline
            if x['type'] != 'DecompressPose'
        ]

        self.model = init_recognizer(config, checkpoint_path, device=device)
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.pipeline = Compose(config.data.test.pipeline)
        self.num_classes = config.model.cls_head.num_classes

    @torch.no_grad()
    def infer(self, keypoint, keypoint_score, img_shape):
        """
        Args:
            keypoint: (M, T, 17, 2) 像素坐标
            keypoint_score: (M, T, 17)
            img_shape: (H, W)

        Returns:
            probs: np.array (num_classes,) 各类概率
        """
        M, T = keypoint.shape[:2]

        fake_anno = dict(
            frame_dir='',
            label=-1,
            img_shape=img_shape,
            original_shape=img_shape,
            start_index=0,
            modality='Pose',
            total_frames=T,
            keypoint=keypoint.astype(np.float32),
            keypoint_score=keypoint_score.astype(np.float32),
        )

        data = self.pipeline(fake_anno)
        data = collate([data], samples_per_gpu=1)
        data = scatter(data, [self.device])[0]

        probs = self.model(return_loss=False, **data)[0]  # (num_classes,)
        return probs


# ============================================================
# YOLO 小物体检测封装（可选）
# ============================================================
class SmallObjectDetector:
    """封装 YOLO11s 小物体检测"""

    def __init__(self, model_path, class_map=None, conf=0.3, imgsz=1280):
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz
        self.class_map = class_map or {0: 'cigarette', 1: 'phone'}

    def detect(self, frame):
        """
        Returns:
            list of {'class': str, 'bbox': [x1,y1,x2,y2], 'conf': float}
        """
        results = self.model(frame, conf=self.conf, imgsz=self.imgsz, verbose=False)
        detections = []
        for box in results[0].boxes:
            cls_id = int(box.cls.item())
            cls_name = self.class_map.get(cls_id)
            if cls_name is None:
                continue
            detections.append({
                'class': cls_name,
                'bbox': box.xyxy[0].tolist(),
                'conf': float(box.conf.item()),
            })
        return detections


# ============================================================
# 可视化
# ============================================================
def draw_skeleton(frame, kps, scores, color=(0, 255, 0), threshold=0.3):
    """在帧上画骨骼"""
    for i, (x, y) in enumerate(kps):
        if scores[i] < threshold:
            continue
        cv2.circle(frame, (int(x), int(y)), 3, color, -1)

    for (i, j) in SKELETON_EDGES:
        if scores[i] < threshold or scores[j] < threshold:
            continue
        pt1 = (int(kps[i][0]), int(kps[i][1]))
        pt2 = (int(kps[j][0]), int(kps[j][1]))
        cv2.line(frame, pt1, pt2, color, 2)


def draw_label(frame, bbox, label, confidence, color):
    """在 bbox 上方画标签"""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    text = f'{label} {confidence:.0%}'
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, text, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def draw_info(frame, fps, frame_idx):
    """画帧信息"""
    text = f'Frame: {frame_idx}  FPS: {fps:.1f}'
    cv2.putText(frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


# ============================================================
# 主推理循环
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Campus Safety Inference Pipeline')
    parser.add_argument('--video', required=True, help='输入视频路径，或 0 表示摄像头')
    parser.add_argument('--pose-config', required=True, help='PoseC3D 配置文件路径')
    parser.add_argument('--pose-checkpoint', required=True, help='PoseC3D 权重路径')
    parser.add_argument('--yolo-pose-model', default='yolo11m-pose.pt', help='YOLO Pose 模型路径')
    parser.add_argument('--small-obj-model', default=None, help='YOLO11s 小物体模型路径（可选）')
    parser.add_argument('--device', default='cuda:0', help='推理设备')
    parser.add_argument('--conf', type=float, default=0.3, help='YOLO 检测置信度阈值')
    parser.add_argument('--show', action='store_true', help='实时显示')
    parser.add_argument('--output', default=None, help='输出视频路径')
    parser.add_argument('--vote-window', type=int, default=5, help='时序投票窗口')
    parser.add_argument('--vote-ratio', type=float, default=0.6, help='投票通过比例')
    args = parser.parse_args()

    # ---------- 加载模型 ----------
    print('[1/3] Loading YOLO Pose model...')
    yolo_pose = YOLO(args.yolo_pose_model)

    print('[2/3] Loading PoseC3D model...')
    posec3d = PoseC3DInferencer(args.pose_config, args.pose_checkpoint, device=args.device)

    small_obj_detector = None
    if args.small_obj_model:
        print('[2.5/3] Loading small object model...')
        small_obj_detector = SmallObjectDetector(args.small_obj_model)

    print('[3/3] Initializing pipeline...')
    skeleton_buf = SkeletonBuffer(clip_len=48, stride=16)
    rule_engine = RuleEngine(
        pose_threshold=0.5,
        vote_window=args.vote_window,
        vote_ratio=args.vote_ratio,
    )

    # ---------- 打开视频 ----------
    source = 0 if args.video == '0' else args.video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f'Error: cannot open video {args.video}')
        return

    fps_video = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    img_shape = (h, w)

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps_video, (w, h))

    # ---------- 每个 track 的最新行为判定（用于持续显示） ----------
    track_labels = {}  # {track_id: {'label': str, 'confidence': float, ...}}

    print(f'\nRunning inference on {args.video} ({w}x{h} @ {fps_video:.0f}fps)')
    print(f'PoseC3D: {posec3d.num_classes} classes, stride=16, clip_len=48')
    print(f'Vote smoothing: window={args.vote_window}, ratio={args.vote_ratio}')
    print('Press Q to quit.\n')

    frame_idx = 0
    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ===== Step 1: YOLO Pose + ByteTrack =====
        results = yolo_pose.track(
            source=frame,
            persist=True,
            tracker='bytetrack.yaml',
            conf=args.conf,
            iou=0.5,
            verbose=False,
        )

        result = results[0]
        current_track_ids = set()
        frame_persons_kps = []   # 当前帧所有人的关键点（给规则引擎判断人数）

        if result.boxes is not None and result.keypoints is not None:
            for i, box in enumerate(result.boxes):
                if box.id is None:
                    continue

                track_id = int(box.id.item())
                current_track_ids.add(track_id)

                kps_data = result.keypoints.data[i].cpu().numpy()  # (17, 3)
                kps_xy = kps_data[:, :2]   # (17, 2)
                kps_sc = kps_data[:, 2]    # (17,)

                frame_persons_kps.append(kps_xy)

                # 更新骨骼缓冲区
                skeleton_buf.update(track_id, kps_xy, kps_sc)

                # 画骨骼
                label_info = track_labels.get(track_id, {'label': 'normal'})
                color = LABEL_COLORS.get(label_info['label'], (200, 200, 200))
                draw_skeleton(frame, kps_xy, kps_sc, color=color)

                # ===== Step 2: PoseC3D 推理（每 stride 帧） =====
                if skeleton_buf.should_infer(track_id):
                    keypoint, keypoint_score = skeleton_buf.get_clip(track_id)
                    pose_probs = posec3d.infer(keypoint, keypoint_score, img_shape)

                    # ===== Step 3: 小物体检测 =====
                    small_objs = []
                    if small_obj_detector is not None:
                        small_objs = small_obj_detector.detect(frame)

                    # ===== Step 4: 规则引擎 =====
                    judgment = rule_engine.judge(
                        track_id=track_id,
                        pose_probs=pose_probs,
                        person_kps=kps_xy,
                        person_scores=kps_sc,
                        all_person_kps=frame_persons_kps,
                        small_obj_detections=small_objs,
                        img_shape=img_shape,
                    )
                    track_labels[track_id] = judgment

                # ===== Step 5: 可视化标签 =====
                if track_id in track_labels:
                    info = track_labels[track_id]
                    bbox = box.xyxy[0].tolist()
                    color = LABEL_COLORS.get(info['label'], (200, 200, 200))
                    draw_label(frame, bbox, info['label'], info['confidence'], color)

        # 清理消失的 track
        skeleton_buf.remove_stale(current_track_ids)
        rule_engine.clear_stale_tracks(current_track_ids)
        stale_tracks = [tid for tid in track_labels if tid not in current_track_ids]
        for tid in stale_tracks:
            del track_labels[tid]

        # FPS 信息
        elapsed = time.time() - t_start
        current_fps = (frame_idx + 1) / elapsed if elapsed > 0 else 0
        draw_info(frame, current_fps, frame_idx)

        # 输出
        if writer:
            writer.write(frame)
        if args.show:
            cv2.imshow('Campus Safety', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_idx += 1

    # 清理
    cap.release()
    if writer:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    total_time = time.time() - t_start
    print(f'\nDone. {frame_idx} frames in {total_time:.1f}s ({frame_idx/total_time:.1f} fps)')


if __name__ == '__main__':
    main()
