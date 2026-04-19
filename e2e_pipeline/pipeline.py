"""
pipeline.py — 校园安防视频行为感知系统 核心推理循环

Pipeline:
  InputSource → YOLO11m-Pose + ByteTrack → SkeletonBuffer → PoseC3D → RuleEngine → 可视化/告警

重构自 main_inference.py，改进：
  1. clip_len 从 PoseC3D config 自动读取（不再硬编码48）
  2. 使用 InputSource 统一视频/摄像头/帧文件夹输入
  3. 支持保存 JSON 检测日志
  4. 结构化 BehaviorResult 输出
"""

import json
import logging
import sys
import time
from collections import defaultdict

import cv2
import numpy as np
import torch

logger = logging.getLogger('e2e_debug')

sys.path.insert(0, '/home/hzcu/BullyDetection/pyskl')

import mmcv
from pyskl.apis import init_recognizer
from pyskl.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
from ultralytics import YOLO

from input_source import InputSource
from rule_engine import RuleEngine, BehaviorResult, POSE_CLASSES, FINAL_CLASSES


# ============================================================
# COCO 17 骨骼连线（可视化用）
# ============================================================
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

LABEL_COLORS = {
    'normal':     (0, 200, 0),
    'fighting':   (0, 0, 255),
    'bullying':   (0, 0, 200),
    'falling':    (0, 140, 255),
    'climbing':   (0, 255, 255),
    'vandalism':  (255, 0, 255),
    'smoking':    (255, 100, 0),
    'phone_call': (255, 200, 0),
    'loitering':  (200, 200, 0),
}


# ============================================================
# 骨骼缓冲区
# ============================================================
class SkeletonBuffer:
    """按 track_id 缓存骨骼帧，推理时均匀采样模拟训练时 UniformSampleFrames"""

    def __init__(self, clip_len, stride=16, max_person=2, buf_max_factor=4, grace_frames=90):
        self.clip_len = clip_len
        self.stride = stride
        self.max_person = max_person
        self.buf_max = clip_len * buf_max_factor  # 保留更长历史用于均匀采样
        self.grace_frames = grace_frames           # 遮挡宽限期（帧数）
        self.tracks = defaultdict(lambda: {
            'kps': [], 'scores': [], 'new_frames': 0,
        })
        self._missing_count = {}  # {track_id: 连续消失帧数}
        self._last_positions = {}  # {track_id: np.array([cx, cy])} 最后已知位置
        self._smooth_kps = {}     # {track_id: np.array(17,2)} EMA平滑关键点
        self._smooth_scores = {}  # {track_id: np.array(17,)} EMA平滑置信度
        self.smooth_alpha = 0.5   # EMA系数：0.5=新旧各半

    def update(self, track_id, keypoints_17x2, scores_17):
        """更新骨骼缓冲区，返回平滑后的关键点用于可视化。
        Returns: (smoothed_kps, smoothed_scores)
        """
        kps = keypoints_17x2.copy()
        sc = scores_17.copy()

        # EMA 时序平滑
        if track_id in self._smooth_kps:
            prev_kps = self._smooth_kps[track_id]
            prev_sc = self._smooth_scores[track_id]
            alpha = self.smooth_alpha
            for j in range(17):
                if sc[j] > 0.3 and prev_sc[j] > 0.3:
                    # 新旧都有效 → 平滑
                    kps[j] = alpha * kps[j] + (1 - alpha) * prev_kps[j]
                    sc[j] = alpha * sc[j] + (1 - alpha) * prev_sc[j]
                elif sc[j] <= 0.3 and prev_sc[j] > 0.3:
                    # 新帧丢失但旧帧有效 → 沿用旧值（减少闪烁）
                    kps[j] = prev_kps[j]
                    sc[j] = prev_sc[j] * 0.9  # 逐渐衰减
                # 新帧有效但旧帧无效 → 用新值（不平滑）

        self._smooth_kps[track_id] = kps.copy()
        self._smooth_scores[track_id] = sc.copy()

        # 存入 buffer（用平滑后的值，推理也更稳定）
        buf = self.tracks[track_id]
        buf['kps'].append(kps)
        buf['scores'].append(sc)
        buf['new_frames'] += 1
        if len(buf['kps']) > self.buf_max:
            buf['kps'] = buf['kps'][-self.buf_max:]
            buf['scores'] = buf['scores'][-self.buf_max:]
        # 更新最后已知位置
        valid = sc > 0.3
        if valid.sum() > 0:
            self._last_positions[track_id] = kps[valid].mean(axis=0)

        return kps, sc

    def should_infer(self, track_id):
        buf = self.tracks[track_id]
        min_frames = max(16, self.clip_len // 2)  # 最少32帧就开始推理，减少track碎片化影响
        return buf['new_frames'] >= self.stride and len(buf['kps']) >= min_frames

    def _uniform_sample(self, kps_list, scores_list):
        """模拟训练时 UniformSampleFrames：从全部缓存帧中均匀采样 clip_len 帧"""
        n = len(kps_list)
        if n >= self.clip_len:
            indices = np.linspace(0, n - 1, self.clip_len, dtype=int)
        else:
            indices = np.arange(n)

        sampled_kps = [kps_list[i] for i in indices]
        sampled_scores = [scores_list[i] for i in indices]
        return sampled_kps, sampled_scores

    def get_clip(self, track_id, secondary_tid=None):
        """
        Returns: keypoint (2, T, 17, 2), keypoint_score (2, T, 17)
        Person 0 = primary track, Person 1 = secondary track (or zeros if None)
        均匀采样以匹配训练时 UniformSampleFrames 的时序分布
        """
        buf = self.tracks[track_id]
        buf['new_frames'] = 0

        keypoint = np.zeros((self.max_person, self.clip_len, 17, 2), dtype=np.float32)
        keypoint_score = np.zeros((self.max_person, self.clip_len, 17), dtype=np.float32)

        # Person 0: primary track — 均匀采样
        sampled_kps, sampled_scores = self._uniform_sample(buf['kps'], buf['scores'])
        n = len(sampled_kps)
        offset = self.clip_len - n
        for i in range(n):
            keypoint[0, offset + i] = sampled_kps[i]
            keypoint_score[0, offset + i] = sampled_scores[i]

        # Person 1: secondary track (nearest neighbor) — 均匀采样
        if secondary_tid is not None and secondary_tid in self.tracks:
            buf2 = self.tracks[secondary_tid]
            sampled_kps2, sampled_scores2 = self._uniform_sample(buf2['kps'], buf2['scores'])
            n2 = len(sampled_kps2)
            offset2 = self.clip_len - n2
            for i in range(n2):
                keypoint[1, offset2 + i] = sampled_kps2[i]
                keypoint_score[1, offset2 + i] = sampled_scores2[i]

        return keypoint, keypoint_score

    def try_reassociate(self, new_tid, new_position, max_dist_ratio=0.15, img_h=1080):
        """新 track 出现时，检查是否匹配某个宽限期内的消失 track。
        匹配条件：空间距离 < max_dist_ratio * img_h（默认15%画面高度）
        Returns: old_tid if matched, else None
        """
        if new_position is None:
            return None
        threshold = img_h * max_dist_ratio
        best_tid, best_dist = None, threshold
        for tid, cnt in self._missing_count.items():
            if cnt > self.grace_frames:
                continue  # 已超出宽限期
            if tid not in self._last_positions:
                continue
            old_pos = self._last_positions[tid]
            d = np.linalg.norm(new_position - old_pos)
            if d < best_dist:
                best_dist = d
                best_tid = tid
        return best_tid

    def migrate(self, old_tid, new_tid):
        """将 old_tid 的 buffer 迁移到 new_tid"""
        if old_tid in self.tracks:
            old_buf = self.tracks[old_tid]
            new_buf = self.tracks[new_tid]
            # 把旧 buffer 的历史帧拼到新 track 前面
            new_buf['kps'] = old_buf['kps'] + new_buf['kps']
            new_buf['scores'] = old_buf['scores'] + new_buf['scores']
            # 截断到 buf_max
            if len(new_buf['kps']) > self.buf_max:
                new_buf['kps'] = new_buf['kps'][-self.buf_max:]
                new_buf['scores'] = new_buf['scores'][-self.buf_max:]
            # new_frames 保留新 track 自身的（触发推理用）
            new_buf['new_frames'] += old_buf['new_frames']
            # 清理旧 track
            del self.tracks[old_tid]
            self._missing_count.pop(old_tid, None)
            self._last_positions.pop(old_tid, None)
            # 迁移平滑状态
            if old_tid in self._smooth_kps:
                self._smooth_kps[new_tid] = self._smooth_kps.pop(old_tid)
                self._smooth_scores[new_tid] = self._smooth_scores.pop(old_tid)
            logger.info(f'[REASSOC] SkeletonBuffer: T{old_tid} → T{new_tid} '
                        f'(migrated {len(old_buf["kps"])} frames)')

    def remove_stale(self, active_ids):
        """遮挡宽限：track 消失后保留 grace_frames 帧，恢复时继承旧 buffer"""
        for tid in list(self.tracks.keys()):
            if tid in active_ids:
                self._missing_count.pop(tid, None)
            else:
                self._missing_count[tid] = self._missing_count.get(tid, 0) + 1
        stale = [tid for tid, cnt in self._missing_count.items()
                 if cnt > self.grace_frames]
        for tid in stale:
            self.tracks.pop(tid, None)
            self._smooth_kps.pop(tid, None)
            self._smooth_scores.pop(tid, None)
            del self._missing_count[tid]


# ============================================================
# PoseC3D 推理封装
# ============================================================
class PoseC3DInferencer:
    """封装 PoseC3D 模型加载和推理，自动从 config 读取 clip_len"""

    def __init__(self, config_path, checkpoint_path, device='cuda:0'):
        config = mmcv.Config.fromfile(config_path)

        # 从 config 自动读取 clip_len
        test_pipeline_cfg = config.data.test.pipeline
        self.clip_len = 48  # fallback
        for step in test_pipeline_cfg:
            if step['type'] == 'UniformSampleFrames':
                self.clip_len = step['clip_len']
                break

        # 检测 with_kp / with_limb 设置
        self.with_kp = True
        self.with_limb = False
        for step in test_pipeline_cfg:
            if step['type'] == 'GeneratePoseTarget':
                self.with_kp = step.get('with_kp', True)
                self.with_limb = step.get('with_limb', False)
                break

        # 移除 DecompressPose（实时推理不需要）
        config.data.test.pipeline = [
            x for x in test_pipeline_cfg
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
            keypoint: (M, T, 17, 2)
            keypoint_score: (M, T, 17)
            img_shape: (H, W)
        Returns:
            probs: np.array (num_classes,)
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

        probs = self.model(return_loss=False, **data)[0]
        return probs


# ============================================================
# YOLO 小物体检测封装
# ============================================================
class SmallObjectDetector:
    """Legacy 统一多类模型封装（unified_3class）。保留向后兼容。
    新 pipeline 应使用 SingleClassDetector + MultiSmallObjectDetector。
    """

    def __init__(self, model_path, class_map=None, conf=0.3, imgsz=1280):
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz
        if class_map is not None:
            self.class_map = class_map
        else:
            self.class_map = dict(self.model.names)

    def detect(self, frame):
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


class SingleClassDetector:
    """单类 YOLO 模型适配层（R11 新增）。

    把单类专用 YOLO 模型（如 laying_yolo11m_v1 / smoking_yolo11m_v1 / phone_yolo11m_v1）
    的任意输出强制映射为固定的 rule_engine 语义类名（falling/smoking/phone）。

    设计前提：每个模型只检测单一目标，其原始 model.names 里的类名不确定
    （可能是 'laying' 不是 'falling'）。此包装层统一输出 rule_engine 认的名字，
    无需修改 rule_engine 字符串匹配逻辑。
    """

    def __init__(self, model_path, target_class, conf=0.3, imgsz=1280):
        self.model = YOLO(model_path)
        self.target_class = target_class
        self.conf = conf
        self.imgsz = imgsz

    def detect(self, frame):
        results = self.model(frame, conf=self.conf, imgsz=self.imgsz, verbose=False)
        detections = []
        for box in results[0].boxes:
            detections.append({
                'class': self.target_class,  # 强制语义类名,忽略模型原始 names
                'bbox': box.xyxy[0].tolist(),
                'conf': float(box.conf.item()),
            })
        return detections


class MultiSmallObjectDetector:
    """三路并列的单类检测器管理器（R11 新增）。

    每帧按需跳过不相关的检测器以节省算力：
      - falling 检测始终运行（安全兜底 —— PoseC3D 对一动不动躺地有盲区）
      - smoking / phone 在任一 track 处于攻击/倒地/攀爬态时跳过（物理互斥）
    """

    def __init__(self, falling_model=None, smoking_model=None, phone_model=None,
                 conf=0.3, imgsz=1280):
        self.falling = SingleClassDetector(falling_model, 'falling', conf, imgsz) \
            if falling_model else None
        self.smoking = SingleClassDetector(smoking_model, 'smoking', conf, imgsz) \
            if smoking_model else None
        self.phone = SingleClassDetector(phone_model, 'phone', conf, imgsz) \
            if phone_model else None

    def any_loaded(self):
        return any(d is not None for d in (self.falling, self.smoking, self.phone))

    def detect(self, frame, need_falling=True, need_smoking=True, need_phone=True):
        detections = []
        ran = []
        skipped = []
        if self.falling and need_falling:
            detections.extend(self.falling.detect(frame))
            ran.append('falling')
        elif self.falling:
            skipped.append('falling')
        if self.smoking and need_smoking:
            detections.extend(self.smoking.detect(frame))
            ran.append('smoking')
        elif self.smoking:
            skipped.append('smoking')
        if self.phone and need_phone:
            detections.extend(self.phone.detect(frame))
            ran.append('phone')
        elif self.phone:
            skipped.append('phone')
        if skipped and logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'  [YOLO-GATE] ran={ran} skipped={skipped}')
        return detections


# ============================================================
# 可视化
# ============================================================
def draw_skeleton(frame, kps, scores, color=(0, 255, 0), threshold=0.3):
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
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f'{label} {confidence:.0%}'
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, text, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def draw_info(frame, fps, frame_idx, total_frames=-1):
    if total_frames > 0:
        text = f'Frame: {frame_idx}/{total_frames}  FPS: {fps:.1f}'
    else:
        text = f'Frame: {frame_idx}  FPS: {fps:.1f}'
    cv2.putText(frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


# ============================================================
# 主推理引擎
# ============================================================
class InferencePipeline:
    """端到端推理 Pipeline"""

    def __init__(self, yolo_pose_model, posec3d_config, posec3d_checkpoint,
                 small_obj_model=None, device='cuda:0', yolo_conf=0.3,
                 stride=16, vote_window=5, vote_ratio=0.6,
                 loiter_time=60.0, loiter_radius=100.0,
                 falling_model=None, smoking_model=None, phone_model=None,
                 on_frame_callback=None):

        print('[1/3] Loading YOLO Pose model...')
        self.yolo_pose = YOLO(yolo_pose_model)

        print('[2/3] Loading PoseC3D model...')
        self.posec3d = PoseC3DInferencer(posec3d_config, posec3d_checkpoint, device=device)

        # R11：优先使用 3 个单类模型；若都没给且 small_obj_model 给了，用 legacy unified 模型
        self.small_obj_detector = None
        self.multi_detector = None
        if any(m for m in (falling_model, smoking_model, phone_model)):
            print('[2.5/3] Loading single-class YOLO models...')
            if falling_model:
                print(f'  falling ← {falling_model}')
            if smoking_model:
                print(f'  smoking ← {smoking_model}')
            if phone_model:
                print(f'  phone ← {phone_model}')
            self.multi_detector = MultiSmallObjectDetector(
                falling_model=falling_model,
                smoking_model=smoking_model,
                phone_model=phone_model,
                conf=yolo_conf,
            )
        elif small_obj_model:
            print(f'[2.5/3] Loading legacy unified small object model... {small_obj_model}')
            self.small_obj_detector = SmallObjectDetector(small_obj_model)

        print('[3/3] Initializing pipeline...')
        clip_len = self.posec3d.clip_len
        self.skeleton_buf = SkeletonBuffer(clip_len=clip_len, stride=stride)
        self.rule_engine = RuleEngine(
            pose_threshold=0.3,
            vote_window=vote_window,
            vote_ratio=vote_ratio,
            loiter_time=loiter_time,
            loiter_radius=loiter_radius,
        )

        self.yolo_conf = yolo_conf
        self.track_labels = {}
        self._label_missing_count = {}  # 遮挡宽限：标签保留
        self._known_track_ids = set()   # 已见过的 track ID，用于检测新 track
        self.event_log = []
        # API 联调钩子：每帧处理完成后回调，payload 见 _emit_frame_callback
        self.on_frame_callback = on_frame_callback

        print(f'  clip_len={clip_len} (auto-detected from config)')
        print(f'  stride={stride}, with_kp={self.posec3d.with_kp}, with_limb={self.posec3d.with_limb}')

    def run(self, source, show=False, output_video=None, output_json=None):
        """
        运行推理。

        Args:
            source: InputSource 或 str（自动创建）
            show: 是否实时显示
            output_video: 输出标注视频路径（None 则不保存）
            output_json: 输出检测日志路径（None 则不保存）
        """
        if isinstance(source, str):
            source = InputSource.create(source)

        w, h = source.get_frame_size()
        fps_video = source.get_fps()
        total_frames = source.get_total_frames()
        img_shape = (h, w)

        writer = None
        if output_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_video, fourcc, fps_video, (w, h))

        print(f'\nRunning inference ({w}x{h} @ {fps_video:.0f}fps)')
        if total_frames > 0:
            print(f'Total frames: {total_frames}')
        print(f'PoseC3D: {self.posec3d.num_classes} classes')
        print('Press Q to quit.\n')

        frame_idx = 0
        t_start = time.time()

        while True:
            ret, frame = source.read()
            if not ret:
                break

            self._process_frame(frame, frame_idx, img_shape)

            # FPS overlay
            elapsed = time.time() - t_start
            current_fps = (frame_idx + 1) / elapsed if elapsed > 0 else 0
            draw_info(frame, current_fps, frame_idx, total_frames)

            if writer:
                writer.write(frame)
            if show:
                cv2.imshow('Campus Safety', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_idx += 1

        # 清理
        source.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()

        total_time = time.time() - t_start
        avg_fps = frame_idx / total_time if total_time > 0 else 0
        print(f'\nDone. {frame_idx} frames in {total_time:.1f}s ({avg_fps:.1f} fps)')

        if output_json and self.event_log:
            with open(output_json, 'w', encoding='utf-8') as f:
                json.dump(self.event_log, f, ensure_ascii=False, indent=2)
            print(f'Event log saved: {output_json} ({len(self.event_log)} events)')

    @staticmethod
    def _person_center(kps_xy, kps_sc, threshold=0.3):
        """计算人物中心（有效关键点的均值）"""
        valid = kps_sc > threshold
        if valid.sum() == 0:
            return None
        return kps_xy[valid].mean(axis=0)

    def _find_nearest_track(self, track_id, track_positions, max_dist=500):
        """找空间上最近的另一个 track，返回 track_id 或 None"""
        if track_id not in track_positions or len(track_positions) < 2:
            return None
        pos = track_positions[track_id]
        best_tid, best_dist = None, max_dist
        for tid, tpos in track_positions.items():
            if tid == track_id:
                continue
            d = np.linalg.norm(pos - tpos)
            if d < best_dist:
                best_dist = d
                best_tid = tid
        return best_tid

    def _process_frame(self, frame, frame_idx, img_shape):
        """处理单帧"""

        # Step 1: YOLO Pose + ByteTrack
        results = self.yolo_pose.track(
            source=frame,
            persist=True,
            tracker='bytetrack.yaml',
            conf=self.yolo_conf,
            iou=0.5,
            verbose=False,
        )

        result = results[0]
        current_track_ids = set()
        frame_persons_kps = []
        track_positions = {}  # {tid: np.array([cx, cy])}
        track_kps = {}        # {tid: (kps_xy, kps_sc)}
        track_bboxes = {}     # {tid: [x1, y1, x2, y2]}

        if result.boxes is not None and result.keypoints is not None:
            # 第一遍: 收集所有 track 位置
            for i, box in enumerate(result.boxes):
                if box.id is None:
                    continue
                track_id = int(box.id.item())
                current_track_ids.add(track_id)

                kps_data = result.keypoints.data[i].cpu().numpy()
                kps_xy = kps_data[:, :2]
                kps_sc = kps_data[:, 2]

                track_bboxes[track_id] = box.xyxy[0].tolist()

                # 平滑关键点（减少抖动）
                smooth_kps, smooth_sc = self.skeleton_buf.update(track_id, kps_xy, kps_sc)

                frame_persons_kps.append(smooth_kps)
                track_kps[track_id] = (smooth_kps, smooth_sc)

                center = self._person_center(smooth_kps, smooth_sc)
                if center is not None:
                    track_positions[track_id] = center

                self.rule_engine.update_track_position(track_id, smooth_kps, smooth_sc)

            # 重关联：新 track 出现时匹配宽限期内的消失 track，迁移状态
            for track_id in current_track_ids:
                if track_id not in self._known_track_ids:
                    self._known_track_ids.add(track_id)
                    pos = track_positions.get(track_id)
                    old_tid = self.skeleton_buf.try_reassociate(
                        track_id, pos, img_h=img_shape[0])
                    if old_tid is not None:
                        logger.info(f'[F{frame_idx}] REASSOC: T{old_tid} → T{track_id} '
                                    f'(位置匹配，迁移状态)')
                        self.skeleton_buf.migrate(old_tid, track_id)
                        self.rule_engine.migrate_track(old_tid, track_id)
                        if old_tid in self.track_labels:
                            self.track_labels[track_id] = self.track_labels.pop(old_tid)
                        self._label_missing_count.pop(old_tid, None)

            # R13 (P15)：撤销 R11 的场景级 gating —— gating 原本想用"物理互斥"
            # 省算力(一个人在 fighting/falling 时不会同时吸烟),但实现是场景级的：
            # 任一 track 进攻击/倒地态就全场跳过 smoking/phone。
            # 真实场景里 T1 抽烟 + T3 倒地会互相干扰 —— T3 误判 falling 会把 T1
            # smoking 模型的输入彻底掐断。per-track gating 在 YOLO 帧级调用下无法实现
            # (YOLO 一次检测整张画面),所以干脆全部跑。
            # 保留接口(need_falling/smoking/phone)用于未来可能的扩展,当前固定 True。
            gate_need_falling = True
            gate_need_smoking = True
            gate_need_phone = True

            # R12 (P13)：场景人数 = 本帧检测到的 + grace 期内仍存活的 buffered track
            # 修复 fighting 中一方被遮挡时 len(all_person_kps)==1 → vandalism 误判
            # grace 期内的 track 物理上仍在场景中（SkeletonBuffer 保留了骨骼），
            # 只是本帧未被 YOLO 检测到（被遮挡 / 肢体压缩）。
            scene_track_ids = set(current_track_ids) | set(self.skeleton_buf.tracks.keys())
            scene_person_count = len(scene_track_ids)
            # 每帧推送一次场景人数到 rule_engine（与 per-track judge 解耦）
            self.rule_engine.push_scene_count(scene_person_count)

            # 帧级缓存 —— 同一帧多 track 推理时复用检测结果，避免 N× YOLO 调用
            small_objs_cache = None

            # 第二遍A: 推理 + 画骨骼（标签绘制延后到耦合之后）
            inferred_tids = []
            for track_id, (kps_xy, kps_sc) in track_kps.items():
                # 画骨骼（用上一帧的标签色，当前帧推理还没完成）
                label_info = self.track_labels.get(track_id)
                color = LABEL_COLORS.get(
                    label_info.label if label_info else 'normal', (200, 200, 200))
                draw_skeleton(frame, kps_xy, kps_sc, color=color)

                # Step 2: PoseC3D 推理（双人配对）
                buf = self.skeleton_buf.tracks[track_id]
                buf_len = len(buf['kps'])

                if self.skeleton_buf.should_infer(track_id):
                    nearest_tid = self._find_nearest_track(track_id, track_positions)
                    keypoint, keypoint_score = self.skeleton_buf.get_clip(track_id, nearest_tid)

                    kp_nonzero = np.any(keypoint[0] != 0, axis=(1, 2))
                    n_valid_frames = int(kp_nonzero.sum())
                    kp2_nonzero = np.any(keypoint[1] != 0, axis=(1, 2))
                    n_valid_p2 = int(kp2_nonzero.sum())
                    logger.debug(f'[F{frame_idx}] T{track_id} INFER: buf_total={buf_len}, '
                                 f'sampled={n_valid_frames}/64, pair=T{nearest_tid}(P1:{n_valid_p2}/64)')

                    pose_probs = self.posec3d.infer(keypoint, keypoint_score, img_shape)

                    # Step 3: 小物体检测（按需 + 帧级缓存）
                    if small_objs_cache is None:
                        if self.multi_detector is not None:
                            small_objs_cache = self.multi_detector.detect(
                                frame,
                                need_falling=gate_need_falling,
                                need_smoking=gate_need_smoking,
                                need_phone=gate_need_phone,
                            )
                        elif self.small_obj_detector is not None:
                            small_objs_cache = self.small_obj_detector.detect(frame)
                        else:
                            small_objs_cache = []
                    small_objs = small_objs_cache

                    # Step 4: 规则引擎
                    all_kps_scores = [(kps, sc) for kps, sc in track_kps.values()]
                    judgment = self.rule_engine.judge(
                        track_id=track_id,
                        pose_probs=pose_probs,
                        person_kps=kps_xy,
                        person_scores=kps_sc,
                        all_person_kps=frame_persons_kps,
                        small_obj_detections=small_objs,
                        img_shape=img_shape,
                        all_person_kps_scores=all_kps_scores,
                        track_kps_dict=track_kps,
                        track_bboxes_dict=track_bboxes,
                        scene_person_count=scene_person_count,
                    )
                    self.track_labels[track_id] = judgment
                    inferred_tids.append(track_id)
                else:
                    min_frames = max(16, self.skeleton_buf.clip_len // 2)
                    if buf_len < min_frames:
                        logger.debug(f'[F{frame_idx}] T{track_id} SKIP: buffer不足 {buf_len}/{min_frames}')

            # 第二遍B: Pair coupling —— bbox 重叠的 track 必须共享攻击态
            # 对所有当前活跃 track（不只本帧推理的）跑耦合，
            # 因为非推理帧的 track_labels 可能保留攻击态，需要传播给新出现的 normal 邻居
            active_judgments = {tid: self.track_labels[tid]
                                for tid in current_track_ids
                                if tid in self.track_labels}
            self.rule_engine.couple_overlapping_pairs(active_judgments, track_bboxes)

            # 记录事件（用耦合后的标签）
            for track_id in inferred_tids:
                judgment = self.track_labels[track_id]
                logger.debug(f'[F{frame_idx}] T{track_id} FINAL → {judgment.label} '
                             f'(conf={judgment.confidence:.3f}, src={judgment.source}, smooth={judgment.smoothed})')
                if judgment.label != 'normal':
                    self.event_log.append({
                        'frame': frame_idx,
                        **judgment.to_dict(),
                    })

            # 第二遍C: 可视化标签（用耦合后的标签）
            for track_id in track_kps.keys():
                bbox = track_bboxes.get(track_id)
                if bbox:
                    if track_id in self.track_labels:
                        info = self.track_labels[track_id]
                        color = LABEL_COLORS.get(info.label, (200, 200, 200))
                        draw_label(frame, bbox, info.label, info.confidence, color)
                    else:
                        draw_label(frame, bbox, 'normal', 1.0, LABEL_COLORS['normal'])

        # API 联调回调：推送本帧所有活跃 track 的标签
        if self.on_frame_callback is not None:
            try:
                self._emit_frame_callback(frame_idx, img_shape, current_track_ids, track_bboxes)
            except Exception as cb_err:
                logger.debug(f'[F{frame_idx}] on_frame_callback 失败: {cb_err}')

        # 清理消失的 track（带遮挡宽限期）
        self.skeleton_buf.remove_stale(current_track_ids)
        self.rule_engine.clear_stale_tracks(current_track_ids)
        grace = self.skeleton_buf.grace_frames
        for tid in list(self.track_labels.keys()):
            if tid in current_track_ids:
                self._label_missing_count.pop(tid, None)
            else:
                self._label_missing_count[tid] = self._label_missing_count.get(tid, 0) + 1
        stale = [tid for tid, cnt in self._label_missing_count.items() if cnt > grace]
        for tid in stale:
            self.track_labels.pop(tid, None)
            self._known_track_ids.discard(tid)
            del self._label_missing_count[tid]

    def _emit_frame_callback(self, frame_idx, img_shape, current_track_ids, track_bboxes):
        """构造 API 回调 payload。API 层可按需转换为 WEB frame 事件格式。"""
        h, w = img_shape
        tracks = []
        for tid in current_track_ids:
            bbox = track_bboxes.get(tid)
            if bbox is None:
                continue
            info = self.track_labels.get(tid)
            if info is None:
                label = 'normal'
                conf = 1.0
                source = 'default'
            else:
                label = info.label
                conf = float(info.confidence)
                source = info.source
            tracks.append({
                'track_id': int(tid),
                'label': label,
                'confidence': conf,
                'source': source,
                'bbox_xyxy': [float(x) for x in bbox],
            })
        payload = {
            'frame_index': int(frame_idx),
            'img_width': int(w),
            'img_height': int(h),
            'tracks': tracks,
        }
        self.on_frame_callback(payload)
