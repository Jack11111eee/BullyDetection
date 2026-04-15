"""
rule_engine.py — 行为判定规则引擎

输入：
  - PoseC3D 预测结果（5类概率）
  - YOLO11s 小物体检测结果（bbox + class）
  - YOLO11m-Pose 骨骼关键点（17个COCO关键点）
  - 轨迹信息（用于徘徊检测）

输出：
  - BehaviorResult 结构化判定结果

COCO 17 关键点索引：
  0-鼻子  1-左眼  2-右眼  3-左耳  4-右耳
  5-左肩  6-右肩  7-左肘  8-右肘  9-左腕  10-右腕
  11-左髋 12-右髋 13-左膝 14-右膝 15-左踝 16-右踝
"""

import time
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict

import numpy as np

logger = logging.getLogger('e2e_debug')


# ============================================================
# 行为类别定义
# ============================================================
POSE_CLASSES = ['normal', 'fighting', 'bullying', 'falling', 'climbing']

FINAL_CLASSES = [
    'normal',       # 0 正常
    'fighting',     # 1 打架
    'bullying',     # 2 霸凌
    'falling',      # 3 摔倒
    'climbing',     # 4 攀爬
    'vandalism',    # 5 破坏公物（规则引擎判定）
    'smoking',      # 6 吸烟（YOLO小物体 + 规则）
    'phone_call',   # 7 打电话（YOLO小物体 + 规则）
    'loitering',    # 8 徘徊（轨迹分析）
]


@dataclass
class BehaviorResult:
    """结构化行为判定结果"""
    label: str
    confidence: float
    source: str          # 'posec3d' / 'rule_yolo_falling' / 'rule_yolo_bullying' / 'rule_smoking' / 'rule_phone' / 'rule_vandalism' / 'rule_loitering'
    smoothed: bool       # 是否经过时序平滑
    track_id: int = -1
    timestamp: float = 0.0

    def to_dict(self):
        return asdict(self)


# ============================================================
# 工具函数
# ============================================================
def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def get_keypoint(keypoints, idx):
    return keypoints[idx][0], keypoints[idx][1]


def keypoint_valid(keypoints, scores, idx, threshold=0.3):
    if scores is not None:
        return scores[idx] > threshold
    return keypoints[idx][0] > 0 or keypoints[idx][1] > 0


# ============================================================
# 规则判定函数
# ============================================================
def check_smoking(person_kps, person_scores, small_obj_detections, img_shape):
    """吸烟判定：检测到香烟 + 在人手或嘴附近"""
    cigarettes = [d for d in small_obj_detections if d['class'] in ('cigarette', 'smoking')]
    if not cigarettes:
        return False, 0.0

    h, w = img_shape
    ref_dist = h * 0.08

    if (keypoint_valid(person_kps, person_scores, 5) and
        keypoint_valid(person_kps, person_scores, 6)):
        shoulder_w = distance(get_keypoint(person_kps, 5), get_keypoint(person_kps, 6))
        if shoulder_w > 10:
            ref_dist = shoulder_w * 0.8

    check_kps = [0, 9, 10]  # 鼻子、左腕、右腕

    for cig in cigarettes:
        cig_center = bbox_center(cig['bbox'])
        for kp_idx in check_kps:
            if not keypoint_valid(person_kps, person_scores, kp_idx):
                continue
            kp_pos = get_keypoint(person_kps, kp_idx)
            if distance(cig_center, kp_pos) < ref_dist:
                return True, cig['conf']

    return False, 0.0


def check_phone_call(person_kps, person_scores, small_obj_detections, img_shape):
    """打电话判定：检测到手机 + 在耳朵附近"""
    phones = [d for d in small_obj_detections if d['class'] == 'phone']
    if not phones:
        return False, 0.0

    h, w = img_shape
    ref_dist = h * 0.08

    if (keypoint_valid(person_kps, person_scores, 5) and
        keypoint_valid(person_kps, person_scores, 6)):
        shoulder_w = distance(get_keypoint(person_kps, 5), get_keypoint(person_kps, 6))
        if shoulder_w > 10:
            ref_dist = shoulder_w * 0.8

    for phone in phones:
        phone_center = bbox_center(phone['bbox'])
        for kp_idx in [3, 4]:  # 左耳、右耳
            if not keypoint_valid(person_kps, person_scores, kp_idx):
                continue
            kp_pos = get_keypoint(person_kps, kp_idx)
            if distance(phone_center, kp_pos) < ref_dist:
                return True, phone['conf']

    return False, 0.0


def check_vandalism(all_person_kps, person_scores, pose_probs, small_obj_detections):
    """破坏公物：fighting 概率高 + 只检测到 1 个人"""
    fighting_prob = pose_probs[1]
    if fighting_prob > 0.5 and len(all_person_kps) == 1:
        return True, fighting_prob * 0.8
    return False, 0.0


def _person_height(kps, scores, threshold=0.3):
    """估算人物站立高度：头顶到脚踝的垂直距离"""
    # 头部：取鼻子(0)或眼睛(1,2)中最高的
    head_y = None
    for idx in [0, 1, 2]:
        if keypoint_valid(kps, scores, idx, threshold):
            y = kps[idx][1]
            if head_y is None or y < head_y:  # y 轴向下，越小越高
                head_y = y
    # 脚部：取脚踝(15,16)中最低的
    foot_y = None
    for idx in [15, 16]:
        if keypoint_valid(kps, scores, idx, threshold):
            y = kps[idx][1]
            if foot_y is None or y > foot_y:
                foot_y = y
    if head_y is not None and foot_y is not None:
        return foot_y - head_y  # 正值，越大越高
    return None


def _head_above_hip_ratio(kps, scores, threshold=0.3):
    """头部相对于髋部的位置比：< 1.0 表示头低于正常站立位置（蜷缩/弯腰）"""
    head_y = None
    for idx in [0, 1, 2]:
        if keypoint_valid(kps, scores, idx, threshold):
            head_y = kps[idx][1]
            break
    hip_y = None
    for idx in [11, 12]:
        if keypoint_valid(kps, scores, idx, threshold):
            y = kps[idx][1]
            if hip_y is None:
                hip_y = y
            else:
                hip_y = (hip_y + y) / 2
    if head_y is None or hip_y is None:
        return None
    # 正常站立时 head_y << hip_y（头在髋上面很多），ratio 很大
    # 蜷缩时 head_y ≈ hip_y，ratio 接近 0
    return hip_y - head_y  # 正值越大=头越高于髋=越直立


def _nearest_person_dist(person_kps, person_scores, all_person_kps_scores, threshold=0.3):
    """计算当前人和最近邻人的距离（像素）。返回 (距离, 邻居身高) 或 (inf, None)"""
    valid = person_scores > threshold
    if valid.sum() == 0:
        return float('inf'), None
    my_center = person_kps[valid].mean(axis=0)

    best_dist = float('inf')
    best_height = None
    for other_kps, other_scores in all_person_kps_scores:
        if np.array_equal(other_kps, person_kps):
            continue
        other_valid = other_scores > threshold
        if other_valid.sum() == 0:
            continue
        other_center = other_kps[other_valid].mean(axis=0)
        d = np.linalg.norm(my_center - other_center)
        if d < best_dist:
            best_dist = d
            best_height = _person_height(other_kps, other_scores)
    return best_dist, best_height


def _is_upright_posture(kps, scores, img_shape, threshold=0.3):
    """检查当前姿态是否为直立/坐姿（躯干竖直）— 用于排除 falling 误判。
    返回 True 表示躯干直立，不应判为 falling。
    """
    head_hip = _head_above_hip_ratio(kps, scores, threshold)
    if head_hip is None:
        return False  # 无法判断，不阻止

    h = img_shape[0]
    # head_hip 是 hip_y - head_y，正值越大=头越高于髋=越直立
    # 正常站立/坐着时，head_hip > 画面高度的 8%（头明显在髋之上）
    # 摔倒时，head_hip 接近 0 或为负（头和髋同高甚至头更低）
    upright_threshold = h * 0.08
    is_upright = head_hip > upright_threshold

    if is_upright:
        logger.debug(f'  [RULE] 姿态直立检查: head_hip={head_hip:.1f} > {upright_threshold:.1f} → 直立(非falling)')
    return is_upright


def check_fallen_by_yolo(person_kps, person_scores, small_obj_detections, img_shape):
    """YOLO 辅助摔倒检测：检测到躺地人体 + 与当前人物重叠。
    用于补偿 PoseC3D 对一动不动躺地者的漏检。
    """
    fallen = [d for d in small_obj_detections if d['class'] == 'falling']
    if not fallen:
        return False, 0.0

    valid = person_scores > 0.3
    if valid.sum() == 0:
        return False, 0.0
    person_center = person_kps[valid].mean(axis=0)  # (2,)

    for det in fallen:
        x1, y1, x2, y2 = det['bbox']
        # 扩展 bbox 20% 容忍骨骼偏移
        bw, bh = x2 - x1, y2 - y1
        margin = max(bw, bh) * 0.2
        if (x1 - margin <= person_center[0] <= x2 + margin and
                y1 - margin <= person_center[1] <= y2 + margin):
            return True, det['conf']

    return False, 0.0


def check_bullying_asymmetry(person_kps, person_scores, all_person_kps_scores, img_shape):
    """
    判断 fighting 是否应改判为 bullying：检测姿态不对称性
    - fighting：双方都有大幅度动作（对称）
    - bullying：一方攻击，另一方被动/蜷缩（不对称）

    Returns: (is_bullying, confidence)
    """
    if len(all_person_kps_scores) < 2:
        return False, 0.0

    # 找当前人的高度和头-髋比
    my_height = _person_height(person_kps, person_scores)
    my_head_hip = _head_above_hip_ratio(person_kps, person_scores)
    if my_height is None or my_head_hip is None:
        return False, 0.0

    # 找最近邻的高度
    my_center = None
    valid_kps = person_scores > 0.3
    if valid_kps.sum() > 0:
        my_center = person_kps[valid_kps].mean(axis=0)
    else:
        return False, 0.0

    best_dist = float('inf')
    other_height = None
    other_head_hip = None
    for other_kps, other_scores in all_person_kps_scores:
        if np.array_equal(other_kps, person_kps):
            continue
        other_valid = other_scores > 0.3
        if other_valid.sum() == 0:
            continue
        other_center = other_kps[other_valid].mean(axis=0)
        d = np.linalg.norm(my_center - other_center)
        if d < best_dist:
            best_dist = d
            other_height = _person_height(other_kps, other_scores)
            other_head_hip = _head_above_hip_ratio(other_kps, other_scores)

    if other_height is None or other_head_hip is None:
        return False, 0.0

    # 近距离约束：两人必须在 2x 较高者身高范围内才算互动
    max_interact_dist = max(my_height, other_height) * 2.0
    if best_dist > max_interact_dist:
        logger.debug(f'  [RULE] bullying距离过远: dist={best_dist:.0f} > {max_interact_dist:.0f} → 不是bullying')
        return False, 0.0

    # 不对称性判断
    # 1. 高度比：一方明显矮于另一方（蜷缩/倒地）
    height_ratio = min(my_height, other_height) / max(my_height, other_height) if max(my_height, other_height) > 0 else 1.0
    # 2. 头-髋差异：一方头接近髋部高度（弯腰/蜷缩）
    head_hip_diff = abs(my_head_hip - other_head_hip)
    h = img_shape[0]
    head_hip_normalized = head_hip_diff / h if h > 0 else 0

    # height_ratio < 0.6 → 一方身高不到另一方的60%（明显蜷缩）
    # head_hip_normalized > 0.1 → 头-髋差异超过画面高度10%（明显不对称）
    is_asymmetric = height_ratio < 0.6 or head_hip_normalized > 0.1

    if is_asymmetric:
        conf = max(1.0 - height_ratio, head_hip_normalized * 5)
        conf = min(conf, 1.0)
        logger.debug(f'  [RULE] bullying不对称检测: height_ratio={height_ratio:.2f}, '
                     f'head_hip_norm={head_hip_normalized:.3f} → bullying')
        return True, conf

    return False, 0.0


# ============================================================
# 主判定引擎
# ============================================================
class RuleEngine:
    """
    行为判定规则引擎

    优先级（从高到低）：
    1. PoseC3D 高置信度结果（falling/climbing > 0.7 直接采信）
    2. YOLO 辅助 falling 检测（一动不动躺地）
    3. 小物体规则（smoking / phone_call）
    4. vandalism 规则
    5. 徘徊检测（轨迹分析）
    6. PoseC3D 默认结果
    """

    def __init__(self, pose_threshold=0.3, vote_window=5, vote_ratio=0.6,
                 loiter_time=60.0, loiter_radius=100.0, grace_frames=90):
        """
        Args:
            pose_threshold: PoseC3D 结果的最低置信度
            vote_window: 时序投票窗口大小
            vote_ratio: 投票通过比例
            loiter_time: 徘徊判定时间阈值（秒）
            loiter_radius: 徘徊判定范围阈值（像素）
            grace_frames: 遮挡宽限帧数，track 消失后保留历史
        """
        self.pose_threshold = pose_threshold
        self.vote_window = vote_window
        self.vote_ratio = vote_ratio
        self.loiter_time = loiter_time
        self.loiter_radius = loiter_radius
        self.grace_frames = grace_frames

        self.history = {}           # {track_id: [label_str, ...]}
        self.track_positions = {}   # {track_id: [(x, y, timestamp), ...]}
        self._missing_count = {}    # {track_id: 连续消失帧数}

    def judge(self, track_id, pose_probs, person_kps, person_scores,
              all_person_kps, small_obj_detections, img_shape,
              all_person_kps_scores=None, track_kps_dict=None):
        """
        对单个人物做最终行为判定。

        Args:
            all_person_kps_scores: [(kps, scores), ...] 所有人的关键点+置信度，用于bullying不对称检测
            track_kps_dict: {track_id: (kps, scores)} 用于 YOLO falling 时查询附近 track 的标签历史

        Returns:
            BehaviorResult
        """
        raw_label, raw_conf, source = self._raw_judge(
            track_id, pose_probs, person_kps, person_scores,
            all_person_kps, small_obj_detections, img_shape,
            all_person_kps_scores=all_person_kps_scores,
            track_kps_dict=track_kps_dict,
        )

        smoothed_label = self._vote_smooth(track_id, raw_label)

        return BehaviorResult(
            label=smoothed_label,
            confidence=raw_conf,
            source=source,
            smoothed=(smoothed_label != raw_label),
            track_id=track_id,
            timestamp=time.time(),
        )

    def _raw_judge(self, track_id, pose_probs, person_kps, person_scores,
                   all_person_kps, small_obj_detections, img_shape,
                   all_person_kps_scores=None, track_kps_dict=None):
        """原始判定（不含时序平滑）"""

        pose_label_idx = int(np.argmax(pose_probs))
        pose_conf = float(pose_probs[pose_label_idx])
        pose_label = POSE_CLASSES[pose_label_idx]

        probs_str = ' '.join(f'{POSE_CLASSES[i]}={pose_probs[i]:.3f}' for i in range(len(POSE_CLASSES)))
        logger.debug(f'  [RAW] T{track_id} PoseC3D: argmax={pose_label}({pose_conf:.3f}) | {probs_str}')

        # 1. 高置信度的紧急行为（仍需姿态验证）
        if pose_label == 'climbing' and pose_conf > 0.7:
            logger.debug(f'  [RAW] T{track_id} → climbing (高置信度直接采信)')
            return 'climbing', pose_conf, 'posec3d'
        if pose_label == 'falling' and pose_conf > 0.7:
            if _is_upright_posture(person_kps, person_scores, img_shape):
                logger.debug(f'  [RAW] T{track_id} → normal (高置信度falling但躯干直立, 可能是坐下)')
                return 'normal', 1.0 - pose_conf, 'rule_upright'
            logger.debug(f'  [RAW] T{track_id} → falling (高置信度+非直立)')
            return 'falling', pose_conf, 'posec3d'

        # 2. YOLO 辅助 falling 检测（一动不动躺地，PoseC3D 识别不出）
        #    躺地 + 附近有被标为 fighting/bullying 的人 → bullying（被霸凌）
        #    躺地 + 附近无攻击者 → falling
        is_fallen_yolo, fallen_yolo_conf = check_fallen_by_yolo(
            person_kps, person_scores, small_obj_detections, img_shape
        )
        if is_fallen_yolo:
            if track_kps_dict and len(track_kps_dict) >= 2:
                valid_self = person_scores > 0.3
                if valid_self.sum() > 0:
                    my_center = person_kps[valid_self].mean(axis=0)
                    my_height = _person_height(person_kps, person_scores)
                    for other_tid, (other_kps, other_sc) in track_kps_dict.items():
                        if other_tid == track_id:
                            continue
                        valid_other = other_sc > 0.3
                        if valid_other.sum() == 0:
                            continue
                        other_center = other_kps[valid_other].mean(axis=0)
                        dist = np.linalg.norm(my_center - other_center)
                        other_height = _person_height(other_kps, other_sc)
                        ref_height = max(my_height or 0, other_height or 0)
                        max_dist = (ref_height * 1.5) if ref_height > 0 else img_shape[0] * 0.25
                        if dist > max_dist:
                            continue
                        # 只有附近这个人的历史标签包含 fighting/bullying 才判 bullying
                        other_history = self.history.get(other_tid, [])
                        if any(h in ('fighting', 'bullying') for h in other_history):
                            logger.debug(f'  [RAW] T{track_id} → bullying (YOLO躺地 + T{other_tid}有'
                                         f'攻击历史: {other_history}, dist={dist:.0f})')
                            return 'bullying', fallen_yolo_conf, 'rule_yolo_bullying'
            logger.debug(f'  [RAW] T{track_id} → falling (YOLO辅助检测: conf={fallen_yolo_conf:.3f})')
            return 'falling', fallen_yolo_conf, 'rule_yolo_falling'

        # 3. 检查吸烟
        is_smoking, smoke_conf = check_smoking(
            person_kps, person_scores, small_obj_detections, img_shape
        )
        if is_smoking:
            return 'smoking', smoke_conf, 'rule_smoking'

        # 4. 检查打电话
        is_phone, phone_conf = check_phone_call(
            person_kps, person_scores, small_obj_detections, img_shape
        )
        if is_phone:
            return 'phone_call', phone_conf, 'rule_phone'

        # 5. 检查破坏公物
        is_vandal, vandal_conf = check_vandalism(
            all_person_kps, person_scores, pose_probs, small_obj_detections
        )
        if is_vandal:
            logger.debug(f'  [RAW] T{track_id} → vandalism (规则引擎: fighting={pose_probs[1]:.3f}, 1人)')
            return 'vandalism', vandal_conf, 'rule_vandalism'

        # 6. 检查徘徊
        is_loiter, loiter_conf = self._check_loitering(track_id, person_kps, person_scores)
        if is_loiter:
            return 'loitering', loiter_conf, 'rule_loitering'

        # 7. 默认用 PoseC3D 结果
        if pose_conf >= self.pose_threshold:
            final_label = pose_label

            # 7a. fighting 额外置信度要求 + 近距离约束
            if final_label == 'fighting':
                if pose_conf < 0.5:
                    logger.debug(f'  [RAW] T{track_id} → normal (fighting置信度不足: {pose_conf:.3f} < 0.5)')
                    return 'normal', 1.0 - pose_conf, 'rule_fight_low_conf'
                if all_person_kps_scores is not None:
                    my_height = _person_height(person_kps, person_scores)
                    nearest_dist, neighbor_height = _nearest_person_dist(
                        person_kps, person_scores, all_person_kps_scores
                    )
                    # 用两人中较高者的身高，适应倒地者身高极小的情况
                    ref_height = max(my_height or 0, neighbor_height or 0)
                    max_fight_dist = (ref_height * 1.5) if ref_height > 0 else img_shape[0] * 0.25
                    if nearest_dist > max_fight_dist:
                        logger.debug(f'  [RAW] T{track_id} → normal (fighting但无人在附近: '
                                     f'nearest={nearest_dist:.0f} > {max_fight_dist:.0f})')
                        return 'normal', 1.0 - pose_conf, 'rule_no_proximity'

                # 7b. fighting → bullying 不对称性修正
                is_bully, bully_conf = check_bullying_asymmetry(
                    person_kps, person_scores, all_person_kps_scores, img_shape
                )
                if is_bully:
                    logger.debug(f'  [RAW] T{track_id} → bullying (fighting改判: 姿态不对称)')
                    return 'bullying', pose_conf * 0.9, 'rule_bullying'

            # 7c. falling 姿态验证：躯干直立（坐下）不算 falling
            if final_label == 'falling':
                if _is_upright_posture(person_kps, person_scores, img_shape):
                    logger.debug(f'  [RAW] T{track_id} → normal (falling但躯干直立, 可能是坐下)')
                    return 'normal', 1.0 - pose_conf, 'rule_upright'

            logger.debug(f'  [RAW] T{track_id} → {final_label} (conf={pose_conf:.3f} >= threshold={self.pose_threshold})')
            return final_label, pose_conf, 'posec3d'

        logger.debug(f'  [RAW] T{track_id} → normal (conf={pose_conf:.3f} < threshold={self.pose_threshold}, 被过滤!)')
        return 'normal', 1.0 - pose_conf, 'posec3d_default'

    def _check_loitering(self, track_id, person_kps, person_scores):
        """
        徘徊检测：同一 track 在小范围内停留超过阈值时间。
        使用髋部中点作为位置参考。
        """
        # 计算人物位置（髋部中点）
        if (keypoint_valid(person_kps, person_scores, 11) and
            keypoint_valid(person_kps, person_scores, 12)):
            lhip = get_keypoint(person_kps, 11)
            rhip = get_keypoint(person_kps, 12)
            cx = (lhip[0] + rhip[0]) / 2
            cy = (lhip[1] + rhip[1]) / 2
        else:
            return False, 0.0

        now = time.time()

        if track_id not in self.track_positions:
            self.track_positions[track_id] = []
        positions = self.track_positions[track_id]
        positions.append((cx, cy, now))

        # 只保留最近 2x loiter_time 的数据
        cutoff = now - self.loiter_time * 2
        self.track_positions[track_id] = [(x, y, t) for x, y, t in positions if t > cutoff]
        positions = self.track_positions[track_id]

        if len(positions) < 2:
            return False, 0.0

        # 检查：在 loiter_time 内，最大位移是否 < loiter_radius
        earliest_in_window = now - self.loiter_time
        window_positions = [(x, y) for x, y, t in positions if t >= earliest_in_window]

        if len(window_positions) < 2:
            return False, 0.0

        # 计算窗口内所有位置到均值中心的最大距离
        arr = np.array(window_positions)
        center = arr.mean(axis=0)
        max_dist = np.max(np.linalg.norm(arr - center, axis=1))

        if max_dist < self.loiter_radius:
            duration = now - positions[0][2]
            if duration >= self.loiter_time:
                conf = min(1.0, duration / (self.loiter_time * 2))
                return True, conf

        return False, 0.0

    # 各类别 vote 最低确认次数
    VOTE_MIN = {
        'falling': 1,    # 紧急，1次即告警
        'climbing': 1,   # 紧急，1次即告警
        'bullying': 2,   # 交互，2次确认
        'fighting': 3,   # 最严格，3次确认（减少站立误判）
    }

    def _vote_smooth(self, track_id, current_label):
        """异常偏向时序平滑，分级响应：
        - falling/climbing（紧急）：窗口内1次即告警
        - bullying：窗口内需2次
        - fighting：窗口内需3次（最严格，减少误报）
        """
        if track_id not in self.history:
            self.history[track_id] = []

        history = self.history[track_id]
        history.append(current_label)

        if len(history) > self.vote_window:
            history.pop(0)

        hist_str = ','.join(history)

        # 统计窗口内非 normal 标签
        anomaly_counts = Counter(h for h in history if h != 'normal')

        if anomaly_counts:
            best_label, best_count = anomaly_counts.most_common(1)[0]
            min_required = self.VOTE_MIN.get(best_label, 2)

            if best_count >= min_required:
                logger.debug(f'  [VOTE] T{track_id} current={current_label} → {best_label}'
                             f'({best_count}>={min_required}) | history=[{hist_str}]')
                return best_label
            else:
                logger.debug(f'  [VOTE] T{track_id} current={current_label} → normal'
                             f'({best_label}只有{best_count}次<{min_required}) | history=[{hist_str}]')
                return 'normal'

        logger.debug(f'  [VOTE] T{track_id} current={current_label} → normal(窗口全normal) | history=[{hist_str}]')
        return 'normal'

    def update_track_position(self, track_id, person_kps, person_scores):
        """在非推理帧也更新位置（更精确的徘徊检测）"""
        if (keypoint_valid(person_kps, person_scores, 11) and
            keypoint_valid(person_kps, person_scores, 12)):
            lhip = get_keypoint(person_kps, 11)
            rhip = get_keypoint(person_kps, 12)
            cx = (lhip[0] + rhip[0]) / 2
            cy = (lhip[1] + rhip[1]) / 2
            if track_id not in self.track_positions:
                self.track_positions[track_id] = []
            self.track_positions[track_id].append((cx, cy, time.time()))

    def migrate_track(self, old_tid, new_tid):
        """将 old_tid 的投票历史和位置迁移到 new_tid"""
        if old_tid in self.history:
            # 旧历史拼到新 track（新 track 可能还没有 history）
            old_hist = self.history.pop(old_tid)
            if new_tid not in self.history:
                self.history[new_tid] = old_hist
            else:
                self.history[new_tid] = old_hist + self.history[new_tid]
                # 截断到 vote_window
                if len(self.history[new_tid]) > self.vote_window:
                    self.history[new_tid] = self.history[new_tid][-self.vote_window:]
            logger.info(f'[REASSOC] RuleEngine: T{old_tid} → T{new_tid} '
                        f'(history={self.history[new_tid]})')
        if old_tid in self.track_positions:
            old_pos = self.track_positions.pop(old_tid)
            if new_tid not in self.track_positions:
                self.track_positions[new_tid] = old_pos
            else:
                self.track_positions[new_tid] = old_pos + self.track_positions[new_tid]
        self._missing_count.pop(old_tid, None)

    def clear_stale_tracks(self, active_track_ids):
        """遮挡宽限：track 消失后保留历史 grace_frames 帧，恢复时继承投票窗口"""
        all_tids = set(self.history.keys()) | set(self.track_positions.keys())
        for tid in all_tids:
            if tid in active_track_ids:
                self._missing_count.pop(tid, None)
            else:
                self._missing_count[tid] = self._missing_count.get(tid, 0) + 1
        stale = [tid for tid, cnt in self._missing_count.items()
                 if cnt > self.grace_frames]
        for tid in stale:
            self.history.pop(tid, None)
            self.track_positions.pop(tid, None)
            del self._missing_count[tid]
