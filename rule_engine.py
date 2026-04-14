"""
rule_engine.py — 行为判定规则引擎

输入：
  - PoseC3D 预测结果（5类概率）
  - YOLO11s 小物体检测结果（bbox + class）
  - YOLO11m-Pose 骨骼关键点（17个COCO关键点）
  - 轨迹信息（可选，用于徘徊检测）

输出：
  - 最终行为标签 + 置信度

COCO 17 关键点索引：
  0-鼻子  1-左眼  2-右眼  3-左耳  4-右耳
  5-左肩  6-右肩  7-左肘  8-右肘  9-左腕  10-右腕
  11-左髋 12-右髋 13-左膝 14-右膝 15-左踝 16-右踝
"""

import numpy as np


# ============================================================
# 行为类别定义
# ============================================================
# PoseC3D 输出的 5 类（Round 6）
POSE_CLASSES = ['normal', 'fighting', 'bullying', 'falling', 'climbing']

# 最终系统输出的完整类别（含规则引擎补充的类别）
FINAL_CLASSES = [
    'normal',       # 0 正常
    'fighting',     # 1 打架
    'bullying',     # 2 霸凌
    'falling',      # 3 摔倒
    'climbing',     # 4 攀爬
    'vandalism',    # 5 破坏公物（规则引擎判定）
    'smoking',      # 6 吸烟（YOLO小物体 + 规则）
    'phone_call',   # 7 打电话（YOLO小物体 + 规则）
]


# ============================================================
# 工具函数
# ============================================================
def bbox_center(bbox):
    """获取 bbox 中心点 (cx, cy)"""
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def point_in_bbox(point, bbox, margin=0):
    """判断点是否在 bbox 内（可选扩展边距）"""
    x, y = point
    x1, y1, x2, y2 = bbox
    return (x1 - margin <= x <= x2 + margin) and (y1 - margin <= y <= y2 + margin)


def distance(p1, p2):
    """两点欧氏距离"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def get_keypoint(keypoints, idx):
    """获取第 idx 个关键点坐标 (x, y)，keypoints shape: (17, 2) 或 (17, 3)"""
    kp = keypoints[idx]
    return kp[0], kp[1]


def keypoint_valid(keypoints, scores, idx, threshold=0.3):
    """判断关键点是否有效"""
    if scores is not None:
        return scores[idx] > threshold
    return keypoints[idx][0] > 0 or keypoints[idx][1] > 0


# ============================================================
# 规则判定函数
# ============================================================
def check_smoking(person_kps, person_scores, small_obj_detections, img_shape):
    """
    吸烟判定：检测到香烟 + 在人手或嘴附近

    Args:
        person_kps: (17, 2) 当前人物的关键点坐标
        person_scores: (17,) 关键点置信度
        small_obj_detections: list of dict {'class': str, 'bbox': [x1,y1,x2,y2], 'conf': float}
        img_shape: (H, W)

    Returns:
        (bool, float) 是否吸烟, 置信度
    """
    cigarettes = [d for d in small_obj_detections if d['class'] == 'cigarette']
    if not cigarettes:
        return False, 0.0

    # 参考距离：用肩宽估算合理的"附近"范围
    h, w = img_shape
    ref_dist = h * 0.08  # 默认用图像高度的 8% 作为距离阈值

    if (keypoint_valid(person_kps, person_scores, 5) and
        keypoint_valid(person_kps, person_scores, 6)):
        shoulder_w = distance(get_keypoint(person_kps, 5), get_keypoint(person_kps, 6))
        if shoulder_w > 10:
            ref_dist = shoulder_w * 0.8

    # 检查每个检测到的香烟是否在嘴/手附近
    mouth_kps = [0]          # 鼻子（近似嘴部位置）
    hand_kps = [9, 10]       # 左腕、右腕

    check_kps = mouth_kps + hand_kps

    for cig in cigarettes:
        cig_center = bbox_center(cig['bbox'])
        for kp_idx in check_kps:
            if not keypoint_valid(person_kps, person_scores, kp_idx):
                continue
            kp_pos = get_keypoint(person_kps, kp_idx)
            dist = distance(cig_center, kp_pos)
            if dist < ref_dist:
                return True, cig['conf']

    return False, 0.0


def check_phone_call(person_kps, person_scores, small_obj_detections, img_shape):
    """
    打电话判定：检测到手机 + 在耳朵附近 + 手腕抬起

    Args: 同 check_smoking
    Returns: (bool, float)
    """
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

    # 耳朵关键点
    ear_kps = [3, 4]  # 左耳、右耳

    for phone in phones:
        phone_center = bbox_center(phone['bbox'])
        for kp_idx in ear_kps:
            if not keypoint_valid(person_kps, person_scores, kp_idx):
                continue
            kp_pos = get_keypoint(person_kps, kp_idx)
            dist = distance(phone_center, kp_pos)
            if dist < ref_dist:
                return True, phone['conf']

    return False, 0.0


def check_vandalism(person_kps, person_scores, pose_probs, small_obj_detections):
    """
    破坏公物判定（规则引擎补回 vandalism）：
    条件：PoseC3D 判定为 fighting 但只检测到 1 个人（打物体而非打人）

    这是一个启发式规则，后续可根据实际表现调整。

    Args:
        person_kps: 当前帧所有人的关键点 list
        person_scores: 当前帧所有人的关键点置信度 list
        pose_probs: PoseC3D 输出的 5 类概率 [normal, fighting, bullying, falling, climbing]
        small_obj_detections: YOLO 小物体检测结果

    Returns:
        (bool, float)
    """
    fighting_prob = pose_probs[1]  # fighting 概率

    # 规则：fighting 概率高 + 场景中只有 1 个人 → 可能是破坏公物
    if fighting_prob > 0.5 and len(person_kps) == 1:
        return True, fighting_prob * 0.8  # 置信度打个折扣

    return False, 0.0


# ============================================================
# 主判定引擎
# ============================================================
class RuleEngine:
    """
    行为判定规则引擎

    优先级（从高到低）：
    1. PoseC3D 高置信度结果（falling/climbing > 0.7 直接采信）
    2. 小物体规则（smoking / phone_call）
    3. vandalism 规则
    4. PoseC3D 默认结果
    """

    def __init__(self, pose_threshold=0.5, vote_window=5, vote_ratio=0.6):
        """
        Args:
            pose_threshold: PoseC3D 结果的最低置信度
            vote_window: 时序投票窗口大小（连续 N 次推理）
            vote_ratio: 投票通过比例（M/N 中的 M/N）
        """
        self.pose_threshold = pose_threshold
        self.vote_window = vote_window
        self.vote_ratio = vote_ratio
        # 每个 track_id 的历史判定结果（用于时序平滑）
        self.history = {}  # {track_id: [label_str, label_str, ...]}

    def judge(self, track_id, pose_probs, person_kps, person_scores,
              all_person_kps, small_obj_detections, img_shape):
        """
        对单个人物做最终行为判定

        Args:
            track_id: ByteTrack 分配的人物 ID
            pose_probs: np.array (5,) PoseC3D 5类概率
            person_kps: (17, 2) 该人物的关键点
            person_scores: (17,) 该人物关键点置信度
            all_person_kps: list 当前帧所有人的关键点（用于判断场景人数）
            small_obj_detections: list of {'class': str, 'bbox': [x1,y1,x2,y2], 'conf': float}
            img_shape: (H, W)

        Returns:
            dict: {
                'label': str,        # 最终行为标签
                'confidence': float,  # 置信度
                'source': str,        # 判定来源 ('posec3d' / 'rule_smoking' / 'rule_phone' / 'rule_vandalism')
                'smoothed': bool,     # 是否经过时序平滑
            }
        """
        raw_label, raw_conf, source = self._raw_judge(
            pose_probs, person_kps, person_scores,
            all_person_kps, small_obj_detections, img_shape
        )

        # 时序平滑
        smoothed_label = self._vote_smooth(track_id, raw_label)

        return {
            'label': smoothed_label,
            'confidence': raw_conf,
            'source': source,
            'smoothed': smoothed_label != raw_label,
        }

    def _raw_judge(self, pose_probs, person_kps, person_scores,
                   all_person_kps, small_obj_detections, img_shape):
        """原始判定（不含时序平滑）"""

        pose_label_idx = int(np.argmax(pose_probs))
        pose_conf = float(pose_probs[pose_label_idx])
        pose_label = POSE_CLASSES[pose_label_idx]

        # 1. 高置信度的紧急行为直接采信
        if pose_label in ('falling', 'climbing') and pose_conf > 0.7:
            return pose_label, pose_conf, 'posec3d'

        # 2. 检查吸烟
        is_smoking, smoke_conf = check_smoking(
            person_kps, person_scores, small_obj_detections, img_shape
        )
        if is_smoking:
            return 'smoking', smoke_conf, 'rule_smoking'

        # 3. 检查打电话
        is_phone, phone_conf = check_phone_call(
            person_kps, person_scores, small_obj_detections, img_shape
        )
        if is_phone:
            return 'phone_call', phone_conf, 'rule_phone'

        # 4. 检查破坏公物
        is_vandal, vandal_conf = check_vandalism(
            all_person_kps, person_scores, pose_probs, small_obj_detections
        )
        if is_vandal:
            return 'vandalism', vandal_conf, 'rule_vandalism'

        # 5. 默认用 PoseC3D 结果
        if pose_conf >= self.pose_threshold:
            return pose_label, pose_conf, 'posec3d'

        return 'normal', 1.0 - pose_conf, 'posec3d_default'

    def _vote_smooth(self, track_id, current_label):
        """
        M/N 帧时序投票平滑

        同一个 track_id，最近 N 次判定中，某个标签出现 >= M 次才采信。
        否则返回 'normal'。
        """
        if track_id not in self.history:
            self.history[track_id] = []

        history = self.history[track_id]
        history.append(current_label)

        # 保留最近 N 次
        if len(history) > self.vote_window:
            history.pop(0)

        # normal 不需要投票，直接通过
        if current_label == 'normal':
            return 'normal'

        # 统计当前标签在窗口内出现的次数
        count = sum(1 for h in history if h == current_label)
        min_votes = int(self.vote_window * self.vote_ratio)

        if count >= min_votes:
            return current_label

        # 未达到投票阈值，保守返回 normal
        return 'normal'

    def clear_track(self, track_id):
        """清除已消失的 track 的历史"""
        self.history.pop(track_id, None)

    def clear_stale_tracks(self, active_track_ids):
        """清除所有不在活跃列表中的 track 历史"""
        stale = [tid for tid in self.history if tid not in active_track_ids]
        for tid in stale:
            del self.history[tid]


# ============================================================
# 使用示例
# ============================================================
if __name__ == '__main__':
    engine = RuleEngine(pose_threshold=0.5, vote_window=5, vote_ratio=0.6)

    # 模拟数据
    fake_pose_probs = np.array([0.1, 0.05, 0.05, 0.7, 0.1])  # falling 高概率
    fake_kps = np.random.rand(17, 2) * 500
    fake_scores = np.ones(17) * 0.9
    fake_small_objs = []  # 没有检测到小物体
    fake_img_shape = (1080, 1920)

    result = engine.judge(
        track_id=1,
        pose_probs=fake_pose_probs,
        person_kps=fake_kps,
        person_scores=fake_scores,
        all_person_kps=[fake_kps],
        small_obj_detections=fake_small_objs,
        img_shape=fake_img_shape,
    )
    print(f"判定结果: {result}")
