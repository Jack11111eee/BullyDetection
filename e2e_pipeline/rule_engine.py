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
    'normal',           # 0 正常
    'fighting',         # 1 打架
    'bullying',         # 2 霸凌
    'falling',          # 3 摔倒
    'climbing',         # 4 攀爬
    'vandalism',        # 5 破坏公物（规则引擎判定）
    'smoking',          # 6 吸烟（YOLO小物体 + 规则）
    'phone_call',       # 7 打电话（YOLO小物体 + 规则）
    'loitering',        # 8 徘徊（轨迹分析）
    'self_harm',        # 9 R25 自伤（撞墙/扶墙撞头，规则引擎 skeleton 速度判定）
    'camera_tampering', # 10 R26 镜头遮挡/黑屏/失焦（scene-level，短路下游推理，track_id=-1）
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
    role: str = None     # R35: 'perpetrator' / 'victim' / None（仅 bullying 有效）

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
    """吸烟判定：检测到香烟 + 在人上半身附近

    R13 P16：扩展关键点范围 + 骨骼 bbox 上半身兜底。
    旧版只查 鼻子/手腕,漏掉"手拿烟在身侧"(肘部)、"胳膊弯起烟在腰前"等姿势。
    新版：
      1. 关键点范围：鼻子 + 两肘 + 两腕 (kp 0, 7, 8, 9, 10)
      2. 兜底：香烟中心落在人骨骼 bbox 的上半身范围内(y < bbox 中线)也算
    """
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

    # P16：扩展到肘部 (kp 7, 8) —— 覆盖"手肘弯起拿烟"姿势
    check_kps = [0, 7, 8, 9, 10]  # 鼻子、左肘、右肘、左腕、右腕

    # 关键点距离判据(主)
    for cig in cigarettes:
        cig_center = bbox_center(cig['bbox'])
        for kp_idx in check_kps:
            if not keypoint_valid(person_kps, person_scores, kp_idx):
                continue
            kp_pos = get_keypoint(person_kps, kp_idx)
            if distance(cig_center, kp_pos) < ref_dist:
                return True, cig['conf']

    # P16 兜底：香烟中心落在骨骼 bbox 上半身(y < 中线)
    # 场景：手拿烟姿势特殊(如烟放下去 kp 位置不稳),但烟显然在人上半身
    # 下半身兜底不启用 —— 避免裤兜里手机/烟盒等物品误判
    valid = person_scores > 0.3
    if valid.sum() >= 3:
        valid_kps = person_kps[valid]
        x_min, y_min = valid_kps.min(axis=0)
        x_max, y_max = valid_kps.max(axis=0)
        y_mid = (y_min + y_max) / 2
        for cig in cigarettes:
            cx, cy = bbox_center(cig['bbox'])
            if x_min <= cx <= x_max and y_min <= cy <= y_mid:
                logger.debug(f'  [RULE] smoking 骨骼 bbox 上半身兜底: cig=({cx:.0f},{cy:.0f}) '
                             f'in bbox ({x_min:.0f},{y_min:.0f})-({x_max:.0f},{y_mid:.0f})')
                return True, cig['conf'] * 0.8  # 兜底判据稍降置信度

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


def check_vandalism(all_person_kps, person_scores, pose_probs, small_obj_detections,
                    scene_person_count=None):
    """破坏公物：fighting 概率高 + 场景只有 1 个人。

    R12 (P14)：`scene_person_count` 优先于 `len(all_person_kps)` —— 前者包含 grace
    期内被遮挡的 track，后者只看本帧 YOLO 检测结果。修复 fighting 中一方被完全
    遮挡时 `len(all_person_kps)==1` → vandalism 误判。

    注意：持续性检查由 RuleEngine._raw_judge 在窗口层面完成，这里只做单帧判定。
    """
    fighting_prob = pose_probs[1]
    effective_count = scene_person_count if scene_person_count is not None else len(all_person_kps)
    if fighting_prob > 0.5 and effective_count == 1:
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


def _is_upright_posture(kps, scores, img_shape, pose_falling_prob=0.0, threshold=0.3):
    """检查当前姿态是否为直立/坐姿（躯干竖直）— 用于排除 falling 误判。
    返回 True 表示躯干直立，不应判为 falling。

    R29 P47: 加两层防御，避免 2D 投影盲区误否决真摔倒：
      (A) dy > |dx| * 1.2：头-髋向量必须以竖直为主（身体横跨画面的侧横躺直接挡）
      (B) pose_falling_prob >= 0.95：PoseC3D 极度确信 falling 时不信任 2D 几何启发式
          —— PoseC3D 看 64 帧时序骨骼动态是强证据，rule_upright 只看单帧 2D 投影是弱证据
    """
    # 收集 head / hip 坐标（与 _head_above_hip_ratio 一致的 kp 选择）
    head_xy = None
    for idx in [0, 1, 2]:
        if keypoint_valid(kps, scores, idx, threshold):
            head_xy = kps[idx]
            break
    hip_pts = []
    for idx in [11, 12]:
        if keypoint_valid(kps, scores, idx, threshold):
            hip_pts.append(kps[idx])
    if head_xy is None or not hip_pts:
        return False  # 无法判断，不阻止
    hip_xy = np.mean(hip_pts, axis=0) if len(hip_pts) == 2 else hip_pts[0]

    dy = float(hip_xy[1] - head_xy[1])  # >0 = 头在髋上方（图像 Y 更小）
    dx_abs = float(abs(hip_xy[0] - head_xy[0]))
    h = img_shape[0]
    upright_threshold = h * 0.03

    # (1) 基础 dy 门槛
    if dy <= upright_threshold:
        return False

    # (A) 向量横向主导 → 身体横跨画面，不是直立
    if dy <= dx_abs * 1.2:
        logger.debug(f'  [RULE] 姿态直立检查失败: dy={dy:.1f} <= dx={dx_abs:.1f}*1.2 '
                     f'(身体横向主导, 躺地而非直立)')
        return False

    # (B) PoseC3D 极度确信 falling 时豁免（救 2D 盲区: 头朝相机躺, dy 主导但非直立）
    if pose_falling_prob >= 0.95:
        logger.debug(f'  [RULE] 姿态直立检查豁免: PoseC3D falling={pose_falling_prob:.3f}>=0.95 '
                     f'信任 PoseC3D 时序证据 (即使 dy={dy:.1f} 看似直立)')
        return False

    logger.debug(f'  [RULE] 姿态直立检查: dy={dy:.1f} > {upright_threshold:.1f} '
                 f'且 > dx={dx_abs:.1f}*1.2, PoseC3D falling={pose_falling_prob:.3f}<0.95 '
                 f'→ 直立(非falling)')
    return True


def _is_sitting_posture(kps, scores, img_shape, threshold=0.3):
    """检查骨骼包围框纵横比 — 纵向展开=坐/站，横向展开=躺倒。
    只需要任意 3+ 个有效关键点即可计算，不依赖特定关键点。
    返回 True 表示非倒地姿态，不应判为 falling。

    R21 P27: h/w>1.1 之后追加 head-vs-hip 二层校验 —
    头朝相机躺地时骨骼在图像里也是竖直（h/w>1），仅凭纵横比会误判为坐姿。
    头在图像里明显低于髋（head_hip < -h*0.02）→ 翻转结论，判定躺倒。
    头朝远处躺地为 2D 投影盲区，本函数无法区分（交由调用方的 YOLO conf 豁免处理）。
    """
    valid = scores > threshold
    if valid.sum() < 3:
        return False  # 有效关键点太少

    valid_kps = kps[valid]  # (N, 2)
    x_min, y_min = valid_kps.min(axis=0)
    x_max, y_max = valid_kps.max(axis=0)
    bbox_w = x_max - x_min
    bbox_h = y_max - y_min

    if bbox_w < 1:
        return True  # 宽度几乎为0，人是竖直的
    aspect_ratio = bbox_h / bbox_w

    # 纵横比 > 1.1 → 骨骼纵向展开 → 坐着/站着（身体竖直）
    # 纵横比 < 1.1 → 骨骼横向展开 → 躺倒（身体水平）
    if aspect_ratio <= 1.1:
        return False

    # R21 P27: head-vs-hip 第二层校验
    # head_hip = hip_y - head_y，正值=头在髋上方（直立），负值=头在髋下方（头朝相机躺）
    head_hip = _head_above_hip_ratio(kps, scores, threshold)
    if head_hip is not None:
        h = img_shape[0]
        head_below_hip_margin = -h * 0.02
        if head_hip < head_below_hip_margin:
            logger.debug(
                f'  [RULE] h/w={aspect_ratio:.2f}>1.1 但 head_hip={head_hip:.1f}<{head_below_hip_margin:.1f}'
                f' → 头低于髋，判定躺倒（头朝相机）'
            )
            return False

    logger.debug(f'  [RULE] 骨骼纵横比检测: h/w={aspect_ratio:.2f} > 1.1 → 非倒地(坐姿/站姿)')
    return True


def _bbox_overlap_ratio(bbox1, bbox2):
    """计算两个 bbox 的重叠面积占较小 bbox 面积的比例。
    返回 0~1，0=无重叠，1=完全包含较小者。
    用 min(area) 而非 union (IoU) 是因为攻击者站在躺地者上方时，
    bbox 大小可能差异很大，IoU 会偏低导致漏检。
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area1 = max(0, (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]))
    area2 = max(0, (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]))
    smaller = min(area1, area2)
    if smaller <= 0:
        return 0.0
    return inter / smaller


def _has_vertical_movement(track_positions_list, img_h, min_samples=3, ratio=0.05):
    """检查 track 近期是否有显著垂直位移 — 用于验证 climbing。
    climbing 必须有垂直方向运动，坐着的人位置固定。
    Args:
        track_positions_list: [(x, y, timestamp), ...] 该 track 的历史位置
        img_h: 画面高度
        min_samples: 最少需要的位置样本数
        ratio: 垂直位移占画面高度的最小比例（默认5%）
    Returns: True 如果有明显垂直位移
    """
    if not track_positions_list or len(track_positions_list) < min_samples:
        return True  # 数据不足，不阻止（保守策略）

    # 取最近的位置样本
    recent = track_positions_list[-min(30, len(track_positions_list)):]
    ys = [y for _, y, _ in recent]
    y_range = max(ys) - min(ys)
    threshold = img_h * ratio

    has_movement = y_range > threshold
    if not has_movement:
        logger.debug(f'  [RULE] 垂直位移检测: y_range={y_range:.0f} < {threshold:.0f} → 无垂直运动(非climbing)')
    return has_movement


def check_fallen_by_yolo(person_kps, person_scores, small_obj_detections, img_shape):
    """YOLO 辅助摔倒检测：检测到躺地人体 + 与当前人物重叠。
    用于补偿 PoseC3D 对一动不动躺地者的漏检。
    Returns: (is_fallen, confidence, bbox_is_horizontal)
        bbox_is_horizontal: True 如果检测框宽>高（人是水平躺着的）
    """
    # R14 实验：conf 下限过滤 conf > 0.4
    # 抛弃低置信度边界检测（laying 模型对竖直姿态频繁输出 0.3~0.4 假阳）
    # 迭代:初版 (0.7, 0.95) → 回调到 (0.52, 0.95) → 去掉上限只留 > 0.52 → 回调到 > 0.4（放宽以减少漏检风险）
    fallen = [d for d in small_obj_detections
              if d['class'] == 'falling' and d['conf'] > 0.4]
    if not fallen:
        return False, 0.0, False

    valid = person_scores > 0.3
    if valid.sum() == 0:
        return False, 0.0, False
    person_center = person_kps[valid].mean(axis=0)  # (2,)

    for det in fallen:
        x1, y1, x2, y2 = det['bbox']
        # 扩展 bbox 20% 容忍骨骼偏移
        bw, bh = x2 - x1, y2 - y1
        margin = max(bw, bh) * 0.2
        if (x1 - margin <= person_center[0] <= x2 + margin and
                y1 - margin <= person_center[1] <= y2 + margin):
            # R19 P24: 下半身缺失 → 视为坐姿/画面裁切，不判 falling
            # 物理依据：躺着的人身体横向展开，膝/踝至少一个会落入可检测范围;
            # 坐在桌/椅后或画面边缘的人下半身（kp 13-16）常被裁掉，只剩上半身。
            # 三重门槛防误杀：
            #   - 下半身 4 个 kp 全缺（最严，1 个可见即不触发）
            #   - 上半身 ≥ 5 个 kp 可信（骨骼本身质量足够，避免整体漏检时误用此规则）
            #   - 位置内聚到 check_fallen_by_yolo 内部（一次修好，所有 YOLO falling 出口统一受益）
            lower_kp_visible = int(sum(1 for i in [13, 14, 15, 16]
                                       if person_scores[i] > 0.3))
            upper_kp_visible = int(sum(1 for i in range(11)
                                       if person_scores[i] > 0.3))
            if lower_kp_visible == 0 and upper_kp_visible >= 5:
                logger.debug(
                    f'  [RULE] YOLO falling 被下半身缺失否决 '
                    f'(上半身 {upper_kp_visible}/11 可见, 下半身 0/4) → 非倒地'
                )
                return False, 0.0, False
            bbox_horizontal = bw > bh  # 宽>高 → 人水平躺着
            return True, det['conf'], bbox_horizontal

    return False, 0.0, False


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

    # P5 收紧（R9）：OR→AND + 阈值收紧，降低正常场景（身高差、弯腰、蹲下）误触发
    # height_ratio < 0.5 → 一方身高不到另一方的50%（明显蜷缩/倒地）
    # head_hip_normalized > 0.15 → 头-髋差异超过画面高度15%（明显姿态不对称）
    # 两个条件必须同时满足，避免单一噪声触发
    is_asymmetric = height_ratio < 0.5 and head_hip_normalized > 0.15

    if is_asymmetric:
        conf = max(1.0 - height_ratio, head_hip_normalized * 5)
        conf = min(conf, 1.0)
        # R35: 矮的一方 = victim（蜷缩/倒地），高的一方 = perpetrator（站立施暴）
        role = 'victim' if my_height <= other_height else 'perpetrator'
        logger.debug(f'  [RULE] bullying不对称检测: height_ratio={height_ratio:.2f}, '
                     f'head_hip_norm={head_hip_normalized:.3f} → bullying({role})')
        return True, conf, role

    return False, 0.0, None


def check_self_harm(head_vel_hist, hip_vel_hist,
                    exceed_thr=0.08, exceed_window=60,
                    exceed_min_count=2, exceed_max_gap=5,
                    ratio_thr=3.5, ratio_window=60,
                    ratio_head_max_min=0.06, ratio_hip_max_min=0.02,
                    eps=1e-6):
    """R25 撞墙/扶墙撞头自伤判定（基于 skeleton 速度滑窗）。

    输入:
        head_vel_hist: (T,) 归一化 head kp 速度历史（最新 push 在末尾）
        hip_vel_hist:  (T,) 归一化 hip 中心速度历史

    判据（OR，任一满足即 True）:
        A) 簇发判据（R27 P43 收紧）：最近 exceed_window 帧里有 ≥ exceed_min_count
           次 head_vel > exceed_thr，且其中至少两次过阈索引差 ≤ exceed_max_gap。
           物理动机：真撞墙是"加速→撞击→急停"的连续 2-3 帧高速；孤立单帧
           跳变（YOLO-Pose nose kp 漂移）天然不满足"簇发"。
        B) 最近 ratio_window 帧内 max(head_vel) / max(hip_vel) >= ratio_thr
           → 补强路径：headbang 大多数 ratio >= 5，normal max=2.62、impact max=3.29

    返回 (triggered: bool, conf: float, source: str).
    source 用于区分两条路径，便于后续调阈。
    """
    # R32: self_harm 判定逻辑整体禁用（误报过多，待数据重新标定阈值）。
    # 保留函数签名与调用点，直接返回 False —— 不影响 FINAL_CLASSES / VOTE /
    # pipeline 速度滑窗等外围结构；需要恢复时把下面 return 删掉即可。
    return False, 0.0, None
    # --- 以下原判定逻辑暂时禁用 ---
    # if not head_vel_hist or not hip_vel_hist:
    #     return False, 0.0, None
    #
    # # A) 主路径 — 簇发判据
    # head_recent = head_vel_hist[-exceed_window:]
    # exceed_idx = [i for i, v in enumerate(head_recent)
    #               if v is not None and v > exceed_thr]
    # burst = False
    # if len(exceed_idx) >= exceed_min_count:
    #     # 检查是否存在两次过阈索引差 ≤ exceed_max_gap
    #     for i in range(len(exceed_idx) - 1):
    #         if exceed_idx[i + 1] - exceed_idx[i] <= exceed_max_gap:
    #             burst = True
    #             break
    # if burst:
    #     exceed_count = len(exceed_idx)
    #     conf = min(1.0, 0.5 + 0.1 * exceed_count)
    #     logger.debug(f'  [RULE] self_harm A路径簇发: head_vel>{exceed_thr} '
    #                  f'count={exceed_count} (gap≤{exceed_max_gap}) '
    #                  f'in last {exceed_window}f → conf={conf:.2f}')
    #     return True, conf, 'rule_self_harm_vel'
    #
    # # B) 补强路径 —— R28 P46：加双绝对下限
    # #   问题（F637 T10 坐姿误判）：坐姿身体不动 → hip_max≈0.011 → 任何 head
    # #   微抖(0.05) ÷ 0.011 = ratio 飙升。分母接近 0 使 ratio 失控，不是因为
    # #   head 真在快速运动。R25 探查 sample bias：normal 样本(走路) hip 持续
    # #   移动 → ratio max=2.62 看似安全；坐姿未覆盖，hip 极小时 ratio 无上限。
    # #   双门槛物理语义：
    # #     head_max ≥ 0.06  —— head 必须达撞击量级（< A 路径 0.08,留余量给 B 补强）
    # #     hip_max  ≥ 0.02  —— 身体必须有基本动作(扶墙撞头时身体会跟着微晃)
    # hip_recent = hip_vel_hist[-ratio_window:]
    # head_max = max((v for v in head_recent if v is not None), default=0.0)
    # hip_max = max((v for v in hip_recent if v is not None), default=0.0)
    # if head_max < ratio_head_max_min:
    #     logger.debug(f'  [RULE] self_harm B路径门槛失败: '
    #                  f'head_max={head_max:.3f}<{ratio_head_max_min} (head 未达撞击量级)')
    # elif hip_max < ratio_hip_max_min:
    #     ratio_noisy = head_max / (hip_max + eps)
    #     logger.debug(f'  [RULE] self_harm B路径门槛失败: '
    #                  f'hip_max={hip_max:.3f}<{ratio_hip_max_min} '
    #                  f'(身体静止, ratio={ratio_noisy:.2f} 分母不可信)')
    # else:
    #     ratio = head_max / hip_max
    #     if ratio >= ratio_thr:
    #         conf = min(1.0, 0.4 + 0.05 * (ratio - ratio_thr))
    #         logger.debug(f'  [RULE] self_harm B路径: head_max/hip_max={ratio:.2f} '
    #                      f'>= {ratio_thr} (head_max={head_max:.3f}, '
    #                      f'hip_max={hip_max:.3f}) → conf={conf:.2f}')
    #         return True, conf, 'rule_self_harm_ratio'
    #
    # return False, 0.0, None


# ============================================================
# 主判定引擎
# ============================================================
class RuleEngine:
    """
    行为判定规则引擎

    优先级（从高到低）：
    1. PoseC3D 高置信度结果（falling/climbing > 0.7 直接采信）
    2. YOLO 辅助 falling 检测（一动不动躺地，有攻击信号时让路）
    3. 小物体规则（smoking / phone_call）
    4. vandalism 规则
    5. 攻击概率主导（fighting 或 bullying 概率 >= 0.3 触发，
       不依赖 argmax，proximity + asymmetry 强约束）
    6. argmax 路径（normal/falling/climbing 的姿态验证）
    7. 徘徊检测（轨迹分析，5分钟阈值）
    """

    def __init__(self, pose_threshold=0.3, vote_window=5, vote_ratio=0.6,
                 loiter_time=300.0, loiter_radius=100.0, grace_frames=90):
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
        self._last_smoothed = {}    # {track_id: 上次 smoothed 输出的标签}（用于 hysteresis）
        self._last_pose_probs = {}  # R18: {track_id: 最近一次 PoseC3D 输出} 供 demote_unsupported_attacks (E) 用
        self._scene_count_history = []  # R12 P14: 最近 N 次推理的 scene_person_count（场景级）
        self._scene_count_window = 5    # vandalism 持续性窗口
        self._pending_bully_role = None  # R35: _raw_judge → judge 传递 bullying 角色

    def reset(self):
        """R17 修 B：跨 task 状态清空。InferencePipeline.reset() 中调用。"""
        self.history.clear()
        self.track_positions.clear()
        self._missing_count.clear()
        self._last_smoothed.clear()
        self._last_pose_probs.clear()
        self._scene_count_history.clear()

    def judge(self, track_id, pose_probs, person_kps, person_scores,
              all_person_kps, small_obj_detections, img_shape,
              all_person_kps_scores=None, track_kps_dict=None, track_bboxes_dict=None,
              scene_person_count=None,
              head_vel_hist=None, hip_vel_hist=None):
        """
        对单个人物做最终行为判定。

        Args:
            all_person_kps_scores: [(kps, scores), ...] 所有人的关键点+置信度，用于bullying不对称检测
            track_kps_dict: {track_id: (kps, scores)} 用于 YOLO falling 时查询附近 track 的标签历史
            track_bboxes_dict: {track_id: [x1,y1,x2,y2]} 用于 bbox 重叠判定（替代距离）
            scene_person_count: R12 P13 场景人数（本帧检测 + grace 期内 buffered track），
                                用于 vandalism 判定，修复遮挡导致的 len(all_person_kps)==1 误判

        Returns:
            BehaviorResult
        """
        raw_label, raw_conf, source = self._raw_judge(
            track_id, pose_probs, person_kps, person_scores,
            all_person_kps, small_obj_detections, img_shape,
            all_person_kps_scores=all_person_kps_scores,
            track_kps_dict=track_kps_dict,
            track_bboxes_dict=track_bboxes_dict,
            head_vel_hist=head_vel_hist,
            hip_vel_hist=hip_vel_hist,
            scene_person_count=scene_person_count,
        )

        # R18: 缓存 pose_probs 供 demote_unsupported_attacks (E) 用
        self._last_pose_probs[track_id] = pose_probs

        smoothed_label = self._vote_smooth(
            track_id, raw_label, raw_source=source,
            pose_normal_prob=float(pose_probs[0]),
        )

        # R35: 读取 _raw_judge 设置的 bullying 角色
        role = self._pending_bully_role if smoothed_label == 'bullying' else None
        self._pending_bully_role = None

        return BehaviorResult(
            label=smoothed_label,
            confidence=raw_conf,
            source=source,
            smoothed=(smoothed_label != raw_label),
            track_id=track_id,
            timestamp=time.time(),
            role=role,
        )

    def _raw_judge(self, track_id, pose_probs, person_kps, person_scores,
                   all_person_kps, small_obj_detections, img_shape,
                   all_person_kps_scores=None, track_kps_dict=None, track_bboxes_dict=None,
                   scene_person_count=None,
                   head_vel_hist=None, hip_vel_hist=None):
        """原始判定（不含时序平滑）"""

        pose_label_idx = int(np.argmax(pose_probs))
        pose_conf = float(pose_probs[pose_label_idx])
        pose_label = POSE_CLASSES[pose_label_idx]

        probs_str = ' '.join(f'{POSE_CLASSES[i]}={pose_probs[i]:.3f}' for i in range(len(POSE_CLASSES)))
        logger.debug(f'  [RAW] T{track_id} PoseC3D: argmax={pose_label}({pose_conf:.3f}) | {probs_str}')

        # 1. 高置信度的紧急行为（仍需姿态验证）
        if pose_label == 'climbing' and pose_conf > 0.7:
            positions = self.track_positions.get(track_id, [])
            if _has_vertical_movement(positions, img_shape[0]):
                logger.debug(f'  [RAW] T{track_id} → climbing (高置信度+有垂直位移)')
                return 'climbing', pose_conf, 'posec3d'
            else:
                logger.debug(f'  [RAW] T{track_id} → normal (climbing但无垂直位移, 可能坐着)')
                return 'normal', 1.0 - pose_conf, 'rule_no_vertical'
        if pose_label == 'falling' and pose_conf > 0.7:
            if _is_upright_posture(person_kps, person_scores, img_shape,
                                   pose_falling_prob=float(pose_probs[3])):
                logger.debug(f'  [RAW] T{track_id} → normal (高置信度falling但躯干直立, 可能是坐下)')
                return 'normal', 1.0 - pose_conf, 'rule_upright'
            if _is_sitting_posture(person_kps, person_scores, img_shape):
                logger.debug(f'  [RAW] T{track_id} → normal (高置信度falling但检测到坐姿)')
                return 'normal', 1.0 - pose_conf, 'rule_sitting'
            logger.debug(f'  [RAW] T{track_id} → falling (高置信度+非直立+非坐姿)')
            return 'falling', pose_conf, 'posec3d'

        # 1.5 R25 自伤（撞墙 / 扶墙撞头）—— R32 禁用
        # 误报过多（坐姿头转动 / 写字 / PoseC3D 灰度带均会触发），暂时整块注释。
        # 函数 check_self_harm 已短路 return False；外围 FINAL_CLASSES / VOTE /
        # pipeline 速度滑窗保留不动，恢复只需把下面块解注释即可。
        # 位置理由：
        #   - step 1 高置信 falling/climbing (> 0.7) 被判定后直接 return，不会走到这里；
        #     真摔倒 / 真攀爬不会被自伤抢走
        #   - step 1 的退出分支（rule_no_vertical / rule_upright / rule_sitting）后继续
        #     走 self_harm —— 那些是"PoseC3D 误判的 normal"，若 head 速度高仍可能是撞墙
        #   - 放 step 2 之前：撞墙时 YOLO laying 可能误触发 falling bbox，
        #     self_harm 比 rule_yolo_falling 更准确
        # 探查数据支持（probe R24 二轮，见 TRAINING_LOG R25）：
        #   head_vel_exceed_008_w60 ≥ 1: normal 0/7, impact 4/4, headbang 7/16
        #   head_to_hip_peak_ratio_w60 ≥ 3.5: normal max=2.62, impact max=3.29, headbang p50=5.9
        # if head_vel_hist is not None and hip_vel_hist is not None:
        #     # R27 P42: PoseC3D normal_prob >= 0.9 强 normal 前置 veto
        #     # 探查 normal 样本仅 7 个走路视频，未覆盖坐姿头部转动 / 看屏幕等场景
        #     # （F1130 T9 坐姿被 self_harm 误判 = R25 已知局限 #1 兑现）。
        #     # PoseC3D 极度确信 normal 时不入态，比事后退 HOLD 更根本。
        #     # R30 P48: 扩展灰度带 —— normal 显著主导（>= 0.7 且 >= 2× 异常类最大值）
        #     # F1247 T1 坐桌前写字：normal=0.756 fighting=0.240 → P42 不触发但 A 路径误判。
        #     # PoseC3D 说 normal 主语义明确时（即使不到 0.9），skeleton 速度应让路。
        #     normal_prob_sh = float(pose_probs[0])
        #     attack_max_sh = max(float(pose_probs[i]) for i in (1, 2, 3, 4))
        #     strong_normal = normal_prob_sh >= 0.9
        #     dominant_normal = (normal_prob_sh >= 0.7 and
        #                        normal_prob_sh >= attack_max_sh * 2)
        #     if strong_normal:
        #         logger.debug(f'  [RAW] T{track_id} self_harm 前置veto (P42 强normal): '
        #                      f'PoseC3D normal={normal_prob_sh:.3f}>=0.9 → 跳过 self_harm 判据')
        #     elif dominant_normal:
        #         logger.debug(f'  [RAW] T{track_id} self_harm 前置veto (P48 normal主导): '
        #                      f'PoseC3D normal={normal_prob_sh:.3f}>=0.7 且 '
        #                      f'>=2*异常类max({attack_max_sh:.3f}) → 跳过 self_harm 判据')
        #     else:
        #         triggered, conf_sh, src_sh = check_self_harm(head_vel_hist, hip_vel_hist)
        #         if triggered:
        #             logger.debug(f'  [RAW] T{track_id} → self_harm ({src_sh}, conf={conf_sh:.2f})')
        #             return 'self_harm', conf_sh, src_sh

        # 2. YOLO 辅助 falling 检测（一动不动躺地，PoseC3D 识别不出）
        #    躺地 + 附近有被标为 fighting/bullying 的人 → bullying（被霸凌）
        #    躺地 + 附近无攻击者 → falling
        # P7 (R9)：YOLO falling 信号若被 PoseC3D 弱攻击信号抢先跳过，需在 step 6 之后
        # 确保被消费，避免 attack_prob 没过门槛或 proximity 失败时信号被 normal 吞掉。
        yolo_falling_deferred = None  # (conf, bbox_horizontal) if 暂存给 step 6 后消费
        is_fallen_yolo, fallen_yolo_conf, bbox_horizontal = check_fallen_by_yolo(
            person_kps, person_scores, small_obj_detections, img_shape
        )
        if is_fallen_yolo:
            # 用 bbox 重叠（而非距离）判断攻击者：在 3D 纵深场景中
            # 攻击者站在躺地者上方时两人 bbox 必然重叠，距离判断不可靠
            my_bbox = track_bboxes_dict.get(track_id) if track_bboxes_dict else None
            # P8 (R10)：PoseC3D 否决门 —— fighting 强主导时 YOLO 躺地视为误检
            # 场景：fighting 中肢体交缠 / 倾斜，YOLO unified_3class 对此类姿态有误检，
            # 日志 F546–F627 显示 PoseC3D fighting=0.998 被 step 2 无视 → bullying 自激 82 帧
            # R17：加 proximity 前置门 —— 孤立场景下 fighting 信号语义失效
            # （fighting 必须有对象），不应用其否决 YOLO falling
            # 日志 F15228 显示孤立摔倒者 PoseC3D fighting=0.775 噪声把真摔倒信号吞掉
            fighting_prob_pre = float(pose_probs[1])
            bullying_prob_pre = float(pose_probs[2])
            proximity_ok_pre = False
            nearest_dist_pre_log = -1.0
            max_fight_dist_pre_log = -1.0
            if all_person_kps_scores is not None and len(all_person_kps_scores) >= 2:
                my_height_pre = _person_height(person_kps, person_scores)
                nearest_dist_pre, neighbor_height_pre = _nearest_person_dist(
                    person_kps, person_scores, all_person_kps_scores
                )
                ref_height_pre = max(my_height_pre or 0, neighbor_height_pre or 0)
                max_fight_dist_pre = ((ref_height_pre * 1.5)
                                      if ref_height_pre > 0
                                      else img_shape[0] * 0.25)
                nearest_dist_pre_log = nearest_dist_pre
                max_fight_dist_pre_log = max_fight_dist_pre
                proximity_ok_pre = (0 < nearest_dist_pre <= max_fight_dist_pre)
            yolo_veto = (proximity_ok_pre and
                         fighting_prob_pre >= 0.7 and
                         bullying_prob_pre < fighting_prob_pre * 0.3)
            if yolo_veto:
                logger.debug(
                    f'  [RAW] T{track_id} YOLO躺地否决 '
                    f'(PoseC3D fighting={fighting_prob_pre:.3f}强主导, '
                    f'bullying={bullying_prob_pre:.3f}, proximity_ok '
                    f'nearest={nearest_dist_pre_log:.0f}<={max_fight_dist_pre_log:.0f}) '
                    f'→ 跳过rule_yolo_bullying/falling'
                )
            elif (fighting_prob_pre >= 0.7 and
                  bullying_prob_pre < fighting_prob_pre * 0.3 and
                  not proximity_ok_pre):
                logger.debug(
                    f'  [RAW] T{track_id} YOLO躺地否决跳过 '
                    f'(fighting={fighting_prob_pre:.3f}强主导但proximity失败 '
                    f'nearest={nearest_dist_pre_log:.0f}>{max_fight_dist_pre_log:.0f}, '
                    f'孤立场景视fighting为PoseC3D噪声)'
                )
            if my_bbox and track_bboxes_dict and len(track_bboxes_dict) >= 2 and not yolo_veto:
                for other_tid, other_bbox in track_bboxes_dict.items():
                    if other_tid == track_id:
                        continue
                    overlap = _bbox_overlap_ratio(my_bbox, other_bbox)
                    if overlap < 0.1:  # 重叠不足 10% → 不算近距离互动
                        continue
                    # P8 (R10)：邻居条件收紧 —— 只有 bullying 才算 bullying 的证据
                    # 原逻辑 {fighting, bullying} 均触发；但邻居是 fighting 反而说明
                    # 两人是对称对打，不应升级本 track 为 bullying 受害者
                    other_history = self.history.get(other_tid, [])
                    other_smoothed = self._last_smoothed.get(other_tid, 'normal')
                    neighbor_bullying = other_smoothed == 'bullying'
                    recent_bullying = any(h == 'bullying' for h in other_history[-3:])
                    if neighbor_bullying or recent_bullying:
                        bully_count = sum(1 for h in other_history if h == 'bullying')
                        logger.debug(
                            f'  [RAW] T{track_id} → bullying (YOLO躺地 + T{other_tid} '
                            f'smoothed={other_smoothed}, recent_bullying={recent_bullying}, '
                            f'history_bully={bully_count}/{len(other_history)}, overlap={overlap:.2f})'
                        )
                        # 双向传播（修复 C）：向邻居 raw history 注入一次 bullying，
                        # 帮助攻击者靠滞回维持攻击状态（不覆盖 smoothed，只加弱证据）
                        self._inject_raw_history(other_tid, 'bullying')
                        self._pending_bully_role = 'victim'
                        return 'bullying', fallen_yolo_conf, 'rule_yolo_bullying'
            # PoseC3D 已检出 fighting/bullying 弱信号（>=0.3）时，跳过 YOLO falling，
            # 让 step 6 走攻击概率主导逻辑（proximity + asymmetry）。
            # YOLO falling 本意是补偿「一动不动+PoseC3D 输出 normal」的盲区，
            # 不应覆盖 PoseC3D 已有的攻击信号。
            fighting_prob_early = float(pose_probs[1])
            bullying_prob_early = float(pose_probs[2])
            if yolo_veto:
                # P8 (R10)：fighting 强主导时 YOLO 检测整体不可信，既不判 bullying 也不判 falling
                # 不设置 yolo_falling_deferred —— 让 step 6 正常返回 fighting
                pass
            elif fighting_prob_early >= self.pose_threshold or bullying_prob_early >= self.pose_threshold:
                # P7 (R9)：暂存 YOLO falling 信号给 step 6 之后兜底消费，
                # 不直接丢弃 — 否则 step 6 攻击判定失败时会被 normal argmax 吞掉。
                yolo_falling_deferred = (fallen_yolo_conf, bbox_horizontal)
                logger.debug(f'  [RAW] T{track_id} YOLO躺地但PoseC3D有攻击信号 '
                             f'(conf={fallen_yolo_conf:.3f}, '
                             f'fighting={fighting_prob_early:.3f}, bullying={bullying_prob_early:.3f}), '
                             f'暂存YOLO falling给step 6后兜底')
            else:
                # R15 修复 B：骨骼坐姿软否决（压误报）
                # 条件：骨骼纵横比纵向展开 + 至少 8 个有效关键点 + PoseC3D normal >= 0.25
                # 三重门槛是为了在"骨骼可信 + 模型交叉验证"时才否决，避免遮挡/低质骨骼导致漏检
                # R21 P28: YOLO 高 conf 豁免 — 头朝远处躺地骨骼 2D 盲区，若 YOLO conf>=0.7 则信任检测器
                # R34: 加 normal<0.9 门槛 — PoseC3D 极度确信 normal 时不豁免，
                #   让 _is_sitting_posture 裁决（坐着 h/w>1.1 → veto; 真躺地 h/w<1.1 → 放行）
                valid_kp_count = int((person_scores > 0.3).sum())
                normal_prob_here = float(pose_probs[0])
                if fallen_yolo_conf >= 0.7 and normal_prob_here < 0.9:
                    logger.debug(
                        f'  [RAW] T{track_id} YOLO conf={fallen_yolo_conf:.3f}>=0.6 '
                        f'normal={normal_prob_here:.3f}<0.9 → 高置信豁免坐姿否决'
                    )
                elif (valid_kp_count >= 8 and
                        _is_sitting_posture(person_kps, person_scores, img_shape) and
                        normal_prob_here >= 0.25):
                    logger.debug(
                        f'  [RAW] T{track_id} YOLO falling 被坐姿否决 '
                        f'(conf={fallen_yolo_conf:.3f}, valid_kp={valid_kp_count}, '
                        f'normal={normal_prob_here:.3f}, bbox={"水平" if bbox_horizontal else "竖直"})'
                    )
                    return 'normal', 1.0 - fallen_yolo_conf, 'rule_sitting_veto_yolo'
                # 信任 YOLO falling 检测
                logger.debug(f'  [RAW] T{track_id} → falling (YOLO辅助检测: conf={fallen_yolo_conf:.3f}, '
                             f'bbox={"水平" if bbox_horizontal else "竖直"})')
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
        # R12 P14: 用 scene_person_count（含 grace 期遮挡 track）代替 len(all_person_kps)
        # 且要求持续性 —— 近 _scene_count_window 次推理场景人数全为 1 才判 vandalism
        # 避免 fighting 中一方瞬时被遮挡 → 1 人 → vandalism 误判
        is_vandal, vandal_conf = check_vandalism(
            all_person_kps, person_scores, pose_probs, small_obj_detections,
            scene_person_count=scene_person_count,
        )
        if is_vandal:
            history = self._scene_count_history
            window = self._scene_count_window
            sustained_solo = (len(history) >= window and
                              all(c == 1 for c in history[-window:]))
            if sustained_solo:
                logger.debug(f'  [RAW] T{track_id} → vandalism (规则引擎: '
                             f'fighting={pose_probs[1]:.3f}, scene={scene_person_count}, '
                             f'持续{window}次单人)')
                return 'vandalism', vandal_conf, 'rule_vandalism'
            else:
                logger.debug(f'  [RAW] T{track_id} vandalism 未达持续性 '
                             f'(scene_history={history[-window:] if history else []}, '
                             f'需{window}次全1)，不触发')

        # 6. 攻击概率主导路径（fighting 或 bullying 概率 >= threshold）
        #    不依赖 argmax —— 只要任一攻击类概率过 0.3，就进入完整的攻击判定流程。
        #    这是为了捕获 PoseC3D 输出被 normal 分散的弱攻击信号
        #    （例: normal=0.5, fighting=0.4, 施暴者站在躺地者上方的典型分布）。
        #    保留 proximity + asymmetry 强约束防止误报。
        # P1 (R9)：增加相对优势判定 —— 仅在 attack_prob 不被 normal_prob 压倒时触发，
        # 避免 [normal=0.60, fighting=0.32, ...] 这种 normal 主导分布也进入攻击路径。
        fighting_prob = float(pose_probs[1])
        bullying_prob = float(pose_probs[2])
        normal_prob = float(pose_probs[0])
        attack_prob = max(fighting_prob, bullying_prob)

        # 相对优势：attack_prob 必须至少是 normal_prob 的 70%
        # 例: normal=0.50 要求 attack>=0.35；normal=0.40 要求 attack>=0.30（与绝对阈值一致）
        relative_ok = attack_prob >= normal_prob * 0.7

        if attack_prob >= self.pose_threshold and relative_ok:
            # 6a. proximity 必要条件：孤立个体不算攻击
            proximity_ok = True
            nearest_dist_log = -1.0
            max_fight_dist_log = -1.0
            if all_person_kps_scores is not None and len(all_person_kps_scores) >= 2:
                my_height = _person_height(person_kps, person_scores)
                nearest_dist, neighbor_height = _nearest_person_dist(
                    person_kps, person_scores, all_person_kps_scores
                )
                ref_height = max(my_height or 0, neighbor_height or 0)
                max_fight_dist = (ref_height * 1.5) if ref_height > 0 else img_shape[0] * 0.25
                nearest_dist_log = nearest_dist
                max_fight_dist_log = max_fight_dist
                if nearest_dist > max_fight_dist:
                    proximity_ok = False
            else:
                # 场景只有 1 人 → 无攻击对象
                proximity_ok = False

            if proximity_ok:
                # 6b. 不对称检测：一方倒地/蜷缩 → bullying
                is_bully, bully_conf, bully_role = check_bullying_asymmetry(
                    person_kps, person_scores, all_person_kps_scores, img_shape
                )
                if is_bully:
                    logger.debug(f'  [RAW] T{track_id} → bullying (攻击概率'
                                 f'f={fighting_prob:.3f},b={bullying_prob:.3f} + 姿态不对称)')
                    # 双向传播：向 bbox 重叠最多的邻居 raw history 注入 bullying
                    # （攻击者和受害者是同一事件，必须同步升级 ENTRY 计数）
                    self._inject_to_overlapping_neighbor(
                        track_id, 'bullying', track_bboxes_dict
                    )
                    self._pending_bully_role = bully_role
                    return 'bullying', attack_prob * 0.9, 'rule_bullying'

                # 6c. 对称攻击 → 按概率选 fighting 或 bullying
                # P10 (R10)：bullying 需显著优势才采信 —— PoseC3D R11 训练数据中
                # bullying 样本仅 446（fighting 12772，28:1 不平衡），bullying_prob 在
                # fighting 场景下边界不稳（常见 b=0.49,f=0.51 翻转）。
                # 旧阈值 `b >= f` 导致单帧概率抖动即翻转标签。
                # 新阈值：b >= f*1.5 且 b >= 0.4 —— 要求 bullying 显著压倒且绝对置信度足够。
                if bullying_prob >= fighting_prob * 1.5 and bullying_prob >= 0.4:
                    logger.debug(f'  [RAW] T{track_id} → bullying (攻击概率显著优势: '
                                 f'b={bullying_prob:.3f}>=f={fighting_prob:.3f}*1.5且>=0.4)')
                    self._inject_to_overlapping_neighbor(
                        track_id, 'bullying', track_bboxes_dict
                    )
                    # R35: step 6c 用高度比较决定角色
                    my_h = _person_height(person_kps, person_scores)
                    best_other_h = None
                    for o_kps, o_sc in all_person_kps_scores:
                        if np.array_equal(o_kps, person_kps):
                            continue
                        oh = _person_height(o_kps, o_sc)
                        if oh is not None:
                            if best_other_h is None or oh > best_other_h:
                                best_other_h = oh
                    if my_h is not None and best_other_h is not None:
                        self._pending_bully_role = 'victim' if my_h <= best_other_h else 'perpetrator'
                    return 'bullying', bullying_prob, 'posec3d'
                else:
                    logger.debug(f'  [RAW] T{track_id} → fighting (攻击概率: '
                                 f'f={fighting_prob:.3f}, b={bullying_prob:.3f} 未达bullying门槛)')
                    self._inject_to_overlapping_neighbor(
                        track_id, 'fighting', track_bboxes_dict
                    )
                    return 'fighting', max(fighting_prob, attack_prob), 'posec3d'
            else:
                logger.debug(f'  [RAW] T{track_id} 攻击信号但proximity失败 (nearest='
                             f'{nearest_dist_log:.0f} > {max_fight_dist_log:.0f})，落到argmax路径')
        elif attack_prob >= self.pose_threshold and not relative_ok:
            # P1 日志：被 relative_ok 阻挡的 case
            logger.debug(f'  [RAW] T{track_id} attack_prob={attack_prob:.3f}过线但被normal压制 '
                         f'(normal={normal_prob:.3f}, 要求attack>=normal*0.7={normal_prob*0.7:.3f})')

        # P7 (R9)：step 6 攻击判定没成功返回 → 消费暂存的 YOLO falling 信号
        # 避免 proximity 失败 / attack_prob 相对劣势 / PoseC3D argmax=normal 吞掉 YOLO 信号
        if yolo_falling_deferred is not None:
            conf, horizontal = yolo_falling_deferred
            # R16 P20：P7 兜底路径同样加坐姿软否决（与 step 2 主路径 R15 P18 对称）
            # 场景：PoseC3D 弱攻击信号触发 defer，但被 normal 压制失败 → 落到 P7 兜底
            # 日志 F2575/F2591：T3 坐着被持续判 falling；R15 P18 只覆盖主路径未覆盖此处
            # R21 P28: YOLO 高 conf 豁免（头朝远处躺地 2D 盲区兜底，与 P18 对称）
            # R34: 加 normal<0.9 门槛（与 P18 对称）
            valid_kp_count = int((person_scores > 0.3).sum())
            normal_prob_here = float(pose_probs[0])
            if conf >= 0.6 and normal_prob_here < 0.9:
                logger.debug(
                    f'  [RAW] T{track_id} P7兜底 YOLO conf={conf:.3f}>=0.6 '
                    f'normal={normal_prob_here:.3f}<0.9 → 高置信豁免坐姿否决'
                )
            elif (valid_kp_count >= 8 and
                    _is_sitting_posture(person_kps, person_scores, img_shape)):
                # R33: P7 路径去掉 normal_prob >= 0.25 门槛。
                # P7 进入条件要求 PoseC3D 有攻击信号(≥0.3) → normal 被压低 →
                # normal >= 0.25 在 P7 上结构性失效。
                # 真正一动不动躺地(YOLO 兜底核心场景)走 step 2 主路径,不走 P7。
                # step 2 主路径(P18) normal >= 0.25 门槛保留不动。
                logger.debug(
                    f'  [RAW] T{track_id} P7兜底YOLO falling 被坐姿否决 '
                    f'(conf={conf:.3f}, valid_kp={valid_kp_count}, '
                    f'normal={normal_prob_here:.3f}, bbox={"水平" if horizontal else "竖直"})'
                )
                return 'normal', 1.0 - conf, 'rule_sitting_veto_yolo'
            logger.debug(f'  [RAW] T{track_id} → falling (step 6未判攻击, YOLO falling兜底, '
                         f'conf={conf:.3f}, bbox={"水平" if horizontal else "竖直"})')
            return 'falling', conf, 'rule_yolo_falling'

        # 6d. argmax 路径（处理 normal/falling/climbing；fighting/bullying 已在上方处理）
        if pose_conf >= self.pose_threshold:
            final_label = pose_label

            # fighting/bullying argmax 但 proximity 失败 → 降级 normal
            if final_label in ('fighting', 'bullying'):
                logger.debug(f'  [RAW] T{track_id} → normal (argmax={final_label}但proximity失败)')
                return 'normal', 1.0 - pose_conf, 'rule_no_proximity'

            # climbing 垂直位移验证：无垂直运动不算 climbing
            if final_label == 'climbing':
                positions = self.track_positions.get(track_id, [])
                if not _has_vertical_movement(positions, img_shape[0]):
                    logger.debug(f'  [RAW] T{track_id} → normal (climbing但无垂直位移, 可能坐着)')
                    return 'normal', 1.0 - pose_conf, 'rule_no_vertical'

            # falling 姿态验证：躯干直立或坐姿不算 falling
            if final_label == 'falling':
                if _is_upright_posture(person_kps, person_scores, img_shape,
                                       pose_falling_prob=float(pose_probs[3])):
                    logger.debug(f'  [RAW] T{track_id} → normal (falling但躯干直立, 可能是坐下)')
                    return 'normal', 1.0 - pose_conf, 'rule_upright'
                if _is_sitting_posture(person_kps, person_scores, img_shape):
                    logger.debug(f'  [RAW] T{track_id} → normal (falling但检测到坐姿)')
                    return 'normal', 1.0 - pose_conf, 'rule_sitting'

            logger.debug(f'  [RAW] T{track_id} → {final_label} (conf={pose_conf:.3f} >= threshold={self.pose_threshold})')
            return final_label, pose_conf, 'posec3d'

        # 7. 检查徘徊（低于 PoseC3D 优先级，避免覆盖 bullying/fighting）
        is_loiter, loiter_conf = self._check_loitering(track_id, person_kps, person_scores)
        if is_loiter:
            return 'loitering', loiter_conf, 'rule_loitering'

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

    # 滞回阈值（修复 A）
    # ENTRY_MIN：进入该异常态所需的窗口内计数
    # HOLD_MIN：已在该异常态时维持所需的窗口内计数（更低，防止闪断）
    VOTE_ENTRY_MIN = {
        'falling': 2,    # 紧急，2次确认即进入
        'climbing': 2,
        'bullying': 3,   # 交互，3次确认
        'fighting': 3,   # 最严格，3次确认（减少站立误判）
        'smoking': 2,    # R13 P17: 稀疏信号(烟小+间歇遮挡),2次确认
        'phone_call': 2,
        'self_harm': 1,  # R25: 判据本身已内嵌 60 帧窗口持续性,单次 RAW 即可入态
    }
    # P2 (R9)：攻击类 HOLD_MIN 1→2，降低永锁风险。
    # 配合 P1（attack_prob 相对优势）+ P5（asymmetry 收紧）+ pair coupling 注入，
    # HOLD=1 在 5 窗口内只要 1 帧异常就维持，等价于进入后几乎永远退不出。
    # 攻击类改为 2（5 窗口 2/5 维持），防止单帧噪声把异常态锁死。
    # falling/climbing 保持 1（姿态延续性高，倒地/攀爬一旦发生很少一闪而过）。
    VOTE_HOLD_MIN = {
        'falling': 1,    # 进入后只要 1 次即维持
        'climbing': 1,
        'bullying': 2,   # R9: 1→2 防永锁
        'fighting': 2,   # R9: 1→2 防永锁
        'smoking': 2,    # R13 P17: 小物体检测稀疏,2次维持防闪断
        'phone_call': 2,
        'self_harm': 1,  # R27 P41: 判据已内嵌 60 帧持续性, HOLD=1 只需防闪断
    }
    # R13 P17：per-label 投票窗口 —— smoking 等稀疏信号用更长窗口累积证据
    # 默认所有标签用 self.vote_window (=5),此处 override
    VOTE_WINDOW_LABEL = {
        'smoking': 7,     # 吸烟是持续几分钟行为,但单帧检测稀疏,放大窗口
        'phone_call': 7,
        'self_harm': 3,   # R27 P41: 判据 60 帧内爆发, VOTE 层只做小窗口平滑防单帧抖动
    }

    def _window_for(self, label):
        """返回指定标签的投票窗口长度"""
        return self.VOTE_WINDOW_LABEL.get(label, self.vote_window)

    def _max_window(self):
        """history buffer 应保留的最大长度(所有标签窗口的最大值)"""
        return max(self.vote_window, *self.VOTE_WINDOW_LABEL.values())

    def _count_in_label_window(self, history, label):
        """在 label 专属窗口内计 label 出现次数"""
        window = self._window_for(label)
        return sum(1 for h in history[-window:] if h == label)

    def _vote_smooth(self, track_id, current_label, raw_source=None,
                     pose_normal_prob=None):
        """异常偏向时序平滑 + 滞回阈值（R13 P17：per-label 投票窗口）
        - 进入异常：ENTRY_MIN（严格，避免误报）
        - 维持异常：HOLD_MIN（宽松，避免告警闪断）
        - 不同标签用不同窗口长度: 攻击/倒地类用 self.vote_window (=5),
          smoking/phone_call 用 7 (稀疏信号需更长观察面)

        R15 修复 C：raw_source 为姿态物理规则级的 normal 时（rule_upright/
        rule_sitting/rule_no_vertical/rule_sitting_veto_yolo），强制退出 HOLD
        —— 姿态规则是"强证据 normal"，权重高于时序惯性。

        R18 P22 (Solution B)：当 PoseC3D 自身明确 normal_prob >= 0.9 且 raw_source=
        'posec3d' 时，配合最近 3 帧 raw 至少 2 帧 normal 一致性检查，强制退出
        fighting/bullying HOLD —— 解决 pair-inference 污染 + HOLD 锁死场景
        （日志 F7210 T3 静坐被误判 fighting）。
        """
        if track_id not in self.history:
            self.history[track_id] = []

        history = self.history[track_id]
        history.append(current_label)

        # 用最大窗口截断,保证所有标签的 window 都能拿到完整数据
        if len(history) > self._max_window():
            history.pop(0)

        hist_str = ','.join(history)

        # per-label 计数：每个异常标签用它自己的窗口
        anomaly_labels = set(h for h in history if h != 'normal')
        anomaly_counts = {L: self._count_in_label_window(history, L) for L in anomaly_labels}
        last = self._last_smoothed.get(track_id, 'normal')

        def _top_anomaly():
            """返回 (label, count) 或 (None, 0)"""
            if not anomaly_counts:
                return None, 0
            best = max(anomaly_counts, key=anomaly_counts.get)
            return best, anomaly_counts[best]

        # R15 修复 C：强证据 normal 否决 HOLD
        # 姿态物理规则（rule_upright / rule_sitting / rule_no_vertical /
        # rule_sitting_veto_yolo）级别的 normal 权重高于时序惯性，强制退出 falling/climbing HOLD
        STRONG_NORMAL_SOURCES = {
            'rule_upright', 'rule_sitting', 'rule_no_vertical',
            'rule_sitting_veto_yolo',
        }
        if (current_label == 'normal' and raw_source in STRONG_NORMAL_SOURCES
                and last in ('falling', 'climbing')):
            logger.debug(f'  [VOTE] T{track_id} current=normal → normal '
                         f'(STRONG_NORMAL src={raw_source} 否决 HOLD {last}) '
                         f'| history=[{hist_str}]')
            self._last_smoothed[track_id] = 'normal'
            return 'normal'

        # R18 P22 (Solution B)：PoseC3D 强 normal 否决异常态 HOLD
        # 场景：pair inference 污染（T_self 是旁观者但与施暴者配对 → RAW 被带偏 fighting）
        #      或 _inject_to_overlapping_neighbor / couple_overlapping_pairs 写入攻击态
        #      → HOLD 在 fighting 窗口里硬撑。
        # R22 扩展：last 从 (fighting, bullying) 扩到 (fighting, bullying, falling, climbing) ——
        #      F1439 场景 T1 站起后 PoseC3D normal=0.934，但 last=falling HOLD 锁死。
        #      P19 STRONG_NORMAL_SOURCES 只接 rule_*（即时退出），P22 配合 recent3 一致性
        #      接 posec3d source，覆盖剩余的所有异常态 HOLD。
        # 触发条件（三重防误退）：
        #   a) 当前 raw label='normal' 且 raw_source='posec3d'（不是 rule_no_proximity 等）
        #   b) PoseC3D normal_prob >= 0.9（自身输出极度明确）
        #   c) 最近 3 帧 raw 至少 2 帧 normal（持续性，避免真施暴/摔倒 1 帧抖动误退）
        # 失败则走原 HOLD 逻辑（真异常者窗口里本来就有多数异常 raw，a/c 不会同时满足）。
        if (current_label == 'normal'
                and raw_source == 'posec3d'
                and pose_normal_prob is not None
                and pose_normal_prob >= 0.9
                and last in ('fighting', 'bullying', 'falling', 'climbing', 'self_harm')):
            recent3 = history[-3:]  # 已 append 当前 label，含本帧
            if len(recent3) >= 2 and recent3.count('normal') >= 2:
                logger.debug(f'  [VOTE] T{track_id} current=normal → normal '
                             f'(STRONG_NORMAL posec3d normal={pose_normal_prob:.3f} '
                             f'否决 HOLD {last}, recent3={recent3}) '
                             f'| history=[{hist_str}]')
                self._last_smoothed[track_id] = 'normal'
                return 'normal'

        # 1) 滞回维持：上次为异常 L，只要 L 在 L 的窗口内 ≥ HOLD_MIN[L] 就继续
        #    但若另一个异常(按各自窗口计数)严格多于 L，允许切换
        if last != 'normal':
            last_count = anomaly_counts.get(last, 0)
            hold_min = self.VOTE_HOLD_MIN.get(last, 1)
            if last_count >= hold_min:
                top_label, top_count = _top_anomaly()
                if top_label is not None and top_label != last and top_count > last_count:
                    logger.debug(f'  [VOTE] T{track_id} current={current_label} → {top_label}'
                                 f' (UPGRADE {top_label}:{top_count}>{last}:{last_count}) | history=[{hist_str}]')
                    self._last_smoothed[track_id] = top_label
                    return top_label
                logger.debug(f'  [VOTE] T{track_id} current={current_label} → {last}'
                             f' (HOLD {last_count}>={hold_min}, window={self._window_for(last)}) | history=[{hist_str}]')
                self._last_smoothed[track_id] = last
                return last

        # 2) 进入逻辑：选 per-label 窗口内计数最高的异常，达到 ENTRY_MIN 才切换
        best_label, best_count = _top_anomaly()
        if best_label is not None:
            entry_min = self.VOTE_ENTRY_MIN.get(best_label, 2)
            if best_count >= entry_min:
                logger.debug(f'  [VOTE] T{track_id} current={current_label} → {best_label}'
                             f' (ENTRY {best_count}>={entry_min}, window={self._window_for(best_label)}) '
                             f'| history=[{hist_str}]')
                self._last_smoothed[track_id] = best_label
                return best_label
            logger.debug(f'  [VOTE] T{track_id} current={current_label} → normal'
                         f' ({best_label}只有{best_count}次<ENTRY{entry_min}, window={self._window_for(best_label)}) '
                         f'| history=[{hist_str}]')
            self._last_smoothed[track_id] = 'normal'
            return 'normal'

        logger.debug(f'  [VOTE] T{track_id} current={current_label} → normal(窗口全normal) | history=[{hist_str}]')
        self._last_smoothed[track_id] = 'normal'
        return 'normal'

    def _inject_raw_history(self, track_id, label):
        """双向传播（修复 C）：向指定 track 的 raw history 追加一条弱证据标签。
        避免重复注入（最后一项已是该标签则跳过），并截断到 vote_window。
        不修改 _last_smoothed —— 邻居自己的 _vote_smooth 会自然处理。

        P9 (R10)：注入条件化 —— 邻居最近 3 帧 ≥2 帧 fighting 时拒绝注入 bullying。
        场景：两人对称 fighting 时，一方因 YOLO 误检短暂判 bullying 会向对方
        cross-inject bullying；若对方 history 正稳定在 fighting，这条注入会污染
        ENTRY 计数使对方也 UPGRADE 到 bullying，形成自激传播（日志 F546–F627 即为此）。
        """
        if track_id not in self.history:
            self.history[track_id] = []
        hist = self.history[track_id]
        if hist and hist[-1] == label:
            return  # 已注入，避免重复撑满
        if label == 'bullying':
            recent = hist[-3:]
            if len(recent) >= 2 and recent.count('fighting') >= 2:
                logger.debug(
                    f'  [INJECT] T{track_id} 拒绝bullying注入 '
                    f'(最近{len(recent)}帧fighting={recent.count("fighting")}, '
                    f'history=[{",".join(hist)}])'
                )
                return
        hist.append(label)
        if len(hist) > self._max_window():
            hist.pop(0)
        logger.debug(f'  [INJECT] T{track_id} raw_history += {label} | history=[{",".join(hist)}]')

    def _inject_to_overlapping_neighbor(self, track_id, label, track_bboxes_dict, min_overlap=0.1):
        """向 bbox 重叠最多的邻居注入攻击标签弱证据。
        R8.5 扩展双向传播：asymmetry/对称攻击判定时同步升级邻居 ENTRY 计数，
        避免「bbox 重叠但一个 bullying 一个 normal」的结构性不合理。

        R20 P26：target 若 PoseC3D 自身 normal_prob ≥ 0.9 时拒绝注入 —— 强 normal
        证据说明 target 没在攻击，注入只会污染它的 history 误导后续 self_active 判定
        （日志 F959：T1 误判 fighting → 向 T3(normal=1.000) 注入 fighting → 污染 E 判据）。
        """
        if not track_bboxes_dict:
            return
        my_bbox = track_bboxes_dict.get(track_id)
        if not my_bbox:
            return
        best_tid, best_overlap = None, min_overlap
        for other_tid, other_bbox in track_bboxes_dict.items():
            if other_tid == track_id:
                continue
            ov = _bbox_overlap_ratio(my_bbox, other_bbox)
            if ov > best_overlap:
                best_overlap = ov
                best_tid = other_tid
        if best_tid is None:
            return
        # R20 P26：target 强 normal 守门
        target_probs = self._last_pose_probs.get(best_tid)
        if target_probs is not None and len(target_probs) >= 1:
            target_normal = float(target_probs[0])
            if target_normal >= 0.9:
                logger.debug(
                    f'  [INJECT-SKIP] T{track_id}→T{best_tid} '
                    f'(overlap={best_overlap:.2f}) {label}: '
                    f'target normal={target_normal:.3f} >= 0.9 拒绝注入'
                )
                return
        logger.debug(f'  [INJECT] T{track_id}→T{best_tid} (overlap={best_overlap:.2f}) {label}')
        self._inject_raw_history(best_tid, label)

    def couple_overlapping_pairs(self, judgments, track_bboxes_dict, min_overlap=0.1):
        """Pair coupling 后处理（R8.5）：
        bbox 重叠 ≥ min_overlap 的 track 对必须共享攻击状态。
        - A 在攻击态 {fighting, bullying} + B normal → B 升级为 A 的标签
        - A bullying + B falling → B 升级 bullying（受害者在霸凌场景中应判 bullying）
        - 同步 _last_smoothed 让后续 HOLD 生效

        Args:
            judgments: {track_id: BehaviorResult} 已经 smooth 后的判定
            track_bboxes_dict: {track_id: bbox}
        Returns:
            judgments (就地修改 label/confidence/source)
        """
        if not track_bboxes_dict or len(judgments) < 2:
            return judgments

        ATTACK_LABELS = ('fighting', 'bullying')
        # 找出所有当前在攻击态的 track
        attackers = [(tid, j) for tid, j in judgments.items()
                     if j.label in ATTACK_LABELS]
        if not attackers:
            return judgments

        for tid, j in list(judgments.items()):
            if j.label in ATTACK_LABELS:
                continue  # 已在攻击态
            my_bbox = track_bboxes_dict.get(tid)
            if not my_bbox:
                continue
            # 找 overlap 最高的攻击者
            best_atk_tid, best_atk_label, best_overlap = None, None, min_overlap
            for a_tid, a_j in attackers:
                if a_tid == tid:
                    continue
                a_bbox = track_bboxes_dict.get(a_tid)
                if not a_bbox:
                    continue
                ov = _bbox_overlap_ratio(my_bbox, a_bbox)
                if ov > best_overlap:
                    best_overlap = ov
                    best_atk_tid = a_tid
                    best_atk_label = a_j.label
            if best_atk_tid is None:
                continue
            # R20 P25：target 强 normal 守门 —— 自身 PoseC3D normal_prob ≥ 0.9 时
            # 拒绝被升级。场景：攻击者 PoseC3D 误判 fighting，target 却明确 normal，
            # bbox overlap 只是摄像头角度造成 —— 不该因物理重叠就继承对方误判标签。
            # 日志 F959：T1 误判 fighting(0.906) overlap T3(normal=1.000) → 旧行为升级 T3 为 fighting，
            # 新行为直接跳过 → T1 失去 partner → 由 E 降级 → 单人 fighting 被消灭。
            target_probs = self._last_pose_probs.get(tid)
            if target_probs is not None and len(target_probs) >= 1:
                target_normal = float(target_probs[0])
                if target_normal >= 0.9:
                    logger.info(
                        f'  [COUPLE-SKIP] T{tid} normal={target_normal:.3f} >= 0.9, '
                        f'拒绝被 T{best_atk_tid} 升级 {best_atk_label} '
                        f'(overlap={best_overlap:.2f})'
                    )
                    continue
            # 应用耦合：
            #   normal / loitering / 其他轻量态 → 升级为攻击态
            #   falling → 如果攻击者是 bullying，升级为 bullying（受害者被霸凌）
            #   falling + 攻击者是 fighting → 保持 falling（倒地独立事件）
            if j.label == 'falling' and best_atk_label != 'bullying':
                continue
            new_label = best_atk_label
            # 受害者场景：bullying + falling overlap → bullying
            if j.label == 'falling' and best_atk_label == 'bullying':
                new_label = 'bullying'
            old_label = j.label
            j.label = new_label
            j.source = f'pair_couple({best_atk_tid})'
            j.smoothed = True
            # R35: couple 升级到 bullying 时根据升级前 label 决定角色
            if new_label == 'bullying':
                if old_label == 'falling':
                    j.role = 'victim'
                else:
                    j.role = 'perpetrator'
            self._last_smoothed[tid] = new_label
            logger.info(f'  [COUPLE] T{tid} {old_label} → {new_label} '
                        f'(T{best_atk_tid} {best_atk_label}, overlap={best_overlap:.2f})')
        return judgments

    def demote_unsupported_attacks(self, judgments, track_bboxes_dict,
                                    min_overlap=0.1, recent_n=3):
        """R18 P23 (Solution E): 攻击态需 partner + self_active 双重支持。

        用户原则: fighting/bullying 是 ≥2 人行为，不存在孤立行为。
        post-couple 阶段对每个被标为攻击态的 track 检查两个条件:
          条件 1 (partner): 场景里另一 track 处于攻击上下文 + bbox overlap ≥ min_overlap
            R31 修复 (1): partner 候选扩到"当前攻击态 ∪ 近 recent_n 帧 raw 含攻击
                         ∪ _last_smoothed ∈ 攻击态" —— 覆盖施暴者 PoseC3D 间歇期
          条件 2 (self_active): 近 recent_n 帧 raw 含攻击 OR 当前 attack_prob ≥ 0.3

        任一失败即降级 normal。覆盖 4 条污染路径:
          P-1 pair inference RAW pollution  (T_self 与施暴者配对 → PoseC3D 输出 fighting)
          P-2 _inject_to_overlapping_neighbor (邻居施暴时主动注入)
          P-3 couple_overlapping_pairs propagation (couple 升级 normal 邻居)
          P-4 HOLD inheritance (history 残留攻击 entries 被 HOLD 锁住)

        R31 修复 (2): pair-based source 白名单 —— rule_yolo_bullying / rule_bullying /
        pair_couple(*) 这些判定本身就基于邻居证据形成，再做 partner 检查重复且危险
        （间歇期被打成 normal 会反过来吞掉合法判定）。

        Args:
            judgments: {track_id: BehaviorResult} couple 后的判定
            track_bboxes_dict: {track_id: [x1,y1,x2,y2]}
            min_overlap: bbox overlap 阈值
            recent_n: self_active 检查的 raw history 窗口
        Returns:
            judgments (就地修改 label/source)
        """
        if not judgments:
            return judgments

        ATTACK_LABELS = ('fighting', 'bullying')
        PAIR_BASED_SOURCES_PREFIX = ('rule_yolo_bullying', 'rule_bullying', 'pair_couple')
        attack_tids = [tid for tid, j in judgments.items() if j.label in ATTACK_LABELS]
        if not attack_tids:
            return judgments

        # R31 修复 (1): partner 候选池扩展 —— 不限于"当前攻击态"，也接受
        #   - 近 recent_n 帧 raw history 含攻击（施暴者 PoseC3D 刚打过、当前 PoseC3D 噪声输出 normal）
        #   - _last_smoothed ∈ 攻击态（施暴者 HOLD 中但本帧 VOTE 刚退出）
        # 涵盖 judgments 以外的所有活跃 track（未推理帧也算）
        partner_candidate_tids = set(attack_tids)
        all_tids = set(judgments.keys())
        if track_bboxes_dict:
            all_tids |= set(track_bboxes_dict.keys())
        all_tids |= set(self.history.keys())
        all_tids |= set(self._last_smoothed.keys())
        for other_tid in all_tids:
            if other_tid in partner_candidate_tids:
                continue
            recent_raw = self.history.get(other_tid, [])[-recent_n:]
            if any(r in ATTACK_LABELS for r in recent_raw):
                partner_candidate_tids.add(other_tid)
                continue
            if self._last_smoothed.get(other_tid) in ATTACK_LABELS:
                partner_candidate_tids.add(other_tid)

        # Pass 1: 决定降级（基于降级前的 attack_tids 集合）
        demote = {}  # tid → (old_label, reason)
        for tid in attack_tids:
            j = judgments[tid]

            # R31 修复 (2): pair-based source 白名单豁免
            #   rule_yolo_bullying: YOLO 躺地 + 邻居 smoothed=bullying / 近 3 帧含 bullying
            #   rule_bullying:      proximity_ok + asymmetry + 双向 inject
            #   pair_couple(*):     couple_overlapping_pairs 基于 overlap 对的升级
            # 三者本身就是基于多人证据形成的判定，再让 demote 审一次 partner 等价于双重惩罚
            src = j.source or ''
            if any(src.startswith(p) for p in PAIR_BASED_SOURCES_PREFIX):
                logger.debug(f'  [DEMOTE-SKIP] T{tid} {j.label} source={src} '
                             f'属 pair-based 判定，跳过 partner/self_active 检查')
                continue

            # 条件 1: partner check（从扩展后的候选池找）
            my_bbox = track_bboxes_dict.get(tid) if track_bboxes_dict else None
            partner_tid = None
            best_ov = min_overlap
            if my_bbox:
                for other_tid in partner_candidate_tids:
                    if other_tid == tid:
                        continue
                    other_bbox = track_bboxes_dict.get(other_tid)
                    if not other_bbox:
                        continue
                    ov = _bbox_overlap_ratio(my_bbox, other_bbox)
                    if ov >= best_ov:
                        best_ov = ov
                        partner_tid = other_tid
            if partner_tid is None:
                demote[tid] = (j.label, f'isolated_no_partner({j.label})')
                continue

            # 条件 2: self_active check
            recent_raw = self.history.get(tid, [])[-recent_n:]
            recent_attack = any(r in ATTACK_LABELS for r in recent_raw)
            cur_probs = self._last_pose_probs.get(tid)
            cur_attack_prob = 0.0
            if cur_probs is not None and len(cur_probs) >= 3:
                cur_attack_prob = max(float(cur_probs[1]), float(cur_probs[2]))
            self_active = recent_attack or cur_attack_prob >= 0.3
            if not self_active:
                demote[tid] = (
                    j.label,
                    f'bystander({j.label}, partner=T{partner_tid} ov={best_ov:.2f}, '
                    f'recent_raw={recent_raw}, cur_attack={cur_attack_prob:.2f})'
                )

        # Pass 2: 应用降级
        for tid, (old_label, reason) in demote.items():
            j = judgments[tid]
            j.label = 'normal'
            j.role = None
            j.source = f'demote_{reason.split("(")[0]}'
            j.smoothed = True
            self._last_smoothed[tid] = 'normal'
            logger.info(f'  [DEMOTE] T{tid} {old_label} → normal ({reason})')
        return judgments

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
                # 截断到 max_window (兼容 per-label 窗口,如 smoking=7)
                max_w = self._max_window()
                if len(self.history[new_tid]) > max_w:
                    self.history[new_tid] = self.history[new_tid][-max_w:]
            logger.info(f'[REASSOC] RuleEngine: T{old_tid} → T{new_tid} '
                        f'(history={self.history[new_tid]})')
        if old_tid in self.track_positions:
            old_pos = self.track_positions.pop(old_tid)
            if new_tid not in self.track_positions:
                self.track_positions[new_tid] = old_pos
            else:
                self.track_positions[new_tid] = old_pos + self.track_positions[new_tid]
        # 迁移滞回状态：优先保留旧 track 的 smoothed（新 track 通常尚未进入异常态）
        if old_tid in self._last_smoothed:
            old_smoothed = self._last_smoothed.pop(old_tid)
            if self._last_smoothed.get(new_tid, 'normal') == 'normal':
                self._last_smoothed[new_tid] = old_smoothed
        # R18: 迁移 _last_pose_probs
        if old_tid in self._last_pose_probs:
            old_probs = self._last_pose_probs.pop(old_tid)
            self._last_pose_probs.setdefault(new_tid, old_probs)
        self._missing_count.pop(old_tid, None)

    def push_scene_count(self, scene_person_count):
        """R12 P14：pipeline 每帧推理前调用一次，记录场景人数。
        与 per-track judge 解耦，避免同帧多 track 重复追加。
        """
        self._scene_count_history.append(int(scene_person_count))
        if len(self._scene_count_history) > self._scene_count_window * 2:
            self._scene_count_history = self._scene_count_history[-self._scene_count_window:]

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
            self._last_smoothed.pop(tid, None)
            self._last_pose_probs.pop(tid, None)  # R18
            del self._missing_count[tid]
