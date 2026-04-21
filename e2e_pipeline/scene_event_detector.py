"""scene_event_detector.py — 帧级 / scene-level 异常检测器

当前只含 CameraTamperingDetector (R26 P37)：
  - 突然变黑 (brightness)
  - 突然失焦 (Laplacian var)
  - 突然遮挡 (Edge Disappearance Rate)

逻辑源自根目录 step5_camera_tampering.py，去掉 CLI/VideoWriter。
pipeline._process_frame 开头调用 update(frame) 短路：触发即跳过 YOLO/PoseC3D/规则。
"""
import cv2
import numpy as np


class CameraTamperingDetector:
    """镜头遮挡 / 黑屏 / 失焦检测。

    - 首帧作为基准（首帧就被挡住的区域视为"正常背景"，不会因此误报）。
    - 无报警时每 refresh_interval 帧刷新基准，适应光照缓慢变化。
    - 报警时不刷新（防止遮挡物被误采为基准）。
    """

    def __init__(self, edr_threshold=0.83, drop_ratio=0.5, refresh_interval=75,
                 confirm_frames=3):
        # edr_threshold: 基准边缘被破坏的比例阈值（83% 算突发遮挡）
        # drop_ratio: 亮度/清晰度骤降阈值（降到基准 50% 以下算异常）
        # refresh_interval: 静默刷新基准的帧间隔（默认 75 ≈ 25fps × 3s）
        # confirm_frames: 连续触发帧数才算真 tamper（防单帧闪断误报）
        self.edr_threshold = edr_threshold
        self.drop_ratio = drop_ratio
        self.refresh_interval = max(1, int(refresh_interval))
        self.confirm_frames = max(1, int(confirm_frames))
        self.reset()

    def reset(self):
        self.ref_edges = None
        self.ref_brightness = None
        self.ref_focus = None
        self.frame_count = 0
        self._consecutive_tamper = 0

    @staticmethod
    def _extract(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        focus = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        return edges, brightness, focus

    def update(self, frame):
        """处理一帧。

        Returns
        -------
        (is_tamper, alarms) : (bool, list[str])
            alarms 的 str 形如 'blackout' / 'defocus' / 'occlusion(edr=0.92)'
        """
        self.frame_count += 1
        edges, brightness, focus = self._extract(frame)

        # 首帧：设为基准，不判定
        if self.ref_edges is None:
            self.ref_edges = edges
            self.ref_brightness = brightness
            self.ref_focus = focus
            return False, []

        alarms = []

        # 1. 突然变黑
        if brightness < 20 or brightness < self.ref_brightness * self.drop_ratio:
            alarms.append('blackout')

        # 2. 突然失焦（基准本身模糊就不报，避免误报低质输入）
        if self.ref_focus > 50 and focus < self.ref_focus * self.drop_ratio:
            alarms.append('defocus')

        # 3. 突然遮挡（EDR）
        ref_edge_count = int(np.sum(self.ref_edges > 0))
        if ref_edge_count > 100:  # 纯色墙防除零
            common = cv2.bitwise_and(self.ref_edges, edges)
            edr = 1.0 - float(np.sum(common > 0)) / float(ref_edge_count)
            if edr > self.edr_threshold:
                alarms.append(f'occlusion(edr={edr:.2f})')

        if alarms:
            self._consecutive_tamper += 1
        else:
            self._consecutive_tamper = 0
            if self.frame_count % self.refresh_interval == 0:
                self.ref_edges = edges
                self.ref_brightness = brightness
                self.ref_focus = focus

        confirmed = self._consecutive_tamper >= self.confirm_frames
        return confirmed, alarms if confirmed else []


def render_tamper_overlay(frame, alarms):
    """tamper 触发时在帧上渲染红色警示 + 文字。

    与 step5_camera_tampering.py 的渲染风格一致：
    - 顶部红字 "CAMERA TAMPER ALERT!"
    - 下方列出触发的 alarm
    - 整帧泛红 (addWeighted 0.3)
    """
    h, w = frame.shape[:2]
    cv2.putText(frame, 'CAMERA TAMPER ALERT!', (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    y = 90
    for a in alarms:
        cv2.putText(frame, a, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        y += 30
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
