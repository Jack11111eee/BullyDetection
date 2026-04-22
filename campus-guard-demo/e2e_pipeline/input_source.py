"""
input_source.py — 统一输入源抽象

支持三种输入：
  1. 视频文件 (mp4, avi, ...)
  2. 摄像头 / RTSP 流 (0, 1, rtsp://...)
  3. 帧文件夹 (jpg/png 序列)
"""

import os
import glob
import cv2
import numpy as np


class InputSource:
    """统一的帧读取接口"""

    def read(self):
        """读取下一帧。Returns: (success: bool, frame: np.ndarray)"""
        raise NotImplementedError

    def get_fps(self):
        """返回帧率"""
        return 25.0

    def get_frame_size(self):
        """返回 (width, height)"""
        raise NotImplementedError

    def get_total_frames(self):
        """返回总帧数，实时流返回 -1"""
        return -1

    def release(self):
        """释放资源"""
        pass

    @staticmethod
    def create(source):
        """
        根据 source 类型自动创建对应的 InputSource。

        Args:
            source: 视频路径 / 摄像头ID(数字字符串或int) / RTSP URL / 帧文件夹路径
        """
        source_str = str(source)

        if os.path.isdir(source_str):
            return FrameFolderSource(source_str)
        elif source_str.isdigit():
            return CameraSource(int(source_str))
        elif source_str.startswith('rtsp://') or source_str.startswith('http://'):
            return CameraSource(source_str)
        else:
            return VideoFileSource(source_str)


class VideoFileSource(InputSource):
    """视频文件输入"""

    def __init__(self, path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Video not found: {path}")
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        self.path = path

    def read(self):
        return self.cap.read()

    def get_fps(self):
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        return fps if fps > 0 else 25.0

    def get_frame_size(self):
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h

    def get_total_frames(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def release(self):
        self.cap.release()


class CameraSource(InputSource):
    """摄像头 / RTSP 流输入"""

    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera/stream: {source}")
        self.source = source

    def read(self):
        return self.cap.read()

    def get_fps(self):
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        return fps if fps > 0 else 25.0

    def get_frame_size(self):
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h

    def get_total_frames(self):
        return -1  # 实时流没有总帧数

    def release(self):
        self.cap.release()


class FrameFolderSource(InputSource):
    """帧文件夹输入（按文件名排序读取）"""

    EXTENSIONS = ('*.jpg', '*.jpeg', '*.png', '*.bmp')

    def __init__(self, folder_path, fps=25.0):
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Frame folder not found: {folder_path}")

        self.files = []
        for ext in self.EXTENSIONS:
            self.files.extend(glob.glob(os.path.join(folder_path, ext)))
        self.files.sort()

        if not self.files:
            raise RuntimeError(f"No image files found in {folder_path}")

        self.idx = 0
        self.fps = fps

        # 读第一帧获取尺寸
        sample = cv2.imread(self.files[0])
        self.h, self.w = sample.shape[:2]

    def read(self):
        if self.idx >= len(self.files):
            return False, None
        frame = cv2.imread(self.files[self.idx])
        self.idx += 1
        if frame is None:
            return False, None
        return True, frame

    def get_fps(self):
        return self.fps

    def get_frame_size(self):
        return self.w, self.h

    def get_total_frames(self):
        return len(self.files)

    def release(self):
        pass
