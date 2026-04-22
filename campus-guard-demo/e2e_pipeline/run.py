"""
run.py — 校园安防视频行为感知系统 入口

用法:
  cd campus-guard-demo && python e2e_pipeline/run.py \
    --source demo.mp4 \
    --posec3d-config pyskl/configs/posec3d/finetune_campus_mil.py \
    --posec3d-ckpt models/epoch_50.pth \
    --show

示例:
  %(prog)s --source video.mp4 --posec3d-config cfg.py --posec3d-ckpt ckpt.pth --show
  %(prog)s --source 0 --posec3d-config cfg.py --posec3d-ckpt ckpt.pth --show
  %(prog)s --source rtsp://ip:port/stream --posec3d-config cfg.py --posec3d-ckpt ckpt.pth
  %(prog)s --source /path/to/frames/ --posec3d-config cfg.py --posec3d-ckpt ckpt.pth
"""

import argparse
import logging
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import InferencePipeline


def _derive_source_tag(source):
    """从 --source 派生一个适合做文件名的短标签。
    - 文件路径(mp4/jpg 等): basename 去扩展名
    - 纯数字: cam{id}
    - RTSP/HTTP URL: url_{host}
    - 文件夹: 目录名
    """
    if source is None:
        return 'unknown'
    s = str(source).strip()
    if re.fullmatch(r'\d+', s):
        return f'cam{s}'
    if s.startswith(('rtsp://', 'http://', 'https://', 'rtmp://')):
        m = re.search(r'://([^/:@]+)', s)
        return 'url_' + (m.group(1) if m else 'remote')
    base = os.path.basename(os.path.normpath(s))
    if not base:
        return 'source'
    stem = os.path.splitext(base)[0]
    # 文件名清理: 只留字母数字/下划线/连字符/点
    stem = re.sub(r'[^\w.\-]+', '_', stem).strip('_.')
    return stem or 'source'


def parse_args():
    parser = argparse.ArgumentParser(
        description='校园安防视频行为感知系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # 必选
    parser.add_argument('--source', required=True,
                        help='输入源: 视频文件 / 摄像头ID / RTSP / 帧文件夹')
    parser.add_argument('--posec3d-config', required=True,
                        help='PoseC3D config 路径')
    parser.add_argument('--posec3d-ckpt', required=True,
                        help='PoseC3D checkpoint 路径')

    # 模型
    parser.add_argument('--yolo-pose', default='models/yolo11m-pose.pt',
                        help='YOLO Pose 模型路径 (default: models/yolo11m-pose.pt)')
    parser.add_argument('--small-obj-model',
                        default=None,
                        help='[Legacy] 统一 3 类 YOLO 模型（unified_3class）。若指定则覆盖 3 路单类模型。'
                             '设为 none 则显式禁用')
    parser.add_argument('--falling-model',
                        default=None,
                        help='单类 YOLO 躺地检测模型（输出语义类 falling）。设为 none 禁用')
    parser.add_argument('--smoking-model',
                        default=None,
                        help='单类 YOLO 吸烟检测模型（输出语义类 smoking）。设为 none 禁用')
    parser.add_argument('--phone-model',
                        default=None,
                        help='单类 YOLO 手机检测模型（输出语义类 phone）。设为 none 禁用')
    parser.add_argument('--device', default='cuda:0',
                        help='推理设备 (default: cuda:0)')

    # 推理参数
    parser.add_argument('--yolo-conf', type=float, default=0.3,
                        help='YOLO 置信度阈值 (default: 0.3)')
    parser.add_argument('--stride', type=int, default=16,
                        help='PoseC3D 推理步长 (default: 16)')
    parser.add_argument('--vote-window', type=int, default=5,
                        help='时序投票窗口大小 (default: 5)')
    parser.add_argument('--vote-ratio', type=float, default=0.4,
                        help='时序投票阈值 (default: 0.4)')
    parser.add_argument('--loiter-time', type=float, default=60.0,
                        help='徘徊判定时间（秒）(default: 60)')
    parser.add_argument('--loiter-radius', type=float, default=100.0,
                        help='徘徊判定半径（像素）(default: 100)')

    # 输出
    parser.add_argument('--show', action='store_true',
                        help='实时显示可视化窗口')
    parser.add_argument('--output-video', default=None,
                        help='输出标注视频路径')
    parser.add_argument('--output-json', default=None,
                        help='输出检测事件 JSON 日志路径')

    # Debug
    parser.add_argument('--debug', action='store_true',
                        help='启用详细 debug 日志（输出到 stderr + debug.log）')

    return parser.parse_args()


def main():
    args = parse_args()

    # Debug logging setup
    debug_logger = logging.getLogger('e2e_debug')
    if args.debug:
        debug_logger.setLevel(logging.DEBUG)
        fmt = logging.Formatter('%(message)s')
        # 按视频名+时间戳命名日志,每次运行独立存档
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        tag = _derive_source_tag(args.source)
        ts = time.strftime('%Y%m%d_%H%M%S')
        log_path = os.path.join(log_dir, f'debug_{tag}_{ts}.log')
        fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
        fh.setFormatter(fmt)
        debug_logger.addHandler(fh)
        # 也输出到 stderr
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        debug_logger.addHandler(sh)
        print(f'[DEBUG] Debug logging enabled → {log_path}')
    else:
        debug_logger.setLevel(logging.WARNING)

    def _norm(v):
        return None if (v is None or (isinstance(v, str) and v.lower() == 'none')) else v

    small_obj = _norm(args.small_obj_model)
    falling_model = _norm(args.falling_model)
    smoking_model = _norm(args.smoking_model)
    phone_model = _norm(args.phone_model)

    # 若 legacy --small-obj-model 给了路径，优先使用它（pipeline 里 multi 会被跳过）
    if small_obj:
        falling_model = smoking_model = phone_model = None

    pipeline = InferencePipeline(
        yolo_pose_model=args.yolo_pose,
        posec3d_config=args.posec3d_config,
        posec3d_checkpoint=args.posec3d_ckpt,
        small_obj_model=small_obj,
        falling_model=falling_model,
        smoking_model=smoking_model,
        phone_model=phone_model,
        device=args.device,
        yolo_conf=args.yolo_conf,
        stride=args.stride,
        vote_window=args.vote_window,
        vote_ratio=args.vote_ratio,
        loiter_time=args.loiter_time,
        loiter_radius=args.loiter_radius,
    )

    pipeline.run(
        source=args.source,
        show=args.show,
        output_video=args.output_video,
        output_json=args.output_json,
    )


if __name__ == '__main__':
    main()
