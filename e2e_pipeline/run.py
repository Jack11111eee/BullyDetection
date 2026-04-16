"""
run.py — 校园安防视频行为感知系统 入口

用法:
  cd /home/hzcu/BullyDetection && python e2e_pipeline/run.py \
    --source demo.mp4 \
    --posec3d-config pyskl/configs/posec3d/finetune_campus_mil.py \
    --posec3d-ckpt pyskl/work_dirs/posec3d_campus_mil/epoch_50.pth \
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
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import InferencePipeline


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
    parser.add_argument('--yolo-pose', default='yolo11m-pose.pt',
                        help='YOLO Pose 模型路径 (default: yolo11m-pose.pt)')
    parser.add_argument('--small-obj-model',
                        default='/home/hzcu/yjm/home/yjm/VideoDetection/v6/runs/detect/campus_A28/unified_3class_model/weights/best.pt',
                        help='小物体检测模型路径（phone/smoking/falling），设为 none 则跳过')
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
        # 输出到文件
        fh = logging.FileHandler('debug.log', mode='w', encoding='utf-8')
        fh.setFormatter(fmt)
        debug_logger.addHandler(fh)
        # 也输出到 stderr
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        debug_logger.addHandler(sh)
        print('[DEBUG] Debug logging enabled → debug.log')
    else:
        debug_logger.setLevel(logging.WARNING)

    # 支持 --small-obj-model none 显式禁用
    small_obj = args.small_obj_model
    if small_obj and small_obj.lower() == 'none':
        small_obj = None

    pipeline = InferencePipeline(
        yolo_pose_model=args.yolo_pose,
        posec3d_config=args.posec3d_config,
        posec3d_checkpoint=args.posec3d_ckpt,
        small_obj_model=small_obj,
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
