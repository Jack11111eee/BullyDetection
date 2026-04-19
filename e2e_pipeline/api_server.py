"""
api_server.py — 校园安防模型 REST + SSE 服务

提供给 Web 后端（Spring Boot）调用的 HTTP API:
  POST /api/v1/analyze/start              上传视频 + metadata → 返回 taskId
  GET  /api/v1/analyze/{taskId}/stream    SSE 推送 frame/alert/done/error 事件
  POST /api/v1/analyze/{taskId}/stop      停止任务
  GET  /api/v1/analyze/{taskId}/status    查询任务状态与进度
  GET  /health                            健康检查（模型是否加载）

启动:
  cd /home/hzcu/BullyDetection
  python e2e_pipeline/api_server.py \\
    --posec3d-config pyskl/configs/posec3d/finetune_campus_mil.py \\
    --posec3d-ckpt pyskl/work_dirs/posec3d_campus_mil/epoch_50.pth \\
    --host 0.0.0.0 --port 8000

依赖:
  pip install fastapi uvicorn[standard] sse-starlette python-multipart
"""

import argparse
import asyncio
import logging
import os
import queue
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
import uvicorn

from pipeline import InferencePipeline


logger = logging.getLogger('api_server')


# ============================================================
# 标签映射：pipeline 内部名 → WEB 枚举
# WEB-HANDOFF §7 定义的 5 类 + 对接商定新增 3 类
# ============================================================
LABEL_MAP = {
    # 原 WEB-HANDOFF 定义
    'fighting':   ('FIGHT',     'fight',     '打架'),
    'falling':    ('FALL',      'fall',      '跌倒'),
    'smoking':    ('SMOKING',   'smoking',   '抽烟'),
    'climbing':   ('CLIMB',     'climb',     '翻越围栏'),
    'loitering':  ('LOITER',    'loiter',    '徘徊'),
    # 新增枚举（需 Web 同学同步扩展后端）
    'bullying':   ('BULLY',     'bully',     '霸凌'),
    'phone_call': ('PHONE',     'phone',     '打电话'),
    'vandalism':  ('VANDALISM', 'vandalism', '破坏公物'),
}

LEVEL_MAP = {
    'fighting':   'HIGH',
    'bullying':   'HIGH',
    'falling':    'HIGH',
    'climbing':   'SENSITIVE',
    'vandalism':  'SENSITIVE',
    'smoking':    'SENSITIVE',
    'phone_call': 'SUSPICIOUS',
    'loitering':  'SUSPICIOUS',
}
LEVEL_NAME = {
    'HIGH':       '高风险异常',
    'SENSITIVE':  '敏感行为',
    'SUSPICIOUS': '可疑行为',
}

# bbox 基准分辨率（WEB-HANDOFF §6.3 指定）
BBOX_BASE_W = 640
BBOX_BASE_H = 360


def scale_bbox_xyxy_to_640x360_xywh(bbox_xyxy, img_w, img_h):
    """原分辨率 [x1,y1,x2,y2] → 640x360 基准的 [x,y,w,h]"""
    x1, y1, x2, y2 = bbox_xyxy
    sx = BBOX_BASE_W / float(img_w) if img_w > 0 else 1.0
    sy = BBOX_BASE_H / float(img_h) if img_h > 0 else 1.0
    return [
        int(round(x1 * sx)),
        int(round(y1 * sy)),
        int(round((x2 - x1) * sx)),
        int(round((y2 - y1) * sy)),
    ]


def adapt_pipeline_to_web_event(payload, task):
    """将 pipeline 回调 payload 转换为 WEB-HANDOFF §6.3 frame 事件格式。

    告警去重由 Spring Boot 端处理（WEB 同学确认），此处每帧任何非 normal
    track 都会生成 alerts 条目。
    """
    img_w = payload['img_width']
    img_h = payload['img_height']
    frame_index = payload['frame_index']
    ts = datetime.now().isoformat()

    targets = []
    alerts = []
    for t in payload['tracks']:
        label = t['label']
        bbox_xywh = scale_bbox_xyxy_to_640x360_xywh(t['bbox_xyxy'], img_w, img_h)

        if label == 'normal':
            # normal track 也推 target，方便前端画绿框（不进 alerts）
            targets.append({
                'trackId': t['track_id'],
                'bbox': bbox_xywh,
                'behavior': 'normal',
                'confidence': t['confidence'],
            })
            continue

        mapping = LABEL_MAP.get(label)
        if mapping is None:
            # 未知标签降级为 normal
            targets.append({
                'trackId': t['track_id'],
                'bbox': bbox_xywh,
                'behavior': 'normal',
                'confidence': t['confidence'],
            })
            continue
        upper, lower, cn = mapping
        level = LEVEL_MAP[label]

        targets.append({
            'trackId': t['track_id'],
            'bbox': bbox_xywh,
            'behavior': lower,
            'confidence': t['confidence'],
        })
        alerts.append({
            'type': upper,
            'behaviorName': cn,
            'level': level,
            'levelName': LEVEL_NAME[level],
            'timestamp': ts,
            'sourceId': task.source_id,
            'slotIndex': task.slot_index,
            'sourceName': task.source_name,
            'locationName': task.location_name,
            'confidence': t['confidence'],
            'trackId': t['track_id'],
            'frameIndex': frame_index,
            'processStatus': 'UNPROCESSED',
            'processStatusName': '未处理',
        })

    return {
        'sourceId': task.source_id,
        'slotIndex': task.slot_index,
        'sourceName': task.source_name,
        'timestamp': ts,
        'frameIndex': frame_index,
        'imageWidth': img_w,
        'imageHeight': img_h,
        'targets': targets,
        'alerts': alerts,
    }


# ============================================================
# 任务管理
# ============================================================
class PipelineStoppedException(Exception):
    """pipeline 回调抛出以中断 run() 循环"""


class AnalyzeTask:
    def __init__(self, task_id, video_path, source_id, slot_index,
                 source_name, location_name):
        self.task_id = task_id
        self.video_path = video_path
        self.source_id = source_id
        self.slot_index = slot_index
        self.source_name = source_name
        self.location_name = location_name

        # 'pending' -> 'running' -> 'done' / 'stopped' / 'error'
        self.status = 'pending'
        self.progress = 0.0
        self.total_frames = 0
        self.current_frame = 0
        self.error_message = None

        self.stop_event = threading.Event()
        self.event_queue = queue.Queue()
        self.thread = None
        self.created_at = time.time()
        self.finished_at = None

    def push_sse(self, event_name, data):
        """线程安全地入队一条 SSE 事件"""
        self.event_queue.put({'event': event_name, 'data': data})

    def mark_finished(self, status, error_msg=None):
        self.status = status
        self.error_message = error_msg
        self.finished_at = time.time()


# 全局状态
TASKS = {}  # task_id → AnalyzeTask
PIPELINE = None  # InferencePipeline 实例（全局加载）
PIPELINE_LOCK = threading.Lock()  # 同一时刻只允许 1 个任务跑（GPU 显存限制）
UPLOAD_DIR = None  # 视频保存目录


def _run_task(task):
    """在 worker 线程里跑 pipeline.run()，通过回调把每帧结果推到 SSE queue"""
    global PIPELINE

    def on_frame(payload):
        if task.stop_event.is_set():
            raise PipelineStoppedException('stop requested')
        task.current_frame = payload['frame_index'] + 1
        if task.total_frames > 0:
            task.progress = min(1.0, task.current_frame / task.total_frames)
        event_data = adapt_pipeline_to_web_event(payload, task)
        task.push_sse('frame', event_data)

    with PIPELINE_LOCK:
        task.status = 'running'
        try:
            # 预探视频总帧数（进度用）
            import cv2
            cap = cv2.VideoCapture(task.video_path)
            task.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()

            PIPELINE.on_frame_callback = on_frame
            PIPELINE.run(task.video_path, show=False, output_video=None, output_json=None)
            # run() 正常结束
            task.progress = 1.0
            task.push_sse('done', {
                'taskId': task.task_id,
                'totalFrames': task.current_frame,
            })
            task.mark_finished('done')
        except PipelineStoppedException:
            task.push_sse('stopped', {'taskId': task.task_id})
            task.mark_finished('stopped')
        except Exception as e:
            err = f'{type(e).__name__}: {e}'
            logger.exception(f'[task {task.task_id}] pipeline 异常')
            task.push_sse('error', {'taskId': task.task_id, 'message': err})
            task.mark_finished('error', err)
        finally:
            PIPELINE.on_frame_callback = None
            # 清理上传的临时视频文件
            try:
                if os.path.exists(task.video_path):
                    os.remove(task.video_path)
            except Exception:
                pass


# ============================================================
# FastAPI 应用
# ============================================================
app = FastAPI(title='Campus Safety Inference API', version='1.0.0')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=False,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.get('/health')
async def health():
    return {
        'status': 'ok' if PIPELINE is not None else 'loading',
        'modelLoaded': PIPELINE is not None,
        'activeTasks': sum(1 for t in TASKS.values() if t.status == 'running'),
        'totalTasks': len(TASKS),
    }


@app.post('/api/v1/analyze/start')
async def analyze_start(
    file: UploadFile = File(...),
    sourceId: str = Form(...),
    slotIndex: int = Form(None),
    sourceName: str = Form(None),
    locationName: str = Form(None),
):
    if PIPELINE is None:
        raise HTTPException(503, 'Model not loaded yet')

    task_id = str(uuid.uuid4())
    # 保存上传视频到 UPLOAD_DIR（处理完删除）
    suffix = os.path.splitext(file.filename or 'video.mp4')[1] or '.mp4'
    video_path = os.path.join(UPLOAD_DIR, f'{task_id}{suffix}')
    with open(video_path, 'wb') as f:
        while True:
            chunk = await file.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)

    task = AnalyzeTask(
        task_id=task_id,
        video_path=video_path,
        source_id=sourceId,
        slot_index=slotIndex,
        source_name=sourceName or 'unknown',
        location_name=locationName,
    )
    TASKS[task_id] = task
    task.thread = threading.Thread(target=_run_task, args=(task,), daemon=True)
    task.thread.start()

    return {
        'taskId': task_id,
        'status': 'started',
        'videoSize': os.path.getsize(video_path),
        'sourceId': sourceId,
        'slotIndex': slotIndex,
    }


@app.get('/api/v1/analyze/{task_id}/stream')
async def analyze_stream(task_id: str):
    task = TASKS.get(task_id)
    if task is None:
        raise HTTPException(404, f'task {task_id} not found')

    async def event_gen():
        # 连接确认（对齐 WEB-HANDOFF §6.3 的 connected 事件）
        yield {'event': 'connected', 'data': 'stream-connected'}

        loop = asyncio.get_event_loop()
        while True:
            # 轮询 queue，非阻塞交给线程池，0.5s 超时
            try:
                item = await loop.run_in_executor(
                    None, lambda: task.event_queue.get(timeout=0.5)
                )
            except queue.Empty:
                # 空 queue 时检查任务是否已结束
                if task.status in ('done', 'stopped', 'error'):
                    break
                # 保持连接活跃
                yield {'event': 'ping', 'data': str(int(time.time()))}
                continue

            yield item
            if item['event'] in ('done', 'stopped', 'error'):
                break

    return EventSourceResponse(event_gen())


@app.post('/api/v1/analyze/{task_id}/stop')
async def analyze_stop(task_id: str):
    task = TASKS.get(task_id)
    if task is None:
        raise HTTPException(404, f'task {task_id} not found')
    if task.status in ('done', 'stopped', 'error'):
        return {'taskId': task_id, 'status': task.status, 'message': 'already finished'}
    task.stop_event.set()
    return {'taskId': task_id, 'status': 'stopping'}


@app.get('/api/v1/analyze/{task_id}/status')
async def analyze_status(task_id: str):
    task = TASKS.get(task_id)
    if task is None:
        raise HTTPException(404, f'task {task_id} not found')
    return {
        'taskId': task_id,
        'status': task.status,
        'progress': round(task.progress, 4),
        'currentFrame': task.current_frame,
        'totalFrames': task.total_frames,
        'sourceId': task.source_id,
        'slotIndex': task.slot_index,
        'sourceName': task.source_name,
        'errorMessage': task.error_message,
        'createdAt': task.created_at,
        'finishedAt': task.finished_at,
    }


@app.get('/api/v1/tasks')
async def list_tasks(status: str = None, limit: int = 50):
    """辅助调试 endpoint：列出近期任务"""
    items = list(TASKS.values())
    if status:
        items = [t for t in items if t.status == status]
    items.sort(key=lambda t: t.created_at, reverse=True)
    items = items[:limit]
    return {
        'total': len(TASKS),
        'returned': len(items),
        'tasks': [
            {
                'taskId': t.task_id,
                'status': t.status,
                'progress': round(t.progress, 4),
                'sourceId': t.source_id,
                'sourceName': t.source_name,
                'createdAt': t.created_at,
                'finishedAt': t.finished_at,
            }
            for t in items
        ],
    }


# ============================================================
# 启动
# ============================================================
def build_pipeline(args):
    def _norm(v):
        return None if (v is None or (isinstance(v, str) and v.lower() == 'none')) else v

    return InferencePipeline(
        yolo_pose_model=args.yolo_pose,
        posec3d_config=args.posec3d_config,
        posec3d_checkpoint=args.posec3d_ckpt,
        small_obj_model=_norm(args.small_obj_model),
        falling_model=_norm(args.falling_model),
        smoking_model=_norm(args.smoking_model),
        phone_model=_norm(args.phone_model),
        device=args.device,
        yolo_conf=args.yolo_conf,
        stride=args.stride,
    )


def parse_args():
    p = argparse.ArgumentParser(description='Campus Safety Inference API Server')
    # 模型（必选 / 默认）
    p.add_argument('--posec3d-config', required=True)
    p.add_argument('--posec3d-ckpt', required=True)
    p.add_argument('--yolo-pose', default='yolo11m-pose.pt')
    p.add_argument('--small-obj-model', default=None)
    p.add_argument('--falling-model',
                   default='/home/hzcu/yjm/home/yjm/VideoDetection/v8/falling/runs/laying_yolo11m_v1/weights/best.pt')
    p.add_argument('--smoking-model',
                   default='/home/hzcu/yjm/home/yjm/VideoDetection/v8/smoking/runs/smoking_yolo11m_v1/weights/best.pt')
    p.add_argument('--phone-model',
                   default='/home/hzcu/yjm/home/yjm/VideoDetection/v8/phone/runs/phone_yolo11m_v1/weights/best.pt')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--yolo-conf', type=float, default=0.3)
    p.add_argument('--stride', type=int, default=16)

    # 服务
    p.add_argument('--host', default='0.0.0.0')
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--upload-dir', default='api_uploads')
    return p.parse_args()


def main():
    global PIPELINE, UPLOAD_DIR

    args = parse_args()
    UPLOAD_DIR = os.path.abspath(args.upload_dir)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    print(f'Upload dir: {UPLOAD_DIR}')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(message)s',
    )

    print('Loading pipeline (this takes a few seconds)...')
    PIPELINE = build_pipeline(args)
    print('Pipeline ready.')

    uvicorn.run(app, host=args.host, port=args.port, log_level='info')


if __name__ == '__main__':
    main()
