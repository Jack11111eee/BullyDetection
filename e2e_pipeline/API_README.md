# 校园安防模型 API 对接说明

本文档面向 Web 后端（Spring Boot）同学。Python 端提供 REST + SSE 接口，你方把上传视频转发过来，订阅 SSE 拿到推理结果后按现有 `/api/monitor/stream` 结构推给前端即可。

更新日期：2026-04-19

---

## 1. 服务信息

- 语言栈：Python 3.10 + FastAPI + sse-starlette
- 运行环境：学校机房服务器（3 × A6000, conda env `dtc`）
- **Base URL**：`http://10.61.190.21:8000`
- **访问要求**：校园网内（有线/WiFi），外网需要先挂学校 VPN
- 模型启动需 ~10 秒加载，启动完成后 `/health` 返回 `modelLoaded: true`

自测命令（校园网内任意机器）：

```bash
curl http://10.61.190.21:8000/health
# 期望: {"status":"ok","modelLoaded":true,...}
```

---

## 2. 端点一览

| 方法 | 路径 | 用途 |
|---|---|---|
| `GET` | `/health` | 健康检查，确认模型已加载 |
| `POST` | `/api/v1/analyze/start` | 上传视频 + metadata，启动分析 |
| `GET` | `/api/v1/analyze/{taskId}/stream` | SSE 推 frame / alert / done / error |
| `POST` | `/api/v1/analyze/{taskId}/stop` | 主动停止任务 |
| `GET` | `/api/v1/analyze/{taskId}/status` | 查询任务进度 |
| `GET` | `/api/v1/tasks` | 列出近期任务（调试用） |

---

## 3. 典型联调流程

```
Spring Boot (LIVE 模式 /api/monitor/start)
  ├─ 1. POST /api/v1/analyze/start (multipart: file + sourceId + slotIndex + sourceName)
  │      ← { taskId, status: 'started' }
  ├─ 2. GET  /api/v1/analyze/{taskId}/stream (保持长连接)
  │      ← event: connected
  │      ← event: frame      (每处理 stride=16 帧推一次)
  │      ← event: frame
  │      ← ...
  │      ← event: done       (视频处理完)
  └─ Spring Boot 把每个 frame 事件转成你方 /api/monitor/stream 的 frame 结构推给前端
```

---

## 4. 接口详细说明

### 4.1 `POST /api/v1/analyze/start`

`multipart/form-data`:

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `file` | file | 是 | 视频文件（mp4 / avi / mov 均可） |
| `sourceId` | string | 是 | 与你方数据库里的 sourceId 一致，会透传到每个 alert |
| `slotIndex` | int | 否 | 1-4，与监控位绑定 |
| `sourceName` | string | 否 | 视频源名称，会透传 |
| `locationName` | string | 否 | 地点名称，会透传 |

返回：

```json
{
  "taskId": "uuid",
  "status": "started",
  "videoSize": 12345678,
  "sourceId": "你传的 sourceId",
  "slotIndex": 1
}
```

错误：
- `503` — 模型未加载完毕，稍后重试
- `422` — multipart 参数缺失

### 4.2 `GET /api/v1/analyze/{taskId}/stream` — SSE

**事件类型**：

| event | 何时触发 | data 格式 |
|---|---|---|
| `connected` | 连接建立时 | 纯文本 `stream-connected` |
| `frame` | 每处理完一批帧（~16 帧/次，~0.5s/次） | JSON（见下） |
| `ping` | 0.5s 内无新事件时 | 纯文本时间戳，用于保活 |
| `done` | 视频处理完成 | `{ "taskId": "...", "totalFrames": N }` |
| `stopped` | 收到 `/stop` 请求 | `{ "taskId": "..." }` |
| `error` | 异常 | `{ "taskId": "...", "message": "..." }` |

**`frame` 事件 data 结构**（JSON，与 WEB-HANDOFF §6.3 对齐）：

```json
{
  "sourceId": "你传的 sourceId",
  "slotIndex": 1,
  "sourceName": "Cam-01",
  "timestamp": "2026-04-19T01:35:00.123456",
  "frameIndex": 128,
  "videoFps": 30.0,
  "videoTime": 4.2667,
  "totalFrames": 1800,
  "imageWidth": 1920,
  "imageHeight": 1080,
  "targets": [
    {
      "trackId": 3,
      "bbox": [120, 80, 60, 180],
      "behavior": "fight",
      "confidence": 0.93
    }
  ],
  "alerts": [
    {
      "type": "FIGHT",
      "behaviorName": "打架",
      "level": "HIGH",
      "levelName": "高风险异常",
      "timestamp": "2026-04-19T01:35:00.123456",
      "sourceId": "你传的 sourceId",
      "slotIndex": 1,
      "sourceName": "Cam-01",
      "locationName": "教学楼 2 层走廊",
      "confidence": 0.93,
      "trackId": 3,
      "frameIndex": 128,
      "processStatus": "UNPROCESSED",
      "processStatusName": "未处理"
    }
  ]
}
```

**关键约定**：

- `bbox` 格式：`[x, y, width, height]`，**已按 640×360 基准缩放**，可直接塞进你方 `targets[].bbox`
- 如果你方前端按别的基准渲染，改 `scale_bbox_xyxy_to_640x360_xywh()` 的 `BBOX_BASE_W/H` 即可，告诉我你要哪个
- `imageWidth / imageHeight` 是**原视频分辨率**，备用字段
- `videoFps / videoTime / totalFrames` —— **方案 A 关键字段**（见 §4.6 前端帧对齐）
- `targets` 包含**所有** track（含 `behavior: "normal"` 的），前端可据此画绿框。不想要 normal target 告诉我可以过滤
- `alerts` **每帧都会发**，去重（按 sourceId + type + trackId + 时间窗）请你方入库前处理。我方不做去重以保留完整原始判定（和 WEB 同学商定）
- `snapshotUrl` 和 `clipUrl` **不由我方生成**。Spring Boot 侧可基于 `frameIndex + videoFps` 算时间戳从原视频截图/切片

### 4.3 `POST /api/v1/analyze/{taskId}/stop`

主动停止。任务会在当前帧处理完后停止，SSE 会先推 `stopped` 事件再断开。

返回：
```json
{ "taskId": "...", "status": "stopping" }
```

### 4.4 `GET /api/v1/analyze/{taskId}/status`

```json
{
  "taskId": "...",
  "status": "pending|running|done|stopped|error",
  "progress": 0.42,
  "currentFrame": 420,
  "totalFrames": 1000,
  "sourceId": "...",
  "slotIndex": 1,
  "sourceName": "Cam-01",
  "errorMessage": null,
  "createdAt": 1729382400.0,
  "finishedAt": null
}
```

### 4.5 `GET /health`

```json
{
  "status": "ok",
  "modelLoaded": true,
  "activeTasks": 1,
  "totalTasks": 42
}
```

### 4.6 前端帧对齐（方案 A，重要）

**背景**：后端推理速度（~15 fps）比视频播放帧率（~30 fps）慢。如果前端 `<video>` 按原 fps 自由播放，bbox 会落后视频 2-3 秒，框停在旧位置，用户体验差。

**方案 A（已商定）**：前端不让 video 自由播，**跟着后端推理节奏**走。每收到一个 `frame` SSE 事件，前端主动把视频跳到对应时刻。

**实现方式**：

```javascript
// 前端 Vue/JS 伪代码
const video = document.querySelector('#monitor-video');  // 播放原视频的 <video>
video.pause();  // 关键：禁用 video 自己的时间推进

// 收到 SSE frame 事件时
eventSource.addEventListener('frame', (e) => {
  const data = JSON.parse(e.data);

  // 1. 视频跳到对应时刻（videoTime 是秒）
  video.currentTime = data.videoTime;

  // 2. 在 canvas 上画 bbox（用 data.targets）
  drawBoxes(data.targets);

  // 3. 处理 alerts（入库、推右侧面板、时间轴等）
  handleAlerts(data.alerts);
});
```

**关键字段**：

| 字段 | 含义 | 前端用途 |
|---|---|---|
| `videoTime` | 当前帧对应的视频秒数（= frameIndex / videoFps） | `video.currentTime = videoTime` 直接跳帧 |
| `videoFps` | 原视频帧率 | 前端可显示"处理进度 X/Y fps"等信息 |
| `frameIndex` | 当前帧号（0-based） | 进度条 / 告警时间戳 |
| `totalFrames` | 视频总帧数 | 进度条 / `%` 显示 |

**注意事项**：

1. **视频必须 pause**：否则 video 会同时按 30fps 自己推进 + 被 `currentTime` 覆盖，出现闪跳
2. **频繁 seek 可能抖动**：HTML5 video 在非关键帧位置 seek 会有小延迟。实测校园网 + mp4（h264 + GOP 约 30）应该可以接受；如果明显卡，让我方在推理端用 ffmpeg 对上传视频重编码成关键帧密集格式（每 1 秒 1 个 I 帧），但会增加启动延迟
3. **告警时间戳用 videoTime 计算**：`clipUrl = /media/videos/{id}.mp4#t=${videoTime - 2}` 之类的片段链接，基于 videoTime 比用 wall-clock 更准
4. **任务结束后 video 可恢复正常播放**：收到 `done` 事件后若用户想回放，调用 `video.play()` 即可

**调试技巧**：如果看到 bbox 位置对不上画面，在浏览器控制台执行 `video.currentTime` 看看是不是确实在对应时刻；对不上就说明 seek 还没完成（监听 `seeked` 事件再画 bbox 可解）。

---

## 5. 枚举值（behavior / type / level）

### 5.1 behavior（`targets[].behavior`，小写）

| 值 | 中文 | 对应 alert type |
|---|---|---|
| `fight` | 打架 | FIGHT |
| `bully` | 霸凌 | **BULLY**（新增，需扩展 Web 后端枚举） |
| `fall` | 跌倒 | FALL |
| `climb` | 翻越围栏 | CLIMB |
| `smoking` | 抽烟 | SMOKING |
| `phone` | 打电话 | **PHONE**（新增） |
| `vandalism` | 破坏公物 | **VANDALISM**（新增） |
| `loiter` | 徘徊 | LOITER |
| `normal` | 正常 | （不生成 alert） |

### 5.2 level（`alerts[].level`）

| 值 | 对应 behavior |
|---|---|
| `HIGH` | FIGHT / BULLY / FALL |
| `SENSITIVE` | CLIMB / VANDALISM / SMOKING |
| `SUSPICIOUS` | PHONE / LOITER |

级别是我方按行为严重性固定映射的，如果 WEB 端有不同的风险等级策略，告诉我可以改。

---

## 6. 环境部署

Python 端启动（我方负责）：

```bash
pip install fastapi 'uvicorn[standard]' sse-starlette python-multipart

cd /home/hzcu/BullyDetection
python e2e_pipeline/api_server.py \
  --posec3d-config pyskl/configs/posec3d/finetune_campus_mil.py \
  --posec3d-ckpt pyskl/work_dirs/posec3d_campus_mil/epoch_50.pth \
  --host 0.0.0.0 --port 8000
```

---

## 7. 你方需要做的 4 件事

1. **扩展 `BehaviorType` 枚举**：加 `BULLY / PHONE / VANDALISM` 三项。对应 `behaviorName` 建议用 `霸凌 / 打电话 / 破坏公物`。`level` 分别建议 `HIGH / SUSPICIOUS / SENSITIVE`
2. **LIVE 模式接入**：在 `/api/monitor/start` 的 LIVE 分支调我方 `/api/v1/analyze/start`，并订阅 `/stream`
3. **告警去重**：`alerts` 我方每帧都推，入库前请按 `(sourceId, type, trackId)` 作为滑窗键去重（建议同键 5 秒内只入库一次）
4. **snapshotUrl / clipUrl 生成**：基于 `frameIndex` + 原视频 fps 算时间戳，调 ffmpeg 截图/切片（示例：`clipUrl = /media/videos/{id}.mp4#t={frameIndex/fps}`）

---

## 8. 联调清单

- [ ] 确认 Python 端 `/health` 返回 `modelLoaded: true`
- [ ] `curl -F 'file=@test.mp4' -F 'sourceId=test-01' http://autodl-host:8000/api/v1/analyze/start` 能拿到 `taskId`
- [ ] `curl -N http://autodl-host:8000/api/v1/analyze/{taskId}/stream` 能看到 `event: connected` 和连续的 `event: frame`
- [ ] Spring Boot 转换后前端 `/monitor` 页面能看到 bbox + 告警
- [ ] 确认 `BULLY / PHONE / VANDALISM` 在前端展示正常

有问题直接在群里 @我。
