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

### 4.6 前端帧对齐（方案 A4，重要 — 已替换原 A 方案）

**背景**：后端推理速度（~15 fps）比视频播放帧率（~30 fps）慢。原 A 方案（`video.pause()` + 每帧 `video.currentTime = X`）会触发频繁 HTML5 seek，浏览器跟不上造成卡顿。

**方案 A4（已商定）**：video **自由播放不 pause**，前端缓存 SSE frame 事件，用 `requestAnimationFrame` 循环每帧查 `video.currentTime`，从缓冲里取时间最接近的 bbox 叠加到 canvas。

**核心思路**：

```
video 30fps 自由流畅播 (浏览器原生解码)
      ↕ 无耦合,不互相阻塞
SSE 15fps 持续来 → 缓存 {videoTime: targets}
      ↕ 查找最近
canvas rAF 60fps 重绘 → 当前 video.currentTime 找 bbox 叠加
```

- video 完全流畅（浏览器原生播放，无 seek 抖动）
- bbox 滞后 ≤ 1 个推理周期（~67ms），**肉眼不可分辨**
- 网络抖动只影响 bbox 滞后程度，不影响视频

**实现方式（Vue/JS 伪代码）**：

```javascript
const video = document.querySelector('#monitor-video');
const canvas = document.querySelector('#bbox-overlay');  // 叠加在 video 上层
const ctx = canvas.getContext('2d');

// 1. 缓冲：videoTime(秒) → targets
const bboxBuffer = [];  // [{videoTime, targets}, ...] 按 videoTime 升序

eventSource.addEventListener('frame', (e) => {
  const data = JSON.parse(e.data);
  bboxBuffer.push({ videoTime: data.videoTime, targets: data.targets });
  // 丢弃已过期的(早于当前 video 时间 2 秒以上的)
  while (bboxBuffer.length > 0 && bboxBuffer[0].videoTime < video.currentTime - 2) {
    bboxBuffer.shift();
  }
  // alerts 独立处理(入库、右侧面板、时间轴)
  handleAlerts(data.alerts, data.videoTime);
});

// 2. 渲染循环：video 播放时实时叠加最近 bbox
function renderLoop() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const t = video.currentTime;
  // 二分或线性查找 videoTime <= t 的最大者
  let latest = null;
  for (const item of bboxBuffer) {
    if (item.videoTime <= t) latest = item;
    else break;
  }
  if (latest) drawBoxes(ctx, latest.targets);
  requestAnimationFrame(renderLoop);
}

video.play();  // video 正常播
renderLoop();  // 启动 bbox 叠加循环
```

**关键字段**：

| 字段 | 含义 | 前端用途 |
|---|---|---|
| `videoTime` | 当前帧对应的视频秒数（= frameIndex / videoFps） | 缓冲索引键,查最近 bbox |
| `videoFps` | 原视频帧率 | 进度显示/告警时间戳换算 |
| `frameIndex` | 当前帧号（0-based） | 进度条 / 告警帧号 |
| `totalFrames` | 视频总帧数 | 进度条百分比 |

**注意事项**：

1. **video 不要 pause**：让浏览器原生播放保证流畅
2. **缓冲过期清理**：`bboxBuffer` 是单调增长的，必须定期 shift 掉已过期项，否则内存泄漏
3. **启动同步**：收到第一个 frame 事件时调用一次 `video.play()`，确保 video 从 0 开始（前端收到 Spring Boot 推的 `/media/videos/xxx.mp4` 地址后就 `.load()` 但不立即 play，等 frame 事件到位才 play，让视频和 bbox 流同时启动）
4. **告警时间戳**：每条 alert 带 `frameIndex` 和 `videoTime`(等价)，`clipUrl = /media/videos/{id}.mp4#t=${alert.videoTime - 2}` 这类跳片段链接继续好使
5. **处理完成时**：收到 `done` 事件后 video 自然播完（如果推理比视频快）或用户已看完（如果推理慢于视频）。停止 rAF 循环或让它空转均可

**为什么不用严格对齐**：
推理速度 ~15 fps = 每 67ms 更新一次 bbox。video 以 30fps 显示，两帧之间是 33ms。最坏情况 bbox 对应 video 往前 2 帧（67ms），**肉眼无感**。这是监控 NVR + OSD 字幕的标准做法，工业级可靠。

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
