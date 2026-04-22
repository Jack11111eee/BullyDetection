# 双模式接口对接协议

更新时间：2026-04-21

本文档面向前端同学、联调同伴或远程模型服务对接方，说明当前系统的两种分析模式、Spring Boot 对外接口、SSE 事件协议、结果结构以及对接注意事项。

当前仓库基线：

- Spring Boot 对外接口：`/api/upload/*`、`/api/sources/*`、`/api/settings/*`、`/api/monitor/*`、`/api/alerts/*`
- 远程模型上游接口：`/health`、`/api/v1/analyze/start`、`/api/v1/analyze/{taskId}/stream`、`/api/v1/analyze/{taskId}/status`、`/api/v1/analyze/{taskId}/stop`

## 1. 总体说明

系统当前支持两种模式：

| 模式枚举 | 中文名称 | 业务语义 |
| --- | --- | --- |
| `PRELOAD_BATCH` | 预加载分析 | 四宫格内 1-4 路视频全部先推理完成，结果回齐后前端再开始播放视频，并按完整结果叠加标注框 |
| `SEVEN_FRAME_REALTIME` | 10 帧实时推理 | 后端先将原视频转成 `10fps` 稀疏视频，再送远程模型实时推理；四宫格不实时叠框，但抓拍图实时带框 |

统一规则：

- 只有已绑定到 `slotIndex=1..4` 的视频源才会参与分析
- `bbox` 坐标格式统一为 `[x, y, width, height]`
- `bbox` 的基准坐标系固定为 `640x360`
- 上传进度不是后端单独接口，而是前端 `multipart/form-data` 上传进度回调
- 推理进度统一通过 `GET /api/monitor/status` 获取
- 远程模型仍然只接收视频，不接收图片

## 2. 对接拓扑

```text
前端上传本地视频
  -> Spring Boot 落盘
  -> 绑定到四宫格监控位
  -> 根据 mode 启动分析
     -> PRELOAD_BATCH: 原视频直接送远程模型
     -> SEVEN_FRAME_REALTIME: 先转 10fps 稀疏视频，再送远程模型
  -> Spring Boot 接收上游 SSE / 进度
  -> Spring Boot 对前端输出：
     - /api/monitor/status
     - /api/monitor/results
     - /api/monitor/stream
     - /api/alerts
```

## 3. 枚举定义

### 3.1 模式枚举 `MonitorMode`

| 值 | 含义 |
| --- | --- |
| `PRELOAD_BATCH` | 预加载分析 |
| `SEVEN_FRAME_REALTIME` | 10 帧实时推理 |

### 3.2 行为枚举 `BehaviorType`

| 值 | 中文 |
| --- | --- |
| `FIGHT` | 打架 |
| `FALL` | 跌倒 |
| `BULLY` | 霸凌 |
| `SMOKING` | 抽烟 |
| `PHONE` | 玩手机 |
| `LOITER` | 徘徊 |
| `CLIMB` | 翻越围栏 |
| `VANDALISM` | 破坏公物 |
| `VICTIM_PERPETRATOR` | 受害者-施害者 |

### 3.3 风险等级 `RiskLevel`

| 值 | 中文 |
| --- | --- |
| `HIGH` | 高风险异常 |
| `SENSITIVE` | 敏感行为 |
| `SUSPICIOUS` | 可疑行为 |

### 3.4 告警处理状态 `ProcessStatus`

| 值 | 含义 |
| --- | --- |
| `UNPROCESSED` | 未处理 |
| `PROCESSED` | 已处理 |
| `FALSE_POSITIVE` | 误报 |
| `VIEWED` | 已查看，接口返回时会归一为已处理 |

## 4. 通用接口

### 4.1 上传视频

`POST /api/upload/video`

请求类型：`multipart/form-data`

字段：

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `file` | file | 是 | 视频文件 |
| `slotIndex` | int | 否 | 绑定监控位，允许 `1-4` 或不传 |
| `sourceName` | string | 否 | 视频源名称 |
| `locationName` | string | 否 | 地点名称 |
| `remark` | string | 否 | 备注 |
| `resolution` | string | 否 | 分辨率，如 `1920x1080` |
| `fps` | int | 否 | 原视频帧率 |
| `durationSec` | int | 否 | 视频时长，单位秒 |

响应：`SourceView`

```json
{
  "sourceId": "8f8cfadc-59b0-432b-8080-bec760800b67",
  "slotIndex": 1,
  "sourceName": "Cam-01",
  "sourceType": "LOCAL_VIDEO",
  "status": "READY",
  "locationName": "3号教学楼2层走廊",
  "remark": "东向走廊",
  "resolution": "1920x1080",
  "fps": 25,
  "durationSec": 60,
  "fileName": "demo.mp4",
  "fileSize": 12345678,
  "playUrl": "/media/videos/xxxxx.mp4",
  "createdAt": "2026-04-21T20:30:15",
  "updatedAt": "2026-04-21T20:30:15"
}
```

### 4.2 查询视频源列表

`GET /api/sources`

响应：`SourceView[]`

### 4.3 修改视频源元信息

`PUT /api/sources/{id}/meta`

请求：

```json
{
  "sourceName": "Cam-01",
  "locationName": "3号教学楼2层走廊",
  "remark": "东向走廊",
  "slotIndex": 1
}
```

### 4.4 修改绑定监控位

`PUT /api/sources/{id}/binding`

请求：

```json
{
  "slotIndex": 2
}
```

### 4.5 删除视频源

`DELETE /api/sources/{id}`

响应：

```json
{
  "success": true
}
```

### 4.6 切换系统模式

`PUT /api/settings/mode`

请求：

```json
{
  "mode": "PRELOAD_BATCH",
  "autoStart": false,
  "showSkeletonDebug": false
}
```

响应：

```json
{
  "mode": "PRELOAD_BATCH",
  "autoStart": false,
  "showSkeletonDebug": false,
  "springBootStatus": "UP",
  "eventStreamStatus": "IDLE",
  "modelServiceStatus": "PRELOAD_READY"
}
```

注意：

- 切换模式会取消当前分析任务
- 切换模式会清空当前轮次缓存结果

## 5. 分析控制接口

### 5.1 开始分析

`POST /api/monitor/start`

响应：

```json
{
  "success": true
}
```

### 5.2 暂停分析

`POST /api/monitor/pause`

响应：

```json
{
  "success": true
}
```

### 5.3 停止分析

`POST /api/monitor/stop`

响应：

```json
{
  "success": true
}
```

### 5.4 查询监控状态

`GET /api/monitor/status`

响应：`MonitorStatusView`

字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `mode` | enum | 当前模式 |
| `running` | boolean | 当前是否仍在分析 |
| `onlineSourceCount` | int | 当前绑定的监控源数量 |
| `activeAlertCount` | long | 当前活跃告警数 |
| `highRiskCount` | long | 当前高风险告警数 |
| `modelStatus` | string | 当前模式下的模型状态 |
| `streamStatus` | string | 前端事件流连接状态 |
| `mysqlStatus` | string | DB 状态 |
| `updatedAt` | datetime | 最近更新时间 |
| `latestLogs` | string[] | 最近日志 |
| `analysisReady` | boolean | 仅 `PRELOAD_BATCH` 有业务意义，表示全量结果是否已就绪 |
| `overallProgress` | double | 总体进度，`0-1` |
| `taskProgresses` | `MonitorTaskProgressView[]` | 每一路任务进度 |

`MonitorTaskProgressView`：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `sourceId` | string | 视频源 ID |
| `slotIndex` | int | 监控位 |
| `sourceName` | string | 视频源名称 |
| `taskId` | string | 远程模型任务 ID |
| `status` | string | `pending/running/done/stopped/error` 等 |
| `progress` | double | `0-1` |
| `currentFrame` | int | 已处理帧 |
| `totalFrames` | int | 总帧数 |
| `bufferedFrames` | int | 后端已缓存帧数 |
| `completed` | boolean | 该路任务是否完成 |

示例：

```json
{
  "mode": "PRELOAD_BATCH",
  "running": true,
  "onlineSourceCount": 4,
  "activeAlertCount": 0,
  "highRiskCount": 0,
  "modelStatus": "PRELOAD_RUNNING",
  "streamStatus": "STREAM_CONNECTED(1)",
  "mysqlStatus": "UP",
  "updatedAt": "2026-04-21T21:00:15",
  "latestLogs": [
    "2026-04-21T21:00:14.010 | 开始分析，模式: 预加载分析"
  ],
  "analysisReady": false,
  "overallProgress": 0.52,
  "taskProgresses": [
    {
      "sourceId": "src-01",
      "slotIndex": 1,
      "sourceName": "Cam-01",
      "taskId": "task-01",
      "status": "running",
      "progress": 0.61,
      "currentFrame": 610,
      "totalFrames": 1000,
      "bufferedFrames": 39,
      "completed": false
    }
  ]
}
```

### 5.5 查询预加载结果

`GET /api/monitor/results`

仅 `PRELOAD_BATCH` 有业务意义。

响应：`MonitorAnalysisResultsView`

字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `mode` | enum | 当前模式 |
| `ready` | boolean | 结果是否可用 |
| `generatedAt` | datetime | 结果生成时间 |
| `sources` | `MonitorSourceResultView[]` | 每一路完整结果 |

`MonitorSourceResultView`：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `sourceId` | string | 视频源 ID |
| `slotIndex` | int | 监控位 |
| `sourceName` | string | 视频源名称 |
| `totalFrames` | int | 总帧数 |
| `videoFps` | double | 视频帧率 |
| `frames` | `MonitorFrameResultView[]` | 该路完整帧结果 |

`MonitorFrameResultView`：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `frameIndex` | int | 帧序号 |
| `videoFps` | double | 视频帧率 |
| `videoTime` | double | 视频时间戳，秒 |
| `totalFrames` | int | 总帧数 |
| `imageWidth` | int | 原视频宽 |
| `imageHeight` | int | 原视频高 |
| `targets` | `TargetView[]` | 该帧目标框 |

示例：

```json
{
  "mode": "PRELOAD_BATCH",
  "ready": true,
  "generatedAt": "2026-04-21T21:02:10.123",
  "sources": [
    {
      "sourceId": "src-01",
      "slotIndex": 1,
      "sourceName": "Cam-01",
      "totalFrames": 1000,
      "videoFps": 25.0,
      "frames": [
        {
          "frameIndex": 0,
          "videoFps": 25.0,
          "videoTime": 0.0,
          "totalFrames": 1000,
          "imageWidth": 1920,
          "imageHeight": 1080,
          "targets": [
            {
              "trackId": 3,
              "bbox": [120, 80, 60, 180],
              "behavior": "fight",
              "confidence": 0.93
            }
          ]
        }
      ]
    }
  ]
}
```

## 6. SSE 协议

### 6.1 前端事件流

`GET /api/monitor/stream`

响应类型：`text/event-stream`

当前 Spring Boot 向前端输出的事件：

| 事件名 | 说明 | 哪个模式会用 |
| --- | --- | --- |
| `connected` | 前端事件流已建立 | 两种模式 |
| `analysis-ready` | 全量结果已就绪 | `PRELOAD_BATCH` |
| `frame` | 单帧结果 | `SEVEN_FRAME_REALTIME` |
| `done` | 某一路任务完成 | 两种模式 |
| `stopped` | 某一路任务停止 | 两种模式 |
| `error` | 某一路任务异常 | 两种模式 |

注意：

- 后端不会向前端透传模型上游的 `ping`
- `PRELOAD_BATCH` 模式不会向前端持续推 `frame`
- `SEVEN_FRAME_REALTIME` 模式不会推 `analysis-ready`

### 6.2 `connected`

```text
event: connected
data: stream-connected
```

### 6.3 `analysis-ready`

仅 `PRELOAD_BATCH`。

```text
event: analysis-ready
data: {"mode":"PRELOAD_BATCH","ready":true,"generatedAt":"2026-04-21T21:02:10.123"}
```

### 6.4 `frame`

仅 `SEVEN_FRAME_REALTIME`。

对应结构：`FrameEventView`

字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `sourceId` | string | 视频源 ID |
| `slotIndex` | int | 监控位 |
| `sourceName` | string | 视频源名称 |
| `timestamp` | datetime | 当前推理时间 |
| `frameIndex` | int | 帧序号 |
| `videoFps` | double | 稀疏视频或原视频的帧率 |
| `videoTime` | double | 当前帧对应视频时间，秒 |
| `totalFrames` | int | 总帧数 |
| `imageWidth` | int | 原视频宽 |
| `imageHeight` | int | 原视频高 |
| `targets` | `TargetView[]` | 当前帧目标框 |
| `alerts` | `AlertView[]` | 当前帧触发的告警 |
| `systemLog` | string | 系统日志文本 |

示例：

```text
event: frame
data: {
  "sourceId":"src-01",
  "slotIndex":1,
  "sourceName":"Cam-01",
  "timestamp":"2026-04-21T21:15:00.123",
  "frameIndex":128,
  "videoFps":7.0,
  "videoTime":18.2857,
  "totalFrames":420,
  "imageWidth":1920,
  "imageHeight":1080,
  "targets":[
    {
      "trackId":3,
      "bbox":[120,80,60,180],
      "behavior":"fight",
      "confidence":0.93
    }
  ],
  "alerts":[
    {
      "alertId":"a-1001",
      "type":"FIGHT",
      "behaviorName":"打架",
      "level":"HIGH",
      "levelName":"高风险异常",
      "timestamp":"2026-04-21T21:15:00.123",
      "sourceId":"src-01",
      "slotIndex":1,
      "sourceName":"Cam-01",
      "locationName":"3号教学楼2层走廊",
      "confidence":0.93,
      "trackId":3,
      "frameIndex":128,
      "snapshotUrl":"/media/snapshots/src-01_18286_ab12cd.jpg",
      "clipUrl":"/media/videos/xxxxx.mp4#t=18.286",
      "processStatus":"UNPROCESSED",
      "processStatusName":"未处理",
      "remark":null
    }
  ],
  "systemLog":"检测到高风险异常：打架 @ Cam-01"
}
```

### 6.5 `done` / `stopped` / `error`

这三个事件的 payload 结构一致，按任务维度返回。

示例：

```json
{
  "taskId": "task-01",
  "sourceId": "src-01",
  "slotIndex": 1,
  "sourceName": "Cam-01",
  "status": "done",
  "progress": 1.0,
  "totalFrames": 420
}
```

异常示例：

```json
{
  "taskId": "task-01",
  "sourceId": "src-01",
  "slotIndex": 1,
  "sourceName": "Cam-01",
  "status": "error",
  "progress": 0.31,
  "message": "remote model timeout"
}
```

## 7. 两种模式的详细协议

## 7.1 模式一：`PRELOAD_BATCH`

### 业务语义

- 四宫格中的 1-4 路视频全部发送给远程模型
- Spring Boot 不把单帧结果实时转发给前端
- Spring Boot 先缓存所有帧结果
- 当所有任务结束后，Spring Boot 触发 `analysis-ready`
- 前端收到 `analysis-ready` 后调用 `/api/monitor/results`
- 前端再开始播放原视频并按完整结果叠框
- 告警抓拍图已经带框

### 对接时序

```text
1. PUT /api/settings/mode             mode=PRELOAD_BATCH
2. POST /api/upload/video             上传并绑定 1-4 路视频
3. POST /api/monitor/start            启动分析
4. GET  /api/monitor/stream           建立 SSE
5. GET  /api/monitor/status           轮询进度
6. 收到 event: analysis-ready
7. GET  /api/monitor/results          拉取完整结果
8. GET  /api/alerts?processStatus=UNPROCESSED&limit=200
9. 前端开始播放视频并叠加 bbox
```

### 前端判定规则

- `running=true` 表示任务仍在执行
- `analysisReady=true` 表示可开始播放
- `running=false` 且 `analysisReady=true` 表示本轮成功完成
- `running=false` 且 `analysisReady=false` 不可直接视为成功

### 四宫格展示规则

- 视频播放源仍然是 `SourceView.playUrl`
- bbox 来源于 `/api/monitor/results`
- 应按 `video.currentTime` 在 `frames[]` 中找到不大于当前时间的最近一帧画框
- 告警详情来源于 `/api/alerts`

### 该模式关键点

- 不依赖 SSE `frame`
- 必须等待 `analysis-ready` 或 `analysisReady=true`
- `results` 是播放时叠框的唯一完整数据源

## 7.2 模式二：`SEVEN_FRAME_REALTIME`

### 业务语义

- Spring Boot 先把原视频转成 `10fps` 稀疏视频
- 再把稀疏视频上传给远程模型
- Spring Boot 持续把上游帧结果转发给前端 `frame`
- 前端播放四宫格，但不在四宫格实时叠框
- 告警抓拍图实时生成，且抓拍中带框

### 对接时序

```text
1. PUT /api/settings/mode             mode=SEVEN_FRAME_REALTIME
2. POST /api/upload/video             上传并绑定视频
3. POST /api/monitor/start
4. Spring Boot 生成 sparse-videos/{sourceId}_10fps.mp4
5. GET  /api/monitor/stream           建立 SSE
6. GET  /api/monitor/status           轮询进度
7. 持续接收 event: frame
8. 从 frame.alerts 或 /api/alerts 展示告警
```

### 四宫格展示规则

- 视频直接播放原始 `playUrl`
- 不用 `frame.targets` 在四宫格实时画框
- 仅抓拍图需要标注框
- 告警详情里的 `snapshotUrl` 已经是带框抓拍

### 该模式关键点

- `results` 接口不是主数据源
- 主要看 `frame` 事件和 `/api/monitor/status`
- 该模式名虽然叫 `10 帧实时推理`，但实现上是“10fps 稀疏视频推理”，不是“7 张图片接口”

## 8. 目标框和告警字段

### 8.1 `TargetView`

```json
{
  "trackId": 3,
  "bbox": [120, 80, 60, 180],
  "behavior": "fight",
  "confidence": 0.93
}
```

字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `trackId` | int | 跟踪 ID |
| `bbox` | int[4] | `[x, y, width, height]`，基准 `640x360` |
| `behavior` | string | 小写行为名 |
| `confidence` | double | 置信度 |

### 8.2 `AlertView`

```json
{
  "alertId": "a-1001",
  "type": "FIGHT",
  "behaviorName": "打架",
  "level": "HIGH",
  "levelName": "高风险异常",
  "timestamp": "2026-04-21T21:15:00.123",
  "sourceId": "src-01",
  "slotIndex": 1,
  "sourceName": "Cam-01",
  "locationName": "3号教学楼2层走廊",
  "confidence": 0.93,
  "trackId": 3,
  "frameIndex": 128,
  "snapshotUrl": "/media/snapshots/xxx.jpg",
  "clipUrl": "/media/videos/xxx.mp4#t=18.286",
  "processStatus": "UNPROCESSED",
  "processStatusName": "未处理",
  "remark": null
}
```

字段含义：

| 字段 | 说明 |
| --- | --- |
| `snapshotUrl` | 抓拍图地址，当前已带标注框 |
| `clipUrl` | 事件片段地址 |
| `trackId` | 对应跟踪 ID |
| `frameIndex` | 对应帧序号 |
| `processStatus` | 当前处理状态 |

## 9. 告警接口

### 9.1 查询告警

`GET /api/alerts`

常用查询参数：

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `type` | enum | 行为类型 |
| `level` | enum | 风险等级 |
| `processStatus` | enum | 处理状态 |
| `sourceId` | string | 视频源 ID |
| `startTime` | datetime | 开始时间 |
| `endTime` | datetime | 结束时间 |
| `limit` | int | 数量限制 |

示例：

```text
GET /api/alerts?processStatus=UNPROCESSED&limit=200
```

### 9.2 查询告警详情

`GET /api/alerts/{id}`

### 9.3 更新告警状态

`PUT /api/alerts/{id}/status`

请求：

```json
{
  "processStatus": "PROCESSED",
  "remark": "已核查"
}
```

### 9.4 批量更新告警状态

`PUT /api/alerts/status/batch`

请求：

```json
{
  "ids": ["a-1001", "a-1002"],
  "currentStatus": "UNPROCESSED",
  "processStatus": "PROCESSED",
  "remark": "批量确认"
}
```

## 10. Spring Boot 到远程模型的上游协议

当前两种模式对远程模型共用同一套接口：

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/health` | 健康检查 |
| `POST` | `/api/v1/analyze/start` | 启动任务 |
| `GET` | `/api/v1/analyze/{taskId}/stream` | 上游 SSE |
| `GET` | `/api/v1/analyze/{taskId}/status` | 查询进度 |
| `POST` | `/api/v1/analyze/{taskId}/stop` | 停止任务 |

`POST /api/v1/analyze/start` 的 multipart 字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `file` | file | 视频文件 |
| `sourceId` | string | 视频源 ID |
| `slotIndex` | int | 监控位 |
| `sourceName` | string | 视频源名称 |
| `locationName` | string | 地点 |

两种模式唯一区别：

| 模式 | 实际上传给远程模型的 `file` |
| --- | --- |
| `PRELOAD_BATCH` | 原始上传视频 |
| `SEVEN_FRAME_REALTIME` | Spring Boot 生成的 `10fps` 稀疏视频 |

这意味着：

- 模型侧不需要为第二种模式新增图片接口
- 只要继续支持“视频上传 + SSE 返回帧结果”的原协议即可

## 11. 对接注意事项

最容易出错的点：

1. `PRELOAD_BATCH` 模式不要等 `frame` 事件，前端收不到 `frame`
2. `PRELOAD_BATCH` 必须等待 `analysis-ready` 或 `analysisReady=true`
3. `PRELOAD_BATCH` 下 `running=false` 不代表失败，必须同时看 `analysisReady`
4. `SEVEN_FRAME_REALTIME` 下四宫格不实时叠框，抓拍图才带框
5. 上传进度不是服务端状态字段，而是浏览器上传回调
6. 推理进度统一看 `GET /api/monitor/status`
7. `bbox` 是 `640x360` 基准坐标，不是原视频像素坐标
8. 第二种模式当前实现是“10fps 稀疏视频”，不是“7 张图片逐次上传”

建议的前端轮询策略：

- `GET /api/monitor/status`：每 `2s` 轮询一次
- `GET /api/alerts`：`PRELOAD_BATCH` 在结果就绪后加载；`SEVEN_FRAME_REALTIME` 可每 `2-10s` 拉一次兜底
- `GET /api/monitor/results`：仅 `PRELOAD_BATCH` 在 `analysis-ready` 后拉取一次

## 12. 一句话总结

- `PRELOAD_BATCH`：先全量推理，后统一播放，四宫格叠完整结果
- `SEVEN_FRAME_REALTIME`：先转 `10fps` 稀疏视频，再实时推理，四宫格不实时叠框，抓拍图实时带框
