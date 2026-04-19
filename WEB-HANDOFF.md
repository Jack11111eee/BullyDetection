# 网页对接说明

更新日期：2026-04-19

这份说明基于当前代码仓库 `D:\code\web1` 的实际实现整理，适合直接发给需要联调或接手该网页的同伴。

## 1. 项目概况

这是一个前后端分离项目，当前实现的是“校园异常行为检测系统”网页。

- 前端：`Vue 3 + Vite + Element Plus`
- 后端：`Spring Boot 3.3 + Spring Web + JPA + MySQL`
- 数据库：`MySQL 8`
- 实时通信：`SSE`
- 当前已接入的视频源类型：`本地上传视频`

目录说明：

- `frontend`：前端工程
- `backend`：后端工程
- `docs`：产品和对接文档
- `design`：设计说明

## 2. 启动方式

### 2.1 环境要求

- `Java 17+`
- `Maven 3.8+`
- `Node.js 18+`
- `MySQL 8+`

### 2.2 数据库配置

后端默认读取配置如下：

- 地址：`localhost:3306`
- 数据库：`campus_guard`
- 用户名：`root`
- 密码：`12138`

也可以通过环境变量覆盖：

- `MYSQL_USER`
- `MYSQL_PASSWORD`

当前后端配置开启了 `createDatabaseIfNotExist=true` 和 `ddl-auto=update`，首次启动会自动建库和补表结构。

### 2.3 启动后端

在 `backend` 目录执行：

```powershell
mvn spring-boot:run
```

默认端口：`8080`

说明：

- 上传文件和抓拍图片保存在 `uploads` 目录下。
- 这个路径是相对“后端进程启动时的工作目录”的。
- 如果你是在 `D:\code\web1\backend` 下执行 `mvn spring-boot:run`，那么实际目录就是：
  - `D:\code\web1\backend\uploads\videos`
  - `D:\code\web1\backend\uploads\snapshots`

### 2.4 启动前端

在 `frontend` 目录执行：

```powershell
npm install
npm run dev
```

默认端口：`5173`

开发环境访问地址：

- 前端：`http://localhost:5173`
- 后端：`http://localhost:8080`

## 3. 前后端连接方式

前端通过 `axios` 访问后端，统一配置如下：

- 前端 API 基址：`/api`
- 开发代理：Vite 会把以下路径代理到 `http://localhost:8080`
  - `/api`
  - `/media`
  - `/mock`

这意味着：

- 前端开发时不需要额外改接口地址。
- 后端返回的媒体地址可以直接用相对路径，例如 `/media/videos/xxx.mp4`。

生产部署时有两种方式：

1. 用网关或 Nginx 反向代理 `/api`、`/media`、`/mock` 到后端。
2. 如果前后端分开域名直连，需要修改前端 [`http.js`](D:/code/web1/frontend/src/api/http.js) 的 `baseURL`，并同步处理媒体路径。

## 4. 页面职责

当前前端路由如下：

| 路由 | 页面职责 |
| --- | --- |
| `/monitor` | 实时监测页，展示 2x2 监控位、SSE 实时事件、待处理告警、事件详情、系统日志 |
| `/sources` | 设备接入页，上传本地视频、绑定监控位、编辑元信息、删除视频源 |
| `/history` | 事件记录页，支持筛选、查看详情、更新处理状态、导出 CSV/JSON |
| `/settings` | 系统设置页，切换模式、查看服务状态、查看数据库连通状态 |

## 5. 当前页面联调流程

建议同伴按下面顺序联调：

1. 启动 MySQL。
2. 启动后端 `8080`。
3. 启动前端 `5173`。
4. 打开 `/sources` 页面上传一个本地视频。
5. 绑定到 `监控 1-4` 中任一槽位。
6. 进入 `/monitor` 页面点击“开始分析”。
7. 前端会通过 `/api/monitor/stream` 建立 SSE 连接。
8. 如果当前模式是 `MOCK`，后端会按固定周期生成模拟告警。
9. 告警会同步出现在：
   - `/monitor` 的右侧待处理列表
   - `/monitor` 的时间轴
   - `/history` 的事件记录

## 6. 关键接口说明

### 6.1 视频源接口

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/api/sources` | 查询全部视频源 |
| `POST` | `/api/upload/video` | 上传本地视频 |
| `PUT` | `/api/sources/{id}/meta` | 更新视频源名称、地点、备注等 |
| `PUT` | `/api/sources/{id}/binding` | 更新监控位绑定 |
| `DELETE` | `/api/sources/{id}` | 删除视频源 |
| `GET` | `/api/upload/video/{id}` | 查询单个视频源元信息 |

`POST /api/upload/video` 使用 `multipart/form-data`，字段如下：

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `file` | file | 是 | 视频文件 |
| `slotIndex` | number | 否 | 监控位，允许 `1-4`，也可不传 |
| `sourceName` | string | 否 | 视频源名称 |
| `locationName` | string | 否 | 地点名称 |
| `remark` | string | 否 | 备注 |
| `resolution` | string | 否 | 例如 `1920x1080` |
| `fps` | number | 否 | 帧率 |
| `durationSec` | number | 否 | 时长，秒 |

返回对象 `SourceView` 结构：

```json
{
  "sourceId": "uuid",
  "slotIndex": 1,
  "sourceName": "Cam-01",
  "sourceType": "LOCAL_VIDEO",
  "status": "READY",
  "locationName": "教学楼 2 层走廊",
  "remark": "东向",
  "resolution": "1920x1080",
  "fps": 25,
  "durationSec": 120,
  "fileName": "demo.mp4",
  "fileSize": 12345678,
  "playUrl": "/media/videos/xxx.mp4",
  "createdAt": "2026-04-19T01:23:45",
  "updatedAt": "2026-04-19T01:23:45"
}
```

绑定规则：

- 一个监控位只能绑定一个视频源。
- `slotIndex` 只能是 `1-4` 或 `null`。
- 如果监控位已被占用，后端会返回 `409 Conflict`。

### 6.2 实时监控接口

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `POST` | `/api/monitor/start` | 开始分析 |
| `POST` | `/api/monitor/pause` | 暂停分析 |
| `POST` | `/api/monitor/stop` | 停止分析 |
| `GET` | `/api/monitor/status` | 获取当前监控状态 |
| `GET` | `/api/monitor/stream` | 建立 SSE 实时流 |

`GET /api/monitor/status` 返回结构：

```json
{
  "mode": "MOCK",
  "running": true,
  "onlineSourceCount": 2,
  "activeAlertCount": 5,
  "highRiskCount": 2,
  "modelStatus": "Mock 模型在线",
  "streamStatus": "已连接(1)",
  "mysqlStatus": "已连接",
  "updatedAt": "2026-04-19T01:30:00",
  "latestLogs": [
    "2026-04-19T01:29:58 | 前端已连接实时事件流",
    "2026-04-19T01:29:59 | 检测到高风险异常"
  ]
}
```

### 6.3 SSE 对接说明

SSE 地址：

- `GET /api/monitor/stream`

服务端会先发送一个连接确认事件：

```text
event: connected
data: stream-connected
```

后续主要发送 `frame` 事件，数据结构如下：

```json
{
  "sourceId": "uuid",
  "slotIndex": 1,
  "sourceName": "Cam-01",
  "timestamp": "2026-04-19T01:35:00",
  "targets": [
    {
      "bbox": [100, 60, 180, 220],
      "behavior": "fight",
      "confidence": 0.93
    }
  ],
  "alerts": [
    {
      "alertId": "uuid",
      "type": "FIGHT",
      "behaviorName": "打架",
      "level": "HIGH",
      "levelName": "高风险异常",
      "timestamp": "2026-04-19T01:35:00",
      "sourceId": "uuid",
      "slotIndex": 1,
      "sourceName": "Cam-01",
      "locationName": "教学楼 2 层走廊",
      "confidence": 0.93,
      "snapshotUrl": "/media/snapshots/xxx.jpg",
      "clipUrl": "/media/videos/xxx.mp4#t=12.5",
      "processStatus": "UNPROCESSED",
      "processStatusName": "未处理",
      "remark": null
    }
  ],
  "systemLog": "异常已推送"
}
```

补充说明：

- 当前前端监听的事件名是 `frame`，不是默认 message。
- `targets[].bbox` 当前按 `640x360` 基准坐标渲染，格式为 `[x, y, width, height]`。
- `clipUrl` 可能带 `#t=秒数`，用于跳到视频片段时间点。
- 当前前端已实现断线 3 秒自动重连。

### 6.4 告警接口

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/api/alerts` | 查询告警列表 |
| `GET` | `/api/alerts/{id}` | 查询单条告警详情 |
| `PUT` | `/api/alerts/{id}/status` | 更新处理状态 |
| `GET` | `/api/alerts/export` | 导出告警数据 |

`GET /api/alerts` 支持的查询参数：

| 参数 | 说明 |
| --- | --- |
| `type` | `FIGHT / FALL / SMOKING / LOITER / CLIMB` |
| `level` | `HIGH / SENSITIVE / SUSPICIOUS` |
| `processStatus` | `UNPROCESSED / PROCESSED / FALSE_POSITIVE` |
| `sourceId` | 视频源 ID |
| `startTime` | 开始时间，格式如 `2026-04-19T01:00:00` |
| `endTime` | 结束时间，格式如 `2026-04-19T02:00:00` |
| `limit` | 返回条数限制 |

更新状态请求体：

```json
{
  "processStatus": "PROCESSED",
  "remark": "已人工确认"
}
```

导出接口说明：

- `GET /api/alerts/export?format=csv`
- `GET /api/alerts/export?format=json`

返回的是文件字节流，不是普通 JSON。

### 6.5 设置接口

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/api/settings` | 获取系统设置 |
| `PUT` | `/api/settings/mode` | 更新模式和开关项 |
| `GET` | `/api/settings/db-status` | 查询数据库状态 |

更新设置请求体：

```json
{
  "mode": "MOCK",
  "autoStart": false,
  "showSkeletonDebug": false
}
```

`autoStart` 会持久化到数据库，并在后端下次启动时生效。

## 7. 枚举值说明

### 7.1 行为类型

| 值 | 含义 |
| --- | --- |
| `FIGHT` | 打架 |
| `FALL` | 跌倒 |
| `SMOKING` | 抽烟 |
| `LOITER` | 徘徊 |
| `CLIMB` | 翻越围栏 |

### 7.2 风险等级

| 值 | 含义 |
| --- | --- |
| `HIGH` | 高风险异常 |
| `SENSITIVE` | 敏感行为 |
| `SUSPICIOUS` | 可疑行为 |

### 7.3 告警处理状态

| 值 | 含义 |
| --- | --- |
| `UNPROCESSED` | 未处理 |
| `PROCESSED` | 已处理 |
| `FALSE_POSITIVE` | 误报 |
| `VIEWED` | 后端内部兼容状态，前端会被归一成 `PROCESSED` |

### 7.4 运行模式

| 值 | 含义 |
| --- | --- |
| `MOCK` | 模拟模式，当前默认模式，可自动生成演示告警 |
| `LIVE` | 预留真实模型模式，当前代码里还没有接入真实模型服务 |

## 8. 媒体资源路径

后端通过静态资源映射暴露以下路径：

| 路径前缀 | 实际来源 |
| --- | --- |
| `/media/**` | `uploads` 目录 |
| `/mock/**` | `classpath:/static/mock/` |

因此：

- 视频播放地址通常是 `/media/videos/xxx.mp4`
- 抓拍图地址通常是 `/media/snapshots/xxx.jpg`
- 默认占位图是 `/mock/snapshot-default.svg`

## 9. 错误返回格式

后端统一错误格式如下：

```json
{
  "timestamp": "2026-04-19T01:40:00",
  "status": 409,
  "error": "Conflict",
  "message": "绑定失败，监控 1 已被其他视频源占用"
}
```

常见状态码：

- `400`：参数错误
- `404`：资源不存在
- `409`：资源冲突，例如监控位被占用
- `500`：服务内部异常

## 10. 当前实现边界

这几个点建议提前跟同伴说明，避免误解：

- 当前真正可跑通的是“本地视频 + Mock 事件流”。
- `LIVE` 模式目前只是配置项和状态位，还没有真实算法服务接入。
- 前端默认假设后端能直接返回可访问的相对媒体路径。
- 前端开发环境依赖 Vite 代理，线上如果没有反向代理会请求失败。
- 实时框坐标渲染使用固定比例，如果后续算法输出坐标系变化，前端也要同步改。

## 11. 可以直接发给同伴的简版说明

可以直接复制下面这段给同伴：

```text
这个网页是前后端分离项目，前端在 frontend，后端在 backend。

启动顺序：
1. 先启动 MySQL，库名 campus_guard，默认账号 root/12138。
2. 在 backend 目录执行 mvn spring-boot:run，端口 8080。
3. 在 frontend 目录执行 npm install && npm run dev，端口 5173。

前端通过 /api 调后端，开发环境已经在 vite.config.js 里把 /api、/media、/mock 代理到 8080 了。

主要页面：
- /sources 上传本地视频并绑定监控位
- /monitor 看实时监控和告警
- /history 查历史事件和导出
- /settings 切模式和看服务状态

联调时先在 /sources 上传视频并绑定到监控 1-4，再去 /monitor 点开始分析。
当前默认是 MOCK 模式，后端会通过 /api/monitor/stream 按 SSE 推送 frame 事件，前端会实时显示告警。

如果你要接接口，重点看这几个：
- POST /api/upload/video
- GET /api/sources
- PUT /api/sources/{id}/binding
- GET /api/monitor/status
- GET /api/monitor/stream
- GET /api/alerts
- PUT /api/alerts/{id}/status
- GET /api/settings
- PUT /api/settings/mode

注意：
- 一个监控位只能绑定一个视频源，占用时后端会返回 409。
- 返回的视频和截图地址是 /media/... 相对路径。
- LIVE 模式目前还没接真实模型，只能完整演示 MOCK 链路。
```
