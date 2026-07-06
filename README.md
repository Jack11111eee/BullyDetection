# 面向校园安防的视频行为感知与异常事件智能预警系统

本项目面向校园监控场景，构建了一套从视频接入、人体检测、人物跟踪、行为识别、异常告警到人工处置与历史追溯的智能预警系统。系统目标不是替代安保人员作最终判断，而是把海量监控画面中的风险线索结构化、可视化、可追溯，帮助学校从“被动查看录像”转向“主动接收预警”。

项目已上传至 GitHub：<https://github.com/Yolene-Star/BullyDetection>

## 项目背景

教学楼、宿舍区、实验室、图书馆、操场和公共通道等校园区域人员密集、活动复杂。传统监控系统通常以画面留存和事后回放为主，值班人员需要长时间关注多路视频，容易在遮挡、光照变化、人员重叠、画面模糊或事件持续时间较短时发生漏看和误判。

本项目围绕校园安全管理中的典型异常事件，结合计算机视觉、姿态估计、骨骼时序建模、小目标检测、规则融合和 Web 可视化，尝试让监控系统具备“看懂行为、主动提醒、辅助处置”的能力。

## 核心能力

- 视频接入与监控位管理：支持本地视频上传、视频源元信息维护、监控位绑定和多路监控展示。
- 人体检测与人物跟踪：基于 YOLO11m-Pose 提取人体框与 17 个 COCO 关键点，并结合跟踪机制维持人物 ID 连续性。
- 骨骼时序行为识别：基于 PoseC3D 对连续骨骼序列进行建模，识别 normal、fighting、bullying、falling、climbing 等行为。
- 小目标与静态风险补充：使用 YOLO 单类检测器补充躺地、吸烟、手机使用等仅靠骨骼动作不易覆盖的场景。
- 规则引擎融合判断：融合 PoseC3D 概率、小目标检测、骨骼几何、轨迹历史、多人交互关系和时序投票，降低单帧抖动造成的误报。
- 结构化告警输出：生成包含事件类型、风险等级、人物 ID、置信度、视频来源、时间、地点和处理状态的告警事件。
- Web 可视化闭环：提供实时监控、目标框叠加、待处理告警、事件详情、历史记录、导出和系统状态查看。
- 人工复核与追溯：支持将事件标记为已处理或误报，填写备注，并进入历史记录用于后续查询。

## 识别类别

| 类别 | 说明 | 主要识别方式 |
| --- | --- | --- |
| normal | 正常行为 | PoseC3D |
| fighting | 打架 | PoseC3D + 多人关系规则 |
| bullying | 霸凌 | PoseC3D + 跨人交互规则 |
| falling | 摔倒 | PoseC3D + 姿态几何 + 躺地检测 |
| climbing | 翻越 | PoseC3D + 位移/姿态规则 |
| smoking | 吸烟 | YOLO 小目标 + 关键点位置规则 |
| phone_call | 使用手机 | YOLO 小目标 + 耳部/手部位置规则 |
| loitering | 徘徊/逗留 | 轨迹分析 |
| vandalism | 破坏公物 | PoseC3D 概率 + 场景人数规则 |
| camera_tampering | 镜头遮挡/黑屏/失焦 | 场景级画面状态检测 |

## 系统架构

```text
视频源 / 摄像头 / 帧序列
        |
        v
YOLO11m-Pose 人体检测与关键点提取
        |
        v
人物 ID 跟踪 + SkeletonBuffer 骨骼序列缓存
        |
        +--------------------------+
        |                          |
        v                          v
PoseC3D 骨骼时序识别        YOLO 小目标检测
        |                          |
        +------------+-------------+
                     v
        RuleEngine 规则融合与时序平滑
                     |
                     v
FastAPI 推理服务 REST / SSE
                     |
                     v
Spring Boot 后端任务调度、告警入库、接口聚合
                     |
                     v
Vue Web 前端实时监控、事件处置、历史追溯
```

当前部署包的主要服务拓扑：

```text
浏览器
  -> Spring Boot 后端 + Vue 前端一体 JAR（默认 8915）
      -> Python FastAPI 推理服务（默认 8000）
          -> YOLO + PoseC3D + RuleEngine
      -> MySQL（默认 3306）
```

## 仓库结构

```text
BullyDetection/
├── campus-guard-demo/              # 可部署演示包
│   ├── campus-guard-backend-1.0.0.jar
│   ├── e2e_pipeline/               # Python 推理服务
│   │   ├── api_server.py           # FastAPI + SSE 服务入口
│   │   ├── pipeline.py             # YOLO -> PoseC3D -> RuleEngine 推理流水线
│   │   ├── rule_engine.py          # 多行为规则融合与时序平滑
│   │   ├── scene_event_detector.py # 镜头遮挡、黑屏、失焦等场景异常检测
│   │   ├── input_source.py         # 视频/摄像头/帧序列输入抽象
│   │   └── API_README.md           # 推理服务接口说明
│   └── README.md                   # 部署指南
├── e2e_pipeline/                   # 端到端推理相关代码与调试版本
├── mil_cleaning/                   # 数据清洗与 MIL 相关脚本
├── training_docs/                  # 训练过程文档
├── build_pkl.py                    # 骨骼数据集构建脚本
├── build_kfold_data.py             # K 折数据构建
├── eval_*.py                       # 模型评估脚本
├── main_inference.py               # 推理入口/历史版本
├── rule_engine.py                  # 根目录规则引擎版本
├── DUAL-MODE-API-PROTOCOL.md       # 双模式 Web/API 对接协议
├── WEB-HANDOFF.md                  # 前后端联调说明
├── YOLO_HANDOFF.md                 # YOLO 模型交接说明
└── PROJECT_PROGRESS.md             # 项目进展记录
```

说明：训练数据、视频、模型权重、pyskl 框架、日志和 pkl 文件体积较大，已通过 `.gitignore` 排除。部署或复现实验时需要按 `campus-guard-demo/README.md` 准备相应模型权重和依赖目录。

## 快速启动

详细部署步骤请查看 [campus-guard-demo/README.md](campus-guard-demo/README.md)。下面是最小启动流程摘要。

### 1. 环境要求

| 组件 | 推荐版本 | 说明 |
| --- | --- | --- |
| Python | 3.10 | 推荐 Conda 环境 |
| PyTorch | 2.1.0+ | GPU 推理推荐 CUDA 11.8 |
| Java | 17 | 运行 Spring Boot JAR |
| MySQL | 8.0+ | 告警、视频源、配置等业务数据 |
| CUDA | 11.8 | 可选，仅 GPU 推理需要 |

### 2. 启动 Python 推理服务

在 `campus-guard-demo/` 目录下执行：

```bash
python e2e_pipeline/api_server.py \
  --posec3d-config pyskl/configs/posec3d/finetune_campus_mil.py \
  --posec3d-ckpt models/epoch_50.pth \
  --host 0.0.0.0 \
  --port 8000
```

无 GPU 环境可追加：

```bash
--device cpu
```

推理服务启动后可通过以下命令检查：

```bash
curl http://localhost:8000/health
```

### 3. 启动 Web 后端与前端

在 `campus-guard-demo/` 目录下新开终端：

```bash
java -jar campus-guard-backend-1.0.0.jar
```

默认数据库配置：

| 参数 | 默认值 |
| --- | --- |
| 数据库 | `campus_guard` |
| 用户名 | `root` |
| 密码 | `12138` |
| 后端端口 | `8915` |
| 推理服务地址 | `http://localhost:8000` |

如需覆盖 MySQL 密码：

```bash
java -jar campus-guard-backend-1.0.0.jar \
  --spring.datasource.password=你的密码
```

浏览器访问：

```text
http://localhost:8915
```

## 推理服务 API

Python 端提供 REST + SSE 接口供 Spring Boot 后端调用：

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/health` | 健康检查 |
| `POST` | `/api/v1/analyze/start` | 上传视频并启动分析任务 |
| `GET` | `/api/v1/analyze/{taskId}/stream` | 订阅 SSE 推理结果流 |
| `POST` | `/api/v1/analyze/{taskId}/stop` | 停止任务 |
| `GET` | `/api/v1/analyze/{taskId}/status` | 查询任务状态 |
| `GET` | `/api/v1/tasks` | 查看近期任务，主要用于调试 |

典型联调流程：

```text
Spring Boot /api/monitor/start
  -> POST /api/v1/analyze/start
  <- { taskId, status }
  -> GET /api/v1/analyze/{taskId}/stream
  <- event: frame / alert / done / error
  -> Spring Boot 转换并推送给前端 /api/monitor/stream
```

SSE `frame` 事件会包含当前帧目标框、人物 ID、行为标签、置信度、告警列表、视频时间戳和原始分辨率等字段。坐标统一按 `640x360` 基准输出，便于前端叠加渲染。

## Web 功能模块

| 模块 | 功能 |
| --- | --- |
| 校园态势总览 | 展示告警数量、风险分布、区域排行和运行状态 |
| 值守工作台 | 多路视频查看、目标框叠加、行为标签、待处理告警、事件时间轴 |
| 接入与点位管理 | 上传视频源、维护地点/楼层/监控位、绑定四宫格监控位 |
| 事件处置中心 | 查看告警详情、截图/片段、置信度、处理状态和备注 |
| 历史记录 | 按类型、等级、状态、时间筛选事件并导出 CSV/JSON |
| 系统与模型状态 | 查看推理服务、数据库、运行模式和系统日志 |

## 模型与训练

项目围绕骨骼动作识别和小目标补充检测构建模型链路：

- 使用 YOLO11m-Pose 对视频逐帧提取人体框与 17 点骨骼。
- 将骨骼序列打包为 pyskl 支持的 PKL 格式。
- 基于 PoseC3D / SlowOnly-R50 backbone 训练校园异常行为识别模型。
- 使用 Ultralytics YOLO 训练躺地、吸烟、手机等单类检测器。
- 通过规则引擎将模型输出映射为校园安防业务事件。

相关脚本包括：

| 文件 | 用途 |
| --- | --- |
| `yolo11-base.py` | 提取人体关键点 |
| `build_pkl.py` / `build_new_pkl.py` | 构建 PoseC3D 骨骼数据 |
| `build_kfold_data.py` | 构建 K 折训练数据 |
| `eval_kfold.py` / `eval_ensemble.py` / `eval_multiclip.py` | 评估不同训练与推理策略 |
| `confusion_matrix.py` | 生成混淆矩阵 |
| `plot_training_curves.py` | 绘制训练曲线 |
| `preprocess_datasets.py` | 数据预处理 |

## 当前测试效果

根据项目方案文档和现有训练记录，当前核心行为识别模型在团队自建真实场景测试集上达到：

| 指标 | 当前结果 |
| --- | --- |
| Overall Top-1 | 93.3% |
| Mean Class Accuracy | 91.8% |
| 正常行为误报率 | 约 6.7% |
| 异常行为漏报率 | 约 4.9% |
| 核心分析模块耗时 | 约 25ms - 30ms |
| 默认分析帧率 | 约 12fps |

类别表现：

| 类别 | 识别效果 |
| --- | --- |
| falling | 99.1% |
| climbing | 98.4% |
| normal | 93.3% |
| fighting | 89.7% |
| bullying | 78.4% |

说明：上述结果来自团队自建测试集和项目文档，端到端延迟还会受到视频读取、硬件环境、告警生成、前端显示和多路并发等因素影响。真实校园试点时仍需补充长期运行、不同摄像头视角、夜间低照度、人员密集和严重遮挡条件下的统计数据。

## 项目创新点

### 多模型融合

系统并不依赖单一分类器，而是融合人体姿态、骨骼时序动作、小目标检测、轨迹分析和规则判断。骨骼序列适合表达打架、摔倒、翻越等动态行为，小目标检测用于补充吸烟、手机等细粒度风险，规则引擎负责把多源结果转换成校园安防业务语义。

### 面向真实视频的稳定机制

真实校园视频中会出现遮挡、人物 ID 切换、多人重叠、动作抖动、静态倒地和画面异常。系统在 `SkeletonBuffer` 和 `RuleEngine` 中加入了时序缓存、平滑、投票、异常保持、跨人关系判断和降级保护等机制，减少单帧误判。

### 从算法到业务闭环

项目不仅展示模型预测结果，还完成了视频源管理、分析任务调度、SSE 实时推送、告警入库、前端展示、人工复核和历史导出，形成“接入 -> 识别 -> 告警 -> 处置 -> 追溯”的完整流程。

## 应用场景

- 教学楼和公共通道：打架、摔倒、翻越、异常聚集等风险发现。
- 宿舍区和楼道：霸凌、吸烟、手机使用、徘徊等敏感行为识别。
- 实验室和重点区域：异常闯入、疑似破坏、摄像头遮挡等事件预警。
- 校园监控中心：多路视频统一查看、告警处理、历史检索与导出。

项目也可进一步扩展到中小学、幼托机构、培训机构、园区、宿舍楼、图书馆、体育场馆和社区公共区域等场景。

## 合规与风险边界

- 系统输出应作为辅助预警线索，不能作为处理事件的唯一依据。
- 真实部署应限定授权区域、授权人员和授权用途。
- 视频数据、告警记录和人员隐私应遵循最小必要原则进行采集、存储和访问。
- 在摄像头角度极端、画面压缩严重、低照度、严重遮挡或人员高度密集时，关键点、小目标检测和人物跟踪可能不稳定。
- 后续试点应建立误报、漏报复核机制，并保留人工处置记录。

## 后续优化方向

- 补充更多真实校园场景数据，提升跨场景泛化能力。
- 优化多人重叠和霸凌场景中的角色区分能力。
- 完善端到端延迟、多路并发、显存占用和长时间运行稳定性测试。
- 增加部署脚本、环境检查脚本和模型权重下载/校验说明。
- 完善权限控制、日志审计和隐私保护策略。
- 支持更多摄像头协议和现有校园安防平台对接。

## 参考文档

- [部署指南](campus-guard-demo/README.md)
- [推理服务 API 文档](campus-guard-demo/e2e_pipeline/API_README.md)
- [双模式接口对接协议](DUAL-MODE-API-PROTOCOL.md)
- [Web 交接说明](WEB-HANDOFF.md)
- [项目进展记录](PROJECT_PROGRESS.md)
- [YOLO 模型交接说明](YOLO_HANDOFF.md)
