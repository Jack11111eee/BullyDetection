# 校园异常行为检测系统 — 部署指南

## 目录结构

```
campus-guard-demo/
├── campus-guard-backend-1.0.0.jar   # Spring Boot 后端 + Vue 前端（一体 JAR）
├── e2e_pipeline/                    # Python 推理服务
│   ├── api_server.py                #   FastAPI + SSE 服务入口
│   ├── pipeline.py                  #   YOLO → PoseC3D → RuleEngine 推理流水线
│   ├── rule_engine.py               #   11 类行为判定 + 时序投票
│   ├── scene_event_detector.py      #   镜头遮挡/移动检测
│   ├── input_source.py              #   视频/摄像头/帧序列输入抽象
│   └── run.py                       #   CLI 入口（离线推理，非服务模式）
├── pyskl/                           #   PoseC3D 骨架行为识别框架
│   ├── pyskl/                       #     核心 Python 包
│   ├── configs/posec3d/             #     模型配置文件
│   ├── requirements.txt             #     Python 依赖
│   └── setup.py                     #     安装脚本
├── models/                          # 模型权重
│   ├── yolo11m-pose.pt              #   YOLO11m 姿态检测（40MB）
│   ├── epoch_50.pth                 #   PoseC3D 行为识别 checkpoint（15MB）
│   ├── joint.pth                    #   PoseC3D 预训练 backbone（8MB）
│   ├── best-falling.pt              #   YOLO 躺地/摔倒检测（39MB）
│   ├── best-smoke.pt                #   YOLO 吸烟��测（39MB）
│   └── best-phone.pt                #   YOLO 手机使用检测（39MB）
└── sample_videos/                   # 演示视频（自行放入）
```

## 系统架构

```
浏览器 ──→ Spring Boot (8080) ──→ Python 推理服务 (8000)
              │                       │
              │ JPA                    │ YOLO + PoseC3D
              ▼                       ▼
           MySQL              模型推理 + SSE 推送
```

- **前端**：Vue 3 + Element Plus，已打包进 JAR，访问 `http://localhost:8080` 即可
- **后端**：Spring Boot 3.3，负责视频管理、告警记录、转发推理请求
- **推理**：FastAPI 服务，加载 YOLO11m-Pose + PoseC3D 模型，通过 SSE 推送实时结果

---

## 一、环境要求

| 组件 | 版本要求 | 说明 |
|------|---------|------|
| **操作系统** | Linux / Windows / macOS | 推荐 Ubuntu 20.04+ |
| **Python** | 3.10 | 推荐使用 Conda 管理 |
| **PyTorch** | 2.1.0+ | 有 NVIDIA GPU 则安装 CUDA 版；无 GPU 可用 CPU 版 |
| **CUDA**（可选） | 11.8 | 仅 GPU 推理需要 |
| **Java** | 17 | 运行 Spring Boot JAR |
| **MySQL** | 8.0+ | 数据库，表会自动创建 |

> **注意**：不需要 Node.js — 前端已打包进 JAR 中。

---

## 二、环境安装

### 2.1 安装 MySQL

```bash
# Ubuntu
sudo apt update && sudo apt install mysql-server -y
sudo systemctl start mysql

# 登录并设置密码（默认密码 12138，可自定义）
sudo mysql -u root
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY '12138';
FLUSH PRIVILEGES;
EXIT;
```

MySQL 启动后无需手动建库建表，Spring Boot 会自动创建 `campus_guard` 数据库和所有表。

### 2.2 安装 Java 17

```bash
# Ubuntu
sudo apt install openjdk-17-jre-headless -y

# 验证
java -version
```

### 2.3 安装 Python 环境

推荐使用 Conda：

```bash
# 创建环境
conda create -n campus python=3.10 -y
conda activate campus
```

#### 2.3.1 安装 PyTorch

**有 NVIDIA GPU（推荐）：**
```bash
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
```

**无 GPU（CPU 推理）：**
```bash
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu
```

#### 2.3.2 安装 mmcv-full

```bash
# GPU 版
pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html

# CPU 版
pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.1/index.html
```

#### 2.3.3 安装其他依赖

```bash
# 推理服务依赖
pip install fastapi uvicorn[standard] sse-starlette python-multipart

# 模型依赖
pip install ultralytics opencv-python numpy scipy

# mmdet / mmpose（pyskl 框架需要）
pip install mmdet==2.23.0 mmpose==0.24.0 decord fvcore
```

#### 2.3.4 安装 pyskl

```bash
cd campus-guard-demo/pyskl
pip install -e .
```

---

## 三、路径配置

### 3.1 项目路径（已自动处理）

推理代码中的 Python 路径已使用相对路径，**只要从 `campus-guard-demo/` 目录启动**，无需额外配置。

### 3.2 Spring Boot 配置

JAR 包内置了默认配置，可通过启动参数覆盖：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--spring.datasource.password` | `12138` | MySQL 密码 |
| `--spring.datasource.username` | `root` | MySQL 用户名 |
| `--spring.datasource.url` | `jdbc:mysql://localhost:3306/campus_guard?...` | MySQL 连接地址 |
| `--app.live.analyze-url` | `http://localhost:8000` | Python 推理服务地址 |

默认配置假定推理服务和后端在同一台机器，直接启动即可：

```bash
java -jar campus-guard-backend-1.0.0.jar

# 如果 MySQL 密码不是 12138
java -jar campus-guard-backend-1.0.0.jar \
  --spring.datasource.password=你的密码
```

### 3.3 推理服务配置

通过命令行参数配置，无需修改文件：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--posec3d-config` | 无（必填） | PoseC3D 模型配置文件路径 |
| `--posec3d-ckpt` | 无（必填） | PoseC3D 模型权重路径 |
| `--yolo-pose` | `models/yolo11m-pose.pt` | YOLO 姿态模型路径 |
| `--device` | `cuda:0` | 推理设备，无 GPU 需改为 `cpu` |
| `--host` | `0.0.0.0` | 绑定地址 |
| `--port` | `8000` | 服务端口 |
| `--falling-model` | `models/best-falling.pt` | 躺地检测 YOLO 模型 |
| `--smoking-model` | `models/best-smoke.pt` | 吸烟检测 YOLO 模型 |
| `--phone-model` | `models/best-phone.pt` | 手机检测 YOLO 模型 |

---

## 四、启动步骤

**所有命令均在 `campus-guard-demo/` 目录下执行。**

### 步骤 1：启动 MySQL

```bash
# 确认 MySQL 已运行
sudo systemctl status mysql
```

### 步骤 2：启动 Python 推理服务（端口 8000）

```bash
conda activate campus

# GPU 推理（yolo-pose / host / port 等均有正确默认值，可省略）
python e2e_pipeline/api_server.py \
  --posec3d-config pyskl/configs/posec3d/finetune_campus_mil.py \
  --posec3d-ckpt models/epoch_50.pth

# CPU 推理（无 GPU 时加 --device cpu）
python e2e_pipeline/api_server.py \
  --posec3d-config pyskl/configs/posec3d/finetune_campus_mil.py \
  --posec3d-ckpt models/epoch_50.pth \
  --device cpu
```

启动成功后会看到：
```
[1/3] Loading YOLO Pose model...
[2/3] Loading PoseC3D model...
[3/3] Building rule engine...
Uvicorn running on http://0.0.0.0:8000
```

### 步骤 3：启动 Spring Boot 后端 + 前端（端口 8080）

新开一个终端：

```bash
java -jar campus-guard-backend-1.0.0.jar
```

如果 MySQL 密码不是 `12138`：
```bash
java -jar campus-guard-backend-1.0.0.jar \
  --spring.datasource.password=你的密码
```

### 步骤 4：打开浏览器

访问 **http://localhost:8080**

---

## 五、功能说明

| 页面 | 功能 |
|------|------|
| **监控页** | 实时四宫格监控，带 bbox 叠加和行为标签 |
| **视频源管理** | 上传视频文件进行分析 |
| **告警历史** | 查看历史告警记录，支持筛选和 CSV/JSON 导出 |
| **设置页** | 切换分析模式、查看服务状态 |

### 行为识别类别

| 类别 | 识别方式 |
|------|---------|
| 正常 (normal) | PoseC3D 骨架识别 |
| 打架 (fighting) | PoseC3D 骨架识别 |
| 霸凌 (bullying) | PoseC3D 骨架识别 |
| 摔倒 (falling) | PoseC3D 骨架识别 |
| 翻越 (climbing) | PoseC3D 骨架识别 |
| 躺地 (laying) | 单类 YOLO 模型（best-falling.pt） |
| 吸烟 (smoking) | 单类 YOLO 模型（best-smoke.pt） |
| 使用手机 (phone) | 单类 YOLO 模型（best-phone.pt） |
| 镜头遮挡 (camera_blocked) | Canny 边缘密度检测 |

---

## 六、健康检查

```bash
# 检查推理服务
curl http://localhost:8000/health

# 检查后端服务
curl http://localhost:8080/actuator/health
```

---

## 七、常见问题

### Q: 启动推理服务报 CUDA 错误

如果没有 GPU，请加 `--device cpu` 参数。如果有 GPU 但版本不匹配，检查 CUDA 和 PyTorch 版本是否对应。

### Q: Spring Boot 启动报数据库连接失败

1. 确认 MySQL 已启动：`sudo systemctl status mysql`
2. 确认密码正确：通过 `--spring.datasource.password=xxx` 覆盖
3. 确认 MySQL 允许本地连接

### Q: 前端页面打不开

确认 JAR 正常启动且 8080 端口未被占用：
```bash
lsof -i :8080
```

### Q: 上传视频后没有分析结果

1. 确认推理服务（端口 8000）已启动
2. 确认后端的 `analyze-url` 指向了正确的推理服务地址
3. 查看推理服务终端的日志输出

### Q: 躺地/吸烟/手机检测不生效

确认 `models/` 目录下存在 `best-falling.pt`、`best-smoke.pt`、`best-phone.pt` 三个文件。启动时默认自动加载，如需禁用某项可设为 `none`：
```bash
python e2e_pipeline/api_server.py \
  --posec3d-config pyskl/configs/posec3d/finetune_campus_mil.py \
  --posec3d-ckpt models/epoch_50.pth \
  --smoking-model none
```

---

## 八、测试接口

### 8.1 推理服务 REST API（端口 8000）

详细接口文档见 `e2e_pipeline/API_README.md`，核心端点：

| 方法 | 路径 | 用途 |
|------|------|------|
| `GET` | `/health` | 健康检查，确认模型已加载 |
| `POST` | `/api/v1/analyze/start` | 上传视频 + metadata，启动分析任务 |
| `GET` | `/api/v1/analyze/{taskId}/stream` | SSE 推送 frame / alert / done 事件 |
| `POST` | `/api/v1/analyze/{taskId}/stop` | 主动停止任务 |
| `GET` | `/api/v1/analyze/{taskId}/status` | 查询任务进度 |

快速测试：

```bash
# 1. 健康检查
curl http://localhost:8000/health

# 2. 提交视频分析
curl -F "file=@sample_videos/test.mp4" \
     -F "sourceId=test-01" \
     http://localhost:8000/api/v1/analyze/start
# 返回: {"taskId":"xxx", "status":"started", ...}

# 3. 订阅 SSE 结果流
curl -N http://localhost:8000/api/v1/analyze/{taskId}/stream
# 输出: event: connected → event: frame (持续) → event: done
```

### 8.2 Web 后端 REST API（端口 8080）

| 方法 | 路径 | 用途 |
|------|------|------|
| `POST` | `/api/sources` | 创建视频源 |
| `POST` | `/api/sources/{id}/upload` | 上传视频文件 |
| `GET` | `/api/monitor/stream?sourceIds=...` | SSE 实时监控流（前端四宫格） |
| `POST` | `/api/monitor/start` | 启动分析（前端触发） |
| `GET` | `/api/events` | 查询告警事件列表 |
| `GET` | `/api/events/export?format=csv` | 导出告警记录 |
| `GET` | `/actuator/health` | Spring Boot 健康检查 |

---

## 九、模型训练接口

### 9.1 PoseC3D 行为识别模型训练

训练框架：pyskl（基于 MMAction2），训练脚本位于 `pyskl/tools/train.py`。

```bash
cd campus-guard-demo

# 单卡训练
python pyskl/tools/train.py \
  pyskl/configs/posec3d/finetune_campus_mil.py \
  --work-dir pyskl/work_dirs/posec3d_campus_mil \
  --seed 0 --deterministic

# 多卡训练（示例：2 卡）
bash pyskl/tools/dist_train.sh \
  pyskl/configs/posec3d/finetune_campus_mil.py 2
```

训练配置文件 `pyskl/configs/posec3d/finetune_campus_mil.py` 中的关键参数：

| 参数 | 当前值 | 说明 |
|------|--------|------|
| `num_classes` | 5 | 正常/打架/霸凌/摔倒/翻越 |
| `clip_len` | 64 | 输入片段帧数 |
| `total_epochs` | 50 | 训练轮数 |
| `lr` | 0.005 | 学习率（SGD） |
| `videos_per_gpu` | 16 | 单卡 batch size |
| `load_from` | `models/joint.pth` | 预训练 backbone（SlowOnly-R50 on NTU-120） |
| `ann_file` | `data/campus/campus_mil_kfold_0.pkl` | 标注数据（PKL 骨架格式） |

### 9.2 模型评估

```bash
# 单卡测试
python pyskl/tools/test.py \
  pyskl/configs/posec3d/finetune_campus_mil.py \
  models/epoch_50.pth \
  --eval top_k_accuracy mean_class_accuracy

# 多卡测试
bash pyskl/tools/dist_test.sh \
  pyskl/configs/posec3d/finetune_campus_mil.py \
  models/epoch_50.pth 2 \
  --eval top_k_accuracy mean_class_accuracy
```

### 9.3 YOLO 单类检测模型训练

躺地/吸烟/手机三个检测模型使用 Ultralytics YOLOv11 训练：

```python
from ultralytics import YOLO

model = YOLO('yolo11m.pt')  # 预训练 base

model.train(
    data='path/to/dataset.yaml',  # YOLO 格式数据集配置
    epochs=100,
    imgsz=640,
    batch=16,
    name='falling_yolo11m',
)
```

数据集格式为标准 YOLO 检测格式（images/ + labels/，每行 `class x_center y_center width height`）。

### 9.4 训练数据说明

| 数据 | 格式 | 说明 |
|------|------|------|
| PoseC3D 骨架数据 | PKL (pyskl格式) | 每条包含 keypoint 坐标序列 + 标签，由 YOLO-Pose 提取 |
| YOLO 检测数据 | YOLO txt | 标准目标检测标注 |

骨架数据提取流程：原始视频 → YOLO11m-Pose 逐帧检测 → 17 关键点坐标序列 → 打包为 PKL。

### 9.5 涉及的训练及预训练模型

| 模型文件 | 类型 | 来源 | 说明 |
|----------|------|------|------|
| `joint.pth` | 预训练 | NTU-RGB+D 120 上训练的 SlowOnly-R50 | PoseC3D 的 backbone 初始化权重 |
| `epoch_50.pth` | 训练产出 | 校园数据集 MIL 清洗后 50 epoch | 最终行为识别模型 (val 89.9%, test 93.3%) |
| `yolo11m-pose.pt` | 预训练 | Ultralytics 官方 COCO-Pose | 17 关键点姿态检测 |
| `best-falling.pt` | 训练产出 | 校园场景躺地数据 | YOLO11m 单类躺地检测 |
| `best-smoke.pt` | 训练产出 | 校园场景吸烟数据 | YOLO11m 单类吸烟检测 |
| `best-phone.pt` | 训练产出 | 校园场景手机数据 | YOLO11m 单类手机检测 |

---

## 十、端口一览

| 服务 | 端口 | 说明 |
|------|------|------|
| Python 推理服务 | 8000 | FastAPI + SSE |
| Spring Boot 后端 + 前端 | 8080 | 浏览器访问此端口 |
| MySQL | 3306 | 数据库 |
