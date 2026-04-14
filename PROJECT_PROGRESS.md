# 校园安防视频行为感知系统 — 项目进度文档

> 第十七届服务创新大赛 A28，西安电子科技大学
> 更新时间：2026-03-26

---

## 一、系统架构

```
视频流 → YOLO11m-Pose + ByteTrack → 骨骼缓冲区（48帧滑窗）
                                   ↓
                         PoseC3D（每16帧推理）→ 打架/霸凌/摔倒/攀爬/破坏
                         YOLO11s（小目标）    → 香烟/手机 + 规则引擎 → 吸烟/打电话
                         轨迹分析（纯规则）   → 徘徊/逗留
                         帧差法（无模型）     → 摄像头遮挡
                                   ↓
                         时序平滑（M/N帧投票）→ 告警
```

**目标延迟**：~25ms（YOLO11m-Pose ~6ms，ByteTrack ~1ms，PoseC3D ~15ms，YOLO11s ~5ms）

---

## 二、行为类别定义

### 当前版本（7类，第二轮训练）

| ID | 类别名 | 说明 |
|----|--------|------|
| 0 | normal | 正常行为 |
| 1 | fighting | 打架 |
| 2 | bullying | 霸凌（攻击者+受害者合并） |
| 3 | falling | 摔倒 |
| 4 | climbing | 攀爬围栏 |
| 5 | vandalism | 破坏公物 |
| 6 | self_harm | 自残（暂无训练数据） |

> ⚠️ 第一轮训练为8类（bullying_attack/bullying_victim分开），因骨骼序列完全相同导致 victim 准确率 0%，已合并。

---

## 三、服务器环境

- **平台**：AutoDL，RTX 3090
- **环境**：Python 3.10，PyTorch 2.1.0，CUDA 11.8
- **conda 环境名**：`dtc`
- **工作目录**：`/home/hzcu/BullyDetection/`
- **PYSKL 路径**：`/home/hzcu/BullyDetection/pyskl/`

### 关键环境变量（每次启动必须设置）

```bash
export LD_LIBRARY_PATH=/home/hzcu/miniconda3/pkgs/cuda-cudart-11.8.89-0/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
```

---

## 四、完整步骤流程

### Step 1：环境安装

```bash
pip install ultralytics
pip install openmim
mim install mmengine
pip install mmcv-full==1.7.0  # 必须用1.7.0，不能用mmcv 2.x
mim install mmdet mmpose
git clone https://github.com/kennymckormick/pyskl.git
cd pyskl && pip install -e .
pip install opencv-python numpy scipy tqdm tensorboard scikit-learn
```

### Step 2：下载预训练权重

- 模型：PYSKL Model Zoo → `slowonly_r50_ntu120_xsub/joint`
- 下载 `joint.pth`，放至：`/home/hzcu/BullyDetection/joint.pth`

### Step 3：骨骼提取（`yolo11-base.py`）

**脚本路径**：`/home/hzcu/BullyDetection/yolo11-base.py`

功能：用 YOLO11m-Pose 提取骨骼关键点，支持视频文件和帧文件夹。

输出 JSON 格式：
```json
{
  "label": "fighting",
  "img_shape": [H, W],
  "frames": {
    "0": {"1": {"kps": [[x,y]×17], "score": [×17]}}
  }
}
```

### Step 4：构建 PKL 数据集（`step4_build_pkl.py`）

**脚本路径**：`/home/hzcu/BullyDetection/step4_build_pkl.py`

关键参数：
```python
CLIP_LEN = 48    # 每个样本48帧
STRIDE = 16      # 滑窗步长
MAX_PERSON = 2   # 最多2人

LABEL_MAP = {
    'normal':          0,
    'fighting':        1,
    'bullying_attack': 2,
    'bullying_victim': 2,  # 合并为同一类
    'falling':         3,
    'climbing':        4,
    'vandalism':       5,
    'self_harm':       6,
}
```

低置信度关键点（< 0.3）用线性插值填补，不丢弃。

### Step 5：格式化 PKL（`reformat_pkl.py`）

**脚本路径**：`/home/hzcu/BullyDetection/reformat_pkl.py`

输出：`/home/hzcu/BullyDetection/data/campus/campus.pkl`

PYSKL 要求的格式（split 必须用 frame_dir 字符串，不能用整数索引）：
```python
data = {
    'split': {
        'train': [s['frame_dir'] for s in train_samples],
        'val':   [s['frame_dir'] for s in val_samples]
    },
    'annotations': all_samples
}
```

### Step 6：训练 PoseC3D

**配置文件**：`/home/hzcu/BullyDetection/pyskl/configs/posec3d/finetune_campus.py`

**训练命令**：
```bash
cd /home/hzcu/BullyDetection/pyskl && \
LD_LIBRARY_PATH=/home/hzcu/miniconda3/pkgs/cuda-cudart-11.8.89-0/lib:$LD_LIBRARY_PATH \
CUDA_VISIBLE_DEVICES=0 \
python -m torch.distributed.run --nproc_per_node=1 \
  tools/train.py configs/posec3d/finetune_campus.py \
  --launcher pytorch
```

**评估命令**：
```bash
cd /home/hzcu/BullyDetection/pyskl && \
LD_LIBRARY_PATH=/home/hzcu/miniconda3/pkgs/cuda-cudart-11.8.89-0/lib:$LD_LIBRARY_PATH \
CUDA_VISIBLE_DEVICES=0 \
python -m torch.distributed.run --nproc_per_node=1 \
  tools/test.py configs/posec3d/finetune_campus.py \
  -C work_dirs/posec3d_campus/epoch_30.pth \
  --launcher pytorch \
  --eval top_k_accuracy mean_class_accuracy
```

---

## 五、数据集来源

| 数据集 | 类别 | 路径 |
|--------|------|------|
| RLVS | fighting | `/home/hzcu/zjc/dataset/RLVS` |
| RWF-2000 | fighting | `/home/hzcu/zjc/dataset/RWF-2000/RWF-2000` |
| ShanghaiTech Campus | normal | `/home/hzcu/zjc/dataset/SHT` |
| UCF-Crime | fighting/assault | `/home/hzcu/zjc/dataset/UCF-crime` |
| UR Fall Detection | falling | `/home/hzcu/zjc/dataset/Fall UR Fall Detection` |
| Multiple Cameras Fall | falling | `/home/hzcu/zjc/dataset/Fall Multiple Cameras Fall Dataset/dataset_extracted/dataset` |
| fall_floor | falling | `/home/hzcu/zjc/dataset/fall_floor/fall_floor_extracted/fall_floor` |
| Vandalism | vandalism | `/home/hzcu/zjc/dataset/Vandalism/Vandalism_extracted/Vandalism` |
| punch | bullying | `/home/hzcu/zjc/dataset/punch/punch_extracted/punch` |
| climb | climbing | `/home/hzcu/zjc/dataset/climb/climb_extracted/climb` |

**骨骼 JSON 输出目录**：`/home/hzcu/BullyDetection/data/raw_skeletons/`

### 数据量（第二轮 PKL，7类合并后）

| 类别 | 验证集样本 |
|------|-----------|
| normal | 5739 |
| fighting | 3828 |
| bullying | 620 |
| falling | 2893 |
| climbing | 139 |
| vandalism | 1949 |
| self_harm | 0 |
| **总计** | **15168** |

训练集：57460，验证集：14365，总计：71825

---

## 六、当前进度

### ✅ 已完成

| 步骤 | 状态 | 说明 |
|------|------|------|
| Step 1 环境搭建 | ✅ | mmcv-full 1.7.0 + PYSKL |
| Step 2 预训练权重 | ✅ | joint.pth 已下载 |
| Step 3 骨骼提取 | ✅ | 所有数据集处理完成 |
| Step 4 PKL 构建 | ✅ | 7类合并，71825样本 |
| Step 5 格式化 PKL | ✅ | campus.pkl 已生成 |
| Step 6 第一轮训练（8类）| ✅ | 30 epoch，top1=41.5%（不合格） |
| Step 6 第二轮训练（7类）| 🔄 | **正在进行中** |

### 🔄 正在进行

- **第二轮 PoseC3D 训练（7类）**，当前在服务器后台运行
  - 改动：合并 bullying 两类 + 修正 class_weight（vandalism: 6.0→0.3）
  - 预计运行：30 epoch，约 6-8 小时
  - checkpoint 保存路径：`work_dirs/posec3d_campus/`

### ⏳ 待完成

| 步骤 | 说明 |
|------|------|
| Step 6 评估第二轮 | 训练结束后运行 test.py 查看 per-class 准确率 |
| Step 7 YOLO 小目标微调 | 香烟/手机检测，需从 Roboflow 下载数据 |
| Step 8 规则引擎 | 吸烟/打电话判断逻辑（guide 已有框架） |
| Step 9 主推理循环 | 集成所有模块，接入视频流 |
| 自录数据 | 摄像头遮挡、攀爬围栏、疑似自残（无公开数据集） |

---

## 七、历史问题与解决方案

### 1. mmcv 版本冲突
- **问题**：mmcv 2.x 没有 `Config` 类，PYSKL 依赖 1.x API
- **解决**：安装 `mmcv-full==1.7.0`（OpenMMLab 预编译 wheel，CUDA 11.8 + PyTorch 2.1）

### 2. libcudart.so.11.0 找不到
- **问题**：训练时报 CUDA 库缺失
- **解决**：每次启动前设置 `LD_LIBRARY_PATH=/home/hzcu/miniconda3/pkgs/cuda-cudart-11.8.89-0/lib`

### 3. torchrun 使用了错误的 Python
- **问题**：`torchrun` 调用系统 Python 3.13，不在 conda 环境里
- **解决**：改用 `python -m torch.distributed.run`

### 4. PYSKL split 格式错误（最关键的 Bug）
- **问题**：`reformat_pkl.py` 中 split 用整数索引 `list(range(N))`，导致 PYSKL 过滤后 0 个样本，训练无法进行
- **解决**：split 必须用 `frame_dir` 字符串列表
  ```python
  train_ids = [s['frame_dir'] for s in train_samples]  # ✅
  # 不能用 list(range(len(train_samples)))              # ❌
  ```

### 5. RepeatDataset times 参数报错
- **问题**：base config 使用 `RepeatDataset(times=10)`，直接 override `data.train` 会冲突
- **解决**：在 data 字典中加 `_delete_=True`

### 6. dropout_ratio 不支持
- **问题**：`I3DHead` 不接受 `dropout_ratio` 参数
- **解决**：改为 `dropout=0.5`

### 7. MMDistributedDataParallel 属性缺失（test.py）
- **问题**：mmcv 1.7.0 的 DDP 没有 `_use_replicated_tensor_module` 属性（PyTorch 2.x 新增）
- **解决**：patch mmcv 源文件
  ```bash
  sed -i 's/self\._use_replicated_tensor_module else self\.module/getattr(self, "_use_replicated_tensor_module", False) else self.module/' \
    /home/hzcu/miniconda3/envs/dtc/lib/python3.10/site-packages/mmcv/parallel/distributed.py
  ```

### 8. test.py checkpoint 参数格式
- **问题**：checkpoint 不能作为位置参数，必须用 `-C` 标志
- **解决**：`python tools/test.py config.py -C epoch_30.pth --launcher pytorch`

### 9. RWF-2000 文件名过长（Cyrillic 字符）
- **问题**：俄语文件名 UTF-8 编码后超过系统限制
- **解决**：`preprocess_datasets.py` 中加 `safe_stem()` 函数，超长时用 MD5 hash 代替

### 10. UCF-Crime 帧文件直接在类别目录下
- **问题**：UCF-Crime 帧混在 `Fighting/` 文件夹里，没有子目录（格式：`Fighting002_x264_1000.png`）
- **解决**：按 `_x264_` 前缀分组，创建临时 symlink 目录模拟子目录结构

### 11. 第一轮训练结果差（top1=41.5%）根因分析
- **vandalism 吞噬其他类**：vandalism class_weight=6.0 过高，预测 vandalism 的概率占 52%，precision 仅 23%
- **bullying_victim 准确率 0%**：attack 和 victim 的骨骼序列完全相同，模型无法区分
- **解决**：合并 bullying 两类 + 将 vandalism 权重从 6.0 降至 0.3，重新训练

---

## 八、训练配置文件关键内容

**路径**：`/home/hzcu/BullyDetection/pyskl/configs/posec3d/finetune_campus.py`

```python
_base_ = './slowonly_r50_ntu120_xsub/joint.py'

model = dict(
    cls_head=dict(
        type='I3DHead',
        in_channels=512,
        num_classes=7,          # 7类（第二轮）
        dropout=0.5,
        loss_cls=dict(
            type='CrossEntropyLoss',
            class_weight=[1.0, 3.0, 6.0, 2.0, 10.0, 0.3, 1.0]
            # [normal, fighting, bullying, falling, climbing, vandalism, self_harm]
        )
    )
)

load_from = '/home/hzcu/BullyDetection/joint.pth'
ann_file  = '/home/hzcu/BullyDetection/data/campus/campus.pkl'

data = dict(
    videos_per_gpu=8, workers_per_gpu=4,
    train=dict(_delete_=True, type='PoseDataset', ann_file=ann_file, split='train', pipeline=train_pipeline),
    val=dict(_delete_=True,   type='PoseDataset', ann_file=ann_file, split='val',   pipeline=val_pipeline),
    test=dict(_delete_=True,  type='PoseDataset', ann_file=ann_file, split='val',   pipeline=test_pipeline),
)

optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0003)
lr_config  = dict(policy='CosineAnnealing', by_epoch=False, min_lr=0)
total_epochs = 30
fp16 = dict(loss_scale='dynamic')
work_dir = '/home/hzcu/BullyDetection/pyskl/work_dirs/posec3d_campus'
evaluation = dict(interval=999, metrics=['top_k_accuracy', 'mean_class_accuracy'])
```

---

## 九、评估目标

| 指标 | 目标 | 第一轮结果 | 第二轮预期 |
|------|------|-----------|-----------|
| Overall top1 | ≥ 90% | 41.5% | 待评估 |
| False Positive Rate | ≤ 10% | 未达标 | 待评估 |
| False Negative Rate | ≤ 10% | 未达标 | 待评估 |

---

## 十、下一步行动

### 训练结束后（Step 6 完成）

1. 运行评估：
```bash
cd /home/hzcu/BullyDetection/pyskl && \
LD_LIBRARY_PATH=/home/hzcu/miniconda3/pkgs/cuda-cudart-11.8.89-0/lib:$LD_LIBRARY_PATH \
CUDA_VISIBLE_DEVICES=0 \
python -m torch.distributed.run --nproc_per_node=1 \
  tools/test.py configs/posec3d/finetune_campus.py \
  -C work_dirs/posec3d_campus/epoch_30.pth \
  --launcher pytorch \
  --eval top_k_accuracy mean_class_accuracy
```

2. 运行 per-class 分析脚本（见历史对话中的 Python 脚本）

### Step 7：YOLO 小目标微调（香烟/手机）

- 从 Roboflow Universe 下载：
  - `cigarette detection`（YOLO 格式）
  - `cell phone detection`（YOLO 格式）
- 使用 `imgsz=1280`（俯视角小目标必须）
- 基础模型：YOLO11s

### Step 8：规则引擎
- 吸烟：检测到香烟 + 人脸附近 → 触发
- 打电话：检测到手机 + 耳边姿势 → 触发

### Step 9：主推理循环集成
- YOLO11m-Pose + ByteTrack → 骨骼缓冲
- PoseC3D 推理（每16帧）
- YOLO11s 小目标检测
- 轨迹分析（纯规则）
- 帧差法摄像头遮挡检测
- M/N 帧时序投票平滑

---

*本文档由 Claude Code 自动生成，记录截至 2026-03-26 的项目状态。*
