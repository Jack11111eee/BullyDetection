# MIL 数据清洗 — 交接文档

## 项目背景

PoseC3D 校园安全行为识别（YOLO11m-Pose + PoseC3D），5 类：normal/fighting/bullying/falling/climbing。

- **当前最佳**: R9 模型，Val 82.4%, Test 84.2%
- **核心瓶颈**: normal ↔ fighting 互混（normal 16% 误判 fighting，fighting 23% 误判 normal）
- **根因**: fighting 视频中大量片段其实是 normal 行为（如 1 分钟视频里 30 秒是 normal），但所有 clip 都继承了视频级 "fighting" 标签 → 标签噪声

## 数据概况

campus.pkl 清洗后有效样本 40895 个：

| Class | Count |
|---|---|
| normal | 16396 |
| fighting | 13981 |
| bullying | 497 |
| falling | 9351 |
| climbing | 670 |

## MIL 清洗方案

用训练好的模型作为 teacher，对每个 clip 打分 P(true_label)，过滤低置信度噪声样本，清洗后重建数据集重新训练。

## 已完成的工作

`mil_cleaning/` 目录下 3 个脚本：

| 文件 | 功能 | 状态 |
|---|---|---|
| `score_samples.py` | 用模型对所有样本打分，保存完整概率向量 | 已写好，有 bug 已修（squeeze fix） |
| `analyze_noise.py` | 分析噪声分布，画直方图，输出统计报告 | 已写好，未运行 |
| `clean_and_rebuild.py` | 根据阈值过滤噪声 + 重建 kfold pkl | 已写好，未运行 |

## 发现的关键问题

**当前 `score_samples.py` 直接用 R9 模型对 campus.pkl 全量打分是错误的。**

原因：R9 模型在 fold 0 训练集上训练到 ~99% train accuracy，对训练集样本过度自信 — 即使是噪声样本 P(true) 也接近 1.0，无法区分噪声。运行时观察到 P(true) 普遍 > 0.9，验证了这个问题。

## 下一步：交叉折打分（Cross-Fold Scoring）

正确做法是让**每个样本只被没见过它的模型打分**：

```
Step 1: 训练 5 个 fold 的模型（用 train_kfold.sh）
  → posec3d_campus_v9_fold0/ (已有)
  → posec3d_campus_v9_fold1/ ~ fold4/ (需要训练)

Step 2: 每个 fold 的模型给对应 val 数据打分
  Fold 0 模型 → 打分 Fold 0 的 val 数据
  Fold 1 模型 → 打分 Fold 1 的 val 数据
  ...
  合并所有 val 分数 = 每个样本都有无偏的 P(true_label)

Step 3: 用 analyze_noise.py 分析噪声分布

Step 4: 用 clean_and_rebuild.py 过滤并重建数据集
```

需要修改 `score_samples.py` 为交叉折打分模式：
- 循环 5 个 fold，加载对应 fold 的模型
- 只对该 fold 的 val split 打分
- 合并所有 fold 的分数

## 关键路径

```
Config:     pyskl/configs/posec3d/finetune_campus_v4.py
R9 Fold 0:  pyskl/work_dirs/posec3d_campus_v9/epoch_50.pth
Kfold data:  data/campus/campus_kfold_0.pkl ~ campus_kfold_4.pkl
Test data:   data/campus/campus_test.pkl
原始数据:    data/campus/campus.pkl
训练脚本:    train_kfold.sh
```

## 训练命令参考

```bash
# 单 fold 训练
cd /home/hzcu/BullyDetection/pyskl && \
LD_LIBRARY_PATH=/home/hzcu/miniconda3/pkgs/cuda-cudart-11.8.89-0/lib:$LD_LIBRARY_PATH \
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 \
tools/train.py configs/posec3d/finetune_campus_v4.py --launcher pytorch

# 一键 5-fold
bash train_kfold.sh
```

注意：`train_kfold.sh` 和 v4 config 可能需要确认是否支持指定 fold（需要检查 config 里 ann_file 的 fold 切换方式）。
