# PoseC3D Campus Safety Training Log

> 校园安防视频行为识别项目的完整工程记录：训练历史、数据工程、部署调优、问题编年史。

---

## 目录

1. [项目信息](#1-项目信息)
2. [训练历史（Round 1–11）](#2-训练历史round-111)
3. [准确率趋势总表](#3-准确率趋势总表)
4. [MIL 数据清洗（突破 85% 的关键）](#4-mil-数据清洗突破-85-的关键)
5. [数据管线](#5-数据管线)
6. [关键配置总表](#6-关键配置总表)
7. [训练/评估命令](#7-训练评估命令)
8. [E2E 部署调优历史（Round 1–11）](#8-e2e-部署调优历史round-111)
9. [E2E 规则引擎当前完整逻辑](#9-e2e-规则引擎当前完整逻辑)
10. [问题编年史](#10-问题编年史)
11. [辅助组件](#11-辅助组件)
12. [经验教训](#12-经验教训)

---

## 1. 项目信息

| 项 | 内容 |
|---|---|
| **任务** | 校园安防视频行为识别（第17届服创赛 A28） |
| **主管线** | YOLO11m-Pose + ByteTrack → PoseC3D (PYSKL) |
| **辅助管线** | YOLO11 三类小物体模型（phone / smoking / falling） + 规则引擎 |
| **服务器** | AutoDL, RTX 3090 |
| **环境** | conda `dtc`, Python 3.10, PyTorch 2.1.0, CUDA 11.8, mmcv-full 1.7.0, PYSKL |
| **目标** | top1 ≥ 90% |
| **最终达成** | R11 val 89.9% / test **93.3%** |

---

## 2. 训练历史（Round 1–11）

### Round 1 — 8 Classes Baseline

| Item | Detail |
|---|---|
| Classes | 8 (normal, fighting, bullying, falling, climbing, vandalism, smoking, phone_call) |
| Result | top1 = **41.5%** |
| Problem | vandalism weight=6.0 主导所有预测 |
| Diagnosis | Class weight 过激 |

### Round 2 — 7 Classes

| Item | Detail |
|---|---|
| Classes | 7（移除一类）|
| Result | top1 = **31.8%** |
| Problem | fighting 主导所有预测 |
| Diagnosis | 减类数解决不了类别不平衡 |

### Round 3 — 6 Classes + Balanced Data

| Item | Detail |
|---|---|
| Classes | 6 |
| Changes | 引入 `fix_and_balance.py`：undersample CAP=6000, oversample MAX=3× |
| Result | top1 = **65.1%** |
| Problem | vandalism 过预测 |
| Diagnosis | vandalism 数据质量差（vandalism2 源 70.8% 零关键点） |

### Round 4 — with_limb=True（灾难）

| Item | Detail |
|---|---|
| Classes | 6 |
| Changes | 启用 `with_limb=True` |
| Result | top1 = **33.1%** |
| Diagnosis | 数据有 bug（零关键点 + 脏数据），limb 特征放大了噪声 |

### Round 5 — Fixed keypoint_score

| Item | Detail |
|---|---|
| Classes | 6 |
| Changes | 修复 keypoint_score 格式（PYSKL 要求 keypoint 和 keypoint_score 分离为 `(M,T,17,2)` 和 `(M,T,17)`）|
| Result | top1 = **56.2%** |
| Problem | **vandalism "垃圾桶类"**：预测 5794 次，真实 1448 次 |
| Diagnosis | vandalism 约 70% 零关键点样本 → 模型把不确定输入丢进 vandalism |

### Round 6 — Removed vandalism (5 Classes)

| Item | Detail |
|---|---|
| Config | `finetune_campus_v3.py`, num_classes=5, dropout=0.5 |
| Classes | 5 (normal, fighting, bullying, falling, climbing) |
| class_weight | `[1.0, 1.0, 1.5, 1.0, 2.5]` |
| Data | `campus_balanced_v5.pkl` |
| Epochs | 50 |
| Result | top1 = **53.8%** |
| Problem | **bullying "垃圾桶类"**：预测 4528 次，真实 288 次，吸掉 40% normal / 36% falling / 18% fighting |
| Diagnosis | 36% 全样本零关键点 → bullying（零率 44%）成为新垃圾桶 |

### Round 7 — Data Quality Fix (v7 Dataset)

| Item | Detail |
|---|---|
| Config | `finetune_campus_v3.py`, num_classes=5 |
| class_weight | `[1.0, 1.0, 1.5, 1.0, 2.5]` |
| Data | `campus_balanced_v7.pkl`（过滤零关键点 + frame_dir 去重）|
| Epochs | 50 |
| Train top1 | ~99% |
| Val top1 | **17.8%** |
| Problem | **极度过拟合**（81 点差距）；**climbing "垃圾桶类"**：2998 normal / 1627 fighting / 1718 falling 全被预测为 climbing |

**Per-class (Round 7 Val):**

| Class | Correct | Total | Accuracy |
|---|---|---|---|
| normal | 0 | 3241 | 0.0% |
| fighting | 1313 | 2949 | 44.5% |
| bullying | 0 | 114 | 0.0% |
| falling | 3 | 1730 | 0.2% |
| climbing | 140 | 154 | 90.9% |

**Root Cause**：清洗掉 36% 零样本后，climbing 只损失 2.7%（最干净）。配合 `class_weight=2.5` + 3× oversampling 双重放大 → 50 epochs 过拟合。

### Round 8 — 5-Fold CV + Remove class_weight + 30 Epochs

| Item | Detail |
|---|---|
| Config | `finetune_campus_v3.py`（移除 class_weight，30 epochs）|
| class_weight | **移除** |
| Data | 5-Fold CV (`campus_kfold_0~4.pkl`) + 独立 test (`campus_test.pkl`) |
| 划分 | 10% test 视频级 hold-out，其余 90% 视频级 5-fold，每 fold 独立做 undersample+oversample |
| Epochs | **30** |

**Fold 0:**

| Metric | Val (7334) | Test (3606) |
|---|---|---|
| Overall top1 | **81.3%** | **83.5%** |
| Mean class acc | 76.3% | 76.5% |

| Class | Val Acc | Test Acc |
|---|---|---|
| normal | 81.1% | 81.8% |
| fighting | 74.7% | 79.0% |
| bullying | 56.4% | 51.0% |
| falling | 94.3% | 96.3% |
| climbing | 75.0% | 74.2% |

**Fold 1 Val top1:** 80.5%（与 Fold 0 差距 <1%，模型稳定）

**成功原因（17.8% → 81.3%）：**
1. 移除 class_weight（最大功臣）— 消除 climbing 垃圾桶
2. 去掉全零关键点数据（第二功臣）
3. 30 epochs 代替 50（辅助）

**当前瓶颈：**
- normal ↔ fighting 互混（两个 fold 一致的最大错误来源）
- bullying 样本极少（val ~100，test ~50）

### Round 9 — clip_len=64 + 更强正则化

| Item | Detail |
|---|---|
| Config | `finetune_campus_v4.py` |
| Changes vs R8 | clip_len 48→64, weight_decay 0.0003→0.001, epochs 30→50, eval interval 5→3 |
| Data | `campus_kfold_0.pkl`（同 fold 0）|
| work_dir | `posec3d_campus_v9` |

**前置实验**：R8 模型上 10-clip 推理取平均 → +0.0%（模型预测已稳定，瓶颈在特征区分）；故 R9 去掉 val num_clips=10。

**label_smooth_val 移除**：PYSKL CrossEntropyLoss 不支持该参数（是 mmaction2 新功能）。

| Metric | Val (7334) | Test (3606) |
|---|---|---|
| Overall top1 | **82.4%** | **84.2%** |
| Mean class acc | 78.9% | 76.3% |

**vs R8：** Val +1.1%, Test +0.7%。提升有限，normal↔fighting 互混仍是瓶颈。

### Round 10 — Limb-only (with_kp=False, with_limb=True)

| Item | Detail |
|---|---|
| Config | `finetune_campus_v5.py` |
| Changes vs R9 | with_kp=False, with_limb=True, resolution 56→64, videos_per_gpu 16→12 |
| work_dir | `posec3d_campus_v10` |

| Metric | Val | Test |
|---|---|---|
| Overall top1 | **80.7%** | **82.5%** |

**关键发现 — kp 与 limb 犯不同的错：**
- kp 擅长 normal / falling
- limb 擅长 fighting（fighting→normal 误判从 553 降到 519）
- **错误互补 → ensemble 是推向 90% 的潜在路线**

### Round 11 — MIL 清洗后训练（达成 90% 目标）

| Item | Detail |
|---|---|
| Config | `finetune_campus_mil.py` |
| Changes vs R9 | 仅改 ann_file 指向 MIL 清洗后数据，其余超参完全一致 |
| Data | `campus_mil_kfold_0.pkl`（清洗后，threshold=0.3，去掉 fighting+normal 噪声）|
| Epochs | 50 |
| work_dir | `posec3d_campus_mil` |

**在清洗后数据上评估：**

| Metric | Val (6401) | Test (3266) |
|---|---|---|
| Overall top1 | **90.5%** | **90.3%** |

**在原始未清洗数据上评估（公平对比）：**

| Metric | Val (7334) | Test (3606) |
|---|---|---|
| Overall top1 | **89.9%** | **93.3%** |
| Mean class acc | 92.7% | 91.8% |

**vs R9（Test）:**

| Class | R9 Test | R11 Test | Delta |
|---|---|---|---|
| normal | 82.2% | 93.3% | **+11.1%** |
| fighting | 80.8% | 89.7% | **+8.9%** |
| bullying | 45.1% | **78.4%** | **+33.3%** |
| falling | 96.1% | 99.1% | +3.0% |
| climbing | 77.4% | **98.4%** | **+21.0%** |

**成功原因：**
1. 清洗掉 4,755 个噪声样本（11.6%），消除了 fighting 视频中的 normal 片段标签噪声
2. normal↔fighting 互混大幅下降
3. bullying 起死回生（45% → 78%），之前被噪声样本误导
4. **Test 93.3% 已超过 90% 目标线**

---

## 3. 准确率趋势总表

| Round | Val top1 | Test top1 | 关键变化 |
|---|---|---|---|
| 1 | 41.5% | — | Baseline 8 classes |
| 2 | 31.8% | — | 7 classes |
| 3 | 65.1% | — | 6 classes + balanced |
| 4 | 33.1% | — | with_limb=True（数据 bug）|
| 5 | 56.2% | — | Fixed keypoint_score |
| 6 | 53.8% | — | 5 classes（移除 vandalism）|
| 7 | 17.8% | — | class_weight+oversample 碰撞 |
| 8 | **81.3%** | **83.5%** | **K-fold CV + 移除 class_weight + 30 epochs** |
| 9 | **82.4%** | **84.2%** | clip_len 48→64 + weight_decay 0.001 |
| 10 | 80.7% | 82.5% | with_limb=True（数据干净后可用，略低于 kp）|
| 11 | **89.9%** | **93.3%** | **MIL 交叉折清洗（threshold=0.3, -4755 噪声）** |

---

## 4. MIL 数据清洗（突破 85% 的关键）

### 背景

R9 卡在 84.2% test，主要瓶颈是 normal ↔ fighting 互混。根本原因：fighting 视频中大量片段其实是 normal 行为（1 分钟里 30 秒正常），但所有 clip 都继承了视频级 "fighting" 标签。

### 方法：交叉折打分（Cross-Fold Scoring）

用 5-fold 的模型对数据打分 — 每个样本只被**没见过它**的模型评分：

```
Fold i 模型（训练在 kfold_i train 上）→ 打分 kfold_i 的 val
合并 → 37,289 个样本都有无偏 P(true_label)
```

**重要教训**：最初尝试直接用 R9 模型对全量 `campus.pkl` 打分，但模型在训练集上达到 ~99% accuracy，对训练样本过度自信（P(true)≈1.0），无法区分噪声。改用交叉折后问题解决。

### 噪声分析结果

5-fold 全量覆盖（37,289 样本），各 fold 分布高度一致：

| 类别 | 总数 | P<0.3（明确噪声）| P<0.5（可疑）| 噪声程度 |
|---|---|---|---|---|
| fighting | 12,772 | 2,559 (20.0%) | 3,301 (25.8%) | 高 |
| normal | 14,917 | 2,196 (14.7%) | 3,045 (20.4%) | 中高 |
| bullying | 446 | 240 (53.8%) | 257 (57.6%) | 极高 |
| climbing | 608 | 142 (23.4%) | 179 (29.4%) | 中高 |
| falling | 8,546 | 347 (4.1%) | 478 (5.6%) | 低（最干净）|

P(true_label) 呈**双峰分布**，threshold=0.3 正好卡在谷底。

### 清洗策略

使用 `--threshold 0.3 --classes fighting normal`：

```
原始有效样本:  40,895
移除噪声:       4,755 (11.6%)
  - normal:     2,196
  - fighting:   2,559
清洗后:         36,140
```

### MIL 工具链

```
mil_cleaning/
├── score_samples.py       ← 交叉折打分
├── analyze_noise.py       ← 噪声分析 + 直方图
├── clean_and_rebuild.py   ← 阈值过滤 + 重建 kfold pkl
├── export_noisy_samples.py← 导出噪声样本清单
├── resolve_video_paths.py ← 查找源视频
├── collect_videos.py      ← 复制到审核文件夹
├── scores.pkl
├── noise_distribution.png
├── noise_report.txt
└── HANDOFF.md
```

---

## 5. 数据管线

```
step4_build_pkl.py    → train.pkl + val.pkl（原始骨骼提取）
        ↓
reformat_pkl.py       → campus.pkl（合并 split dict）
        ↓
build_kfold_data.py   → campus_kfold_0~4.pkl + campus_test.pkl   ← R8+
                         - 过滤 label >= 5
                         - 过滤全零关键点
                         - Dedup frame_dir
                         - 10% 视频级 held-out test
                         - 90% → 5-fold 视频级 CV
                         - 每 fold 独立 undersample(CAP=6000) + oversample(MAX=3×)
        ↓
mil_cleaning/*        → campus_mil_kfold_0~4.pkl + campus_mil_test.pkl  ← R11
                         - 5-fold 交叉折打分
                         - threshold=0.3 过滤 fighting/normal 噪声
```

旧流程（R7 及之前）：`fix_and_balance.py → campus_balanced_v7.pkl`（80/20 split，无 test set）。

---

## 6. 关键配置总表

### 6.1 训练配置（R6–R11）

| Parameter | R6 (v3) | R7 (v3) | R8 (fold) | R9 (v4) | R10 (v5) | R11 (mil) |
|---|---|---|---|---|---|---|
| num_classes | 5 | 5 | 5 | 5 | 5 | 5 |
| dropout | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| class_weight | [1,1,1.5,1,2.5] | [1,1,1.5,1,2.5] | **移除** | 移除 | 移除 | 移除 |
| clip_len | 48 | 48 | 48 | **64** | 64 | 64 |
| total_epochs | 50 | 50 | **30** | **50** | 50 | 50 |
| lr | 0.005 | 0.005 | 0.005 | 0.005 | 0.005 | 0.005 |
| weight_decay | 0.0003 | 0.0003 | 0.0003 | **0.001** | 0.001 | 0.001 |
| with_kp | True | True | True | True | **False** | True |
| with_limb | False | False | False | False | **True** | False |
| train resolution | 56 | 56 | 56 | 56 | **64** | 56 |
| videos_per_gpu | 16 | 16 | 16 | 16 | **12** | 16 |
| ann_file | v5.pkl | v7.pkl | kfold_0 | kfold_0 | kfold_0 | **mil_kfold_0** |

### 6.2 E2E Pipeline 参数（当前版本，Round 8 后）

| 参数 | 值 | 说明 |
|---|---|---|
| clip_len | 64 | PoseC3D 输入帧数 |
| stride | 16 | 推理步长 |
| buf_max | 256 (4×clip_len) | SkeletonBuffer 最大历史帧数 |
| min_infer_frames | 32 (clip_len//2) | 最少帧数即可推理 |
| pose_threshold | 0.3 | PoseC3D 最低置信度（fighting/bullying 也用此阈值）|
| attack_prob_threshold | 0.3 | **新增 R8**：fighting 或 bullying 概率触发攻击判定的阈值（不依赖 argmax）|
| vote_window | 5 | 时序投票窗口 |
| VOTE_ENTRY_MIN.fighting | 3 | 进入 fighting 所需窗口计数 |
| VOTE_ENTRY_MIN.bullying | 3 | 进入 bullying 所需窗口计数 |
| VOTE_ENTRY_MIN.falling | 2 | 进入 falling 所需窗口计数 |
| VOTE_ENTRY_MIN.climbing | 2 | 进入 climbing 所需窗口计数 |
| VOTE_HOLD_MIN.falling / climbing | 1 | R8 / R9 保持 |
| VOTE_HOLD_MIN.fighting / bullying | **2** | **R9**：1→2 防永锁（原 R8.1 设为 1 后进入态几乎退不出）|
| hysteresis upgrade | 允许 | **R8**：HOLD 时另一异常票数严格更多即切换 |
| attack_prob 触发 | ≥0.3 **且** ≥ normal_prob × 0.7 | **R9**：加相对优势判定，normal 主导时攻击门槛抬高 |
| asymmetry 逻辑 | `ratio<0.5 AND head_hip>0.15` | **R9**：OR→AND + 阈值收紧（原 R3: `<0.6 OR >0.1`）|
| YOLO falling deferred | 启用 | **R9 P7**：被 PoseC3D 弱攻击信号抢先时暂存，step 6 未判攻击则兜底返回 falling |
| proximity_factor | 1.5×max(身高) | fighting/bullying 近距离约束 |
| upright_threshold | 3% 画面高度 | falling 姿态验证阈值 |
| smooth_alpha | 0.5 | EMA 关键点平滑系数 |
| bbox overlap threshold | 10% | 替代距离判定（3D 纵深）+ pair coupling 门槛（R8.5）|
| pair coupling | 启用 | **R8.5**：bbox overlap ≥ 10% 的 track 对强制共享攻击态 |
| cross-label injection | 扩展到全攻击路径 | **R8.5**：rule_bullying/fighting 触发时向 bbox overlap 邻居 raw history 注入同标签 |
| reassoc max_dist_ratio | 15% 画面高 | 新 track 匹配旧 track 阈值 |
| vertical movement ratio | 5% 画面高 | climbing 必需垂直位移 |
| grace_frames | 90（~3s @30fps）| 遮挡宽限期 |
| loiter_time | 300s | 徘徊阈值 |
| small_obj_model | 3 路单类 | **R11 (E2E)**：v8 拆分为 3 个独立 YOLO11m 模型 — laying / smoking / phone。SingleClassDetector 统一语义映射 |
| YOLO gating | 帧级按需触发 | **R11 (E2E)**：falling 始终跑（安全兜底）；smoking/phone 在任一 track ∈ {fighting,bullying,falling,climbing} 时跳过（物理互斥）|

---

## 7. 训练/评估命令

### 训练

```bash
cd /home/hzcu/BullyDetection/pyskl && \
LD_LIBRARY_PATH=/home/hzcu/miniconda3/pkgs/cuda-cudart-11.8.89-0/lib:$LD_LIBRARY_PATH \
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 \
tools/train.py configs/posec3d/finetune_campus_mil.py --launcher pytorch
```

### 评估

```bash
cd /home/hzcu/BullyDetection && python per_class_eval_v2.py
```

### E2E 推理

```bash
cd /home/hzcu/mnt/autodl/e2e_pipeline && python run.py \
  --source <video_path> \
  --output <output_path>
```

---

## 8. E2E 部署调优历史（Round 1–11）

### 背景

R11 模型 test 93.3%，但在真实视频流上跑 E2E pipeline 表现差。**核心不是模型问题，是后处理管线多层过度过滤叠加**。

**Debug 统计（最初，542 次推理）：**

| 过滤层 | 吃掉数 | 机制 |
|---|---|---|
| threshold < 0.5 | 67 | fighting=0.498 等被强制归 normal |
| vote: current=normal 直接返回 | 417 | 一次 normal 立刻覆盖所有异常历史 |
| vote: 票数不够 | 21 | 首次 fighting 只有 1/3 票 |
| buffer 不足 SKIP | 3353 | 86% 帧无法推理（track 碎片化）|

### E2E Fix Round 1 — 后处理基础修复

| 修改 | 文件 | 详情 |
|---|---|---|
| 重写 `_vote_smooth` | rule_engine.py | 改为异常偏向：窗口内有 1 次异常就维持告警，全部 normal 才清除 |
| 降低 `pose_threshold` | rule_engine.py + pipeline.py | 0.5 → 0.3 |
| 允许半满 buffer 推理 | pipeline.py | `should_infer` 最低帧数 64 → 32 |
| CLI 默认参数 | run.py | vote-window 5→3, vote-ratio 0.6→0.34 |

**效果：**

| 指标 | 修复前 | 修复后 | 变化 |
|---|---|---|---|
| SKIP（buffer 不足）| 3353 | 622 | -81% |
| 被 threshold 过滤 | 67 | 17 | -75% |
| 被 vote 压成 normal | 21 | **0** | 消除 |
| FINAL fighting | 30 | 77 | +157% |
| FINAL falling | 37 | 78 | +111% |

**残余问题**：75% 推理仍输出 normal — 根因是训练/推理时序采样不一致。

### E2E Fix Round 2 — 均匀采样修复

| 修改 | 详情 |
|---|---|
| 扩大缓冲区 | `buf_max = clip_len * 4 = 256`（~8.5s）|
| 均匀采样 | `get_clip()` 用 `np.linspace` 从全部缓存帧均匀采 64 帧，模拟训练时 `UniformSampleFrames` |
| threshold 硬编码修复 | pipeline.py 硬编码 `pose_threshold=0.5` 覆盖了默认值，修为 0.3 |

### E2E Fix Round 3 — bullying/fighting 区分 + 误报控制

**问题**：坐着被误判 fighting；fighting 和 bullying 难以区分。

| 修改 | 详情 |
|---|---|
| `_vote_smooth` 分级响应 | falling/climbing 1 次即告警，fighting/bullying 需 2 次确认 |
| bullying 不对称检测 | `check_bullying_asymmetry()`：身高比<0.6 或 头-髋差>10%画面高 → fighting 改判 bullying |
| 传递完整骨骼数据 | 收集所有人的 (kps, scores) 传给规则引擎 |

### E2E Fix Round 4 — 正常视频误报修复

**问题**：靠墙站立误判 fighting → bullying；坐下过程误判 falling。

| 修改 | 详情 |
|---|---|
| fighting 近距离约束 | 必须有另一人在附近才算 fighting |
| fighting 置信度门槛 | fighting 单独要求 conf ≥ 0.5 |
| fighting vote 要求 | 窗口内需 3 次 fighting（最严格）|
| falling 姿态验证 | `_is_upright_posture()`：头-髋差 > 8% 画面高 → 躯干直立 → 非 falling |
| 高置信度也验证 | falling conf>0.7 也需过姿态验证 |

### E2E Fix Round 5 — Proximity 放宽 + Bbox 修复

**问题**：bully 视频中倒地者 proximity 太严杀掉 98 个 fighting；踢人者只显示骨架无框。

| 修改 | 详情 |
|---|---|
| Proximity 用两人最大身高 | `ref_height = max(my_height, neighbor_height) × 1.5` |
| Bbox 变量 bug | 每个 track 存储自己的 bbox，不再引用循环残留变量 |
| 始终显示 bbox + 标签 | 未推理时默认显示 normal + 绿色框 |

### E2E Fix Round 6 — YOLO 三类 + YOLO 辅助 falling + 遮挡宽限

**模型路径**：`/home/hzcu/yjm/home/yjm/VideoDetection/v6/runs/detect/campus_A28/unified_3class_model/weights/best.pt`

| 修改 | 详情 |
|---|---|
| SmallObjectDetector 自动读取类名 | class_map 不再硬编码，从 `model.names` 自动读取 |
| check_smoking 兼容新类名 | 匹配 'cigarette' 或 'smoking' |
| YOLO 辅助 falling 检测 | `check_fallen_by_yolo()`：YOLO detection bbox 与人物骨骼中心重叠（20% margin）|
| YOLO falling → bullying 升级 | 躺地 + 附近 track 历史含 fighting/bullying → bullying；否则 → falling |
| 遮挡宽限期（grace_frames=90）| 三组件统一：SkeletonBuffer / RuleEngine / Pipeline.track_labels，track 消失后保留状态 90 帧 |
| 默认加载小物体模型 | `--small-obj-model` 默认指向 unified_3class_model |

### E2E Fix Round 7 — Track 重关联 + 姿态/位移约束 + 投票升级 + Bbox 重叠

**问题**：T3 被遮挡后 ByteTrack 给同一个人新 ID（T4），grace 保留的 falling 白白浪费；坐着的人被识别为 falling/climbing；攻击历史一次即触发 bullying；3D 纵深场景距离判断失效。

| 修改 | 文件 | 详情 |
|---|---|---|
| Track 重关联 | pipeline.py + rule_engine.py | 新 track 出现时，按位置匹配宽限期内消失的旧 track（threshold=15% 画面高），迁移 SkeletonBuffer 帧、投票历史、显示标签 |
| Loitering 5 分钟 + 降优先级 | rule_engine.py | loiter_time 60→300s；优先级降到 PoseC3D 之后（避免 bullying 被误覆盖）|
| 坐姿检测（多次迭代）| rule_engine.py | 最终方案：`_is_sitting_posture` 用骨骼包围框纵横比 h/w > 1.0 → 非倒地；`_is_upright_posture` 阈值 8%→3% 画面高 |
| YOLO falling 路径加坐姿验证 | rule_engine.py | 之前坐姿检测只在 PoseC3D 路径，但 falling 实际从 YOLO 辅助路径走 |
| 信任 YOLO falling | rule_engine.py | 测试俯视角度躺地视频后，移除 YOLO 路径的姿态检查（像素坐标检查受相机角度影响）|
| Climbing 垂直位移约束 | rule_engine.py | `_has_vertical_movement`：近 30 位置 Y 范围 < 5% 画面高 → 无垂直运动 → 非 climbing |
| EMA 关键点平滑 | pipeline.py | SkeletonBuffer.update() 对 17 个关键点做 α=0.5 EMA |
| Bullying 攻击历史阈值 | rule_engine.py | 从「任意 1 次攻击」改为「历史≥2 且攻击占比≥50%」，避免路人误触发 |
| Vote 窗口 3→5 | rule_engine.py + run.py | falling/climbing vote_min 1→2, bullying 2→3, fighting 保持 3 |
| Bbox 重叠替代距离 | rule_engine.py + pipeline.py | YOLO bullying 判定改用 `_bbox_overlap_ratio`（重叠面积/较小 bbox 面积 > 10%）|

**提交链路：** `c9dec2b` 重关联 → `1a93399` loitering 阈值 → `2afec40~09395f0` 坐姿检测迭代 → `1d188fb` 垂直位移 → `e5edb69~add2edc` YOLO 纵横比 → `b081d01` EMA → `e4539df` 攻击历史持续性 → `4b196ae` vote 窗口 → `00d2e02` bbox 重叠。

---

### E2E Fix Round 8 — 滞回投票 + 攻击概率主导（本章核心：R7 部署后残留问题的系统性修复）

**背景**：R7 之后观察到两个严重 bug：
1. 明显的 fight/bully 行为无法持续识别，只有一段后变 normal
2. 倒地同时被 bully 的人持续被识别为 falling/normal 而非 bullying

根本原因链：VOTE 阈值过严 → 窗口内 fighting 计数跌到 2 立刻归 normal；施暴者 raw history 大量 normal → 受害者 50% 占比条件打不穿；falling 锁死在 HOLD 阻止 bullying 接管；YOLO falling 抢先返回 falling 跳过不对称检测；PoseC3D argmax=normal 但 fighting_prob>0.3 被丢弃。

本 Round 分四次提交修复：

#### R8.1 — 滞回投票 + 放宽受害者 + 双向传播（commit `9d8ced2`）

**文件**：`e2e_pipeline/rule_engine.py` (+79/-23)

| 修改 | 机制 |
|---|---|
| A. 滞回投票（hysteresis）| 新增 `VOTE_ENTRY_MIN`（入=3/2）和 `VOTE_HOLD_MIN`（维持=1）分离；新增 `_last_smoothed[tid]` 跟踪上次输出；异常态时只要 ≥HOLD_MIN 就继续 |
| B. 放宽受害者判定 | YOLO 躺地 + bbox overlap≥10% 的邻居判定：旧「攻击历史占比≥50%」→ 新「邻居 smoothed ∈ {fighting,bullying} **或** 最近 3 条 raw history 有攻击」|
| C. 双向传播 | 受害者被判 bullying 时，新增 `_inject_raw_history()` 向邻居 raw history 追加一条 `bullying` 弱证据，防重复（末尾相同则跳过）+ 截断到 vote_window |
| migrate_track / clear_stale_tracks 同步 | 迁移/清理 `_last_smoothed` |

**设计意图**：A 让告警不再因 PoseC3D 闪烁而断裂；B 解决攻击者 raw history 噪声打不穿 50% 阈值的问题；C 让受害者↔施暴者相互强化，形成稳定攻击状态。

#### R8.2 — 滞回升级（commit `1fa3a21`）

**背景**：R8.1 后日志显示受害者 RAW 判定正确输出 bullying，但 FINAL 全被 smooth 压回 falling：
```
[VOTE] T7 current=bullying → falling (HOLD 1>=1) | history=[falling,bullying,bullying,bullying,bullying]
→ 4 bullying vs 1 falling，但 falling HOLD_MIN=1 就锁死
```

**修改**：`_vote_smooth` 的 HOLD 分支内加入升级检查（+9 行）：
```python
if last_count >= hold_min:
    top_label, top_count = anomaly_counts.most_common(1)[0]
    if top_label != last and top_count > last_count:
        return top_label  # UPGRADE
    return last           # HOLD
```

**效果**：
```
history=[falling,bullying,bullying,bullying,bullying]
→ bullying:4 > falling:1 → UPGRADE 到 bullying ✓
```
票数相同或更少时仍 HOLD，避免振荡。

#### R8.3 — PoseC3D 优先于 YOLO falling（commit `ebbf93c`）

**背景**：日志 F330–F760，T4 受害者被持续殴打，PoseC3D 正确输出 fighting=0.5~0.8，但 step 2 的 YOLO falling 抢先返回 falling，跳过了 step 6 的不对称检测：
```
F346: PoseC3D fighting=0.794 → [RAW] falling (YOLO辅助检测)
F362: PoseC3D fighting=0.641 → [RAW] falling (YOLO辅助检测)
```

**修改**：`_raw_judge` 在 YOLO falling 默认返回之前，检查 PoseC3D 是否有攻击信号（+16/-4）：
```python
if pose_label == 'fighting' and pose_conf >= 0.5:
    # 跳过 YOLO falling，交给 step 6 处理
elif pose_label == 'bullying' and pose_conf >= pose_threshold:
    # 跳过 YOLO falling
else:
    return 'falling', ..., 'rule_yolo_falling'
```

**设计原则**：YOLO falling 的本意是补偿「一动不动 + PoseC3D 输出 normal」的盲区。当 PoseC3D 已检出攻击信号时，没有盲区要补偿——攻击信号优先。

#### R8.4 — 攻击概率主导（commit `18fad7c`）

**背景**：日志 F431–F767，T1 一直殴打躺地的 T4，但 PoseC3D 概率分布是：
```
F447: normal=0.526 fighting=0.370   ← fighting 触发 0.3 但 argmax=normal
F463: normal=0.479 fighting=0.403
F591: normal=0.667 fighting=0.314
F751: normal=0.508 fighting=0.480
```
每帧 `fighting ≥ 0.3`，但 argmax=normal → 输出 normal → 不进入不对称检测。

**修改**：重构 step 6（+70/-38），改为**攻击概率主导**，不依赖 argmax：

```python
attack_prob = max(fighting_prob, bullying_prob)
if attack_prob >= 0.3:
    if proximity_ok:                             # 必要保护
        if asymmetry_ok → bullying               # 不对称=霸凌
        elif bullying_prob >= fighting_prob → bullying
        else → fighting                          # 对称攻击
    else: 落到 argmax 路径（fighting/bullying argmax 降 normal）
else: 走 argmax 路径（normal/falling/climbing 的姿态验证）
```

同步调整 step 2 (YOLO falling) 的跳过条件：`argmax==fighting && conf>=0.5` → `fighting_prob>=0.3 or bullying_prob>=0.3`。

**保留的保护**：
- `proximity_ok`（孤立个体不算攻击）
- `check_bullying_asymmetry`（fighting → bullying 升级）
- vote ENTRY=3 在 5 窗口（噪声过滤）

**删除**：`fighting conf<0.5 → normal`（被 0.3 概率阈值替代）

#### R8.5 — Pair Coupling：bbox 重叠 track 强制共享攻击态（commit `20185f3`）

**背景**：R8.4 后日志发现仍有两个结构性 bug：

**Bug 1 (F575–F618)**：T1 持续殴打躺地的 T4，bbox 明显重叠，但 T4 持续被判 bullying，T1 持续被判 normal。

```
F554: T4 RAW=bullying(对称)  history=[b,b,b,b,b]           → bullying ✓
F575: T1 RAW=normal(0.854)   history=[n,n,n,n,n]           → normal  ✗ T1 掉出
F591: T1 RAW=bullying(对称)  history=[n,n,n,n,b] ENTRY 1<3 → normal  ✗
F607: T1 RAW=bullying(对称)  history=[n,n,n,b,b] ENTRY 2<3 → normal  ✗
F618: T4 RAW=normal(0.821)   history=[b,n,n,n,n] HOLD 1>=1 → bullying ✓
```

**Bug 2 (F618–F778)**：受害者 T4 也掉出 bullying 被锁进 falling，直到 F778 才恢复。

```
F634: T4 RAW=normal(0.816)   history=[n,n,n,n,n]           → normal
F682: T4 RAW=falling(YOLO)   history=[n,n,n,fa,fa] ENTRY 2 → falling ✗
       _last_smoothed[T4] = falling
F762: T4 RAW=bullying        HOLD falling 1>=1             → falling ✗
F778: T4 RAW=bullying        UPGRADE bullying:2>falling:1   → bullying ✓
```

**根因**：每个 track 独立判定。bbox 重叠说明 T1 和 T4 是**同一场交互事件**，但判定机制不耦合 → T4 早期入 bullying 的 HOLD 保持，T1 后入场却因 fighting_prob 在 0.15~0.38 波动，永远凑不齐 5/5 窗口中的 3 次。受害者 PoseC3D 看倒地不动输出 normal 冲刷窗口 + YOLO falling 又锁进 falling，双方同时失效。

**修复 A：扩展双向传播到 asymmetry / 对称攻击路径**

R8.1 只在 `rule_yolo_bullying` 路径（YOLO 躺地检测）注入邻居 raw history。R8.5 扩展到 step 5 的 `rule_bullying`（asymmetry）和对称攻击路径：

```python
# 新增 _inject_to_overlapping_neighbor(track_id, label, track_bboxes_dict)
# 找 bbox overlap 最高的邻居（≥10%）→ _inject_raw_history(neighbor, label)
```

每次 rule_bullying/fighting 判定，向配对邻居 raw history 同步注入同一标签 → 加速邻居达成 ENTRY=3。

**修复 B：Post-smoothing pair coupling**

新增 `RuleEngine.couple_overlapping_pairs(judgments, track_bboxes_dict)`，在所有 track smooth 判定完成后，扫描 track 对强制共享攻击态：

```
A ∈ {fighting, bullying} + B normal + bbox overlap ≥ 10%
  → B 升级为 A 的标签，source = 'pair_couple(A_tid)'
  → _last_smoothed[B] = A 的标签（让后续 HOLD 维持）

A = bullying + B falling + overlap ≥ 10%
  → B 升级为 bullying（受害者在霸凌场景中应判 bullying，不是单纯 falling）

A = fighting + B falling + overlap ≥ 10%
  → B 保持 falling（独立倒地事件）
```

**pipeline.py 结构调整**：`_process_frame` 从 `loop(infer+visualize)` 重构为 `loop(infer) → couple → loop(visualize)`，确保耦合作用在所有 track 上（不只本帧推理的）。事件日志也用耦合后的标签。

**保留的保护**：
- ENTRY=3/2 严格入口（防止路人被错误拉入攻击态）
- bbox overlap ≥ 10% 是耦合的必要门槛（只有真的互动才耦合）
- falling + fighting 不耦合（独立事件）

**预期对照 Bug 1/2**：

| 帧 | T1 | T4 | 预期效果 |
|---|---|---|---|
| F586 | normal | bullying (HOLD) | 耦合：T1 ← bullying from T4 |
| F591 | RAW=bullying | bullying | T1 ENTRY 被 A 注入加速 |
| F634 | (holds via couple) | normal → 被 T1 coupling 拉回 bullying | T4 不再掉入 normal |
| F682 | bullying | falling (YOLO) | A=bullying + B=falling → B 升级 bullying |

---

### E2E Fix Round 9 — 规则引擎判定逻辑收紧（反制 R8 过矫正的系统性修复）

**背景**：R8.1–R8.5 五次连续修复都在「把系统往倾向报告攻击的方向推」—— 降阈值、pair coupling、跨 track 注入、攻击概率主导、HOLD=1。叠加后进入攻击态极易、退出极难。R9 做反向收紧，目标"实事求是"，不倾向任何一边。深度审视判定逻辑找到 7 处结构性问题（P1–P7），本轮先实施 ROI 最高的 4 处：P1/P2/P5/P7（纯阈值+少量兜底逻辑，不动框架）。

#### P1 — attack_prob 改为相对优势判定（绝对阈值 → 绝对+相对）

**问题**：`attack_prob >= 0.3` 是绝对阈值，忽略 normal 的压制。PoseC3D 5 类 softmax + R11 仍残留 ~10% normal↔fighting 互混 → `[normal=0.60, fighting=0.32, ...]` 会触发攻击判定，但 normal 是 fighting 两倍。等价于"只要 fighting 不被完全压制就考虑"。

**修改**（rule_engine.py 步骤 6）：
```python
normal_prob = float(pose_probs[0])
relative_ok = attack_prob >= normal_prob * 0.7
if attack_prob >= self.pose_threshold and relative_ok:
    # 进入攻击判定
```

**阈值含义**：normal=0.50 要求 attack≥0.35；normal=0.40 要求 attack≥0.30（与原绝对阈值一致）；normal=0.60 要求 attack≥0.42。normal 越主导，攻击门槛越高。

#### P2 — HOLD_MIN 攻击类 1→2（防永锁）

**问题**：`HOLD_MIN = 1` 意味 5 窗口内只要 1 帧异常就维持。配合 P1 的原低门槛 + pair coupling + raw injection 三重放大 → 进入极易、退出几乎不可能。

**修改**（rule_engine.py `VOTE_HOLD_MIN`）：
```python
VOTE_HOLD_MIN = {
    'falling': 1, 'climbing': 1,
    'bullying': 2, 'fighting': 2,  # R9: 1→2
}
```

falling/climbing 保持 1（姿态延续性高），攻击类要求 5 窗口 2/5 维持。原本 R8 担心的"断裂"来自 ENTRY=3 太严，不是 HOLD=1 太松。

#### P5 — check_bullying_asymmetry 阈值收紧 + OR→AND

**问题**：
```python
is_asymmetric = height_ratio < 0.6 or head_hip_normalized > 0.1
```
身高比<0.6（真实身高差+PoseC3D 漏检下半身频繁触发）、头-髋差>10% 画面高（正常弯腰、蹲下、拾物即满足）。**OR** 让任一条件即触发 + 单帧判定 → 姿态瞬间抖动即误判 bullying。

**修改**（rule_engine.py line 398）：
```python
is_asymmetric = height_ratio < 0.5 and head_hip_normalized > 0.15
```

两条件必须同时满足；阈值分别收紧到 50% 身高比 + 15% 画面高。

#### P7 — YOLO falling 暂存 + 兜底消费（修复结构性 bug）

**问题**：R8.3 的 step 2 跳过逻辑：
```python
if fighting_prob_early >= 0.3 or bullying_prob_early >= 0.3:
    # 跳过 YOLO falling，交给 step 6
```
但 step 6 若 proximity 失败（独自躺地）→ 走 argmax 路径 → argmax=normal → 输出 normal。**YOLO 检测到的 falling bbox 被完全吞掉**。一个独自昏倒的人只要 PoseC3D 噪声输出 fighting=0.3 就不会被标 falling。

**修改**（rule_engine.py）：
1. step 2 跳过时把 YOLO falling 结果存进局部变量 `yolo_falling_deferred`（而非直接丢弃）
2. step 6 所有分支走完后、step 6d argmax 前，若 `yolo_falling_deferred` 仍存在 → return falling

```python
# Step 2
yolo_falling_deferred = None
if is_fallen_yolo:
    if ... # 优先 bullying 检查
    if PoseC3D 有攻击信号:
        yolo_falling_deferred = (conf, horizontal)  # 暂存
    else:
        return 'falling', ...  # 直接返回

# Step 6 攻击判定走完

# P7 兜底：step 6 未判攻击 → 消费暂存
if yolo_falling_deferred is not None:
    return 'falling', conf, 'rule_yolo_falling'
```

#### 未实施的 P3/P4/P6（ROI 较低或改动量大）

- **P3 Pair coupling 退出机制缺失** —— 当前耦合后 `_last_smoothed[B]=攻击态` 无退出条件，需 coupling TTL + bbox 解耦判定（约 15 行）
- **P4 Pair coupling 无次级阈值** —— B 需要自己的 attack_prob ≥ 0.15 才允许被 A 拉升，需存 last_pose_probs（约 20 行）
- **P6 Raw injection 无弱证据标记** —— 注入应 0.5 票计数而非 full-weight（约 30 行）

P1+P2+P5+P7 改动总量 ~50 行，主体是阈值+兜底路径。P3/P4/P6 视 R9 效果再决定。

#### R9 配置快照

| 参数 | R8.5 | R9 |
|---|---|---|
| attack_prob 触发 | ≥ 0.3 | ≥ 0.3 **且** ≥ normal_prob × 0.7 |
| asymmetry 逻辑 | `ratio<0.6 OR head_hip>0.1` | `ratio<0.5 AND head_hip>0.15` |
| VOTE_HOLD_MIN (attack) | 1 | 2 |
| YOLO falling 被跳过 | 直接丢弃 | 暂存兜底 |

**预期行为变化**：
- 走廊并行两人（height_ratio 0.85, head_hip 5%画面）→ 不再误判 bullying（阈值拦截）
- normal=0.55 fighting=0.35 场景 → 不再进入攻击判定（normal 压制）
- 攻击态一旦确认，5 窗口里 2 帧异常即维持（更稳）但 4 帧 normal 将退出（不永锁）
- 独自昏倒 + PoseC3D fighting=0.3 噪声 → 仍标 falling（YOLO 兜底）

---

### E2E Fix Round 10 — fighting/bullying 判定脱敏（消除 YOLO 误检自激）

**背景**：R9 上线后观察到持续对称 fighting 视频里标签在 fighting ↔ bullying 间反复翻转。debug.log 分析：601 行日志中误判 bullying 集中在 F126–F175 / F415–F450 / **F546–F627（82 帧连续自激）**；其中 F546–F627 两个 track 的 PoseC3D 输出稳定在 `fighting=0.998–0.999, bullying=0.000`，FINAL 却全部是 bullying。

**根因（三条独立病灶）**：

1. **step 2 `rule_yolo_bullying` 完全绕过 PoseC3D argmax** —— YOLO unified_3class falling 模型对 fighting 中倾斜/交缠姿态有误检，即使 PoseC3D 99% 确定 fighting 也被升级为 bullying
2. **cross-inject 无差别污染** —— 一方误判 bullying 即向邻居 history 注入 bullying，邻居稳定 fighting 时也被污染到 UPGRADE，形成自激
3. **step 6c `b ≥ f` 门槛过松** —— R11 训练数据 bullying 样本 446 vs fighting 12772（28:1 不平衡），softmax 抖动 `b=0.49 f=0.51` 就能翻转

本 Round 三改动（~40 行）：

#### P8 — step 2 邻居收紧 + PoseC3D 否决门（commit `8af7a53`）

**文件**：`e2e_pipeline/rule_engine.py:526-572`

| 修改 | 旧 | 新 |
|---|---|---|
| 邻居触发条件 | `smoothed ∈ {fighting,bullying}` 或最近 3 帧含攻击 | 仅 `smoothed=bullying` 或最近 3 帧含 bullying |
| PoseC3D 否决门 | 无 | `fighting_prob≥0.7 AND bullying_prob<fighting_prob*0.3` 时跳过整个 step 2（bullying+falling 都跳） |

**语义**：邻居是 fighting 说明两人对称对打，不是 bullying 证据；PoseC3D 极度确定 fighting 时无"一动不动躺地"盲区需要 YOLO 补偿，YOLO 信号视为误检。

#### P9 — cross-inject 条件化（commit `8af7a53`）

**文件**：`e2e_pipeline/rule_engine.py:865-892`

**修改**：`_inject_raw_history` 接收 `label='bullying'` 前检查邻居最近 3 帧，若 ≥2 帧 fighting 则拒绝注入。

**语义**：邻居正稳定在 fighting（对称对打）时，一方因 YOLO 误检短暂判 bullying 不应污染对方 ENTRY 计数。

#### P10 — step 6c 相对门槛收紧（commit `8af7a53`）

**文件**：`e2e_pipeline/rule_engine.py:664-680`

| 修改 | 旧 | 新 |
|---|---|---|
| bullying 判定门槛 | `bullying_prob >= fighting_prob` | `bullying_prob >= fighting_prob * 1.5 AND bullying_prob >= 0.4` |

**语义**：要求 bullying 显著压倒 fighting（1.5 倍）且绝对置信度足够（≥0.4）。R11 的 bullying 训练样本太少，概率边界不稳，必须要求显著优势才采信，不然单帧抖动就会翻转。

#### R10 配置快照

| 参数 | R9 | R10 |
|---|---|---|
| rule_yolo_bullying 邻居条件 | `{fighting,bullying}` 或 history 含攻击 | `bullying` only |
| PoseC3D 否决 YOLO | 无 | fighting≥0.7 且 bullying<f*0.3 跳过 step 2 |
| cross-inject bullying 前置检查 | 无 | 邻居最近 3 帧 fighting<2 |
| step 6c bullying 阈值 | `b ≥ f` | `b ≥ f*1.5 且 b ≥ 0.4` |

**预期行为变化**：
- F546–F627 自激完全消除（PoseC3D fighting=0.998 触发 P8 否决门）
- 单帧 PoseC3D 输出 b=0.35 f=0.33 不再翻转 bullying（P10 要求显著优势）
- 即使 step 2 误判 bullying，P9 阻断传染到 fighting 邻居
- 真实 bullying 场景（受害者倒地 + 攻击者单向）仍可识别：邻居 smoothed=bullying 持续时 step 2 正常触发；PoseC3D b=0.6 f=0.3 也满足 P10 门槛

---

### E2E Fix Round 11 — 小物体检测三路拆分 + 帧级 gating（commit `074c47f`）

**背景**：队友重训了小物体检测模型，从一个 unified 3-class 模型拆成 3 个独立单类 YOLO11m（性能更好但每类名字可能不一致）。原 `unified_3class_model/best.pt` 替换为：
- `v8/falling/runs/laying_yolo11m_v1/weights/best.pt`（注意：模型文件夹名 `laying` 而非 `falling`，原始 `model.names` 可能输出 `laying` → 直接换权重会导致 rule_engine 字符串匹配失效）
- `v8/smoking/runs/smoking_yolo11m_v1/weights/best.pt`
- `v8/phone/runs/phone_yolo11m_v1/weights/best.pt`

#### 修改 A — SingleClassDetector 语义适配层（pipeline.py）

新增 `SingleClassDetector` 包装单类模型：任何输出 box 强制打固定 `target_class` 标签（`falling`/`smoking`/`phone`），完全忽略模型原始 `names`。rule_engine 字符串匹配逻辑保持不变。

新增 `MultiSmallObjectDetector` 管理 3 路检测器：`detect(frame, need_falling, need_smoking, need_phone)` 按需跳过。

#### 修改 B — 帧级 gating（pipeline.py `_process_frame`）

- **帧级缓存**：同一帧多 track 推理时只调一次 YOLO（原实现 N 个 track 触发 N 次 detect，浪费）
- **按类延迟触发（物理互斥 gating）**：
  - `falling`：始终运行（PoseC3D 对一动不动躺地的盲区必须补偿）
  - `smoking` / `phone`：任一 track 上一帧标签 ∈ {fighting, bullying, falling, climbing} 时跳过（这些状态下人不可能同时吸烟/打电话）

Gating 基于**上一帧**的 track 标签（因果性，不等当前帧判定完）。

#### 修改 C — run.py CLI

新增 `--falling-model` / `--smoking-model` / `--phone-model`，默认指向 v8 三路新路径。保留 `--small-obj-model`（legacy unified 模型）用于回退；若指定则覆盖 3 路配置。`none` 显式禁用单个检测器。

```bash
# 默认使用 3 路单类模型
python e2e_pipeline/run.py --source demo.mp4 --posec3d-config ... --posec3d-ckpt ...

# 回退到旧 unified 模型
python e2e_pipeline/run.py --small-obj-model /old/unified_3class/best.pt ...

# 禁用 smoking 检测
python e2e_pipeline/run.py --smoking-model none ...
```

#### R11 (E2E) 配置快照

| 参数 | R10 | R11 |
|---|---|---|
| 小物体模型 | 1 个 unified 3-class | 3 个 single-class + SingleClassDetector 语义映射 |
| 类名语义 | 依赖 `model.names` 字典 | 强制覆盖为 rule_engine 预期名（falling/smoking/phone） |
| 检测调用频率 | 每 track 推理时都调 | 每帧缓存一次，多 track 共享 |
| gating | 无 | falling 始终跑；smoking/phone 在攻击/倒地/攀爬态跳过 |

**预期效果**：
- 检测器升级（v6 → v8）精度提升（需视频验证）
- 跳过开销：典型 fighting 段每帧少跑 2 次 YOLO；多 track 场景每帧少跑 (N-1) 次

**未处理 / 已知风险**：
- 若 3 个模型的 `conf/imgsz` 需要不同（如 laying 要更高 imgsz 捕大 bbox），目前共用 `yolo_conf=0.3, imgsz=1280`
- 上一帧 gating 有 1 帧延迟：开始吸烟的第一帧若另一 track 仍在 fighting 态，smoking 会被跳过（下一帧 fighting 结束自然恢复）

---

## 9. E2E 规则引擎当前完整逻辑

```
输入：PoseC3D 5 类概率 + YOLO 小物体 detections + 所有人骨骼 + track bboxes
 │
 ├─ Step 1: 高置信度紧急行为
 │   climbing > 0.7 + 有垂直位移 → climbing
 │   falling > 0.7 + 非直立 + 非坐姿 → falling
 │
 ├─ Step 2: YOLO 辅助 falling 检测
 │   YOLO 检测到 falling bbox + 骨骼中心重叠（20% margin）
 │     ├─ R10 P8 否决门：fighting_prob ≥ 0.7 AND bullying_prob < f*0.3
 │     │   → 整个 step 2 跳过（YOLO 视为误检）
 │     ├─ bbox overlap ≥ 10% 的邻居 smoothed=bullying 或最近 3 帧含 bullying
 │     │   → bullying + 向邻居注入 'bullying'（R10 P8 邻居收紧）
 │     ├─ PoseC3D fighting_prob ≥ 0.3 或 bullying_prob ≥ 0.3
 │     │   → 暂存为 yolo_falling_deferred，交给 step 5/6（R9 P7）
 │     └─ 否则 → falling（信任 YOLO 检测器）
 │
 ├─ Step 3: 小物体规则
 │   检测到香烟 + 在嘴/手附近 → smoking
 │   检测到手机 + 在耳朵附近 → phone_call
 │
 ├─ Step 4: vandalism 规则
 │   fighting_prob > 0.5 + 场景仅 1 人 → vandalism
 │
 ├─ Step 5: 攻击概率主导（R8.4 新增，R9/R10 收紧）
 │   attack_prob = max(fighting_prob, bullying_prob) ≥ 0.3
 │   且 attack_prob ≥ normal_prob × 0.7（R9 P1 相对优势）时：
 │     proximity_ok（邻居在 1.5×max 身高范围内）
 │       ├─ 不对称（ratio<0.5 AND head_hip>0.15, R9 P5 收紧）→ bullying + 注入
 │       ├─ bullying_prob ≥ fighting_prob*1.5 AND bullying_prob≥0.4 (R10 P10)
 │       │    → bullying + 注入
 │       └─ else → fighting + 注入
 │     proximity 失败 → 落到 step 6
 │
 ├─ R9 P7 兜底：step 5 未判攻击且 yolo_falling_deferred 存在 → falling
 │
 ├─ Step 6: argmax 路径
 │   argmax ∈ {fighting, bullying} 但 proximity 失败 → normal
 │   argmax = climbing 但无垂直位移 → normal
 │   argmax = falling 但直立/坐姿 → normal
 │   else → argmax 标签
 │
 └─ Step 7: 徘徊检测（最低优先级）
     同一 track 在 100px 半径内停留 ≥ 300s → loitering

输出：raw_label
 │
 ▼
_vote_smooth（窗口 = 5，滞回）：
  ENTRY（首次进入）：fighting/bullying 需 3 次，falling/climbing 需 2 次
  HOLD（维持）：falling/climbing 需 1 次；fighting/bullying 需 2 次（R9 P2 防永锁）
  UPGRADE（升级）：HOLD 时另一异常票数严格更多即切换
 │
 ▼
couple_overlapping_pairs（R8.5 新增 post-processing）：
  对所有活跃 track，bbox overlap ≥ 10% 的 pair 强制共享攻击态
    - A ∈ {fighting,bullying} + B normal → B 升级为 A 的标签
    - A bullying + B falling → B 升级 bullying（受害者在霸凌场景）
    - A fighting + B falling → B 保持 falling（独立倒地）
  同步 _last_smoothed[B] 让后续 HOLD 维持

最终 FINAL label → 可视化 + event_log
```

### 遮挡宽限期机制

```
track 被遮挡（从画面消失）
  → 宽限期 90 帧内：SkeletonBuffer 保留帧，RuleEngine 保留投票历史，显示旧标签
  → ByteTrack 恢复同一 track_id → 立刻继承旧状态，buffer 继续累积
  → 16 帧后触发新推理，平滑过渡
  → 超过 90 帧仍未出现 → 清除所有状态
  
track 被分配新 ID（重关联）
  → 按位置匹配（15% 画面高阈值）
  → migrate SkeletonBuffer 帧 + RuleEngine history + track_labels + _last_smoothed
```

---

## 10. 问题编年史

### 训练期问题

#### Problem 1: All-Zero Keypoint Samples (36%)

- **发现**：Round 6 post-mortem
- **详情**：36.3% train / 36.2% val 全零关键点（YOLO 未检测到人）
- **来源分布**：

| Source | Total | Zero% |
|---|---|---|
| vandalism2 | 7,160 | **70.8%** |
| ucf | 2,673 | **60.6%** |
| shanghaitech | 14,803 | **53.1%** |
| chute | 10,054 | **41.7%** |
| rlvs | 9,998 | 13.3% |
| rwf | 9,818 | 7.4% |

- **修复**：`fix_and_balance.py` 加入 `np.all(keypoint == 0)` 过滤 → v7 数据集
- **状态**：R7 修复

#### Problem 2: "Garbage Bin Class" Pattern

- **模式**：每 round 一个类吸收所有预测
- **历史**：vandalism (R3, R5) → bullying (R6) → climbing (R7)
- **根因（R3–R6）**：零关键点样本
- **根因（R7）**：class_weight + oversampling 双重放大
- **修复**：移除 class_weight，让 oversampling 单独处理不平衡
- **状态**：R8 修复

#### Problem 3: Label Conflicts (1,033 samples)

- **详情**：同 frame_dir 不同 label（主要 RWF 的 fighting vs normal）
- **修复**：`fix_and_balance.py` 做 frame_dir 去重（保留首次）
- **状态**：v7 修复

#### Problem 4: Duplicate Split IDs

- **详情**：train 883 个、val 445 个重复 ID
- **修复**：数据处理时去重
- **状态**：v7 修复

#### Problem 5: with_limb Catastrophe (R4)

- **详情**：with_limb=True 从 65% 跌到 33%
- **修复**：先还原，干净数据后 R10 再试（80.7%，可用但略弱）
- **状态**：修复

#### Problem 6: keypoint_score Format

- **详情**：PYSKL 要求 `keypoint (M,T,17,2)` 和 `keypoint_score (M,T,17)` 分离
- **修复**：step4_build_pkl.py
- **状态**：R5 修复

#### Problem 7: Overfitting (Train 99% vs Val 17.8%)

- **发现**：R7
- **根因**：50 epochs on cleaner smaller dataset + class_weight 放大
- **修复**：30 epochs + 移除 class_weight
- **状态**：R8 修复

#### Problem 8: normal ↔ fighting 互混

- **发现**：R8 (Fold 0/1 一致)
- **详情**：normal 16% 误判为 fighting，fighting 23% 误判为 normal
- **根因**：fighting 视频中包含大量 normal 片段但继承视频级标签
- **修复**：MIL 交叉折清洗
- **状态**：R11 修复（test 93.3%）

### 部署期问题

#### Problem 9: 训练/推理管线不一致（单人 vs 双人）

- **发现**：R11 部署后
- **详情**：训练时 2 人骨骼 `(M=2, T, 17, 2)`，但 `SkeletonBuffer.get_clip()` 只返回 1 人 → fighting 全部漏检
- **修复**：`SkeletonBuffer.get_clip(track_id, secondary_tid)` 输出 `(2, T, 17, 2)`；`_process_frame` 两遍处理（先收集位置，后配对推理）
- **状态**：已修复

#### Problem 10: 模型 93% 但视频流效果极差

- **根因**：后处理管线多层过度过滤
- **修复**：E2E Fix Round 1（见 §8）
- **状态**：已修复

#### Problem 11: PoseC3D 对静态场景盲区

- **发现**：E2E Fix Round 6
- **详情**：一动不动躺地者（失去意识）无时序变化，PoseC3D 输出 normal
- **修复**：YOLO unified_3class 辅助 falling 检测（bbox 匹配骨骼中心）
- **状态**：已修复

#### Problem 12: YOLO falling 误判 bullying

- **发现**：E2E Fix Round 6 首版
- **详情**：「躺地+附近有人→bullying」路人经过也触发
- **修复**：检查附近 track 投票历史含 fighting/bullying
- **状态**：已修复（后续 R7/R8 持续细化）

#### Problem 13: 遮挡恢复后标签空窗期

- **发现**：E2E Fix Round 6
- **详情**：track 消失下一帧删除所有状态 → 恢复后标签变 normal
- **修复**：三组件统一 grace_frames=90 宽限期
- **状态**：已修复

#### Problem 14: ByteTrack 重分配 ID 导致状态丢失

- **发现**：E2E Fix Round 7
- **详情**：T3 遮挡 → ByteTrack 恢复给新 ID T4 → grace 保留的旧状态永不继承
- **修复**：三组件协同重关联（位置匹配 + 迁移 buffer/history/labels）
- **状态**：已修复

#### Problem 15: 像素坐标姿态检查受摄像头角度影响

- **发现**：E2E Fix Round 7 多次迭代
- **详情**：head-hip Y 差、躯干角度、骨骼纵横比在不同摄像头角度下失效
- **修复**：分层处理 — PoseC3D 路径保留姿态检查（排除坐姿误判 falling/climbing），YOLO falling 路径直接信任检测器
- **状态**：已修复

#### Problem 16: 距离判断在 3D 纵深场景失效

- **发现**：E2E Fix Round 7
- **详情**：攻击者站在躺地者上方时，2D 像素中心距离很大但 bbox 必然重叠
- **修复**：`_bbox_overlap_ratio`（intersection/min(area)）替代距离
- **状态**：已修复

#### Problem 17: 告警间歇性中断

- **发现**：E2E Fix Round 8
- **详情**：vote 窗口内 fighting 计数跌到 2（<ENTRY=3）立刻归 normal，然后要重新凑 3 次才能再次触发 → 大量中间帧变 normal
- **证据**：debug.log 大量 `[VOTE] bullying只有1次<ENTRY3 → normal`
- **修复**：滞回投票 — `VOTE_HOLD_MIN=1` 维持异常态（commit `9d8ced2`）
- **状态**：已修复

#### Problem 18: 滞回锁死阻止异常升级

- **发现**：E2E Fix Round 8.1 后
- **详情**：受害者 RAW 正确输出 bullying，但 HOLD 机制锁在 falling（只要窗口还有 1 次 falling 就不切换，即使 bullying 票数更多）
- **证据**：`history=[falling,bullying,bullying,bullying,bullying] → HOLD falling`
- **修复**：HOLD 分支加入 UPGRADE 检查（票数严格更多允许切换）（commit `1fa3a21`）
- **状态**：已修复

#### Problem 19: YOLO falling 抢先拦截 PoseC3D 攻击信号

- **发现**：E2E Fix Round 8.2 后
- **详情**：受害者持续被殴打，PoseC3D 明确输出 fighting=0.5~0.8，但 step 2 YOLO falling 返回 falling 跳过了 step 6 的不对称检测
- **证据**：`F346: PoseC3D fighting=0.794 → [RAW] falling (YOLO辅助检测)`
- **修复**：step 2 默认返回前检查 PoseC3D 是否有攻击信号，有则跳过交给 step 6（commit `ebbf93c`）
- **状态**：已修复

#### Problem 20: argmax=normal 但 fighting_prob>0.3 被丢弃

- **发现**：E2E Fix Round 8.3 后
- **详情**：施暴者站在躺地者上方时，PoseC3D 概率分散（normal=0.5, fighting=0.4），argmax=normal → 输出 normal → 不进入不对称检测。300+ 连续帧误判
- **证据**：
  ```
  F447: normal=0.526 fighting=0.370 → normal
  F463: normal=0.479 fighting=0.403 → normal
  F751: normal=0.508 fighting=0.480 → normal
  ```
- **修复**：step 6 重构为**攻击概率主导** — `max(fighting_prob, bullying_prob) ≥ 0.3` 触发攻击判定，不依赖 argmax；保留 proximity + asymmetry 强约束（commit `18fad7c`）
- **状态**：已修复

#### Problem 21: bbox 重叠却独立判定导致状态不一致

- **发现**：E2E Fix Round 8.4 后
- **详情**：T1 持续殴打躺地的 T4，bbox 明显重叠（>10%），但 T4 靠 HOLD 保持 bullying 状态，T1 因 fighting_prob 在 0.15~0.38 波动无法凑齐 ENTRY=3，持续 normal。bbox 重叠意味着两人处于同一交互事件，一个 bullying 一个 normal 是结构性不合理
- **证据**：
  ```
  F591: T1 RAW=bullying history=[n,n,n,n,b] ENTRY 1<3 → normal
  F607: T1 RAW=bullying history=[n,n,n,b,b] ENTRY 2<3 → normal
  F618: T4 history=[b,n,n,n,n] HOLD 1>=1 → bullying
  ```
- **根因**：每个 track 独立判定，状态机无跨 track 耦合
- **修复**：
  - 修复 A：扩展双向传播到 asymmetry / 对称攻击路径 — 每次 rule_bullying/fighting 触发时向 bbox overlap 最高邻居注入同标签
  - 修复 B：`RuleEngine.couple_overlapping_pairs()` post-smoothing 耦合 — bbox overlap ≥10% 的 pair 强制共享攻击态，同步 `_last_smoothed`（commit `20185f3`）
- **状态**：已修复

#### Problem 22: 受害者被 YOLO falling 锁死，无法恢复 bullying

- **发现**：E2E Fix Round 8.4 后
- **详情**：倒地受害者 PoseC3D 看到静止躯体普遍输出 normal，几帧后窗口被 normal 填满 → bullying HOLD 失效。此时 YOLO falling 检测躺地，raw=falling，两次确认后 `_last_smoothed[T]=falling`。之后即使受害者 RAW 恢复 bullying，HOLD(falling)=1 优先于 UPGRADE，bullying 只有 1 票时被锁在 falling
- **证据**：
  ```
  F634: T4 history=[n,n,n,n,n] → normal (T4 掉出 bullying)
  F682: T4 RAW=falling ENTRY 2>=2 → falling (_last_smoothed=falling)
  F762: T4 RAW=bullying HOLD falling 1>=1 → falling  (UPGRADE 需 strict more)
  ```
- **修复**：pair coupling 中 `A=bullying + B=falling + overlap≥10% → B=bullying`。受害者在霸凌场景中应判 bullying 而非单纯 falling（commit `20185f3`）
- **状态**：已修复

#### Problem 23: attack_prob 绝对阈值忽略 normal 压制

- **发现**：E2E Fix Round 9 深度审视
- **详情**：R8.4 step 6 的 `attack_prob >= 0.3` 是绝对阈值。PoseC3D 分布 `[normal=0.60, fighting=0.32, ...]` 仍会触发攻击判定，但 normal 是 fighting 两倍——语义上 normal 主导，不该进入攻击路径
- **修复**：P1 — 增加相对优势判定 `attack_prob >= normal_prob * 0.7`（commit `e9cbc11`）
- **状态**：已修复

#### Problem 24: 攻击态 HOLD_MIN=1 永锁

- **发现**：E2E Fix Round 9 深度审视
- **详情**：R8.1 为防告警闪断把 `VOTE_HOLD_MIN=1`，配合 pair coupling 注入 + 低 attack 阈值 + R8.5 cross-label injection 三重放大，5 窗口内只要 1 帧异常就维持攻击态 → 进入极易、退出几乎不可能
- **修复**：P2 — 攻击类（fighting/bullying）HOLD_MIN 1→2，falling/climbing 保持 1（姿态延续性高）（commit `e9cbc11`）
- **状态**：已修复

#### Problem 25: asymmetry OR 逻辑 + 宽阈值误触发 bullying

- **发现**：E2E Fix Round 9 深度审视
- **详情**：`check_bullying_asymmetry` 用 `height_ratio<0.6 OR head_hip>0.1`。单帧判定 + OR → 正常弯腰、蹲下、拾物、真实身高差、PoseC3D 下半身漏检均可触发
- **修复**：P5 — OR→AND + 阈值收紧到 `<0.5 AND >0.15`（commit `e9cbc11`）
- **状态**：已修复

#### Problem 26: YOLO falling 信号被弱攻击信号吞掉

- **发现**：E2E Fix Round 9 深度审视
- **详情**：R8.3 的 step 2 跳过逻辑——PoseC3D `fighting_prob/bullying_prob >= 0.3` 时跳过 YOLO falling 交给 step 6。但 step 6 若 proximity 失败（独自躺地）→ 走 argmax 路径 → argmax=normal → 输出 normal。**YOLO 检测到的 falling bbox 被完全吞掉**
- **修复**：P7 — step 2 跳过时把 YOLO falling 结果存进 `yolo_falling_deferred`，step 6 所有分支走完后若未判攻击则兜底返回 falling（commit `e9cbc11`）
- **状态**：已修复

#### Problem 27: 对称 fighting 被 YOLO 误检触发 bullying 自激

- **发现**：E2E Fix Round 10（R9 上线后真实视频流测试）
- **详情**：两人持续对称 fighting 视频里标签反复翻转 fighting↔bullying。debug.log F546–F627 连续 82 帧两个 track PoseC3D 均输出 `fighting=0.998~0.999, bullying=0.000`，但 FINAL 全部是 bullying。根因是 YOLO unified_3class falling 对 fighting 中倾斜/交缠姿态误检 → step 2 `rule_yolo_bullying` 完全绕过 PoseC3D argmax → 一方误判 bullying 后 cross-inject 污染邻居 history → 邻居也被 UPGRADE 到 bullying → 形成 82 帧自激
- **证据**：
  ```
  F546: T1 PoseC3D fighting=0.998, bullying=0.000
        → [RAW] bullying (YOLO躺地 + T3 smoothed=fighting, overlap=0.35)
        → [INJECT] T3 += bullying
  F557: T3 PoseC3D fighting=0.999
        → [RAW] bullying (T1 现在 smoothed=bullying)
  ... 持续到 F627
  ```
- **根因（三条独立病灶）**：
  - step 2 不看 PoseC3D argmax，YOLO 躺地+邻居攻击即判 bullying
  - `_inject_raw_history` 无差别注入，污染稳定 fighting 邻居
  - step 6c `b >= f` 对不平衡训练数据下的抖动过敏
- **修复**：R10 三改动 P8+P9+P10（commit `8af7a53`）
  - P8：step 2 邻居条件收紧（仅 bullying 算证据）+ `fighting_prob≥0.7 AND bullying<f*0.3` 否决整个 step 2
  - P9：`_inject_raw_history` 拒绝向最近 3 帧 ≥2 帧 fighting 的邻居注入 bullying
  - P10：step 6c 门槛 `b>=f` → `b>=f*1.5 且 b>=0.4`
- **状态**：已修复（待视频验证）

---

## 11. 辅助组件

| 组件 | 文件 | 状态 |
|---|---|---|
| Rule Engine | `e2e_pipeline/rule_engine.py` | 已集成 |
| Inference Pipeline | `e2e_pipeline/pipeline.py` | 已集成 |
| YOLO11 Unified 3-Class | `unified_3class_model/best.pt` | 队友训练（phone/smoking/falling）|
| Training Curves | `plot_training_curves.py` | 已建，解析 mmcv 日志 |
| Data Diagnostics | `diagnose_data.py`, `diagnose_all_data.py` | 已建 |
| Sample Visualization | `visualize_samples.py` | 已建 |
| K-Fold Data Builder | `build_kfold_data.py` | 已建 |
| K-Fold Eval | `eval_kfold.py` | 已建 |
| Multi-clip Eval | `eval_multiclip.py` | 已建（已证明 +0.0% 无效）|
| All-epoch Eval | `eval_all_epochs.py` | 已建 |
| K-Fold Training | `train_kfold.sh` | 已建 |
| Round Eval | `eval_round9.py` | 已建 |
| MIL Cleaning | `mil_cleaning/*` | 已建 |

---

## 12. 经验教训

### 数据与训练

1. **Data quality > everything**：36% 垃圾数据让 6 rounds 超参调优毫无意义
2. **别堆叠 class_weight + oversampling**：类不平衡策略选一个；R7→R8 仅移除 class_weight 就从 17.8% → 81.3%
3. **"Garbage bin class" = 症状不是病**：真正的问题总在数据或 loss 权重
4. **with_limb 在干净数据上可用但略弱于 kp**：R4 灾难因数据 bug；R10 干净数据上 limb 80.7% vs kp 82.4%，错误模式互补适合 ensemble
5. **一定要先验证数据管线**：R1 前就应该跑 diagnostics
6. **视频级 split 是必要的**：防止 train/val 数据泄漏
7. **K-fold CV 验证模型稳定性**：Fold 0 vs Fold 1 差 <1% 确认结果可靠
8. **独立 test set 很重要**：Test 83.5% 与 val 81.3% 一致说明没过拟合 val 超参
9. **Multi-clip 推理无效**：10-clip 取平均 +0.0%，说明预测已稳定，瓶颈在特征表达
10. **clip_len 48→64 提升有限**：+1.1% val / +0.7% test
11. **PYSKL CrossEntropyLoss 不支持 label_smooth_val**：是 mmaction2 较新版本的功能
12. **kp 与 limb 错误互补**：kp 擅长 normal/falling，limb 擅长 fighting → ensemble 是推到 90% 的路线之一
13. **交叉折 MIL 清洗是突破 85% 的关键**：仅去掉 11.6% 噪声带来 +9.1% test 提升，远超任何超参调优

### 部署与推理

14. **训练/推理管线一致性至关重要**：训练 2 人骨骼但推理送 1 人 → fighting 全部漏检
15. **不能用训练集模型给训练集打分**：过度自信（P(true)≈1.0），必须交叉折
16. **PoseC3D 对静态场景有盲区**：一动不动躺地无时序特征 → normal；需要 YOLO 单帧检测兜底
17. **规则引擎判定要有证据链**：「躺地+有人」≠ bullying，必须检查邻居行为历史
18. **遮挡恢复需要宽限期**：track 消失立即清除会导致恢复后标签空窗；grace_frames=90 保留 buffer/history/label
19. **ByteTrack 会重新分配 ID**：grace 只在原 ID 恢复时有效，换 ID 需要主动重关联（位置匹配 + 状态迁移）
20. **像素坐标姿态检查受摄像头角度影响**：分层处理 — PoseC3D 路径用姿态检查，YOLO 路径信任检测器
21. **3D 纵深场景用 bbox 重叠替代距离**：攻击者站在躺地者上方时 2D 距离很大但 bbox 必然重叠
22. **滞回投票防止告警闪断**：ENTRY 严格（防误报）+ HOLD 宽松（防断裂）+ UPGRADE 允许（防锁死）
23. **不能只看 argmax**：施暴者站在躺地者上方时 PoseC3D 概率分散（normal=0.5, fighting=0.4），argmax=normal 丢失信号；改为**攻击概率主导**（`max(fighting_prob, bullying_prob) ≥ 0.3`）配合 proximity + asymmetry 强约束
24. **规则引擎的优先级需要语义对齐**：YOLO falling 的本意是补偿 PoseC3D 盲区，当 PoseC3D 已有攻击信号时应让路；不能让"补偿规则"覆盖"主信号"
25. **独立判定各 track 在强交互场景会失效**：bbox 重叠的两人是同一事件，必须用 pair coupling 强制共享攻击态；交互物理约束（bbox 重叠）比单 track 时序约束（投票窗口）更可靠，应作为 post-processing 覆盖在单 track 判定之上
26. **受害者≠普通倒地**：`A=bullying + B=falling + overlap` 场景下 B 不是独立倒地，是霸凌场景中的受害者 → 升级为 bullying；单纯 falling 只适用于无攻击者的跌倒
27. **连续同向修复累积偏差**：R8.1–R8.5 五连发全部朝"更容易识别攻击"的方向，叠加后 false positive 飙升。修复要有方向记账——每一步是朝"报告攻击"还是朝"报告正常"推，超过 3 次同向就停下来反向审视
28. **绝对阈值要配相对约束**：`attack_prob ≥ 0.3` 在 softmax 分布下忽略了 normal 的压制。同时考虑绝对下限（过滤极低置信）和相对优势（避免被其它类主导）才稳健
29. **HOLD 门槛要分类别设**：falling/climbing 姿态延续性高 HOLD=1 合理；fighting/bullying 是交互事件，PoseC3D 对间歇输出噪声 → HOLD=1 会永锁。延续性类别和交互类别的时序先验不同
30. **OR 条件要小心**：单判据 OR 拼起来等于"任意一条噪声即触发"；AND 才符合"同时满足两个独立维度"的物理语义。姿态检测这类容易抖的判定尤其要用 AND
31. **子模型误检必须有反向守门**：YOLO unified_3class 对 fighting 中交缠/倾斜姿态有 falling 误检，但 step 2 的 rule_yolo_bullying 完全信任 YOLO，不看 PoseC3D。正确做法：当另一个子模型（这里是 PoseC3D）对当前场景有极强的反向信号（fighting≥0.7 + bullying<0.03）时，本子模型的检测应被否决。不要让任一子模型的误检单独驱动关键决策
32. **cross-inject 需要接收方状态检查**：R8.1 引入 inject 是为解决攻击者/受害者 ENTRY 计数不同步，但对称 fighting 场景下它会把一方的误判传染到稳定 fighting 的另一方，形成自激。传播必须看接收方是否"正在坚持不同的稳定标签"—— 稳定 fighting 的邻居不该被 bullying 注入污染
33. **训练样本极不平衡时，该类的 softmax 概率大小比较不可靠**：R11 bullying 样本 446 vs fighting 12772（28:1），bullying_prob 在 fighting 场景下边界抖动。用 `b >= f` 这种"相等即采信"的门槛等于让模型用猜的决策。应该要求显著优势（如 1.5 倍）+ 绝对下限（如 0.4）双重过滤，本质是承认模型在该类的边界不可靠

---

## 附录：Git 提交链（E2E 关键节点）

```
074c47f feat(e2e): R11 small-object 3-way single-class + frame-level gating   (R11 E2E)
8af7a53 fix(e2e): R10 fighting/bullying desensitization — P8/P9/P10           (R10)
e9cbc11 fix(e2e): R9 rule engine tightening — P1/P2/P5/P7                    (R9)
20185f3 fix(e2e): pair coupling — bbox-overlapping tracks must share attack state (R8.5)
18fad7c fix(e2e): attack-prob-driven detection (ignore argmax for fighting/bullying) (R8.4)
ebbf93c fix(e2e): PoseC3D fighting/bullying priority over YOLO falling       (R8.3)
1fa3a21 fix(e2e): allow anomaly upgrade in hysteresis hold (falling→bullying) (R8.2)
9d8ced2 fix(e2e): hysteresis vote + relaxed bullying victim + cross-label propagation (R8.1)
4a16b2b docs: log E2E Fix Round 7                                            (R7)
00d2e02 fix: use bbox overlap (not distance) for YOLO bullying check         (R7)
4b196ae feat: increase vote window 3→5 and adjust vote thresholds            (R7)
e4539df fix: require sustained attack history for YOLO bullying detection    (R7)
b081d01 feat: EMA keypoint smoothing                                         (R7)
add2edc fix: trust YOLO falling (remove posture checks)                      (R7)
1d188fb fix: climbing vertical movement constraint                           (R7)
09395f0 fix: sitting posture detection via bone bbox aspect ratio            (R7)
1a93399 fix: loitering threshold 60s→300s + lower priority                   (R7)
c9dec2b feat: track re-association on ByteTrack ID switch                    (R7)
```
