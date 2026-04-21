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

