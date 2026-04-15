# PoseC3D Campus Safety Training Log

## Project Info

- **Task**: Campus Safety Video Behavior Recognition (17th Service Innovation Competition, A28)
- **Pipeline**: YOLO11m-Pose + ByteTrack -> PoseC3D (PYSKL)
- **Server**: AutoDL, RTX 3090, conda env `dtc`
- **Framework**: mmcv-full 1.7.0 + PYSKL, Python 3.10, PyTorch 2.1.0, CUDA 11.8
- **Target**: top1 accuracy >= 90%

---

## Training Round History

### Round 1 — 8 Classes

| Item | Detail |
|---|---|
| **Classes** | 8 (normal, fighting, bullying, falling, climbing, vandalism, smoking, phone_call) |
| **Result** | top1 = **41.5%** |
| **Problem** | vandalism weight=6.0 dominated all predictions |
| **Diagnosis** | Class weight too aggressive for vandalism |

---

### Round 2 — 7 Classes

| Item | Detail |
|---|---|
| **Classes** | 7 (removed one class) |
| **Result** | top1 = **31.8%** |
| **Problem** | fighting dominated all predictions |
| **Diagnosis** | Class imbalance still not resolved, fewer classes didn't help |

---

### Round 3 — 6 Classes + Balanced Data

| Item | Detail |
|---|---|
| **Classes** | 6 |
| **Changes** | Introduced `fix_and_balance.py` for data balancing (undersample CAP=6000, oversample MAX=3x) |
| **Result** | top1 = **65.1%** |
| **Problem** | vandalism over-predicted |
| **Diagnosis** | vandalism data quality extremely poor (70.8% zero keypoints from vandalism2 source) |

---

### Round 4 — with_limb=True

| Item | Detail |
|---|---|
| **Classes** | 6 |
| **Changes** | Enabled limb heatmap (`with_limb=True`) |
| **Result** | top1 = **33.1%** |
| **Problem** | Catastrophic accuracy drop |
| **Diagnosis** | Limb features introduced noise; reverted `with_limb=False` |

---

### Round 5 — Fixed keypoint_score

| Item | Detail |
|---|---|
| **Classes** | 6 |
| **Changes** | Fixed `keypoint_score` format in data pipeline (PYSKL expects separate keypoint + keypoint_score arrays) |
| **Result** | top1 = **56.2%** |
| **Problem** | **vandalism "garbage bin class"**: predicted 5794 times (actual 1448), absorbed predictions from all other classes |
| **Diagnosis** | vandalism had ~70% zero-keypoint samples; model learned to dump uncertain inputs into vandalism |

---

### Round 6 — Removed vandalism (5 Classes)

| Item | Detail |
|---|---|
| **Config** | `finetune_campus_v3.py`, `num_classes=5`, `dropout=0.5` |
| **Classes** | 5 (normal, fighting, bullying, falling, climbing) |
| **class_weight** | `[1.0, 1.0, 1.5, 1.0, 2.5]` |
| **Data** | `campus_balanced_v5.pkl` |
| **Epochs** | 50 |
| **Result** | top1 = **53.8%** |
| **Problem** | **bullying "garbage bin class"**: predicted 4528 times (actual 288), absorbed 40% of normal, 36% of falling, 18% of fighting |
| **Diagnosis** | Removing vandalism didn't fix root cause — 36% of ALL samples had all-zero keypoints. bullying had 44% zero → became new garbage bin. Same input (all zeros) + different labels = impossible to learn. |

**Per-class accuracy (Round 6):**

| Class | Accuracy |
|---|---|
| normal | Low (40% predicted as bullying) |
| fighting | Moderate |
| bullying | "Garbage bin" - massively over-predicted |
| falling | Low (36% predicted as bullying) |
| climbing | Moderate |

---

### Round 7 — Data Quality Fix (v7 Dataset)

| Item | Detail |
|---|---|
| **Config** | `finetune_campus_v3.py`, `num_classes=5` |
| **Classes** | 5 (normal, fighting, bullying, falling, climbing) |
| **class_weight** | `[1.0, 1.0, 1.5, 1.0, 2.5]` |
| **Data** | `campus_balanced_v7.pkl` (filtered zero keypoints + deduped frame_dir) |
| **Epochs** | 50 |
| **Train top1** | **~99%** (Epoch 50) |
| **Val top1** | **17.8%** |
| **Problem** | **Extreme overfitting** (81 point gap). **climbing "garbage bin class"**: model predicted 2998 normal, 1627 fighting, 1718 falling all as climbing |

**Per-class accuracy (Round 7):**

| Class | Correct | Total | Accuracy |
|---|---|---|---|
| normal | 0 | 3241 | 0.0% |
| fighting | 1313 | 2949 | 44.5% |
| bullying | 0 | 114 | 0.0% |
| falling | 3 | 1730 | 0.2% |
| climbing | 140 | 154 | 90.9% |

**Confusion Matrix (Round 7):**

|  | -> normal | -> fighting | -> bullying | -> falling | -> climbing |
|---|---|---|---|---|---|
| normal | 0 | 237 | 6 | 0 | **2998** |
| fighting | 0 | 1313 | 9 | 0 | **1627** |
| bullying | 0 | 114 | 0 | 0 | 0 |
| falling | 0 | 9 | 0 | 3 | **1718** |
| climbing | 0 | 14 | 0 | 0 | 140 |

**Root Cause**: Data cleaning removed 36% zero samples, but climbing lost only 2.7% (cleanest class). Combined with `class_weight=2.5` AND oversampling 3x, climbing's effective influence was massively amplified. 50 epochs on cleaner but smaller dataset = severe overfitting.

---

### Round 8 — 5-Fold CV + Remove class_weight + 30 Epochs

| Item | Detail |
|---|---|
| **Config** | `finetune_campus_v3.py` (class_weight removed, 30 epochs) |
| **Classes** | 5 (normal, fighting, bullying, falling, climbing) |
| **class_weight** | **Removed** |
| **Data** | 5-Fold CV (`campus_kfold_0~4.pkl`) + held-out test (`campus_test.pkl`) |
| **Data split** | 10% test held-out, 90% → 5-fold (每 fold 72% train / 18% val) |
| **Epochs** | **30** |
| **Data script** | `build_kfold_data.py` (清洗 + 5-fold 划分 + test set) |

**Fold 0 结果：**

| Metric | Val (7334 samples) | Test (3606 samples) |
|---|---|---|
| **Overall top1** | **81.3%** | **83.5%** |
| **Mean class acc** | **76.3%** | **76.5%** |

| Class | Val Acc | Test Acc |
|---|---|---|
| normal | 81.1% | 81.8% |
| fighting | 74.7% | 79.0% |
| bullying | 56.4% | 51.0% |
| falling | 94.3% | 96.3% |
| climbing | 75.0% | 74.2% |

**Fold 0 Val Confusion Matrix:**

|  | -> normal | -> fighting | -> bullying | -> falling | -> climbing |
|---|---|---|---|---|---|
| normal | **[2244]** | *438* | 7 | 71 | 8 |
| fighting | *611* | **[1999]** | 13 | 48 | 4 |
| bullying | 22 | 18 | **[57]** | 3 | 1 |
| falling | 44 | 47 | 2 | **[1579]** | 2 |
| climbing | 10 | 16 | 0 | 3 | **[87]** |

**Fold 1 结果（交叉验证确认）：**

| Metric | Val (7710 samples) |
|---|---|
| **Overall top1** | **80.5%** |
| **Mean class acc** | **72.3%** |

| Class | Fold 0 Val | Fold 1 Val |
|---|---|---|
| normal | 81.1% | 80.7% |
| fighting | 74.7% | 75.1% |
| bullying | 56.4% | 44.9% |
| falling | 94.3% | 93.1% |
| climbing | 75.0% | 67.8% |

**结论：两个 fold 差距 <1%（除 bullying/climbing 因样本极少波动大），模型稳定在 ~81% 水平。**

**成功原因（从 17.8% → 81.3%）：**
1. **去掉 class_weight**（最大功臣）— 消除 climbing 垃圾桶问题
2. **去掉全零 keypoint 数据**（第二功臣）— 消除 36% 噪声样本
3. **30 epochs 代替 50**（辅助）— 减少过拟合

**当前瓶颈：**
- **normal ↔ fighting 互混** 是最大错误来源（两个 fold 一致）
- **bullying 样本极少**（val ~100 个，test ~50 个），准确率波动大

---

### Round 9 — clip_len=64 + 更强正则化

| Item | Detail |
|---|---|
| **Config** | `finetune_campus_v4.py` |
| **Classes** | 5 |
| **Changes vs R8** | clip_len 48→64, weight_decay 0.0003→0.001, epochs 30→50, eval interval 5→3 |
| **Data** | `campus_kfold_0.pkl` (同 fold 0) |
| **Epochs** | 50 |
| **work_dir** | `posec3d_campus_v9` |

**前置实验 — Multi-clip 推理（R8模型，不重训）：**
- 在 R8 fold0 模型上测试 10-clip 推理取平均 → **+0.0%**（val 81.3%, test 83.5% 不变）
- 结论：模型预测已稳定，问题不在采样随机性，在于特征区分能力
- 因此 R9 config 去掉 val num_clips=10（白浪费10x验证时间）

**label_smooth_val=0.1 被移除：**
- PYSKL 的 `CrossEntropyLoss` 不支持 `label_smooth_val` 参数（mmaction2 较新版本才有）
- 训练时报错 `CrossEntropyLoss.__init__() got an unexpected keyword argument 'label_smooth_val'`

**结果：**

| Metric | Val (7334 samples) | Test (3606 samples) |
|---|---|---|
| **Overall top1** | **82.4%** | **84.2%** |
| **Mean class acc** | **78.9%** | **76.3%** |

| Class | Val Acc | Test Acc |
|---|---|---|
| normal | 80.3% | 82.2% |
| fighting | 76.7% | 80.8% |
| bullying | 58.4% | 45.1% |
| falling | 96.3% | 96.1% |
| climbing | 82.8% | 77.4% |

**Val Confusion Matrix:**

|  | -> normal | -> fighting | -> bullying | -> falling | -> climbing |
|---|---|---|---|---|---|
| normal | **[2222]** | *453* | 11 | 77 | 5 |
| fighting | *553* | **[2053]** | 14 | 49 | 6 |
| bullying | 14 | 26 | **[59]** | 2 | 0 |
| falling | 28 | 30 | 2 | **[1612]** | 2 |
| climbing | 5 | 10 | 0 | 5 | **[96]** |

**vs Round 8:** Val +1.1%, Test +0.7%。clip_len 48→64 有小幅提升但不显著。normal↔fighting 互混仍是主要瓶颈。

---

### Round 10 — Limb-only 模型 (with_kp=False, with_limb=True)

| Item | Detail |
|---|---|
| **Config** | `finetune_campus_v5.py` |
| **Classes** | 5 |
| **Changes vs R9** | with_kp=False, with_limb=True, 训练分辨率56→64, videos_per_gpu 16→12 |
| **Data** | `campus_kfold_0.pkl` (同 fold 0) |
| **Epochs** | 50 |
| **work_dir** | `posec3d_campus_v10` |

**背景：**
- PYSKL 不允许 `with_kp=True` 和 `with_limb=True` 同时启用（AssertionError）
- Round 4 时 `with_limb=True` 灾难性失败（33.1%），但当时数据有bug（零keypoint+脏数据）
- 现在数据干净，重新测试纯 limb 模型，目标是后续与 kp 模型 ensemble

**结果：**

| Metric | Val (7334 samples) | Test (3606 samples) |
|---|---|---|
| **Overall top1** | **80.7%** | **82.5%** |
| **Mean class acc** | **77.4%** | **74.0%** |

| Class | R9 Val (kp) | R10 Val (limb) | Delta |
|---|---|---|---|
| normal | 80.3% | 75.9% | -4.4% |
| fighting | 76.7% | **78.5%** | **+1.8%** |
| bullying | 58.4% | 57.4% | -1.0% |
| falling | 96.3% | 93.4% | -2.9% |
| climbing | 82.8% | 81.9% | -0.9% |

| Class | R9 Test (kp) | R10 Test (limb) | Delta |
|---|---|---|---|
| normal | 82.2% | 79.4% | -2.8% |
| fighting | 80.8% | 79.5% | -1.3% |
| bullying | 45.1% | 39.2% | -5.9% |
| falling | 96.1% | 96.1% | 0.0% |
| climbing | 77.4% | 75.8% | -1.6% |

**Val Confusion Matrix (R10 limb):**

|  | -> normal | -> fighting | -> bullying | -> falling | -> climbing |
|---|---|---|---|---|---|
| normal | **[2101]** | *574* | 9 | 76 | 8 |
| fighting | *519* | **[2099]** | 14 | 35 | 8 |
| bullying | 18 | 23 | **[58]** | 2 | 0 |
| falling | 51 | 53 | 0 | **[1563]** | 7 |
| climbing | 1 | 20 | 0 | 0 | **[95]** |

**vs R9:** Val -1.7%, Test -1.7%。Limb-only 模型整体略差于 kp-only。

**关键发现 — 两个模型犯不同的错：**
- **kp 模型(R9)擅长**: normal (80.3% vs 75.9%), falling (96.3% vs 93.4%)
- **limb 模型(R10)擅长**: fighting (78.5% vs 76.7%)
- kp 模型 fighting→normal 误判 553 次，limb 模型只有 519 次
- kp 模型 normal→fighting 误判 453 次，limb 模型有 574 次
- **错误模式互补 → ensemble 有潜力**

---

## Key Problems Discovered & Solved

### Problem 1: All-Zero Keypoint Samples (36%)

- **Discovery**: Round 6 post-mortem diagnostic
- **Details**: 36.3% of train, 36.2% of val had `keypoint == 0` everywhere (YOLO detected no person)
- **Impact**: Identical zero inputs with different labels → model randomly picks one class as default → "garbage bin class"
- **Source breakdown**:

| Source | Total | Zero% |
|---|---|---|
| vandalism2 | 7,160 | **70.8%** |
| ucf | 2,673 | **60.6%** |
| shanghaitech | 14,803 | **53.1%** |
| chute | 10,054 | **41.7%** |
| rlvs | 9,998 | 13.3% |
| rwf | 9,818 | 7.4% |

- **Fix**: Added `np.all(keypoint == 0)` filter in `fix_and_balance.py` → v7 dataset
- **Status**: Fixed in Round 7

### Problem 2: "Garbage Bin Class" Pattern

- **Pattern**: In every round, one class absorbs predictions from all others
- **History**: vandalism (R3, R5) → bullying (R6) → climbing (R7)
- **Root Cause (R3-R6)**: Zero-keypoint samples
- **Root Cause (R7)**: class_weight + oversampling double amplification on clean data
- **Fix**: Remove class_weight, let oversampling alone handle balance
- **Status**: Fix applied for Round 8

### Problem 3: Label Conflicts (1,033 samples)

- **Details**: Same `frame_dir` appears with different labels (mainly `fighting` vs `normal` from RWF dataset)
- **Fix**: frame_dir deduplication in `fix_and_balance.py` (keep first occurrence)
- **Status**: Fixed in v7 dataset

### Problem 4: Duplicate Split IDs

- **Details**: 883 duplicate IDs in train, 445 in val of balanced dataset
- **Fix**: Deduplication during data processing
- **Status**: Fixed in v7 dataset

### Problem 5: with_limb Catastrophe (Round 4)

- **Details**: Enabling `with_limb=True` dropped accuracy from 65% to 33%
- **Fix**: Reverted to `with_limb=False, with_kp=True`
- **Status**: Fixed

### Problem 6: keypoint_score Format

- **Details**: PYSKL expects `keypoint (M, T, 17, 2)` and `keypoint_score (M, T, 17)` as separate arrays
- **Fix**: Fixed in `step4_build_pkl.py`
- **Status**: Fixed in Round 5

### Problem 7: Overfitting (Train 99% vs Val 17.8%)

- **Discovery**: Round 7
- **Root Cause**: 50 epochs on cleaner (smaller) dataset + class_weight amplification
- **Fix**: Reduce to 30 epochs + remove class_weight
- **Status**: **Fixed in Round 8** — overfitting消失，val 81.3%

### Problem 8: normal ↔ fighting 互混

- **Discovery**: Round 8 (Fold 0/1 一致)
- **Details**: normal 的 16% 被判为 fighting，fighting 的 23% 被判为 normal
- **Root Cause**: 骨骼层面两个类的动作模式相似（都是站立/移动），单次 48 帧采样有随机性
- **Status**: **进行中** — Round 9 用 label smoothing + multi-clip 推理优化

---

## Data Pipeline

```
step4_build_pkl.py    → train.pkl + val.pkl (raw skeleton extraction)
        ↓
reformat_pkl.py       → campus.pkl (merge with split dict)
        ↓
build_kfold_data.py   → campus_kfold_0~4.pkl + campus_test.pkl  ← Round 8+
                         - Filter label >= 5
                         - Filter all-zero keypoints
                         - Dedup frame_dir
                         - 10% held-out test (视频级)
                         - 90% → 5-fold CV (视频级)
                         - 每 fold 训练集独立做 undersample(CAP=6000) + oversample(MAX=3x)
```

**旧流程（Round 7 及之前）：** `fix_and_balance.py → campus_balanced_v7.pkl`（80/20 split，无 test set）
**新流程（Round 8+）：** `build_kfold_data.py → 5-fold + test`（有独立 held-out test set）

---

## Key Config Files

| Parameter | R6 (v3) | R7 (v3) | R8 (v3+fold) | R9 (v4) | R10 (v5) |
|---|---|---|---|---|---|
| num_classes | 5 | 5 | 5 | 5 | 5 |
| dropout | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| class_weight | [1,1,1.5,1,2.5] | [1,1,1.5,1,2.5] | **Removed** | Removed | Removed |
| clip_len | 48 | 48 | 48 | **64** | **64** |
| total_epochs | 50 | 50 | **30** | **50** | **50** |
| lr | 0.005 | 0.005 | 0.005 | 0.005 | 0.005 |
| weight_decay | 0.0003 | 0.0003 | 0.0003 | **0.001** | **0.001** |
| with_kp | True | True | True | True | **False** |
| with_limb | False | False | False | False | **True** |
| train resolution | 56x56 | 56x56 | 56x56 | 56x56 | **64x64** |
| videos_per_gpu | 16 | 16 | 16 | 16 | **12** |
| ann_file | v5.pkl | v7.pkl | kfold_0.pkl | kfold_0.pkl | kfold_0.pkl |
| config file | v3.py | v3.py | fold0.py | **v4.py** | **v5.py** |

---

## Training Command

```bash
cd /home/hzcu/BullyDetection/pyskl && \
LD_LIBRARY_PATH=/home/hzcu/miniconda3/pkgs/cuda-cudart-11.8.89-0/lib:$LD_LIBRARY_PATH \
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 \
tools/train.py configs/posec3d/finetune_campus_v3.py --launcher pytorch
```

---

## Evaluation Command

```bash
cd /home/hzcu/BullyDetection && python per_class_eval_v2.py
```

---

## Other Components (Built, Not Yet Integrated)

| Component | File | Status |
|---|---|---|
| Rule Engine | `rule_engine.py` | Built. Adds vandalism/smoking/phone_call via heuristics |
| Inference Pipeline | `main_inference.py` | Built. End-to-end video inference |
| YOLO11 Unified 3-Class | `unified_3class_model/best.pt` | Teammate trained. phone/smoking/falling 三类检测 |
| Training Curves | `plot_training_curves.py` | Built. Parses mmcv logs, plots curves |
| Data Diagnostics | `diagnose_data.py`, `diagnose_all_data.py` | Built. Pipeline-wide data quality checks |
| Sample Visualization | `visualize_samples.py` | Built. Draws skeleton from pkl samples |
| K-Fold Data Builder | `build_kfold_data.py` | Built. 5-fold CV + 10% test set |
| K-Fold Eval | `eval_kfold.py` | Built. 汇总 5-fold + test set 评估 |
| Multi-clip Eval | `eval_multiclip.py` | Built. Multi-clip 推理评估（已证明+0.0%无效） |
| All-epoch Eval | `eval_all_epochs.py` | Built. 遍历 checkpoints 评估 |
| K-Fold Training | `train_kfold.sh` | Built. 一键训练 5 个 fold |
| Round Eval | `eval_round9.py` | Built. 通用模型评估，结果保存到 work_dir/eval_results.txt |
| E2E Pipeline | `e2e_pipeline/` | Built. 端到端推理（YOLO+PoseC3D+规则引擎+可视化） |

---

## Accuracy Trend

| Round | Val top1 | Test top1 | Key Change |
|---|---|---|---|
| 1 | 41.5% | — | Baseline 8 classes |
| 2 | 31.8% | — | 7 classes |
| 3 | 65.1% | — | 6 classes + balanced data |
| 4 | 33.1% | — | with_limb=True (catastrophe, 数据有bug) |
| 5 | 56.2% | — | Fixed keypoint_score |
| 6 | 53.8% | — | 5 classes (removed vandalism) |
| 7 | 17.8% | — | Clean data but class_weight+oversample collision |
| 8 | **81.3%** | **83.5%** | **K-fold CV + no class_weight + 30 epochs** |
| 9 | **82.4%** | **84.2%** | clip_len 48→64 + weight_decay 0.001 |
| 10 | 80.7% | 82.5% | with_limb=True (数据干净后可用，但略低于kp) |

---

## Lessons Learned

1. **Data quality > everything**: 36% garbage data made 6 rounds of hyperparameter tuning meaningless
2. **Don't stack class_weight + oversampling**: Pick one strategy for class imbalance, not both. Round 7→8 仅去掉 class_weight 就从 17.8% → 81.3%
3. **"Garbage bin class" = symptom, not disease**: The real problem is always in the data or loss weighting
4. **with_limb 在干净数据上可用但略弱于 kp**: R4 灾难是因为数据bug，R10 在干净数据上 limb 模型达到 80.7%（vs kp 82.4%），但两者犯不同的错，适合 ensemble
5. **Always validate data pipeline first**: Should have run diagnostics before Round 1
6. **Video-level split is essential**: Prevents data leakage between train/val
7. **K-fold CV 验证模型稳定性**: Fold 0 (81.3%) vs Fold 1 (80.5%) 差距 <1%，确认结果可靠
8. **独立 test set 很重要**: Test 83.5% 与 val 81.3% 一致，说明没有过拟合 val 的超参选择
9. **Multi-clip 推理无效**: 10-clip 取平均 +0.0%，说明模型预测已稳定，瓶颈在特征表达而非采样随机性
10. **clip_len 48→64 提升有限**: +1.1% val / +0.7% test，更长时序窗口帮助不大
11. **PYSKL CrossEntropyLoss 不支持 label_smooth_val**: 这是 mmaction2 较新版本的功能，PYSKL 没有
12. **kp 与 limb 模型互补**: kp 擅长 normal/falling，limb 擅长 fighting → ensemble 是推到 90% 的关键路线
13. **交叉折 MIL 清洗是突破 85% 瓶颈的关键**: 仅去掉 11.6% 的噪声样本就带来 +9.1% test 提升，远超任何超参调优
14. **训练/推理管线一致性至关重要**: 训练用 2 人骨骼但推理只送 1 人 → fighting 全部漏检。必须保证 M 维度一致
15. **不要用训练集模型给训练集打分**: 模型会过度自信（P(true)≈1.0），必须用交叉折让每个样本被未见过它的模型评分
16. **PoseC3D 对静态场景有盲区**: 一动不动躺地者无时序特征 → 输出 normal。需要 YOLO 单帧检测兜底
17. **规则引擎判定要有证据链**: 「躺地+有人」不等于 bullying，必须检查附近人的行为历史（fighting/bullying），避免路人误触发
18. **遮挡恢复需要宽限期**: track 消失后立即清除状态会导致恢复后标签空窗。grace_frames=90 保留 buffer/history/label，恢复时平滑过渡

---

## MIL 数据清洗

### 背景

Round 9 卡在 84.2% test，主要瓶颈是 **normal ↔ fighting 互混**（normal 16% 误判 fighting，fighting 23% 误判 normal）。
根本原因：fighting 视频中大量片段其实是 normal 行为（如 1 分钟视频里 30 秒是 normal），但所有 clip 都继承了视频级 "fighting" 标签。

### 方法：交叉折打分（Cross-Fold Scoring）

用 5-fold 交叉验证的模型对数据打分 — 每个样本只被没见过它的模型评分：

```
Fold 0 模型（训练在 kfold_0 train 上）→ 打分 kfold_0 的 val（模型没见过）
Fold 1 模型（训练在 kfold_1 train 上）→ 打分 kfold_1 的 val（模型没见过）
...
合并 → 37,289 个样本都有无偏的 P(true_label)
```

**重要教训**：最初尝试直接用 R9 模型对全量 campus.pkl 打分，但模型在训练集上达到 ~99% accuracy，对训练样本过度自信（P(true)≈1.0），无法区分噪声。改用交叉折后问题解决。

### 噪声分析结果

5-fold 全量覆盖（37,289 样本），各 fold 噪声分布高度一致：

| 类别 | 总数 | P<0.3 (明确噪声) | P<0.5 (可疑) | 噪声程度 |
|---|---|---|---|---|
| **fighting** | 12,772 | 2,559 (20.0%) | 3,301 (25.8%) | 高 |
| **normal** | 14,917 | 2,196 (14.7%) | 3,045 (20.4%) | 中高 |
| bullying | 446 | 240 (53.8%) | 257 (57.6%) | 极高 |
| climbing | 608 | 142 (23.4%) | 179 (29.4%) | 中高 |
| falling | 8,546 | 347 (4.1%) | 478 (5.6%) | 低（最干净） |

交叉混淆重点：
- fighting 样本被高置信判为 normal：2,803 个（21.9%），其中 2,047 个 P(normal)>0.7
- normal 样本被高置信判为 fighting：2,478 个（16.6%），其中 1,637 个 P(fighting)>0.7

P(true_label) 直方图呈明显**双峰分布**，threshold=0.3 正好卡在谷底，是最自然的分割点。

### 清洗策略

使用 `--threshold 0.3 --classes fighting normal`，移除 fighting 和 normal 中 P(true_label)<0.3 的样本：

```
原始有效样本:  40,895
移除噪声:      4,755 (11.6%)
  - normal:    2,196
  - fighting:  2,559
清洗后:        36,140
```

### MIL 清洗工具链

```
mil_cleaning/
├── score_samples.py       ← 交叉折打分（5-fold 模型各打自己的 val）
├── analyze_noise.py       ← 噪声分析 + 直方图 + 报告
├── clean_and_rebuild.py   ← 根据阈值过滤 + 重建 kfold pkl
├── export_noisy_samples.py← 导出噪声样本清单（追溯源视频）
├── resolve_video_paths.py ← 在数据集目录中查找源视频真实路径
├── collect_videos.py      ← 复制噪声视频到审核文件夹
├── scores.pkl             ← 37,289 个样本的完整概率向量
├── noise_distribution.png ← P(true_label) 直方图
├── noise_report.txt       ← 详细噪声统计报告
└── HANDOFF.md             ← 交接文档
```

---

### Round 11 — MIL 清洗后训练

| Item | Detail |
|---|---|
| **Config** | `finetune_campus_mil.py` |
| **Classes** | 5 |
| **Changes vs R9** | 仅改 ann_file 指向 MIL 清洗后数据，其余超参完全一致 |
| **Data** | `campus_mil_kfold_0.pkl`（清洗后，threshold=0.3，去掉 fighting+normal 噪声） |
| **Epochs** | 50 |
| **work_dir** | `posec3d_campus_mil` |

**在清洗后数据上评估（campus_mil_kfold_0 val + campus_mil_test）：**

| Metric | Val (6401 samples) | Test (3266 samples) |
|---|---|---|
| **Overall top1** | **90.5%** | **90.3%** |
| **Mean class acc** | **82.7%** | **82.0%** |

**在原始未清洗数据上评估（campus_kfold_0 val + campus_test，更公平的对比）：**

| Metric | Val (7334 samples) | Test (3606 samples) |
|---|---|---|
| **Overall top1** | **89.9%** | **93.3%** |
| **Mean class acc** | **92.7%** | **91.8%** |

**vs Round 9（原始数据上对比）：**

| Class | R9 Test | R11 Test | Delta |
|---|---|---|---|
| normal | 82.2% | 93.3% | **+11.1%** |
| fighting | 80.8% | 89.7% | **+8.9%** |
| bullying | 45.1% | **78.4%** | **+33.3%** |
| falling | 96.1% | 99.1% | +3.0% |
| climbing | 77.4% | **98.4%** | **+21.0%** |

**Test Confusion Matrix (R11, 原始数据):**

|  | -> normal | -> fighting | -> bullying | -> falling | -> climbing |
|---|---|---|---|---|---|
| normal | **[1380]** | 81 | 0 | 18 | 0 |
| fighting | 96 | **[1085]** | 5 | 23 | 0 |
| bullying | 7 | 4 | **[40]** | 0 | 0 |
| falling | 2 | 3 | 0 | **[798]** | 2 |
| climbing | 0 | 0 | 0 | 1 | **[61]** |

**成功原因：**
1. 清洗掉 4,755 个噪声样本（11.6%），消除了 fighting 视频中的 normal 片段标签噪声
2. normal↔fighting 互混大幅下降：fighting→normal 误判从 553→96（val），normal→fighting 从 453→81
3. bullying 起死回生（45%→78%），之前被噪声样本误导，清洗后模型终于能学到真实模式
4. **Test 93.3% 已超过 90% 目标线**

---

### Problem 9: 训练/推理管线不一致（单人 vs 双人）

- **Discovery**: Round 11 部署后，真实视频上 fighting 全部漏检
- **Details**: 训练时 `build_pkl.py` 每个样本包含 2 人骨骼 `(M=2, T, 17, 2)`，PoseC3D 将两人骨骼叠加在同一热力图上学习交互模式。但 `e2e_pipeline` 推理时 `SkeletonBuffer.get_clip()` 只返回 1 人骨骼 `(M=1, T, 17, 2)`，模型只看到单人 → 像正常走路 → 预测 normal。
- **Fix**: 修改 `pipeline.py`:
  1. `SkeletonBuffer.get_clip(track_id, secondary_tid)` 改为输出 `(2, T, 17, 2)`
  2. `_process_frame()` 先收集所有 track 位置，为每个推理目标找空间最近邻配对
  3. 两遍处理：第一遍收集位置，第二遍推理（确保配对时能看到当前帧所有人）
- **Status**: 已修复

### Problem 11: PoseC3D 无法识别一动不动躺地者

- **Discovery**: E2E Fix Round 6
- **Details**: 一动不动躺在地上的人（如摔倒后失去意识），PoseC3D 因无时序动作变化普遍输出 normal
- **Fix**: 集成队友的 YOLO11 unified_3class_model（phone/smoking/falling），用 YOLO falling 检测框匹配人物骨骼中心（20% margin），补偿 PoseC3D 的静态场景盲区
- **Status**: 已修复

### Problem 12: YOLO falling 误判 bullying 逻辑

- **Discovery**: E2E Fix Round 6 首版逻辑
- **Details**: 最初设计「躺地+附近有人→bullying」，但路人经过也会触发误报
- **Fix**: 改为检查附近 track 的投票历史（`self.history`），只有附近有 fighting/bullying 标签历史的 track 才判 bullying
- **Status**: 已修复

### Problem 13: 遮挡恢复后标签空窗期

- **Discovery**: E2E Fix Round 6
- **Details**: 人被遮挡后 track 消失 → buffer/history/label 全部清除 → 恢复后标签短暂变 normal
- **Root Cause**: `remove_stale()` 和 `clear_stale_tracks()` 在 track 消失的下一帧就删除所有状态
- **Fix**: 三个组件（SkeletonBuffer / RuleEngine / Pipeline.track_labels）统一添加 grace_frames=90（≈3秒@30fps）宽限期，track 消失后保留状态，恢复时继承旧 buffer 和投票窗口
- **Status**: 已修复

---

## Accuracy Trend (Updated)

| Round | Val top1 | Test top1 | Key Change |
|---|---|---|---|
| 1 | 41.5% | — | Baseline 8 classes |
| 2 | 31.8% | — | 7 classes |
| 3 | 65.1% | — | 6 classes + balanced data |
| 4 | 33.1% | — | with_limb=True (catastrophe, 数据有bug) |
| 5 | 56.2% | — | Fixed keypoint_score |
| 6 | 53.8% | — | 5 classes (removed vandalism) |
| 7 | 17.8% | — | Clean data but class_weight+oversample collision |
| 8 | **81.3%** | **83.5%** | **K-fold CV + no class_weight + 30 epochs** |
| 9 | **82.4%** | **84.2%** | clip_len 48→64 + weight_decay 0.001 |
| 10 | 80.7% | 82.5% | with_limb=True (数据干净后可用，但略低于kp) |
| 11 | **89.9%** | **93.3%** | **MIL 交叉折清洗 (threshold=0.3, -4755 噪声)** |

---

## Key Config Files (Updated)

| Parameter | R9 (v4) | R10 (v5) | R11 (mil) |
|---|---|---|---|
| num_classes | 5 | 5 | 5 |
| dropout | 0.5 | 0.5 | 0.5 |
| clip_len | 64 | 64 | 64 |
| total_epochs | 50 | 50 | 50 |
| lr | 0.005 | 0.005 | 0.005 |
| weight_decay | 0.001 | 0.001 | 0.001 |
| with_kp | True | False | True |
| with_limb | False | True | False |
| ann_file | kfold_0.pkl | kfold_0.pkl | **mil_kfold_0.pkl** |
| config file | v4.py | v5.py | **mil.py** |
| work_dir | campus_v9 | campus_v10 | **campus_mil** |

---

## E2E Pipeline 调优历史

### Problem 10: 模型 93% 准确率但视频流推理效果极差

- **Discovery**: Round 11 部署后，在真实视频流上跑 e2e pipeline，大量异常行为识别不出来，持续显示 normal
- **Root Cause**: 后处理管线多层过度过滤叠加，不是模型问题

**Debug 统计（修复前，542 次推理）：**

| 过滤层 | 吃掉数 | 机制 |
|---|---|---|
| threshold < 0.5 | 67 | fighting=0.498 等被强制归 normal |
| vote: current=normal 直接返回 | 417 | 一次 normal 立刻覆盖所有异常历史 |
| vote: 票数不够 | 21 | 首次 fighting 只有 1/3 票 |
| buffer 不足 SKIP | 3353 | 86%的帧无法推理（track碎片化） |

---

### E2E Fix Round 1 — 后处理基础修复

| 修改 | 文件 | 详情 |
|---|---|---|
| 重写 `_vote_smooth` | rule_engine.py | 改为异常偏向：窗口内有1次异常就维持告警，全部normal才清除 |
| 降低 `pose_threshold` | rule_engine.py + pipeline.py | 0.5 → 0.3 |
| 允许半满 buffer 推理 | pipeline.py | `should_infer` 最低帧数 64 → 32 |
| CLI 默认参数 | run.py | vote-window 5→3, vote-ratio 0.6→0.34 |

**修复后效果：**

| 指标 | 修复前 | 修复后 | 变化 |
|---|---|---|---|
| SKIP (buffer不足) | 3353 | 622 | **-81%** |
| 被threshold过滤 | 67 | 17 | **-75%** |
| 被vote压成normal | 21 | **0** | **消除** |
| FINAL fighting | 30 | **77** | **+157%** |
| FINAL falling | 37 | **78** | **+111%** |

**残余问题**: 75%推理仍输出normal，274个置信度>0.8 — 模型真实输出。根因：训练/推理时序采样不一致。

---

### E2E Fix Round 2 — 均匀采样修复

| 修改 | 文件 | 详情 |
|---|---|---|
| 扩大缓冲区 | pipeline.py | `buf_max = clip_len * 4 = 256` 帧（~8.5秒） |
| 均匀采样 | pipeline.py | `get_clip()` 用 `np.linspace` 从全部缓存帧均匀采64帧，模拟训练时 `UniformSampleFrames` |
| threshold 硬编码修复 | pipeline.py | 发现 pipeline.py 硬编码 `pose_threshold=0.5` 覆盖了默认值，修为 0.3 |

---

### E2E Fix Round 3 — bullying/fighting 区分 + 误报控制

**问题**: 坐着被误判 fighting；fighting 和 bullying 难以区分

| 修改 | 文件 | 详情 |
|---|---|---|
| `_vote_smooth` 分级响应 | rule_engine.py | falling/climbing 1次即告警，fighting/bullying 需2次确认 |
| bullying 不对称检测 | rule_engine.py | `check_bullying_asymmetry()`: 身高比<0.6 或 头-髋差异>10%画面高度 → fighting改判bullying |
| 传递完整骨骼数据 | pipeline.py | 收集所有人的 (kps, scores) 传给规则引擎 |

---

### E2E Fix Round 4 — 正常视频误报修复

**问题**: 靠墙站立误判 fighting → bullying；坐下过程误判 falling

| 修改 | 文件 | 详情 |
|---|---|---|
| fighting 近距离约束 | rule_engine.py | 必须有另一人在附近才算 fighting，否则降级 normal |
| fighting 置信度门槛 | rule_engine.py | fighting 单独要求 conf ≥ 0.5（其他类保持 0.3） |
| fighting vote 要求 | rule_engine.py | 窗口内需 **3次** fighting 才输出（最严格） |
| falling 姿态验证 | rule_engine.py | `_is_upright_posture()`: 头-髋差 > 8%画面高度 = 躯干直立 = 非falling |
| 高置信度也验证 | rule_engine.py | falling conf>0.7 也需过姿态验证，不再无条件直接采信 |

---

### E2E Fix Round 5 — Proximity 放宽 + Bbox 修复

**问题**: bully视频中倒地者proximity太严杀掉98个fighting，踢人者只显示骨架无框

| 修改 | 文件 | 详情 |
|---|---|---|
| Proximity 用两人最大身高 | rule_engine.py | `ref_height = max(my_height, neighbor_height)` × 1.5 |
| Bbox 变量 bug | pipeline.py | 每个track存储自己的bbox，不再引用循环残留变量 |
| 始终显示 bbox+标签 | pipeline.py | 未推理时默认显示 normal + 绿色框 |

---

### E2E Fix Round 6 — YOLO11 三类小物体模型集成 + YOLO辅助falling + 遮挡宽限期

**背景**: 队友训练的 YOLO11 小物体检测模型升级为统一三类模型（phone, smoking, falling），需要嵌入 pipeline 并用于辅助 falling 识别（一动不动躺地者 PoseC3D 普遍识别不出）。同时发现遮挡恢复后标签会有短暂 normal 空窗期。

**模型路径**: `/home/hzcu/yjm/home/yjm/VideoDetection/v6/runs/detect/campus_A28/unified_3class_model/weights/best.pt`

| 修改 | 文件 | 详情 |
|---|---|---|
| SmallObjectDetector 自动读取类名 | pipeline.py | class_map 不再硬编码，从 model.names 自动读取 |
| check_smoking 兼容新类名 | rule_engine.py | 匹配 'cigarette' 或 'smoking' |
| YOLO 辅助 falling 检测 | rule_engine.py | `check_fallen_by_yolo()`: YOLO 检测到 falling bbox + 与人物骨骼中心重叠（20% margin） |
| YOLO falling → bullying 升级 | rule_engine.py | 躺地 + 附近 track 历史含 fighting/bullying → bullying；否则 → falling |
| 遮挡宽限期 (grace_frames=90) | pipeline.py + rule_engine.py | track 消失后保留 buffer/history/label 90帧（≈3秒），恢复时继承旧状态 |
| 默认加载小物体模型 | run.py | `--small-obj-model` 默认指向 unified_3class_model，`--small-obj-model none` 可禁用 |

---

### E2E Pipeline 规则引擎完整逻辑（当前版本）

```
PoseC3D 原始 5 类概率
  → Step 1: climbing>0.7 → 直接采信
             falling>0.7 + 非直立 → 采信；直立 → normal
  → Step 2: YOLO falling 检测（躺地不动）
             躺地 + 附近有 fighting/bullying 历史的 track → bullying
             躺地 + 无攻击者 → falling
  → Step 3: 小物体规则 (smoking / phone_call)
  → Step 4: vandalism (fighting>0.5 + 仅1人)
  → Step 5: 徘徊检测 (轨迹分析)
  → Step 6: PoseC3D 默认 (conf >= 0.3)
       fighting: conf<0.5→normal | 无近距离人→normal | 不对称→bullying
       falling: 躯干直立→normal
  → Vote Smooth (窗口=3):
       fighting:3次 | bullying:2次 | falling/climbing:1次
```

### 遮挡宽限期机制

```
track 被遮挡（从画面消失）
  → 宽限期 90 帧内：保留 SkeletonBuffer + 投票历史 + 显示标签
  → ByteTrack 恢复同一 track_id
  → 立刻显示旧标签（无 normal 空窗），buffer 继续累积
  → 16 帧后触发新推理，平滑过渡
  → 超过 90 帧仍未出现 → 清除所有状态
```

### E2E Pipeline 关键参数（当前版本）

| 参数 | 值 | 说明 |
|---|---|---|
| clip_len | 64 | PoseC3D 输入帧数 |
| stride | 16 | 推理步长 |
| buf_max | 256 (4×clip_len) | SkeletonBuffer 最大历史帧数 |
| min_infer_frames | 32 (clip_len//2) | 最少帧数即可推理 |
| pose_threshold | 0.3 (fighting单独0.5) | PoseC3D 最低置信度 |
| vote_window | 3 | 时序投票窗口 |
| fighting vote_min | 3 | fighting 最低确认次数 |
| bullying vote_min | 2 | bullying 最低确认次数 |
| falling/climbing vote_min | 1 | 紧急行为最低确认次数 |
| proximity_factor | 1.5×max身高 | fighting 近距离约束 |
| upright_threshold | 8% 画面高度 | falling 姿态验证阈值 |
| grace_frames | 90 (≈3秒@30fps) | 遮挡宽限期，track 消失后保留状态 |
| small_obj_model | unified_3class (phone/smoking/falling) | YOLO11 三类检测，辅助 falling + smoking + phone |
