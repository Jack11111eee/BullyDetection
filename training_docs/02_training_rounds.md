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
| Mean class acc | 77.4% | 74.0% |

**Per-class (Round 10 Test):**

| Class | Correct | Total | Accuracy |
|---|---|---|---|
| normal | 1174 | 1479 | 79.4% |
| fighting | 961 | 1209 | 79.5% |
| bullying | 20 | 51 | 39.2% |
| falling | 774 | 805 | 96.1% |
| climbing | 47 | 62 | 75.8% |

**Test 混淆矩阵（行=真实, 列=预测）：**

| | normal | fighting | bullying | falling | climbing |
|---|---|---|---|---|---|
| **normal** | **1174** | 248 | 6 | 44 | 7 |
| **fighting** | 226 | **961** | 5 | 15 | 2 |
| **bullying** | 7 | 22 | **20** | 2 | 0 |
| **falling** | 7 | 21 | 0 | **774** | 3 |
| **climbing** | 3 | 8 | 0 | 4 | **47** |

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

**Test 混淆矩阵（行=真实, 列=预测）：**

| | normal | fighting | bullying | falling | climbing |
|---|---|---|---|---|---|
| **normal** | **1380** | 81 | 0 | 18 | 0 |
| **fighting** | 96 | **1085** | 5 | 23 | 0 |
| **bullying** | 7 | 4 | **40** | 0 | 0 |
| **falling** | 2 | 3 | 0 | **798** | 2 |
| **climbing** | 0 | 0 | 0 | 1 | **61** |

![confusion_r10_r11](confusion_r10_r11.png)

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

