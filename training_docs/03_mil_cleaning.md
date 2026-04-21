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

