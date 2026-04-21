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
| Per-run debug log | `e2e_pipeline/run.py` `_derive_source_tag` | `logs/debug_<source>_<ts>.log` 按运行归档(commit `5ebac76`) |
| API 服务（Web 联调） | `e2e_pipeline/api_server.py` + `API_README.md` | FastAPI + sse-starlette，POST mp4 → SSE 推 frame/alert 事件。对齐 WEB-HANDOFF §6.3。端口默认 8000，GPU 显存限制下串行跑 |
| Pipeline 回调钩子 | `InferencePipeline(on_frame_callback=...)` + `_emit_frame_callback` | `_process_frame` 末尾调用回调，传每 track 的 label/conf/bbox_xyxy。API server 转换成 WEB frame 事件 |
| 撞墙 / 头撞墙探查 | `probe_wall_impact.py` | R24 Probe：YOLO-Pose+ByteTrack 跑给定样本，输出速度 / 滑窗 std / 自相关 / 姿态 / 场景上下文时序 + 跨视频叠加图，用于数据驱动定阈值。`--with-posec3d` 可选叠加 5 类概率对照 |
| 镜头遮挡 / 黑屏 / 失焦 | `e2e_pipeline/scene_event_detector.py` | R26 P37：scene-level Canny EDR + 亮度 + 拉普拉斯方差三通道检测。pipeline `_process_frame` Step 0 短路调用，触发即跳过 YOLO/PoseC3D/规则。逻辑源自根目录 `step5_camera_tampering.py`（保留为独立 CLI 参考实现） |

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
34. **pipeline 内同一语义量必须单一来源**：R12 发现 PoseC3D 用 SkeletonBuffer 视角（grace 期 2 人），rule_engine 用本帧 YOLO 视角（被遮挡变 1 人），导致 vandalism 误判。修理由的"场景人数"类语义，必须有 single source of truth — 任何跨模块的人数/状态判定都应从同一份数据源推导，否则各模块"自以为"的场景会互相冲突
35. **遮挡不等于消失**：被遮挡的人物理上还在场景里只是不被看见。grace_frames 机制保留了骨骼 buffer 但没同步到"场景人数"这类派生量。任何"当前帧检测结果"类统计都要考虑遮挡兼容—— 默认用 buffered 视角，只有姿态/位置这类必须精确的才用当前帧
36. **性能 gating 必须 per-track 而不是场景级**：R11 的"任一 track 在攻击/倒地态就关掉 smoking/phone YOLO"设计把独立事件耦合在一起—— T1 抽烟 vs T3 倒地被同一个 gate 同时切断。"物理互斥"是 per-track 语义（一个人不会同时做两件事），不该跨 track 放大成场景级。而且 YOLO 是帧级调用，per-track gating 在实现上也节省不了算力，干脆每帧都跑
37. **稀疏脉冲信号和持续姿态信号需要不同的时序策略**：fighting/bullying/falling 是持续姿态，5 帧窗口 + HOLD=1 够用；smoking/phone 是小物体 + 间歇可见（烟被遮挡、手放下），需要更大窗口（7+）和更宽的 HOLD（2+）。不能一套 VOTE 参数套所有标签 —— 行为的物理节奏不同
38. **判据关键点要覆盖行为的多种姿势，不只是"典型姿势"**：check_smoking 原设计只查鼻子/手腕（"拿烟在嘴边"姿势），漏掉"手肘弯起在胸前"、"烟垂在身侧"等常见姿势。设计关键点时要枚举该行为的多种常见变体（文化/习惯差异），而不是选一个最典型的姿势

---

## 附录：Git 提交链（E2E 关键节点）

```
839d829 revert(e2e): R31 P50 恢复 rule_bullying 白名单 (删除后效果更差)           (R31 revert)
0043565 fix(e2e): R31 P50 收窄白名单 — 删 rule_bullying                          (R31 hotfix, reverted)
7c88c49 feat(e2e): R35 bullying 角色区分 perpetrator/victim                      (R35)
8796036 fix(e2e): R34 P28 高conf豁免加 normal<0.9 门槛                           (R34)
5342107 fix(e2e): R33 P7 兜底坐姿 veto 去掉 normal>=0.25 门槛                   (R33)
16d7a99 fix(e2e): R32 暂时禁用 self_harm 判定逻辑 (误报过多)                     (R32)
d84e572 fix(e2e): R31 P49+P50 demote 放宽 partner + pair-based source 白名单 (R31)
e8b1f77 fix(e2e): R30 P48 self_harm raw normal 显著主导 veto (P42 灰度带扩展)  (R30)
a55d5d2 fix(e2e): R29 P47 _is_upright_posture 二维投影盲区双层防御            (R29)
54c7d8c fix(e2e): R28 P46 self_harm B 路径双绝对下限 (hip_max 分母失控修复)   (R28)
ff91a5b fix(e2e): R27 P41-P45 self_harm 误报压制 (坐姿头转动 / 簇发 / kp门槛)  (R27)
4ced812 fix(e2e): R26 P40 draw_label 自适应位置                                (R26)
a319fe4 feat(e2e): R26 P37-P39 camera_tampering scene-level 检测             (R26)
da4e546 feat(e2e): R25 self_harm 落地 — 撞墙/扶墙撞头 skeleton 速度双路径       (R25)
8b3c749 fix+feat(probe): R24 二轮 · 修 self-exclusion bug + 加新指标            (R24)
f056547 feat(probe): R24 probe_wall_impact.py for self-harm feature exploration (R24)
2648bc5 revert(e2e): R23 P31 (oversuppressed real lying via PoseC3D static-pose blind spot)
165a0c4 fix(e2e): R23 P31 PoseC3D strong-normal reverse-veto YOLO conf bypass (R23, reverted)
442331b fix(e2e): R22 P30 extend P22 STRONG_NORMAL posec3d HOLD exit to falling/climbing (R22)
5c18eac fix(e2e): R21 P27+P28 head-vs-hip + YOLO-conf bypass for lying-toward-camera (R21)
4a69675 fix(e2e): R20 P25+P26 target strong-normal guard in couple+inject    (R20)
67f40f4 fix(e2e): R19 P24 lower-body-missing veto in check_fallen_by_yolo    (R19)
0f7c1b3 fix(e2e): R18 P22+P23 kill single-person fighting                    (R18)
0655144 fix(e2e): R17 P8 otoriel door add proximity precondition              (R17)
6d3af28 fix(e2e): R13 smoking detection — P15/P16/P17                          (R13)
b4d389d fix(e2e): R12 occlusion-aware scene_person_count — P13/P14            (R12)
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
