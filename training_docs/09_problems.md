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

#### Problem 28: 遮挡导致 fighting → vandalism 误判

- **发现**：E2E Fix Round 12（用户报告）
- **详情**：fighting 中一方站在画面前方持续遮挡另一方，被遮挡者的 YOLO-Pose 骨骼被挤压→丢失。`frame_persons_kps` 只有 1 人。PoseC3D 从 SkeletonBuffer grace 期 buffer 取骨骼仍输出 fighting → step 5 `check_vandalism` 条件满足（`fighting_prob>0.5 AND len(all_person_kps)==1`）→ fighting 误判为 vandalism
- **根因**：pipeline 内两套"场景人数"标准不一致
  - PoseC3D：看 SkeletonBuffer（grace=90 帧保留），视野 = 2 人
  - rule_engine：看 `frame_persons_kps`（仅本帧 YOLO 检测），视野 = 1 人
- **修复**：R12 两改动 P13+P14（commit `b4d389d`）
  - P13：pipeline 计算 `scene_person_count = |current_tracks ∪ skeleton_buf.tracks|`，通过 `RuleEngine.push_scene_count()` 推送
  - P14：`check_vandalism` 改用 `scene_person_count`；新增持续性门槛 —— 近 5 次推理场景人数全=1 才触发
- **状态**：已修复（待视频验证）

#### Problem 29: 持续抽烟只检出一小部分（gating 误关 + VOTE 对稀疏信号过严 + 关键点范围窄）

- **发现**：E2E Fix Round 13（用户报告 + debug.log F831–F1263 分析）
- **详情**：持续抽烟视频里 T1 的 smoking 只有小部分帧被识别。三条独立病灶叠加：
  1. **R11 场景级 gating 错误实现**：T3 被 laying YOLO 误判 falling → `[YOLO-GATE] ran=['falling'] skipped=['smoking', 'phone']` 从 F943 持续到 F1135（约 180 帧 / 6 秒），T1 smoking 模型**完全未运行**，RAW 没有任何 smoking 信号输入
  2. **VOTE 窗口对稀疏信号过严**：gate 开时 T1 RAW smoking 命中率约 4/7，但 5 窗口 HOLD=1 的默认滞回对脉冲式检测不友好，反复进出态
  3. **check_smoking 关键点范围窄**：只查鼻子(0)/左腕(9)/右腕(10)，"手肘弯起拿烟"(kp 7/8) 或香烟位置特殊时漏检
- **证据**：
  ```
  F943: [YOLO-GATE] ran=['falling'] skipped=['smoking', 'phone']
  F975: T1 FINAL → smoking (HOLD 1>=1, 最后一次 HOLD)
  F991–F1135: T1 FINAL → normal (smoking 模型未跑 + history 清空)
  F1151: T1 FINAL → normal (smoking 1 次 < ENTRY 2)
  F1167: T1 FINAL → smoking (ENTRY 2，终于重新入态)
  ```
- **根因（三处独立）**：
  - 场景级 gating ≠ per-track 物理互斥；T1 抽烟 vs T3 倒地不该耦合
  - 窗口+HOLD=1 对"脉冲信号"（每 1-2 秒检出一次）不够宽
  - 关键点清单按"成年人拿烟姿势"设计，覆盖面窄
- **修复**：R13 三改动 P15+P16+P17（commit `6d3af28`）
  - P15：撤销 R11 gating，smoking/phone/falling 固定每帧都跑
  - P16：关键点扩到 `[0, 7, 8, 9, 10]` + 骨骼 bbox 上半身兜底
  - P17：smoking/phone 专用 VOTE 窗口=7, ENTRY=2, HOLD=2
- **状态**：已修复（待视频验证）
- **未处理**：T3 的 laying YOLO 在竖直姿态下误判 falling（PoseC3D `normal=1.000 falling=0.000` 但 laying 持续输出 bbox conf=0.3+），本轮未动 —— 候选修复：`check_fallen_by_yolo` 加 `normal_prob ≥ 0.9` 否决门

#### Problem 30: 孤立摔倒被 P8 否决门吞掉（PoseC3D fighting 噪声 + R10 一刀切）

- **发现**：E2E Fix Round 17（用户报告 + debug.log F15228/F15244/F15260 分析）
- **详情**：单人摔倒视频中 T47 PoseC3D 输出 `fighting=0.775, falling=0.055`（R11 残留 normal↔fighting 互混噪声），proximity 失败（nearest=765 > 210，画面里就一个人）。但 R10 P8 否决门只看 PoseC3D fighting 强主导，不看 proximity → 否决整个 step 2 → YOLO falling bbox 完全吞掉（连 deferred 都不设）→ FINAL=normal。下一帧 fighting 跌到 0.674 才走 P7 deferred 兜底，再隔 16 帧才达 ENTRY=2，FINAL 才变 falling
- **证据**：
  ```
  F15228: fighting=0.775 → P8 否决 → step 2 跳过 → argmax=fighting + proximity失败 → normal
  F15244: fighting=0.674 → P8 不触发 → P7 deferred → falling，但 ENTRY 1<2 → normal
  F15260: fighting=0.624 → P7 deferred → falling，ENTRY 2>=2 → falling ✓
  ```
- **根因**：R10 P8 设计场景（F546-F627 对称对打自激）proximity 必定 ok，作者写规则时默认 fighting 高就一定有对象。但孤立场景下 fighting 信号物理上不可能成立，是 PoseC3D 假阳性，不该用其否决任何东西
- **修复**：R17 P21 — `yolo_veto` 加 `proximity_ok_pre` 前置（commit `0655144`）。step 2 用 step 5 同样的公式（`max(my_h, neighbor_h) * 1.5`，fallback `img_h * 0.25`）算一次 proximity，只有 proximity_ok 时才允许 P8 否决
- **状态**：已修复（待视频验证）

#### Problem 31: 孤立 fighting — 旁观者被 pair inference + HOLD 锁死

- **发现**：E2E Fix Round 18（用户报告 + debug.log F7210/F7215/F7219 分析）
- **详情**：拉扯场景中 T3 只是坐着旁观，但 FINAL=fighting；同帧 T11（施暴）、T26（受害）也 FINAL=fighting 但全靠 HOLD 硬撑，无一帧 RAW=fighting 进入
- **证据**：
  ```
  F7210 T3 PoseC3D normal=1.000 fighting=0.000 → RAW=normal
        history=[N,F,N,F,N,F,N] → VOTE HOLD 2>=2 → fighting ✗
  F7215 T11 PoseC3D fighting=0.889 但 proximity失败 nearest=347>270
        → RAW=normal → VOTE HOLD 2>=2 → fighting
  F7219 T26 PoseC3D fighting=0.883 proximity失败 → HOLD → fighting
  ```
- **根因（4 条污染路径）**：
  - **P-1 pair inference RAW 污染**：T3 被 `_find_nearest_track` 选 T11 为 pair → PoseC3D (T3, T11) 2-person tensor 被 T11 动作主导 → 输出 fighting → 写入 T3.history
  - **P-2 `_inject_to_overlapping_neighbor`**：T11 步骤 5 攻击判定时会主动向 bbox overlap 最高的邻居（若为 T3）注入 fighting
  - **P-3 `couple_overlapping_pairs`**：T11 smoothed=fighting + T3 overlap → couple 升级 T3 为 fighting + 写 `_last_smoothed[T3]=fighting`
  - **P-4 VOTE HOLD 锁死**：T3.history 有 ≥ 2 fighting entries + HOLD_MIN=2 → 每帧维持 fighting，即使当前 RAW=normal=1.000 也救不回来
  - 四条路径都违反用户原则 "fighting/bullying ≥2 人 + 无孤立行为"
- **修复**：R18 P22 + P23（commit `0f7c1b3`）
  - **P22 (Solution B)**：`_vote_smooth` 加 STRONG_NORMAL HOLD exit — 当 raw_source=posec3d + normal_prob≥0.9 + 最近 3 帧 raw 至少 2 帧 normal → 强制退 fighting/bullying HOLD
  - **P23 (Solution E)**：post-couple 新方法 `demote_unsupported_attacks` — 要求 (partner: 另一 track 在攻击态且 bbox overlap≥10%) AND (self_active: 近 3 帧 raw 含攻击 OR 当前 attack_prob≥0.3)，任一失败即降级 normal
  - 新增 `_last_pose_probs` 缓存 + migrate/clear 同步清理
- **状态**：已修复（待视频验证）

#### Problem 32: 坐姿手臂外展被 YOLO falling 误判（_is_sitting_posture 纵横比失效）

- **发现**：E2E Fix Round 19（用户报告 + debug.log F3385 分析）
- **详情**：T2 坐着手臂外展 → YOLO laying 模型触发 falling bbox + bbox=水平 → 规则引擎返回 falling。但 PoseC3D `normal=0.998` 明确说 normal
- **证据**：
  ```
  F3385 T2 PoseC3D normal=0.998 fighting=0.000 falling=0.002
        RAW → falling (YOLO辅助检测, conf=0.746, bbox=水平)
        VOTE HOLD 5>=1 → falling ✗
  ```
- **根因**：R15 P18 / R16 P20 两处坐姿 veto 都只用 `_is_sitting_posture`（骨骼 h/w > 1）一个判据。手臂外展时 w 被肘/腕 kp 拉开 → h/w < 1 → 判据失效。没有第二道独立判据兜底
- **修复**：R19 P24 `check_fallen_by_yolo` 内部加**下半身缺失否决**（commit `67f40f4`）
  - 躺地的人身体横向展开 → 膝/踝（kp 13-16）至少一个落入可检测范围
  - 坐在桌/椅后或画面边缘的人下半身被裁 → 只剩上半身
  - 条件：`lower_kp_visible==0 AND upper_kp_visible>=5` → 返回 `is_fallen=False`
  - 位置内聚到 `check_fallen_by_yolo` 内部，一次修好所有 YOLO falling 出口（还清 R16 记录的设计债）
- **状态**：已修复（待视频验证）

#### Problem 33: 摄像头假重叠导致旁观者被误标 fighting（couple/inject 无 target 守门）

- **发现**：E2E Fix Round 20（用户报告 + debug.log F959 + 视频截图分析）
- **详情**：柜台场景 T1 走过前景被模型误判 fighting=0.906，摄像头俯拍角度使 T1 bbox 和后面坐在柜台后的 T2、T3 物理重叠（0.11 / 0.30），但 T2/T3 PoseC3D 都输出 `normal=1.000`。T3 仍被标 FINAL=fighting
- **证据**：
  ```
  F959 T1 [INJECT] T1→T3 (overlap=0.30) fighting
        T3 history=[n,n,n,n,n,n,fighting]  ← INJECT 污染
  couple: T2/T3 都被升级 fighting
  demote: T2 被降级（recent_raw 干净）
          T3 未降级（recent_raw 含 INJECT 写入的 fighting）
  FINAL: T1/T2/T3 = fighting/normal/fighting ✗
  ```
- **根因**：
  - `_inject_to_overlapping_neighbor` 无条件注入（P-2 污染路径仍在）
  - `couple_overlapping_pairs` 仅看 bbox overlap，不看 target 自身 PoseC3D 证据
  - R18 P23 E 的 `self_active` 不区分 history 里的 raw 是 PoseC3D 自推还是 INJECT 伪造 → 被污染绕过
  - 摄像头 3D 纵深 / 俯拍场景，bbox 物理重叠不等于行为互动
- **修复**：R20 P25 + P26（commit `4a69675`）— 源头守门，不改 E
  - **P25 couple**：target 自身 `normal_prob ≥ 0.9` 时跳过升级
  - **P26 inject**：target 自身 `normal_prob ≥ 0.9` 时拒绝注入
  - 级联：T3 既不被注入也不被升级 → attack_tids=[T1] → T1 partner=None → E 自动降级 T1 → 单人 fighting 消灭
  - 副作用：顺带修好每帧反复 COUPLE/DEMOTE 的日志污染
- **状态**：已修复（待视频验证）

#### Problem 34: 头朝相机躺地被骨骼 h/w 单一判据误判为坐姿

- **发现**：E2E Fix Round 21（用户报告 + debug.log F2284 分析）
- **详情**：T7 朝相机方向躺倒（头在图像下方、脚在上方），骨骼包围框 h/w=1.37 > 1.1 触发坐姿 veto → FINAL=normal。PoseC3D 同时输出 normal=0.443 恰好过 ≥ 0.25 阈值，veto 三重门槛全满足
- **证据**：
  ```
  F2284 T7 PoseC3D normal=0.443 fighting=0.397 falling=0.145
        [RAW] T7 YOLO躺地但PoseC3D有攻击信号 → 暂存
        [RAW] T7 proximity失败 → 落到argmax路径
        [RULE] h/w=1.37 > 1.1 → 非倒地(坐姿/站姿)
        [RAW] T7 P7兜底YOLO falling 被坐姿否决 (valid_kp=17, normal=0.443)
        [FINAL] normal ✗
  ```
- **根因**：R15 P18 / R16 P20 / R19 P24 三处 YOLO falling sitting veto 都只用 `h/w > 1.1` 一个判据。头朝相机方向躺地时身体沿图像 Y 轴展开 → h/w > 1 → 判据失效。继 R19 P24 "手臂外展 w 变大" 之后第二次翻同一个判据盲区。另外 head-away 躺地是 2D 投影本征盲区，骨骼启发式无解
- **修复**：R21 P27 + P28（commit `5c18eac`）
  - **P27**：`_is_sitting_posture` 内聚 head-vs-hip 二层校验 — `head_hip < -h*0.02` 时翻转返回 False（头在髋下方 → 躺倒）
  - **P28**：P18 主路径 + P20 P7 兜底加 YOLO conf 豁免门 — `conf ≥ 0.6` 时跳过 sitting veto（头朝远处躺地 2D 盲区兜底）
  - 日志：3 处 sitting_veto 路径日志补 `conf={x:.3f}`
- **状态**：已修复（待视频验证）

#### Problem 35: 站起后 PoseC3D normal=0.934 被 falling HOLD 锁死（R18 P22 漏补 falling/climbing）

- **发现**：E2E Fix Round 22（用户报告 + debug.log F1439 分析）
- **详情**：T1 站起来，PoseC3D 极度明确 `normal=0.934`，但 VOTE HOLD 把状态保留在 falling
- **证据**：
  ```
  F1439 T1 PoseC3D argmax=normal(0.934)
        [RAW] T1 → normal (source=posec3d)
        [VOTE] T1 current=normal → falling (HOLD 1>=1)
               history=[falling,normal,normal,normal,falling,normal,normal]
        [FINAL] falling ✗
  ```
- **根因**：两套 STRONG_NORMAL HOLD exit 有覆盖缺口
  - R15 P19：接 last ∈ (falling, climbing) 但只认 `rule_*` source（rule_upright 等），不接 `posec3d`
  - R18 P22：接 `posec3d` source 但 last 只 (fighting, bullying)，不含 falling/climbing
  - F1439 正好落在交集外：last=falling + source=posec3d，两层都不开门 → 锁住
- **修复**：R22 P30（commit `442331b`）— 把 P22 的 last 接受集从 `(fighting, bullying)` 扩到 `(fighting, bullying, falling, climbing)`；其他三重防误退条件（`pose_normal_prob≥0.9` + `recent3≥2 normal`）全部保留
- **状态**：已修复（待视频验证）
- **教训**：R18 写 P22 时只针对 F7210 pair-inference 污染攻击态场景，没有扩到所有"异常态 HOLD"。P19 和 P22 本质上是同一类机制（强 normal 退 HOLD）的两个互补分支（source 不同、防误退策略不同），分头写就漏跨交集 case。以后新增 HOLD exit 应先穷举 (raw_source, last) 两维组合表

#### Problem 36: 视频边缘 bbox 让 YOLO 高 conf 误检吞掉 PoseC3D 强 normal（R21 P28 风险兑现）

- **发现**：E2E Fix Round 23（用户报告 + debug.log F1679 / F1695 分析）
- **详情**：T3 位于视频边缘，骨骼 noisy。YOLO laying 对截断/边缘 bbox 输出 conf=0.85 horizontal bbox → R21 P28 豁免门放过 sitting veto → 直接返回 falling。但 PoseC3D 看完整 17 keypoint 时序输出 `normal=0.998` 极度确信
- **证据**：
  ```
  F1679 T3 PoseC3D normal=0.998
        [RAW] YOLO conf=0.853>=0.6 → 高置信豁免 → falling
        history=[n,n,n,n,n,n,falling] ENTRY 1<2 → normal
  F1695 T3 PoseC3D normal=0.998
        [RAW] YOLO conf=0.849>=0.6 → 高置信豁免 → falling
        history=[n,n,n,n,n,falling,falling] ENTRY 2>=2 → falling ✗
  ```
- **根因**：R21 P28 的 YOLO 高 conf 豁免无仲裁逻辑。风险笔记预判到了 —— "若视频测试暴露，可调高阈值到 0.7 或加 normal_prob ≥ 0.9 反向门"。F1679 / F1695 直接把这条风险兑现
- **修复尝试**：R23 P31（commit `165a0c4`）— **已回退 `2648bc5`**
  - 改动：P18 主路径 + P20 P7 兜底对称加 YOLO conf≥0.6 豁免内嵌 PoseC3D `normal_prob ≥ 0.9` 反向 veto；新 source `rule_strong_normal_veto_yolo` 加入 `STRONG_NORMAL_SOURCES`
  - **回退原因**：视频验证发现正常倒地被识别为 normal。R23 假设"PoseC3D normal≥0.9 = 真 normal" 与 Problem 11 的设计前提冲突 —— PoseC3D 对静态躺地有结构性盲区（无时序变化 → 输出 normal 甚至 ≥0.9），这正是 R6 引入 YOLO falling 的全部理由。R23 把"PoseC3D 静态盲区"当成"YOLO 误检"反向 veto，把真倒地的合法 YOLO 信号给吞了
- **状态**：未修复（R23 尝试已回退，等待新方案）
- **教训 1**：两个子模型高置信冲突的仲裁规则不能一刀切给某一方。R23 的"信任 PoseC3D 时序"思路忽略了 PoseC3D 的已知盲区 —— Problem 11 的存在本身就证明 PoseC3D 在某些 case 下不可信，反向 veto 不能拿它当裁判
- **教训 2**：解 F1679/F1695 应该针对"边缘 bbox YOLO 误检"特征而非"PoseC3D 强 normal"全局判据。候选方向：YOLO bbox 触边检测、骨骼总分均值低 → 降 YOLO 可信度、跨帧连续性检查
- **未处理**：F1679/F1695 仍是开问题，下一 Round 重新设计

---

