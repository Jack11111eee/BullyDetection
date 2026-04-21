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
 │   ├─ R19 P24 内聚否决：下半身 kp 13-16 全缺 AND 上半身 kp 0-10 ≥5 可见
 │   │   → check_fallen_by_yolo 返回 False（整个 step 2 跳过，当作无 YOLO falling 信号）
 │     ├─ R10 P8 否决门（R17 P21 加 proximity 前置）：
 │     │   proximity_ok AND fighting_prob ≥ 0.7 AND bullying_prob < f*0.3
 │     │   → 整个 step 2 跳过（YOLO 视为误检）
 │     │   孤立场景（proximity 失败）即使 fighting≥0.7 也不否决（视为 PoseC3D 噪声）
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

