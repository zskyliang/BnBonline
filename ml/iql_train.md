# IQL 纯模型躲泡 V1 训练文档

## 1. 目标与口径
- 目标：训练并接入纯 IQL 无规则兜底躲泡模型（`policy_mode=pure`）。
- KPI：`survival_rate = 1 - bombed_count / spawned_bubbles_effective`，目标 `>= 0.98`。
- 首轮数据：`200k` 帧，混合比例目标 `expert:negative = 6:4`（`negative=random+epsilon`）。
- 轮次推进：Round A `200k`，若未达标依次扩展到 `400k/600k/900k`。

## 2. 训练脚本与数据集
- 训练脚本：`ml/train_iql.py`
- 默认数据集：`output/ml/datasets/dodge_iql_v1_mixed_200k.jsonl`
- 模型产物：
  - `output/ml/models/dodge_iql_v1.pt`
  - `output/ml/models/dodge_iql_v1.onnx`
  - `output/ml/reports/iql_v1_metrics.json`

## 3. 已提取的真实训练样本（来自 200k 混合集）
- 原始单行 JSONL：[`iql_sample_row_raw.jsonl`](/Users/slzeng/Documents/work/vibe/game/BnBonline/output/ml/reports/iql_sample_row_raw.jsonl)
- 可读预览（裁剪为 2x2 局部）：[`iql_sample_row_preview.json`](/Users/slzeng/Documents/work/vibe/game/BnBonline/output/ml/reports/iql_sample_row_preview.json)

样本摘要（`S1`）：
- `id=S1`
- `action=0(wait)`
- `reward=0.03`
- `done=false`
- `policy_tag=expert`
- `risk_label=0`
- `state_map shape=13x15x8`
- `state_vector=[0,0,0,0,0,0,0,1,0]`

预览样本：
```json
{
  "id": "S1",
  "ts": 1776557267793,
  "action": 0,
  "reward": 0.03,
  "done": false,
  "policy_tag": "expert",
  "risk_label": 0,
  "state_vector": [0, 0, 0, 0, 0, 0, 0, 1, 0],
  "next_state_vector": [0, 0, 0, 0, 0, 0, 0, 1, 0],
  "state_map_top_left_2x2": [
    [[0,0,0,0,0,0,1,0], [0,0,0,0,1,0,1,0]],
    [[0,0,0,0,1,0,1,0], [0,0,0,0,1,0,1,0]]
  ],
  "next_state_map_top_left_2x2": [
    [[0,0,0,0,0,0,1,0], [0,0,0,0,1,0,1,0]],
    [[0,0,0,0,1,0,1,0], [0,0,0,0,1,0,1,0]]
  ]
}
```

## 4. 数据字段口径（按 `train_iql.py` 实现）
- `action`：离散动作，合法范围 `[0..4]`，非法样本会被过滤。
- `state_map`：目标维度固定 `13x15x8`。
- `state_vector`：目标维度固定 `9`。
- `policy_tag`：`expert|random|epsilon`，内部映射为 `0|1|2`。
- `risk_label`：若样本缺失该字段，按规则自动推断。
- 训练使用字段：`state/action/reward/done/next_state/risk_label/policy_tag`。

### 4.1 state_map 通道定义（V1）
- `C0` 障碍层
- `C1` 泡泡倒计时层
- `C2` danger ETA 层
- `C3` AI 位置层
- `C4` 安全自由度层
- `C5` Blast Map（未来爆炸十字覆盖）
- `C6` Reachability Map（BFS 可达且时序安全）
- `C7` Half-body Safety Layer（半身安全边界）

### 4.2 兼容旧数据（脚本内自动补齐）
当输入不是 8 通道时，脚本会补齐到 8 通道：
- `<6` 通道时：`C5 <- C2`（blast surrogate）
- `<7` 通道时：`C6 <- (C4 > 0.5)`（reachable surrogate）
- `<8` 通道时：`C7 <- clip(abs(roll(C2,left)-roll(C2,right)))`（half-body surrogate）

### 4.3 state_vector 定义（9维）
`[dx, dy, left_foot_unsafe, right_foot_unsafe, left_foot_eta_norm, right_foot_eta_norm, center_eta_norm, safe_neighbors_norm, active_bombs_norm]`

兼容逻辑：
- 旧向量不足 9 维时自动补零。
- 若缺 `safe_neighbors_norm`，用 `meta.safeNeighbors / 4` 回填到第 8 维。
- 若缺 `active_bombs_norm`，用 `min(1, meta.activeBombs / 10)` 回填到第 9 维。
- 向量最终 clip 到 `[-1, 1]`。

### 4.4 risk_label 推断规则
样本未提供 `risk_label` 时，满足任一条件视为高风险（1）：
- `pre_death=true`
- `done=true`
- `reward < -0.5`
- `meta.activeBombs > 0 && meta.safeNeighbors <= 1`
否则为 0。

## 5. 模型结构（离散 IQL）
- 共享编码器：`Conv(8->32) + Conv(32->64) + Flatten + concat(state_vector) + FC(256)`。
- 头部：
  - `Q1/Q2`：双 Q，输出 5 动作 Q 值
  - `V`：状态值
  - `Policy`：离散 logits
  - `Risk`：二分类 logit

## 6. 损失函数（`train_iql.py`）
- Q 回归：`q_target = r + gamma * (1-done) * V_target(s')`
- V 回归：IQL expectile loss（默认 expectile=0.7）
- Actor：advantage-weighted BC（`exp(beta * adv)`，含上限裁剪）
- Risk：BCEWithLogits
- 总损失：`q_loss + v_loss + actor_loss + risk_loss_weight * risk_loss`

默认超参（脚本参数）：
- `epochs=10`
- `batch_size=512`
- `lr=3e-4`
- `gamma=0.99`
- `expectile=0.7`
- `beta=3.0`
- `adv_max_weight=80`
- `risk_loss_weight=0.35`
- `target_tau=0.01`
- `sampler=balanced`（支持动作与 policy_tag 重加权采样）

## 7. 训练命令
```bash
python ml/train_iql.py \
  --dataset output/ml/datasets/dodge_iql_v1_mixed_200k.jsonl \
  --epochs 10 \
  --batch-size 512 \
  --sampler balanced \
  --out-pt output/ml/models/dodge_iql_v1.pt \
  --out-onnx output/ml/models/dodge_iql_v1.onnx \
  --out-metrics output/ml/reports/iql_v1_metrics.json
```

## 8. 验收与排查
数据正确性检查：
- `state_map` 固定 `13x15x8`
- `state_vector` 固定 `9` 维，归一化范围合理
- `policy_tag` 分布接近 `6:4` 目标
- `risk_label` 正负样本不塌缩
- `action` 全部在 `[0..4]`

训练正确性检查：
- 训练过程无 NaN
- `val_policy_f1`、`val_risk_acc` 持续可用
- ONNX 可加载，输出 `policy_logits/risk_logit`

对战口径检查（pure 模式）：
- `rule_calls_mean` 应接近 `0`
- `pure_violation_count` 应为 `0`
- `fallback_rate` 在纯模式应接近 `0`
