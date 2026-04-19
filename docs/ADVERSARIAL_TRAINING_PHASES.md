# 对抗学习训练阶段说明（1v1 主赛道）

## 总览
- 主赛道：标准 `1v1`。
- 动作空间：`discrete6`（`0..5`，其中 `5=drop_bomb`）。
- 迭代顺序：`Phase 0 -> Phase 1 -> Phase 2 -> Phase 3`。
- 当前启动范围：仅执行 `Phase 0`，不直接进入 PPO/自博弈长跑。

## 工程约定
- 样本协议（JSONL）核心字段：
  - `state.state_map`: `13x15x10`
  - `state.state_vector`: `16`
  - `action`: `0..5`
  - `action_mask`: 长度 `6`
  - `episode_id`, `agent_id`, `opponent_id`, `outcome_tag`
- 报告统一落地：`output/ml/reports/combat_phase0_*.json`
- 模型统一输出：`output/ml/models/combat_phase0_iql_v1.{pt,onnx}`

## Phase 0（离线冷启动）
### 目标
- 用离线数据快速获得可用战斗策略初始化。
- 重点建立 `drop_bomb` 的基本可用性与动作合法性约束。

### 数据
- 规模：`20万~100万` 帧。
- 场景：`1v1`。
- 行为源：增强 heuristic（含放泡、抢道具、封路），可叠加人类回放。
- 风车爱心图 Phase 0 特化：
  - 每局开始执行 `100%清图`：仅保留刚性障碍，清空全部非刚性物体。
  - 随机道具回填：按密度上限补齐，并保证出生点最小安全半径。
  - 禁止复制写入：按 `unique transition` 去重，目标真实 `20万` 帧。
  - 终局字段强制写入：`done / pre_death / outcome_tag(win/loss/draw/self_kill)`。

### 训练
- 主线：IQL（可切 BC）。
- 网络：`10` 通道地图 + `16` 维向量，`ACTION_DIM=6`。
- 迁移：加载 `dodge_iql_v1.pt` 卷积层，前 `50` epochs 冻结。
- 时序：启用 `GRU(256)`。
- actor loss：应用 `action_mask`，抑制非法/必死动作。
- 训练信号修正：
  - 取消每 epoch 截断，恢复完整 epoch 覆盖。
  - 终局重权：强化 `self_kill`、`timeout_draw`、`successful_trap` 片段。
  - 追加对抗辅助特征（用于重权与质量分析）：
    - `post_bomb_escape_steps(<=2/<=4/<=6)`
    - `min_escape_eta_after_bomb`
    - `corridor_deadend_depth`
    - `enemy_recent_bomb_cd`
    - `enemy_heading_delta`
    - `item_race_delta`
    - `local_choke_occupancy`

### 启动命令（本轮执行）
1. ```bash
   node scripts/collect-combat-dataset.js --target-frames=200000 --arena=1v1 --map=windmill-heart --action-space=discrete6 --fresh=1 --clear-nonrigid=1 --random-item-density=0.12
   ```
2. ```bash
   python3 ml/train_iql_combat.py --dataset output/ml/datasets/combat_phase0_v1.jsonl --epochs 60 --batch-size 512 --freeze-conv-epochs 50 --init-pt output/ml/models/dodge_iql_v1.pt --out-pt output/ml/models/combat_phase0_iql_v1.pt --out-onnx output/ml/models/combat_phase0_iql_v1.onnx
   ```
3. ```bash
   node scripts/eval-combat-1v1.js --model-url=/output/ml/models/combat_phase0_iql_v1.onnx --opponent=heuristic_v2 --runs=200
   ```
   评估固定口径：`45s/局 + 200局 + windmill-heart + heuristic_v2`。

### 验收阈值
- `action=5` 召回率 `>= 0.35`（验证集）。
- 离线 `val_policy_f1 >= 0.55`。
- 训练无 NaN。
- 非法动作预测率 `<= 0.5%`。
- 在线 1v1（固定 heuristic）：
  - `win_rate >= 35%`
  - `self_kill_rate <= 25%`

### 进入 Phase 1 门槛
- 以上所有阈值全部达标。
- 连续两次评估（同配置）波动不超过 `±3pp`。

### 回滚策略
- 若 `action=5` 召回低于阈值：优先扩充高压放泡片段并重采样动作分布。
- 若非法动作率超阈值：优先检查 `action_mask` 生成与训练掩码应用。
- 若在线胜率不达标：保持离线主干，先做数据增量（不直接切 PPO）。

## Phase 1（PPO 固定对手）
### 目标
- 将离线初始化策略转为在线可稳定收敛策略。

### 训练设置
- 算法：`PPO-LSTM`。
- 固定对手：Phase 0 最优模型或强规则 bot。
- 奖励：保留现有 shaping，新增：
  - 超时平局惩罚 `-0.2`
  - 卡死小惩罚（长时间无效行为）

### 验收阈值
- 对固定对手 `win_rate >= 90%`（连续 3 轮达标）。
- `self_kill_rate <= 12%`
- `draw_rate <= 25%`

## Phase 2（PBT/FSP 自博弈）
### 目标
- 提升博弈上限并抑制灾难性遗忘。

### 训练设置
- 对手池：`80% latest + 20% history`。
- 奖励退火：训练进度 `30% -> 70%` 线性衰减到零和。
- Elo 看板：checkpoint 对固定评测池 round-robin。

### 验收阈值
- 相对 Phase 1 基线 Elo 提升 `>= 150`。
- 对历史池最低胜率 `>= 45%`。

### Elo 口径
- 统一用固定评测池、固定地图与局时。
- 汇总报告包含：`win_rate / draw_rate / self_kill_rate / item_control / elo_result`。

## Phase 3（微操专精）
### 目标
- 专项强化高压出生与极限脱险微操。

### 训练设置
- `30%` 对局注入高压出生场景。
- near-miss 奖励先开后关：前期 `+2.0`，稳定后衰减到 `0`。

### 验收阈值
- 高压场景存活率较 Phase 2 提升 `>= 20%`。
- 半身位有效利用率提升且总体 `win_rate` 回退不超过 `5%`。

## 测试计划
### 单元测试
- 动作编码/解码（含 `5=drop_bomb`）一致性。
- `action_mask` 屏蔽非法放泡与必死动作。
- `state_map/state_vector` 新旧维度兼容。

### 集成测试
- `collect-combat-dataset` JSONL 合约检查（字段齐全、shape 固定、action 合法）。
- 数据合同额外检查：`rows=200000`、复制率接近 `0`、`done_ratio>0`、`outcome_tag` 非单值。
- ONNX 推理输出动态 `action_dim`（5/6）兼容。

### 回归测试
- 旧 `dodge_iql_v1.onnx` 在 battle 模式可继续运行（不触发越界放泡）。
- 躲泡既有评估基线不显著回退（survival 保持）。
