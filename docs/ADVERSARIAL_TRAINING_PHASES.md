# 对抗学习训练阶段说明（1v1 主赛道）

## 总览
- 主赛道：标准 `1v1`。
- 动作空间：`discrete6`（`0..5`，其中 `5=drop_bomb`）。
- 迭代顺序：`Phase 0 -> Phase 1 -> Phase 2 -> Phase 3`。
- 当前启动范围：仅执行 `Phase 0`，不直接进入 PPO/自博弈长跑。

## 工程约定
- 样本协议（JSONL）核心字段：
  - `state.state_map`: `13x15x10`
  - `state.state_vector`: `32`
  - `action`: `0..5`
  - `action_mask`: 长度 `6`
  - `episode_id`, `agent_id`, `opponent_id`, `outcome_tag`
- 报告统一落地：`output/ml/reports/combat_phase0_*.json`
- 模型统一输出：`output/ml/models/combat_phase0_iql_v4_suddendeath.{pt,onnx}`

## 数据样本行说明（JSONL）
每一行表示一个离散时间步的离线 transition，结构为：
- `s_t -> a_t -> r_t -> s_(t+1)`，并附带合法动作约束与终局标签。

### 顶层字段
| 字段 | 类型 | 含义 |
|---|---|---|
| `id` | string | 样本唯一标识（如 `S1234`）。 |
| `ts` | number | 采样时间戳（毫秒）。 |
| `state` | object | 当前状态 `s_t`。 |
| `action` | int | 当前动作 `a_t`，范围 `0..5`。 |
| `action_mask` | int[6] | 当前状态下动作合法掩码，`1=可执行`，`0=非法/高危屏蔽`。 |
| `reward` | number | 当前步 reward（若终局可被终局奖励覆盖）。 |
| `done` | bool | 是否终局。 |
| `next_state` | object | 下一状态 `s_(t+1)`。 |
| `pre_death` | bool | 是否命中临终/近死亡窗口。 |
| `risk_label` | int | 风险标签（`0/1`，用于辅助训练）。 |
| `policy_tag` | string | 行为来源标记（`expert/random/epsilon` 等）。 |
| `episode_id` | string | 对局分段 ID（如 `m12_a1`）。 |
| `agent_id` | string | 当前被训练体 ID（通常 `role_2`，即 AI）。 |
| `opponent_id` | string | 最近对手 ID（通常 `role_1`）。 |
| `outcome_tag` | string | 结果标签：`ongoing/win/loss/draw/self_kill`。 |
| `meta` | object | 辅助调试与奖励估计信息（位置、eta、周边安全度等）。 |

### 动作编码（`action`）
| 编码 | 动作 |
|---|---|
| `0` | `wait` |
| `1` | `up` |
| `2` | `down` |
| `3` | `left` |
| `4` | `right` |
| `5` | `drop_bomb` |

### `state.state_map`（`13x15x10`）通道定义
`state_map[y][x][c]` 的 `c` 含义：
| 通道 | 含义 |
|---|---|
| `C0` | 障碍强度（空地 `0`，刚性障碍高，非刚性/软阻挡中等）。 |
| `C1` | 炸弹紧迫度（当前格炸弹爆炸倒计时归一化）。 |
| `C2` | 威胁 ETA 归一化（越接近爆炸越高）。 |
| `C3` | 自身位置 one-hot。 |
| `C4` | 局部安全邻居充足性（安全邻居数阈值特征）。 |
| `C5` | 炸线覆盖危险度（blast danger）。 |
| `C6` | 可达安全性（reachability）。 |
| `C7` | 半身位风险结构特征（half-body layer）。 |
| `C8` | 敌方位置层（enemy position）。 |
| `C9` | 道具层（item map，按道具类型编码）。 |

### `state.state_vector`（32维）定义
| 索引 | 含义 |
|---|---|
| `V0` | 角色在格中心内的 `dx` 偏移（归一化）。 |
| `V1` | 角色在格中心内的 `dy` 偏移（归一化）。 |
| `V2` | 左脚点是否危险。 |
| `V3` | 右脚点是否危险。 |
| `V4` | 左脚点威胁 ETA 归一化。 |
| `V5` | 右脚点威胁 ETA 归一化。 |
| `V6` | 当前格威胁 ETA 归一化。 |
| `V7` | 当前格安全邻居数归一化。 |
| `V8` | 当前激活炸弹数量归一化。 |
| `V9` | 自身当前是否可放泡。 |
| `V10` | 自身剩余泡泡容量归一化。 |
| `V11` | 自身威力归一化。 |
| `V12` | 自身速度归一化。 |
| `V13` | 最近敌人距离归一化。 |
| `V14` | 最近敌人是否可放泡。 |
| `V15` | 敌方威胁密度（基于距离衰减）。 |
| `V16` | `post_bomb_escape_steps_norm`（放泡后最短逃生步数归一化，越小越安全）。 |
| `V17` | `min_escape_eta_after_bomb`（放泡后逃生时间裕度，越大越安全）。 |
| `V18` | `corridor_deadend_depth`（当前通道死胡同深度）。 |
| `V19` | `blast_overlap_next_2s`（假设放泡后与未来 2 秒爆线重叠度）。 |
| `V20` | `enemy_escape_options_after_my_bomb`（敌方在我方放泡后的可逃选项归一化）。 |
| `V21` | `trap_closure_score`（封路与击杀转化评分）。 |
| `V22` | `item_race_delta`（关键道具敌我到达步差优势）。 |
| `V23` | `enemy_power_gap`（敌我战力差，越高表示敌方更强）。 |
| `V24` | `self_total_bubble_cap_norm`（自身总泡泡容量归一化）。 |
| `V25` | `self_active_bubble_count_norm`（自身当前激活泡泡数量归一化）。 |
| `V26` | `enemy_total_bubble_cap_norm`（敌方总泡泡容量归一化）。 |
| `V27` | `enemy_active_bubble_count_norm`（敌方当前激活泡泡数量归一化）。 |
| `V28` | `enemy_power_norm`（敌方泡泡威力归一化）。 |
| `V29` | `enemy_speed_norm`（敌方速度归一化）。 |
| `V30` | `enemy_shortest_path_dist_norm`（当前敌我最短路距离归一化）。 |
| `V31` | `spawn_shortest_path_dist_norm`（出生点最短路距离归一化）。 |

### `meta` 字段（常见）
| 字段 | 含义 |
|---|---|
| `roleNumber` | 当前 agent 角色编号。 |
| `x`,`y` | 当前位置格坐标。 |
| `eta` | 当前格危险 ETA（毫秒，可能为空）。 |
| `activeBombs` | 当前激活炸弹数量。 |
| `safeNeighbors` | 当前格可逃生邻居数。 |
| `nextSafeRank` | 策略选择时的候选安全等级。 |
| `action_source` | 本步动作来源（含是否被 mask 修正）。 |

### 训练时如何使用一行样本
- 策略学习：`state` + `action_mask` + `action`（受约束行为克隆/IQL actor）。
- 价值学习：`state`, `action`, `reward`, `next_state`, `done`。
- 风险辅助：`risk_label` 与 `pre_death` 用于抑制高危行为。
- 终局监督：`outcome_tag` 用于终局重权采样与评估切片。

### 当前数据构造默认策略（已启用）
- 地图环境：`sudden-death 1v1`；局前随机清除 `35%~75%` 非刚性障碍，并在空位随机回填部分道具。
- 出生点：敌我在风车爱心图上随机出生，最短路距离目标 `1~10` 格，评估报告统计 `1~3 / 4~6 / 7~10` 分桶。
- 终局规则：任一角色进入 `IsInPaopao` 立即判负；关闭复活；`12s` 无关键事件触发 `stall_abort`。
- 对手池：`heuristic_v1 / heuristic_v2 / aggressive_trapper / coward_runner / item_rusher / randomized_mistake_bot`。
- 反重复：移除全局 exact/near dedupe，仅保留“同一局连续完全相同 transition”的轻量 `burst cap`。
- 并行采样：支持 `10 worker` 分片采样，合并阶段只做合同校验与报告聚合。

## Phase 0（离线冷启动）
### 目标
- 用离线数据快速获得可用战斗策略初始化。
- 重点建立 `drop_bomb` 的基本可用性与动作合法性约束。

### 数据
- 规模：`20万~100万` 帧。
- 场景：`1v1`。
- 行为源：增强 heuristic（含放泡、抢道具、封路），可叠加人类回放。
- 风车爱心图 Phase 0 特化：
  - 每局开始执行 `partial clear`：仅随机清除部分非刚性障碍，保留地图结构多样性。
  - 随机道具回填：按密度上限补齐，并保证出生点最小安全半径。
  - 终局字段强制写入：`done / pre_death / outcome_tag / terminal_reason`。
  - 对手自杀局不进入训练集；自身自杀局保留为负样本。

### 训练
- 主线：IQL（可切 BC）。
- 网络：`10` 通道地图 + `32` 维向量，`ACTION_DIM=6`。
- 迁移：加载 `dodge_iql_v1.pt` 卷积层，前 `50` epochs 冻结。
- 时序：启用 `GRU(256)`。
- actor loss：应用 `action_mask`，抑制非法/必死动作。
- 训练信号修正：
  - 取消每 epoch 截断，恢复完整 epoch 覆盖。
  - 终局重权：强化 `self_kill`、`timeout_draw`、`successful_trap` 片段。
  - 追加对抗辅助特征（用于重权与质量分析）：
    - `post_bomb_escape_steps_norm`
    - `min_escape_eta_after_bomb`
    - `corridor_deadend_depth`
    - `blast_overlap_next_2s`
    - `enemy_escape_options_after_my_bomb`
    - `trap_closure_score`
    - `enemy_recent_bomb_cd`
    - `enemy_heading_delta`
    - `item_race_delta`
    - `local_choke_occupancy`
    - `enemy_power_gap`
    - `power_spike_horizon`

### 启动命令（本轮执行）
1. ```bash
   node scripts/collect-combat-dataset-parallel.js --workers=10 --target-frames=200000 --max-wall-sec=7200 --arena=1v1 --map=windmill-heart --action-space=discrete6 --fresh=1 --clear-nonrigid=1 --sudden-death=1 --disable-revive=1 --ignore-enemy-self-kill=1 --stall-no-progress-ms=12000 --partial-clear-min-ratio=0.35 --partial-clear-max-ratio=0.75 --spawn-shortest-path-min=1 --spawn-shortest-path-max=10 --opponent-pool=heuristic_v1,heuristic_v2,aggressive_trapper,coward_runner,item_rusher,randomized_mistake_bot --dataset-path=output/ml/datasets/combat_phase0_v4_suddendeath.jsonl
   ```
2. ```bash
   python3 ml/train_iql_combat.py --dataset output/ml/datasets/combat_phase0_v4_suddendeath.jsonl --epochs 60 --batch-size 512 --freeze-conv-epochs 50 --init-pt output/ml/models/dodge_iql_v1.pt --out-pt output/ml/models/combat_phase0_iql_v4_suddendeath.pt --out-onnx output/ml/models/combat_phase0_iql_v4_suddendeath.onnx
   ```
3. ```bash
   node scripts/eval-combat-1v1.js --model-url=/output/ml/models/combat_phase0_iql_v4_suddendeath.onnx --opponent=heuristic_v2 --runs=50 --match-duration-sec=45 --map=windmill-heart --clear-nonrigid=1 --sudden-death=1 --disable-revive=1 --ignore-enemy-self-kill=1 --stall-no-progress-ms=12000 --partial-clear-min-ratio=0.35 --partial-clear-max-ratio=0.75 --spawn-shortest-path-min=1 --spawn-shortest-path-max=10 --random-item-density=0.12 --parallel=8
   ```
   评估固定口径：`50局 + sudden-death + windmill-heart + heuristic_v2`。
4. ```bash
   node scripts/train-combat-phase0.js --workers=10 --target-frames=200000 --collect-max-wall-sec=7200 --runs=50 --map=windmill-heart --opponent=heuristic_v2 --sudden-death=1
   ```
   一键流水线：`基线评估 -> 并行采样 -> 训练 -> 评估 -> 指标对比报告`。

### V4 Sudden-Death Sampling（当前迭代）
- 训练与评估统一到 `sudden-death` 规则：先被泡困住/炸到者直接判负，不再等待时间结算。
- 评估新增统计：`terminal_reason_hist / stall_draw_rate / opponent_self_kill_rate / spawn_dist_hist / close_range_win_rate / threatened_bomb_finish_rate`。
- 采样质量优先看“有效局面覆盖”而非全局去重后的唯一样本数：
  - `valid_episode_count`
  - `rows_written`
  - `rows_per_sec`
  - `terminal_reason_hist`
  - `spawn_dist_hist`
  - `action5_ratio`
  - `property_bucket_hist`

### Phase0-Seq（长时信用分配 + 序列推理）
- 目标：在 `agent_expert_duel=1` 的双专家 1v1 场景下，构建 `20w` 样本并引入“事件塑形 + 折扣回报 + 序列窗口”训练闭环。
- 专家对战池统一为：`heuristic_v2,aggressive_trapper`（agent/opponent 同池抽样）。
- 数据合同新增字段（由 `scripts/postprocess-combat-credit.js` 写入）：
  - `reward_train`
  - `meta.reward_raw`
  - `meta.reward_dense`
  - `meta.return_discounted`
  - `meta.credit_horizon_ms`
  - `meta.sequence_index`
- 奖励构造：
  - `reward_dense = reward_raw + w1*danger + w2*item_gain + w3*kill_credit + w4*escape_success - w5*self_kill_penalty`
  - `return_discounted = Σ(gamma^k * reward_dense_{t+k})`（窗口 `credit_horizon_ms` 内，遇终局提前截断）
  - `reward_train = clip((1-alpha)*reward_dense + alpha*return_discounted, [-3,3])`
- 序列训练：
  - `ml/train_iql_combat.py` 支持 `--sequence-len --sequence-stride`，按 `episode_id + ts` 构造窗口。
  - 模型输入支持序列形状：`state_map=[B,T,C,H,W]`、`state_vector=[B,T,V]`。
  - 推理运行时自动识别输入 rank：`5D/3D -> 序列路径`，`4D/2D -> 单帧兼容路径`。
- 新增训练参数：
  - `--sequence-len`（默认 `8`）
  - `--sequence-stride`（默认 `1`）
  - `--credit-gamma`（默认 `0.995`）
  - `--credit-horizon-ms`（默认 `8000`）
  - `--reward-blend-alpha`（默认 `0.65`）
- 新增评估统计：
  - `sequence_path_hit_rate`
  - `sequence_path_hits`
  - `single_path_hits`

#### Phase0-Seq 建议命令
1. 并行采样（双专家对战）：
   ```bash
   node scripts/collect-combat-dataset-parallel.js --workers=10 --target-frames=200000 --max-wall-sec=7200 --arena=1v1 --map=windmill-heart --action-space=discrete6 --fresh=1 --clear-nonrigid=1 --sudden-death=1 --disable-revive=1 --ignore-enemy-self-kill=1 --stall-no-progress-ms=12000 --partial-clear-min-ratio=0.35 --partial-clear-max-ratio=0.75 --spawn-shortest-path-min=1 --spawn-shortest-path-max=10 --agent-expert-duel=1 --agent-pool=heuristic_v2,aggressive_trapper --opponent-pool=heuristic_v2,aggressive_trapper --dataset-path=output/ml/datasets/combat_phase0_seq_raw.jsonl --report-path=output/ml/reports/combat_phase0_seq_collect.json
   ```
2. credit 后处理：
   ```bash
   node scripts/postprocess-combat-credit.js --input=output/ml/datasets/combat_phase0_seq_raw.jsonl --output=output/ml/datasets/combat_phase0_seq_credit.jsonl --report-path=output/ml/reports/combat_phase0_seq_credit.json --credit-gamma=0.995 --credit-horizon-ms=8000 --reward-blend-alpha=0.65
   ```
3. 序列训练：
   ```bash
   python3 ml/train_iql_combat.py --dataset output/ml/datasets/combat_phase0_seq_credit.jsonl --sequence-len 8 --sequence-stride 1 --credit-gamma 0.995 --credit-horizon-ms 8000 --reward-blend-alpha 0.65 --epochs 60 --batch-size 512 --freeze-conv-epochs 50 --init-pt output/ml/models/dodge_iql_v1.pt --out-pt output/ml/models/combat_phase0_iql_phase0_seq.pt --out-onnx output/ml/models/combat_phase0_iql_phase0_seq.onnx --out-metrics output/ml/reports/combat_phase0_seq_train.json
   ```
4. 固定口径评估（`vs heuristic_v2`，50局，实时帧）：
   ```bash
   node scripts/eval-combat-1v1.js --model-url=/output/ml/models/combat_phase0_iql_phase0_seq.onnx --opponent=heuristic_v2 --runs=50 --parallel=8 --match-duration-sec=45 --map=windmill-heart --clear-nonrigid=1 --sudden-death=1 --disable-revive=1 --ignore-enemy-self-kill=1 --stall-no-progress-ms=12000 --partial-clear-min-ratio=0.35 --partial-clear-max-ratio=0.75 --spawn-shortest-path-min=1 --spawn-shortest-path-max=10 --random-item-density=0.12 --live-view=1 --report-path=output/ml/reports/combat_phase0_seq_eval.json
   ```
5. 一键流水线（采样->credit->训练->评估->对比）：
   ```bash
   node scripts/train-combat-phase0-seq.js --workers=10 --target-frames=200000 --collect-max-wall-sec=7200 --runs=50 --parallel=8 --map=windmill-heart --sequence-len=8 --sequence-stride=1 --credit-gamma=0.995 --credit-horizon-ms=8000 --reward-blend-alpha=0.65 --live-view=1
   ```

### V3 Balanced Sampling（当前迭代）
- 不进入 Phase 1；先修 Phase 0 数据分布：`场景分桶采样 + 终局优先保留 + 放泡后果监督`。
- 采样桶目标：`ongoing=60%`、`pre_death=15%`、`drop_bomb_safe=10%`、`drop_bomb_bad=10%`、`terminal=5%`。
- 新增样本字段：
  - `sample_bucket`: `ongoing | pre_death | drop_bomb_safe | drop_bomb_bad | terminal`
  - `aux_labels`: `bomb_escape_success_label / bomb_self_trap_risk / enemy_trap_after_bomb / nearest_safe_tile_eta / commitment_depth / terminal_credit_action`
- 场景注入：随机出生点、软障碍少量回灌、随机道具、随机战力、放泡逃生、通道封锁、关键道具争抢、半死路追击。
- 采样期使用短局口径加速终局产生：默认 `collect-match-duration-sec=18`，评估仍固定 `45s/局`。
- `terminal_tail_ms=3000` 会把终局前 3 秒样本写入终局标签与 reward，用于解决 Phase 0 原始数据 `done/pre_death` 过稀的问题。
- 仅在 `balanced=1` 且 `action_mask[5]` 合法时，采样器允许把部分高价值局面重标注为 `action=5`，并记录 `meta.balanced_relabel_action5=1` 与 `meta.original_action`；该机制用于补足放泡后果监督，不改变 ONNX 主输入。
- 采样质量门槛：`done_ratio>=4%`、`pre_death_ratio>=8%`、`action5_ratio>=8%`、`self_kill/loss` 非零、`terminal_credit` 非零。
- 一键命令：
  ```bash
  node scripts/train-combat-phase0.js --balanced=1 --verify-frames=30000 --workers=10 --target-frames=200000 --collect-max-wall-sec=7200 --collect-match-duration-sec=18 --runs=200 --parallel=8 --epochs=60 --batch-size=512 --freeze-conv-epochs=50 --map=windmill-heart --opponent=heuristic_v2
  ```
- 输出模型：`output/ml/models/combat_phase0_iql_v3_balanced.onnx` 与 `.pt`。

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
- 并行采样聚合检查：`unique_rows_before_cap`、`near_duplicate_rows`、`done_ratio`、`outcome_hist`。
- ONNX 推理输出动态 `action_dim`（5/6）兼容。

### 回归测试
- 旧 `dodge_iql_v1.onnx` 在 battle 模式可继续运行（不触发越界放泡）。
- 躲泡既有评估基线不显著回退（survival 保持）。
- ONNX 输入向量维度兼容：`state_vector` 自动 `pad/truncate`，16/24 维模型可并存运行。
