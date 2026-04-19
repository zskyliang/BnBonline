# Phase 0 增量门禁训练（200k 递增）

本文件对应脚本：`scripts/train-combat-phase0-incremental.js`。

## 目标

- 首轮从 `200,000` 帧开始。
- 采用累计数据重训练：`200k -> 400k -> 600k -> ...`，每次新增 `200k`。
- 仅当新增数据带来有效提升时继续扩量；连续两轮无效则停止并进入优化分支。

## 门禁规则

脚本会在 `output/ml/reports/combat_phase0_rounds.json` 记录每轮：

- `delta_win_rate`（pp）
- `delta_self_kill_rate`（pp）
- `delta_draw_rate`（pp）
- `increment_effective`
- `stop_reason`

默认“有效提升”判定：

- `delta_win_rate >= +2.0pp`
- `delta_self_kill_rate <= +1.0pp`
- `delta_draw_rate <= +3.0pp`

阶段停止条件：

- `win_rate >= 35%` 且 `self_kill_rate <= 25%`
- 或连续 `2` 轮 `increment_effective=false`
- 或到达最大数据量（默认 `1.2M`）

## 固定评估协议

评估脚本：`scripts/eval-combat-1v1.js`

- 固定地图：`windmill-heart`
- 固定对手模式：`scripted_heuristic_v1`
- 固定种子基线：`20260419`
- 默认每轮评估：`200` 局 1v1

## 运行

### 一键运行（默认 200k 递增到 1.2M）

```bash
npm run train:combat:phase0
```

### 常用参数

```bash
node scripts/train-combat-phase0-incremental.js \
  --step-frames=200000 \
  --max-frames=1200000 \
  --eval-matches=500 \
  --eval-match-duration-sec=45
```

### 断点续跑

```bash
node scripts/train-combat-phase0-incremental.js --resume=1
```

### 干跑（只打印命令）

```bash
node scripts/train-combat-phase0-incremental.js --dry-run=1
```

## 优化分支说明

当连续两轮增量无效时，报告中会出现 `optimization_branch`，包含建议的单变量 ablation 列表：

- 特征优先：`C8 enemy`、`C9 item heat`、时序特征
- 数据优先：自杀泡/追击失败/卡位平局重采样，动作再平衡，困难样本过采样

恢复条件：同一固定评估协议下重新达到 `increment_effective=true`，再继续下一轮 `+200k`。
