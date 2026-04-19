Original prompt: 当前ai还是太容易死了，还会经常出现不移动的情况。请训练ai的躲泡能力，在当前风车爱心地图上随机位置生成10个水泡（每个水泡间隔0.3s生成），要求ai能够走位躲开，共生成100轮随机水泡，每躲开一轮+1分，每被炸死一次-2分，要求ai最终分数为正数且越高越好。以上对局每完成一局，总结一次经验修改ai的躲泡逻辑，共进行10局游戏强化ai的躲泡技能。以可视化的方式让我看到训练过程，可以让我看到实际的游戏过程。

- Added configurable bubble fuse timing (`window.BNBPaopaoFuseMs`) and bomb explosion timestamp metadata for AI time-aware threat estimation.
- Strengthened `Role.MoveTo` with return status + stall watchdog to reduce frequent “AI not moving” lockups.
- Added dodge policy persistence (`bnb_ai_dodge_policy`) and threat snapshot model with ETA.
- Added AI unstick behavior + retry throttling in `Monster.MoveToMap` and `Monster.Think`.
- Added dedicated dodge-training behavior for monsters (evade + safe patrol only).
- Added full 10-match training orchestrator:
  - fixed map to `windmill-heart`
  - per round: random 10 bubbles @ 0.3s interval
  - scoring: survive round +1, death -2
  - 100 rounds per match, 10 matches total
  - per-match lesson and policy tuning
- Added visual training panel in the existing right-side match panel.
- Updated page startup:
  - default -> dodge training mode
  - `?mode=battle` -> original battle mode

TODO / next pass:
- Run full-browser validation and inspect screenshots to ensure panel updates + movement behavior look right.
- If deaths remain high on full runs, increase dodge policy conservativeness (`forecastMs`, `safeBufferMs`) and lower stall timeout further.

Validation:
- Ran Playwright client against local server with 6 capture iterations.
- Verified no runtime `errors-*.json` generated in final run.
- Captured training gameplay frames in `output/web-game/shot-0.png` .. `shot-5.png`.
- Confirmed AI moves under random bubble pressure on windmill-heart map.
- Extended validation: 15-frame capture run produced `shot-0.png`..`shot-14.png` with no runtime error files.

User-request update (2026-04-16):
- Fixed trainer freeze root cause: rounds now use tokenized lifecycle + hard timeout guard; death during round no longer stalls spawn/settle timers.
- Training config updated per request:
  - bubblesPerRound: 20
  - bubbleIntervalMs: 200ms
  - bubble strength: 10 (full cross range)
- Training-mode map now strips all non-rigid barriers:
  - added `IsRigidBarrierNo` and `StripNonRigidBarriersFromMap`
  - enabled stripping in `InitGame` when `window.BNBTrainingStripNonRigid` is on
  - runtime cleanup also removes any non-rigid barrier sprites during training.
- Final match completion robustness:
  - reset timers between matches
  - active round token guards against stale callbacks
  - round hard-timeout fallback ensures progression to next round/match
  - trainer now stops monster cleanly after match 10.

Validation (headless, Chrome fallback):
- State sampling over ~80s showed continuous progression from round 1 to round 13 without freezing.
- Confirmed live trainer params: `20 bubbles`, `200ms interval`, `strong=10`.
- Map rigidity check during training: `{ rigid: 3, nonRigid: 0 }`.
- Captured full-page visualization screenshot: `output/web-game/training-latest.png`.

User-request fix batch (2026-04-16, follow-up):
- Restored explosion hit logic to original map-cell based rule (`same tile hit`, no half-body special window).
- Fixed persistent explosion/water-column issue:
  - added robust explosion sprite forced cleanup timeout in `PopoBang`;
  - fixed runtime error in `Barrier.Bomb` when map rows have no barrier storage (common after stripping non-rigid blocks).
- Updated training schedule to 50 rounds per match (10 matches total).
- Re-started training with new config and verified live progression.

Verification:
- Hit logic check: `{ same: true, adjacent: false }`.
- Explosion visibility check: after isolated trigger, visible explosion sprites drop to `0` by ~2.9s.
- No page runtime errors in 6s stress window (`ERR_COUNT=0`).
- Live run state: start `roundsPerMatch=50`, later advanced from round 1 to round 4 while running.

User-request update (2026-04-17):
- Added right-panel AI enemy-count config (`0-4`) in battle mode UI:
  - new select: `#ai-enemy-count-select`
  - new hint: `#ai-enemy-count-hint`
- Added persistent storage for enemy count (`localStorage` key: `bnb_ai_enemy_count`).
- Enemy-count change now triggers immediate match restart (page reload) so the new count takes effect.
- `StartSinglePlayerGame()` now reads stored enemy count by default instead of hardcoded `3`.
- Fixed `0` enemy edge case support:
  - `StartSinglePlayerGame` no longer falls back via `|| 3`
  - `StartMonsters` now accepts `0` and clamps within `0..4`.
- Battle entry updated to `StartSinglePlayerGame()` (no hardcoded count arg).

Validation:
- Syntax checks passed:
  - `node --check public/game/bnb.js`
  - `node --check public/game/bnbMonsters.js`
- Runtime smoke test passed at `http://127.0.0.1:4000/?mode=battle` with Playwright client.

User-request update (2026-04-17, hit-rule-aware training refactor):
- Reworked bomb danger telemetry and AI training flow to align with new explosion-hit + half-body rule set (without changing core hit/half-body rule execution itself).
- Added/extended explosion event snapshot API for AI/trainer consumption:
  - `window.GetBNBExplosionEventSnapshot(now?)`
  - includes `bombs[]`, `clusters[]`, `activeWindows[]`, `unsafeWindowMs`.
- Connected-coverage cluster timing now unified by earliest bomb start/end for pending clusters and active windows.
- Role unsafe resolver now carries explosion event IDs so trainer can attribute per-event fail reasons.
- AI dodge policy now reads joint threat view (pending + active window), adds next-frame movement evaluator, shortest-safe preference, and explicit half-body preference branch.
- Replaced old round trainer with 10-match training mechanism:
  - map fixed to `windmill-heart`
  - strip non-rigid barriers
  - continuous random spawn (`0.2s`, `strong=10`)
  - attempt ends at 10 bombed hits
  - max 3 attempts per match
  - target progression: previous match best + 10
  - event-level success/fail accounting and reason distribution.
- Added event-level logs and expanded result payload in `window.BNBLatestTrainingResult`.

Additional fix (2026-04-17, consistency):
- Active connected explosion windows now enforce fixed duration = earliest start + `unsafeWindowMs`.
- Threat map `dangerEndMap` merge now keeps latest danger end (max), not earliest, to avoid underestimating risk horizon.

Validation:
- Syntax checks passed:
  - `node --check public/game/bnbPaopao.js`
  - `node --check public/game/bnbRole.js`
  - `node --check public/game/bnbMonsters.js`
  - `node --check scripts/run-ai-training-iter5.js`
- Headless smoke (Chrome fallback) passed with no page errors:
  - trainer running and progressing
  - snapshot keys present: `now/unsafeWindowMs/bombs/clusters/activeWindows`
  - verified `unsafeWindowMs = 450` and active window durations consistent.
- Long-run progress check (partial, manually stopped to avoid long session):
  - confirmed match transition and scoring logic:
    - match 1 reached death limit and ended
    - match 2 started with target = previous best + 10
    - attempt rollover at death limit works (attempt 1 -> attempt 2).

User-request update (2026-04-18, Offline ML BC V1 integration):
- Implemented Offline ML modules directly in `public/game/bnbMonsters.js`:
  - `BNBMLFeatureEncoder`: exports `13x15x5` state map + `[dx, dy]`.
  - `BNBMLDatasetCollector`: frame sampling in training mode with JSONL-ready rows (`state/action/reward/done/next_state`) and `pre_death` marking.
  - `BNBMLRuntime`: ONNX runtime loading + inference + safety validation + model-first fallback-aware execution.
- Added browser globals/contracts:
  - `window.BNBMLDatasetCollectorState`
  - `window.BNBMLCollectorDrain(maxRows)`
  - `window.BNBMLRuntimeState`
- Added model-driven dodge integration:
  - Training mode (`ThinkDodgeTraining`) now records dataset frames and tries ML dodge first.
  - Battle mode (`Think`) now uses ML dodge in bomb-danger context before rule evade branch.
- Added policy-freeze support for expert data collection:
  - `ml_freeze=1` / `window.BNBMLFreezeExpertPolicy` disables online auto-tuning branches.
- Added KPI counters and runtime metrics:
  - `spawned_bubbles`, `bombed_count`, `survival_rate`, `fallback_rate`, `avg_latency_ms`.
- Added output static route in server:
  - `app.use('/output', express.static(path.join(__dirname, 'output')))`.
- Added scripts:
  - `scripts/collect-dodge-dataset.js` (headless collection pipeline, JSONL stream write).
  - `scripts/eval-ml-dodge.js` (`ml=0` vs `ml=1` A/B run, unified KPI report JSON).
- Added ML training files:
  - `ml/train_bc.py` (CNN+MLP BC training, class-weighted CE, ONNX export, metrics JSON).
  - `ml/train_cql.py` (Offline RL/CQL contract scaffold + validation report).
  - `ml/requirements.txt`.

Validation (local):
- Syntax checks passed:
  - `node --check public/game/bnbMonsters.js`
  - `node --check public/game/bnb.js`
  - `node --check app.js`
  - `node --check scripts/collect-dodge-dataset.js`
  - `node --check scripts/eval-ml-dodge.js`
  - `python3 -m py_compile ml/train_bc.py ml/train_cql.py`
- Dependency setup:
  - Installed `torch`, `onnx`, `onnxscript`.
- Dataset:
  - Generated `output/ml/datasets/dodge_bc_v1.jsonl` (smoke-sized, manually early-stopped).
- BC artifacts generated:
  - `output/ml/models/best.pt`
  - `output/ml/models/dodge_bc_v1.onnx` (single-file ONNX, no sidecar data file)
  - `output/ml/reports/bc_v1_metrics.json`
- CQL scaffold report generated:
  - `output/ml/reports/cql_scaffold_report.json`
- A/B evaluation report generated:
  - `output/ml/reports/dodge_ml_eval_report.json`
  - latest run (20s each scenario):
    - baseline (`ml=0`): survival_rate = `0.9596` (95.96%)
    - model (`ml=1`): survival_rate = `0.9697` (96.97%)
    - delta = `+0.0101` (+1.01pp)
    - model fallback_rate = `0.3974`

Notes / next pass:
- Current dataset remains wait-heavy; model still predicts wait as top-1 often.
- Runtime now recovers by selecting next best valid action from model distribution, reducing full fallback.
- To approach stronger standalone model behavior (lower fallback_rate), run longer collection and/or add action-balanced sampling and trajectory-level labeling from temporal plan actions.

User-request update (2026-04-18, Offline ML V1.1 metric/logging refactor):
- Survival metric corrected to effective denominator only:
  - added `spawned_bubbles_effective` and `spawned_bubbles_ignored_trapped` in both
    `BNBMLDatasetCollectorState` and `BNBMLRuntimeState`.
  - `survival_rate = 1 - bombed_count / spawned_bubbles_effective` (effective==0 => 1).
  - collector/runtime `OnBubbleSpawned(role)` now counts effective only when `!role.IsDeath && !role.IsInPaopao`.
- Battle observability added (per-decision):
  - `TryOfflineMLDodgeAction` now returns structured decision result (`handled/attempted/reason`).
  - battle-mode logs now emit model/rule source, fallback reason, confidence, start and path.
  - runtime state now exposes `last_decision_source`, `last_fallback_reason`, `decision_trace_tail`.
- Added runtime decision trace buffer + console output prefix `[BNB-ML][battle]`.
- Collection pipeline improvements:
  - `collect-dodge-dataset.js` default target set to 600000 (plan target).
  - report/log fields now include `spawned_bubbles_effective` and `spawned_bubbles_ignored_trapped`.
  - added `--ml-runtime=0|1` option for collection acceleration experiments.
  - enabled configurable wait-frame keep ratio (`ml_wait_keep`, `window.BNBMLCollectWaitKeepProb`).
- Evaluation pipeline improvements:
  - `eval-ml-dodge.js` now supports multi-run evaluation (`--runs`) and reports mean metrics.
  - report includes per-run details and mean KPI (`survival_rate_mean`, `fallback_rate_mean`, etc.).
- Added iterative orchestration script:
  - `scripts/train-bc-iterative.js` for A/B/C rounds (60w->90w->120w) with auto stop when KPI hit.
  - npm script: `npm run train:bc:v1_1`.

Validation (V1.1):
- Syntax checks passed:
  - `node --check public/game/bnbMonsters.js`
  - `node --check scripts/collect-dodge-dataset.js`
  - `node --check scripts/eval-ml-dodge.js`
  - `node --check scripts/train-bc-iterative.js`
  - `python3 -m py_compile ml/train_bc.py ml/train_cql.py`
- Battle trace snapshot generated:
  - JSON: `output/ml/reports/battle_ml_trace_snapshot.json`
  - Screenshot: `output/ml/reports/battle_ml_trace_snapshot.png`
  - trace fields verified (`decision_trace_tail`, `last_decision_source`, `last_fallback_reason`).
- Re-trained BC with refreshed dataset (current interactive run):
  - dataset lines: `5266`
  - train used: `3170` (`drop_pre_death=True`)
  - artifacts updated: `output/ml/models/best.pt`, `output/ml/models/dodge_bc_v1.onnx`
  - metrics: `output/ml/reports/bc_v1_metrics.json`
- Multi-run KPI (corrected metric) from `output/ml/reports/dodge_ml_eval_report.json`:
  - baseline survival mean: `0.9320`
  - model survival mean: `0.9745`
  - delta: `+0.0425`
  - model fallback rate mean: `0.2618`
  - => current corrected KPI already > 95% in this run setup.

Next pass / TODO:
- Run full long-horizon A/B/C data schedule with `npm run train:bc:v1_1` (targets: 600k/900k/1200k).
- Improve action diversity (currently wait/down dominates) to further lower fallback rate.

User-request update (2026-04-18, long-run 200k dataset collection + retrain):
- Strictly followed requirement: only started training after dataset reached >=200000 rows.
- Collected dataset to `201686` rows in `output/ml/datasets/dodge_bc_v1.jsonl` before training.
- Collection process was kept continuous; to speed up without interrupting, ran parallel append collectors (`fresh=0`).
- Post-collection training run:
  - command: `python3 ml/train_bc.py --dataset output/ml/datasets/dodge_bc_v1.jsonl --epochs=6 --batch-size=512`
  - output model: `output/ml/models/dodge_bc_v1.onnx`
  - metrics: `output/ml/reports/bc_v1_metrics.json`
- Training metrics (this run):
  - dataset_size_raw: `201686`
  - dataset_size_used (`drop_pre_death=True`): `102445`
  - val_acc: `0.9882`
  - val_macro_f1: `0.6486`

Evaluation (corrected survival metric, default `ml_conf=0.34`):
- report: `output/ml/reports/dodge_ml_eval_report.json`
- setup: `duration=45000ms`, `runs=3`
- baseline survival mean: `0.95495`
- model survival mean: `0.95469`
- model fallback rate mean: `0.25254`
- delta: `-0.00027` (no uplift in this run)

99% target status:
- Not achieved in this 200k round under current BC pipeline/config.
- Additional quick confidence sweep (`ml_conf` in {0.34,0.5,0.7,0.9}, shorter eval) also did not reach 99%.

Notes / next pass to chase 99%:
- Current dataset is still action-imbalanced (`action 0` dominates), despite larger volume.
- Recommended next iteration:
  1) action-balanced collection (cap wait samples or oversample move actions),
  2) keep 200k+ but enforce per-action minimum quotas,
  3) train BC with balanced sampler,
  4) tighten runtime guardrails and only trust model under immediate-threat context,
  5) then re-evaluate with longer multi-run benchmark.

User-request update (2026-04-18, 99% sprint in progress):
- Started non-interrupting multi-shard quota collection for 200k requirement:
  - active shards: dodge_bc_v1 + part2..part10
  - all shards use corrected survival denominator fields; quota collection enabled.
- Added collector speed control in `scripts/collect-dodge-dataset.js`:
  - new arg `--think-interval-ms` (default 8), wired into in-page trainer think interval.
- Added eval guardrail tunables in `scripts/eval-ml-dodge.js`:
  - new args `--ml-move-conf`, `--ml-margin`, `--ml-force-move-eta`, `--ml-wait-block-eta`, `--ml-move-threat-ms`.
  - report now records `ml_cfg` object for audit.
- Status: still collecting; training is intentionally deferred until dataset total >= 200000.

User-request update (2026-04-18, 99% sprint run completed):
- Enforced "collect >=200k before training":
  - collected multi-shard quota dataset total `201,984` rows.
  - merged and sliced exact training file `output/ml/datasets/dodge_bc_v1_sprint99_200k.jsonl` (`200,000` rows).
- Data distribution check on 200k file:
  - action_hist: `{0:23974,1:79385,2:47523,3:27730,4:21388}`
  - pre_death_ratio: `0.310495`
- Trained BC with balanced sampler:
  - command used `--sampler=balanced --sampler-power=0.7 --max-wait-ratio-train=0.45 --epochs=12`
  - metrics (`output/ml/reports/bc_v1_metrics.json`):
    - dataset_size_raw: `200000`
    - dataset_size_used: `137901`
    - val_acc: `0.811675`
    - val_macro_f1: `0.790233`
  - model artifacts refreshed:
    - `output/ml/models/best.pt`
    - `output/ml/models/dodge_bc_v1.onnx`
- Added guardrail-tunable eval interface (`scripts/eval-ml-dodge.js`):
  - `--ml-move-conf --ml-margin --ml-force-move-eta --ml-wait-block-eta --ml-move-threat-ms`
  - report now records `ml_cfg`.
- 99% sprint eval results:
  - final confirm (45s x 3 runs, aggr cfg):
    - baseline mean: `0.936984`
    - model mean: `0.951542`
    - delta: `+0.014558`
    - fallback mean: `0.516351`
  - short sweep best absolute model mean: `0.970209` (`aggr`)
  - still below target `0.99` under corrected metric.
- Battle observability check:
  - `output/ml/reports/battle_trace_check.json` shows `loaded=true`, `trace_len=20`, source/reason populated.
  - screenshot: `output/ml/reports/battle_trace_check.png`.

Next recommended iteration (if continuing toward 99%):
1) Keep current best model online but run Round-B data expansion to 400k+ with same quotas.
2) Add hard-negative oversampling for trajectories labeled with `fallback_guardrail_rejected_wait` and recent bomb events.
3) Re-train BC and then re-run same aggr guardrail config benchmark.

User-request update (2026-04-18, 角色三道具与初始属性统一):
- 新增统一角色平衡配置 `RoleBalanceConfig`（玩家与 AI 共用）：
  - 初始：水泡数=2、速度=150px/s、威力=2格
  - 增量：水泡+1、速度+25px/s、威力+1格
  - 上限：最大水泡=8、最大速度=300px/s、最大威力=10格
- 新增速度换算与复用函数：`SpeedPxPerSecToMoveStep`（保留内部 MoveStep 机制）。
- 玩家与 AI 初始化改为同一来源配置：
  - `CreateRole` 与 `Monster` 构造均改为读取 `RoleBalanceConfig`。
- 道具效果仅保留 3 种（101/102/103）：
  - 101 -> 水泡数+1（封顶8）
  - 102 -> 速度+25px/s（封顶300px/s）
  - 103 -> 威力+1格（封顶10）
  - 旧道具 104~109 不再生效。
- 地图掉落规则收敛：
  - `NormalGiftPool` 仅保留 `[101,102,103]`
  - 可炸非刚体 `3/8` 被炸后均随机生成三类道具之一。
  - `Barrier.Materials` 道具贴图仅保留 Gift1/2/3。
- 爆炸威力上限去硬编码：
  - `FindPaopaoBombXY` 中由固定 `10` 改为读取 `RoleConstant.MaxPaopaoStrong`。
- AI 道具扫描与评分同步收敛到 101~103（不再追逐旧道具）。
- 速度上限初始化修复：
  - 避免模板默认值覆盖新上限，`InitRoleMaxMoveStepConfig()` 改为以 `RoleConstant.MaxMoveStep` 初始化。
  - 模板默认展示同步更新为上限 6（对应 300px/s）。

Validation:
- `node --check public/game/bnbRole.js` 通过
- `node --check public/game/bnbBarrier.js` 通过
- `node --check public/game/bnb.js` 通过
- `node --check public/game/bnbMonsters.js` 通过
- `node --check public/game/bnbPaopao.js` 通过

User-request update (2026-04-18, 配置面板补齐与单位修正):
- 右侧配置面板新增两项可调上限：
  - `人物最大水泡数`（`max-bubble-count-input`）
  - `人物最大威力(覆盖格)`（`max-power-input`）
- 旧速度配置改为 `px/s` 语义：
  - 标签改为 `人物最大速度 (px/s)`；
  - 输入范围改为 `50..1000`，步进 `25`；
  - 提示文案改为 `当前上限：xxx px/s`。
- 运行时生效逻辑：
  - 速度上限：更新 `RoleBalanceConfig.MaxSpeedPxPerSec` + `RoleConstant.MaxMoveStep`，并同步裁剪在场角色速度。
  - 水泡上限：更新 `RoleBalanceConfig.MaxBubbleCount`，同步裁剪在场角色 `CanPaopaoLength`，并同步 `MonsterMaxPaopaoLength`。
  - 威力上限：更新 `RoleBalanceConfig.MaxPower` + `RoleConstant.MaxPaopaoStrong`，同步裁剪在场角色 `PaopaoStrong`。
- 初始化流程：`InitGame()` 现会初始化三项上限配置（速度/水泡/威力）。
- 模板静态面板同步补齐三项配置块，确保首屏可见。

Validation:
- `node --check public/game/bnb.js` 通过
- `node --check public/game/bnbRole.js` 通过

User-request update (2026-04-18, battle freeze investigation + fix):
- Investigated freeze reports on battle URL with ML enabled.
- Root-cause candidates and fixes applied:
  1) High-frequency explosion safety debug output in battle mode could add heavy runtime overhead.
     - Added debug mode gating in `bnbPaopao.js`: panel/console output is now disabled by default in `mode=battle`.
     - Added optional query flags: `safety_debug=1` and `safety_debug_console=1` for manual enable.
     - Added DOM hard cap: debug `<li>` entries are trimmed to `ExplosionSafetyDebugMaxLogs`.
  2) Monster thinker lifecycle trap bug:
     - In `Monster.Start`, `IsInPaopao` no longer calls `self.Stop()` (which cleared `ThinkInterval` permanently).
     - Now pauses movement during trap and keeps interval alive, so AI resumes after untrap.
  3) Console spam mitigation:
     - Added short-interval duplicate suppression for `[BNB-ML][battle]` console lines; decision trace memory/state is still preserved.
- Validation:
  - Syntax checks passed for modified files.
  - 30s battle probe after fix:
    - no page errors
    - explosion debug panel absent in battle (`panel=false`, `listLen=0`)
    - ML runtime loaded and trace active.
  - 120s soak probe after fix:
    - no page/console errors
    - no sustained stall signature in sampled monster states.
  - explicit trap/untrap interval probe:
    - `ThinkInterval` remains alive before/during/after trap (`true/true/true`).

User-request update (2026-04-18, battle 模式 1~3 分钟卡死排查):
- 复现检查（带参数 `mode=battle&ml=1...`）:
  - 在 classic / windmill-heart 两张图下做了 190s 级别 Playwright 压测（含持续方向键+空格输入）。
  - 未复现 JS 异常（`pageerror=0`），ML 日志量也不高（非日志洪泛导致）。
- 高概率根因定位：
  - 渲染主循环原实现按 `currentZIndex` 从 `0` 递增匹配，如果场景里出现负层级 `ZIndex<0`，循环可能无法收敛，导致主线程卡死、页面无响应。
  - 本轮规则改造后新增了 `8` 可炸块掉落道具；在边界坐标（如 `x=0,y=0`）生成道具时，阴影层级计算为 `zindex-1`，可能得到 `-1`，触发上述风险。
- 修复:
  1) `public/game/bnbBarrier.js`
     - 道具阴影层级改为非负：`shadowobject.ZIndex = zindex > 0 ? zindex - 1 : 0`。
  2) `public/game/game.js`
     - 渲染改为对 `SpriteArray` 按 `ZIndex` 排序后顺序绘制，移除“按层级递增直到全部命中”的循环，避免任何异常层级导致死循环。
- 验证:
  - `node --check public/game/bnbBarrier.js` 通过
  - `node --check public/game/game.js` 通过
  - 额外回归：手工注入 `Barrier.Create(0,0,101)` 后页面仍可持续运行（无卡死）。

User-request update (2026-04-19, IQL pure model V1 implementation):
- Implemented `IQL pure` data/feature/runtime path in `public/game/bnbMonsters.js`:
  - Feature encoder upgraded to `13x15x8` with new channels: blast map / reachability / half-body layer.
  - State vector expanded to 9 dims: `[dx,dy,left/right foot unsafe,left/right eta,center eta,safeNeighbors,activeBombs]`.
  - Dataset rows now carry `policy_tag` and `risk_label` with IQL-ready `reward/done/next_state`.
  - Training-mode mixed behavior corrected to single-sample policy choice per frame (avoid double-mix bias).
  - Reward finalization fixed from constant reward to shaped IQL reward path.
  - Pure mode runtime invariants kept observable (`rule_calls`, `pure_violation_count`) and decode supports `risk_logit` output.
  - Pure-mode confidence gating adjusted so low-confidence no longer hard-blocks model actions.
- Added/updated IQL pipeline scripts:
  - New `ml/train_iql.py`: discrete IQL (Q1/Q2 + V expectile + policy AWBC + risk head), ONNX export.
  - New `scripts/train-iql-iterative.js`: rounds target contract (`200k/400k/600k/900k`).
  - `scripts/collect-dodge-dataset.js` defaults switched to IQL dataset/report names and supports IQL mix params in URL/report.
  - `scripts/eval-ml-dodge.js` supports pure mode metrics (`rule_calls_mean`, `pure_violation_count_mean`) and defaults to IQL model url.
  - `package.json` scripts added: `collect:iql`, `train:iql`, `train:iql:iterative`, `eval:iql`.
  - `public/game/restart-game.sh` default battle link switched to pure IQL ONNX model.

Validation & execution:
- Syntax checks passed:
  - `node --check public/game/bnbMonsters.js`
  - `node --check scripts/collect-dodge-dataset.js`
  - `node --check scripts/eval-ml-dodge.js`
  - `node --check scripts/train-iql-iterative.js`
  - `python3 -m py_compile ml/train_iql.py`
- Completed full Round A data collection (200k exact):
  - dataset: `output/ml/datasets/dodge_iql_v1_mixed_200k.jsonl`
  - rows_written: `200000`
  - policy mix written: expert `119997`, random `40284`, epsilon `39719` (close to 6:2:2)
  - action written: wait-heavy (`157642` waits) due pure-run behavior distribution.
- Trained IQL model artifacts:
  - `output/ml/models/dodge_iql_v1.pt`
  - `output/ml/models/dodge_iql_v1.onnx`
  - `output/ml/reports/iql_v1_metrics.json`
- Evaluation (`3 runs x 60s`, pure mode):
  - Before pure-gating fix: model mean survival `0.9339`.
  - After pure-gating fix: model mean survival `0.9453`, baseline `0.9183`, delta `+0.0270`.
  - Pure invariant confirmed: `rule_calls_mean=0`, `pure_violation_count_mean=0`, `fallback_rate_mean=0`.
- KPI status:
  - Current round A still below target `>=0.98`; next planned step remains data-scale rounds (`400k -> 600k -> 900k`) with retraining.

Runtime launch:
- Restarted server via `public/game/restart-game.sh`.
- Current battle URL:
  - `http://127.0.0.1:4000/?mode=battle&ml=1&ml_policy_mode=pure&ml_conf=0.26&ml_move_conf=0.34&ml_margin=0.03&ml_force_move_eta=460&ml_wait_block_eta=760&ml_move_threat_ms=300&ml_model=/output/ml/models/dodge_iql_v1.onnx`
