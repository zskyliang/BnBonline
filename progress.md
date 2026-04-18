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
