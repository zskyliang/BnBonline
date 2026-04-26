const fs = require("fs");
const path = require("path");

const skillPlaywrightPath = path.join(
    process.env.HOME || "",
    ".codex/skills/develop-web-game/node_modules/playwright"
);
let chromium;
try {
    ({ chromium } = require(skillPlaywrightPath));
} catch (_) {
    ({ chromium } = require("playwright"));
}

const OUT_DIR = path.resolve(__dirname, "../output/ml/reports");
const REPORT_PATH_DEFAULT = path.join(OUT_DIR, `combat_phase0_eval_${Date.now()}.json`);
const LIVE_STATUS_PATH_DEFAULT = path.join(OUT_DIR, "combat_eval_live_status.json");
const LIVE_FRAME_PATH_DEFAULT = path.join(OUT_DIR, "combat_eval_live_frame.png");

function getArg(name, fallback) {
    const prefix = "--" + name + "=";
    for (const arg of process.argv.slice(2)) {
        if (arg.startsWith(prefix)) {
            return arg.slice(prefix.length);
        }
    }
    return fallback;
}

function asInt(v, fallback) {
    const n = parseInt(v, 10);
    return Number.isFinite(n) ? n : fallback;
}

function asBool(v, fallback) {
    if (v === undefined || v === null || v === "") {
        return !!fallback;
    }
    const s = String(v).trim().toLowerCase();
    if (s === "1" || s === "true" || s === "yes" || s === "on") {
        return true;
    }
    if (s === "0" || s === "false" || s === "no" || s === "off") {
        return false;
    }
    return !!fallback;
}

function asPositiveInt(v, fallback) {
    const n = parseInt(v, 10);
    return Number.isFinite(n) && n > 0 ? n : fallback;
}

function asFloatOrNull(v) {
    if (v === "" || v === null || v === undefined) {
        return null;
    }
    const n = parseFloat(v);
    return Number.isFinite(n) ? n : null;
}

function normalizeOpponentMode(raw) {
    const value = String(raw || "heuristic_v1").trim().toLowerCase();
    if ([
        "heuristic_v1",
        "heuristic_v2",
        "scripted_heuristic_v2",
        "stationary_dummy",
        "aggressive_trapper",
        "coward_runner",
        "item_rusher",
        "randomized_mistake_bot"
    ].includes(value)) {
        return value === "scripted_heuristic_v2" ? "heuristic_v2" : value;
    }
    return "heuristic_v1";
}

async function launchBrowser() {
    const isSandboxLikely = String(process.env.CODEX_SANDBOX || "").trim() !== "";
    const chromePath = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
    const launchArgs = [
        "--use-gl=angle",
        "--use-angle=swiftshader",
        "--disable-dev-shm-usage",
        "--disable-gpu",
        "--disable-background-networking",
    ];
    const blockedMarks = [
        "operation not permitted",
        "Target page, context or browser has been closed",
        "signal=SIGTRAP",
        "signal=SIGABRT",
    ];
    let lastErr = null;
    for (let attempt = 1; attempt <= 3; attempt++) {
        try {
            return await chromium.launch({
                headless: true,
                args: launchArgs,
            });
        } catch (err) {
            lastErr = err;
            const msg = String(err && err.message ? err.message : err);
            if (isSandboxLikely && blockedMarks.some((m) => msg.includes(m))) {
                throw new Error("playwright_launch_blocked_by_sandbox: rerun this script with escalated permissions.");
            }
            if (msg.includes("Executable doesn't exist")) {
                if (!fs.existsSync(chromePath)) {
                    throw new Error("playwright_browser_missing: run `npx playwright install chromium-headless-shell` first.");
                }
                return chromium.launch({
                    headless: true,
                    executablePath: chromePath,
                    args: launchArgs,
                });
            }
            if (fs.existsSync(chromePath)) {
                try {
                    return await chromium.launch({
                        headless: true,
                        executablePath: chromePath,
                        args: launchArgs,
                    });
                } catch (fallbackErr) {
                    lastErr = fallbackErr;
                }
            }
            if (attempt < 3) {
                await new Promise((resolve) => setTimeout(resolve, 400 * attempt));
            }
        }
    }
    throw lastErr || new Error("playwright_launch_failed");
}

function computePowerScore(role) {
    if (!role) {
        return 0;
    }
    const canPaopaoLength = Number(role.CanPaopaoLength || 0);
    const moveStep = Number(role.MoveStep || 0);
    const paopaoStrong = Number(role.PaopaoStrong || 0);
    return canPaopaoLength + moveStep + paopaoStrong;
}

async function runMatch(runIndex, cfg, live) {
    const mlFlag = cfg.mlEnabled ? "1" : "0";
    const url =
        "http://127.0.0.1:4000/?autostart=0"
        + "&ml=" + mlFlag
        + "&ml_collect=0"
        + "&ml_freeze=1"
        + "&ml_policy_mode=" + encodeURIComponent(cfg.policyMode)
        + (cfg.modelUrl ? "&ml_model=" + encodeURIComponent(cfg.modelUrl) : "")
        + (typeof cfg.mlConf === "number" ? "&ml_conf=" + encodeURIComponent(String(cfg.mlConf)) : "")
        + (typeof cfg.mlMoveConf === "number" ? "&ml_move_conf=" + encodeURIComponent(String(cfg.mlMoveConf)) : "")
        + (typeof cfg.mlMargin === "number" ? "&ml_margin=" + encodeURIComponent(String(cfg.mlMargin)) : "")
        + (typeof cfg.mlForceMoveEta === "number" ? "&ml_force_move_eta=" + encodeURIComponent(String(cfg.mlForceMoveEta)) : "")
        + (typeof cfg.mlWaitBlockEta === "number" ? "&ml_wait_block_eta=" + encodeURIComponent(String(cfg.mlWaitBlockEta)) : "")
        + (typeof cfg.mlMoveThreatMs === "number" ? "&ml_move_threat_ms=" + encodeURIComponent(String(cfg.mlMoveThreatMs)) : "");

    let browser = null;
    let page = null;
    let timedOut = false;
    const hardTimeoutMs = Math.max(15000, (cfg.matchDurationSec + 20) * 1000);
    const hardTimer = setTimeout(() => {
        timedOut = true;
        if (browser) {
            browser.close().catch(() => {});
        }
    }, hardTimeoutMs);

    const runNo = runIndex + 1;
    const previewEnabled = !!(live && live.enabled && live.tryClaimPreview(runNo));
    let lastFrameAt = 0;

    try {
        browser = await Promise.race([
            launchBrowser(),
            new Promise((_, reject) => {
                setTimeout(() => reject(new Error("browser_launch_timeout")), 12000);
            }),
        ]);
        page = await browser.newPage({ viewport: { width: 1460, height: 940 } });
        await page.goto(url, { waitUntil: "domcontentloaded", timeout: 15000 });
        await page.waitForTimeout(200);

        const seed = (cfg.seedBase >>> 0) + runIndex;
        await page.evaluate((runtimeCfg) => {
            function mulberry32(a) {
                let t = a >>> 0;
                return function() {
                    t += 0x6D2B79F5;
                    let z = Math.imul(t ^ (t >>> 15), 1 | t);
                    z ^= z + Math.imul(z ^ (z >>> 7), 61 | z);
                    return ((z ^ (z >>> 14)) >>> 0) / 4294967296;
                };
            }

            const rng = mulberry32(runtimeCfg.seed >>> 0);
            window.Math.random = function() {
                return rng();
            };
            function clamp01(v) {
                const n = Number(v);
                if (!isFinite(n)) return 0;
                return Math.max(0, Math.min(1, n));
            }

            window.alert = function() {};
            window.__combatEvalController = null;
            window.__combatEvalMonitor = null;
            window.__combatEvalItemRespawnTicker = null;
            window.__combatEvalEnv = {
                clearNonRigid: !!runtimeCfg.clearNonRigid,
                randomItemDensity: runtimeCfg.randomItemDensity,
                randomItemDensityJitter: runtimeCfg.randomItemDensityJitter,
                itemRespawnMs: runtimeCfg.itemRespawnMs,
                itemRespawnJitterRatio: runtimeCfg.itemRespawnJitterRatio,
                itemSafeRadius: runtimeCfg.itemSafeRadius,
                itemSafeRadiusJitter: runtimeCfg.itemSafeRadiusJitter,
                itemMax: runtimeCfg.itemMax,
                suddenDeath: !!runtimeCfg.suddenDeath,
                disableRevive: !!runtimeCfg.disableRevive,
                ignoreEnemySelfKill: !!runtimeCfg.ignoreEnemySelfKill,
                stallNoProgressMs: Number(runtimeCfg.stallNoProgressMs || 12000),
                partialClearMinRatio: Number(runtimeCfg.partialClearMinRatio || 0.35),
                partialClearMaxRatio: Number(runtimeCfg.partialClearMaxRatio || 0.75),
                spawnShortestPathMin: Number(runtimeCfg.spawnShortestPathMin || 1),
                spawnShortestPathMax: Number(runtimeCfg.spawnShortestPathMax || 10),
                currentDensity: runtimeCfg.randomItemDensity,
                currentItemRespawnMs: runtimeCfg.itemRespawnMs,
                currentItemSafeRadius: runtimeCfg.itemSafeRadius,
                currentEpisodeMeta: null,
                originalNonRigidCells: null,
                matchFinalized: false,
                lastTerminal: null,
                lastProgressSignature: "",
                lastProgressAt: Date.now(),
                spawnBucketTarget: Number(runtimeCfg.spawnBucketTarget || 0),
                evalProfile: String(runtimeCfg.evalProfile || "standard"),
                specialBombEscape: !!runtimeCfg.specialBombEscape,
                specialRoundSec: Number(runtimeCfg.specialRoundSec || 60),
                specialItemRespawnMs: Number(runtimeCfg.specialItemRespawnMs || 2000),
                specialRespawnDelayMs: Number(runtimeCfg.specialRespawnDelayMs || 0),
                specialRespawnInvincibleMs: Number(runtimeCfg.specialRespawnInvincibleMs || 300),
                matchStartAt: Date.now(),
            };

            function stopItemRespawn() {
                if (window.__combatEvalItemRespawnTicker) {
                    clearInterval(window.__combatEvalItemRespawnTicker);
                    window.__combatEvalItemRespawnTicker = null;
                }
            }

            function stopIntervals() {
                if (window.__combatEvalController) {
                    clearInterval(window.__combatEvalController);
                    window.__combatEvalController = null;
                }
                if (window.__combatEvalMonitor) {
                    clearInterval(window.__combatEvalMonitor);
                    window.__combatEvalMonitor = null;
                }
                stopItemRespawn();
            }

            function findPlayerAndAiRoles() {
                const state = window.singlePlayerState;
                const fighters = state && Array.isArray(state.Fighters) ? state.Fighters : [];
                let player = null;
                let ai = null;
                for (let i = 0; i < fighters.length; i++) {
                    const f = fighters[i];
                    if (!f || !f.role) {
                        continue;
                    }
                    if (!player && f.id === "player") {
                        player = f.role;
                    }
                    if (!ai && typeof f.id === "string" && f.id.indexOf("ai_") === 0) {
                        ai = f.role;
                    }
                }
                return { player, ai };
            }

            function getRoleMap(role) {
                if (!role || typeof role.CurrentMapID !== "function") {
                    return null;
                }
                return role.CurrentMapID();
            }

            function computeRolePower(role) {
                if (!role) {
                    return 0;
                }
                return Number(role.CanPaopaoLength || 0)
                    + Number(role.MoveStep || 0)
                    + Number(role.PaopaoStrong || 0);
            }

            function isRigidBarrierNo(no) {
                return typeof window.IsRigidBarrierNo === "function"
                    ? !!window.IsRigidBarrierNo(Number(no || 0))
                    : (Number(no || 0) > 0 && Number(no || 0) < 100 && Number(no || 0) !== 3 && Number(no || 0) !== 8);
            }

            function isWalkableEmpty(x, y) {
                return !!(window.townBarrierMap
                    && window.townBarrierMap[y]
                    && window.townBarrierMap[y][x] === 0
                    && (!window.IsAIWalkable || window.IsAIWalkable(x, y)));
            }

            function collectWalkableCells() {
                const out = [];
                if (!window.townBarrierMap || !Array.isArray(window.townBarrierMap)) {
                    return out;
                }
                for (let y = 0; y < window.townBarrierMap.length; y++) {
                    const row = window.townBarrierMap[y];
                    if (!Array.isArray(row)) {
                        continue;
                    }
                    for (let x = 0; x < row.length; x++) {
                        if (isWalkableEmpty(x, y)) {
                            out.push({ X: x, Y: y });
                        }
                    }
                }
                return out;
            }

            function bfsDistance(start, goal) {
                if (!start || !goal || !window.townBarrierMap) {
                    return 999;
                }
                const rows = window.townBarrierMap.length;
                const cols = rows > 0 && Array.isArray(window.townBarrierMap[0]) ? window.townBarrierMap[0].length : 0;
                const queue = [{ X: start.X, Y: start.Y, d: 0 }];
                const seen = {};
                seen[`${start.X}_${start.Y}`] = true;
                let head = 0;
                while (head < queue.length) {
                    const cur = queue[head++];
                    if (cur.X === goal.X && cur.Y === goal.Y) {
                        return cur.d;
                    }
                    const dirs = [[0, -1], [0, 1], [-1, 0], [1, 0]];
                    for (let i = 0; i < dirs.length; i++) {
                        const nx = cur.X + dirs[i][0];
                        const ny = cur.Y + dirs[i][1];
                        const key = `${nx}_${ny}`;
                        if (nx < 0 || ny < 0 || ny >= rows || nx >= cols || seen[key]) {
                            continue;
                        }
                        if (!(typeof window.IsAIWalkable === "function" ? window.IsAIWalkable(nx, ny) : isWalkableEmpty(nx, ny))) {
                            continue;
                        }
                        seen[key] = true;
                        queue.push({ X: nx, Y: ny, d: cur.d + 1 });
                    }
                }
                return 999;
            }

            function ensureOriginalNonRigidSnapshot() {
                if (window.__combatEvalEnv.originalNonRigidCells || !window.townBarrierMap) {
                    return;
                }
                const cells = [];
                for (let y = 0; y < window.townBarrierMap.length; y++) {
                    for (let x = 0; x < window.townBarrierMap[y].length; x++) {
                        const no = Number(window.townBarrierMap[y][x] || 0);
                        if (no > 0 && !isRigidBarrierNo(no)) {
                            cells.push({ X: x, Y: y, No: no });
                        }
                    }
                }
                window.__combatEvalEnv.originalNonRigidCells = cells;
            }

            function disposeBarrierAt(x, y) {
                if (!window.Barrier || !window.Barrier.Storage || !window.Barrier.Storage[y]) {
                    return;
                }
                const cell = window.Barrier.Storage[y][x];
                if (cell && cell.Object && typeof cell.Object.Dispose === "function") {
                    cell.Object.Dispose();
                }
                window.Barrier.Storage[y][x] = null;
            }

            function applyPartialClearMap() {
                ensureOriginalNonRigidSnapshot();
                if (!window.__combatEvalEnv.clearNonRigid || !window.townBarrierMap) {
                    return 0;
                }
                const cells = window.__combatEvalEnv.originalNonRigidCells || [];
                const clearRatio = window.__combatEvalEnv.specialBombEscape
                    ? 1
                    : (window.__combatEvalEnv.partialClearMinRatio
                        + rng() * Math.max(0, window.__combatEvalEnv.partialClearMaxRatio - window.__combatEvalEnv.partialClearMinRatio));
                let cleared = 0;
                for (let i = 0; i < cells.length; i++) {
                    const c = cells[i];
                    disposeBarrierAt(c.X, c.Y);
                    if (rng() < clearRatio) {
                        window.townBarrierMap[c.Y][c.X] = 0;
                        cleared += 1;
                    } else {
                        window.townBarrierMap[c.Y][c.X] = c.No;
                        if (window.Barrier && typeof window.Barrier.Create === "function") {
                            window.Barrier.Create(c.X, c.Y, c.No);
                        }
                    }
                }
                return cleared;
            }

            function collectSpawnAnchors() {
                const anchors = [];
                const roles = findPlayerAndAiRoles();
                const pMap = getRoleMap(roles.player);
                const aiMap = getRoleMap(roles.ai);
                if (pMap) anchors.push({ X: pMap.X, Y: pMap.Y });
                if (aiMap) anchors.push({ X: aiMap.X, Y: aiMap.Y });
                return anchors;
            }

            function isNearAnchors(x, y, anchors, minRadius) {
                for (let i = 0; i < anchors.length; i++) {
                    const a = anchors[i];
                    if (Math.abs(a.X - x) + Math.abs(a.Y - y) < minRadius) {
                        return true;
                    }
                }
                return false;
            }

            function isOccupiedByRole(x, y) {
                const state = window.singlePlayerState;
                const fighters = state && Array.isArray(state.Fighters) ? state.Fighters : [];
                for (let i = 0; i < fighters.length; i++) {
                    const role = fighters[i] ? fighters[i].role : null;
                    if (!role || role.IsDeath || typeof role.CurrentMapID !== "function") {
                        continue;
                    }
                    const map = role.CurrentMapID();
                    if (map && map.X === x && map.Y === y) {
                        return true;
                    }
                }
                return false;
            }

            function countItems() {
                let count = 0;
                if (!window.townBarrierMap || !Array.isArray(window.townBarrierMap)) {
                    return count;
                }
                for (let y = 0; y < window.townBarrierMap.length; y++) {
                    const row = window.townBarrierMap[y];
                    if (!Array.isArray(row)) {
                        continue;
                    }
                    for (let x = 0; x < row.length; x++) {
                        if (Number(row[x]) > 100) {
                            count += 1;
                        }
                    }
                }
                return count;
            }

            function rollMapItemParams() {
                const env = window.__combatEvalEnv;
                if (env.specialBombEscape) {
                    env.currentDensity = Math.max(0, Math.min(0.4, Number(env.randomItemDensity || 0.12)));
                    env.currentItemRespawnMs = Math.max(200, Number(env.specialItemRespawnMs || 2000));
                    env.currentItemSafeRadius = Math.max(0, Math.min(8, Number(env.itemSafeRadius || 2)));
                    return;
                }
                const densityNoise = (rng() * 2 - 1) * Math.max(0, Number(env.randomItemDensityJitter || 0));
                const density = Math.max(0, Math.min(0.4, Number(env.randomItemDensity || 0) + densityNoise));
                const respawnNoise = (rng() * 2 - 1) * Math.max(0, Number(env.itemRespawnJitterRatio || 0));
                const respawn = Math.max(220, Math.round(Number(env.itemRespawnMs || 0) * (1 + respawnNoise)));
                const safeNoise = Math.round((rng() * 2 - 1) * Math.max(0, Number(env.itemSafeRadiusJitter || 0)));
                const safeRadius = Math.max(0, Math.min(8, Number(env.itemSafeRadius || 0) + safeNoise));
                env.currentDensity = density;
                env.currentItemRespawnMs = respawn;
                env.currentItemSafeRadius = safeRadius;
            }

            function trySpawnRandomItems() {
                if (!window.townBarrierMap || !Array.isArray(window.townBarrierMap) || window.townBarrierMap.length === 0) {
                    return 0;
                }
                const rows = window.townBarrierMap.length;
                const cols = Array.isArray(window.townBarrierMap[0]) ? window.townBarrierMap[0].length : 0;
                if (cols <= 0) {
                    return 0;
                }
                const anchors = collectSpawnAnchors();
                const densityTarget = Math.floor(rows * cols * Math.max(0, Math.min(0.4, window.__combatEvalEnv.currentDensity || 0)));
                const hardCap = Math.max(0, Number(window.__combatEvalEnv.itemMax || 0));
                const target = Math.min(hardCap, densityTarget);
                if (target <= 0) {
                    return 0;
                }
                let current = countItems();
                let need = target - current;
                if (need <= 0) {
                    return 0;
                }
                let added = 0;
                let attempts = 0;
                const maxAttempts = rows * cols * 12;
                while (need > 0 && attempts < maxAttempts) {
                    attempts += 1;
                    const x = Math.floor(rng() * cols);
                    const y = Math.floor(rng() * rows);
                    if (window.townBarrierMap[y][x] !== 0) continue;
                    if (typeof window.IsAIWalkable === "function" && !window.IsAIWalkable(x, y)) continue;
                    if (isNearAnchors(x, y, anchors, window.__combatEvalEnv.currentItemSafeRadius || 0)) continue;
                    if (isOccupiedByRole(x, y)) continue;
                    if (window.PaopaoArray
                        && window.PaopaoArray[y]
                        && window.PaopaoArray[y][x]
                        && !window.PaopaoArray[y][x].IsExploded) {
                        continue;
                    }
                    const giftNo = typeof window.CreateRandomGift === "function"
                        ? window.CreateRandomGift()
                        : ([101, 102, 103][Math.floor(rng() * 3)]);
                    window.townBarrierMap[y][x] = giftNo;
                    if (window.Barrier && typeof window.Barrier.Create === "function") {
                        window.Barrier.Create(x, y, giftNo);
                    }
                    added += 1;
                    need -= 1;
                }
                return added;
            }

            function startItemRespawn() {
                stopItemRespawn();
                const respawnMs = Math.max(0, Number(window.__combatEvalEnv.currentItemRespawnMs || 0));
                if (respawnMs <= 0) {
                    return;
                }
                window.__combatEvalItemRespawnTicker = setInterval(function() {
                    if (!window.gameRunning) {
                        return;
                    }
                    trySpawnRandomItems();
                }, respawnMs);
            }

            function tuneRole(role, profile) {
                if (!role) {
                    return;
                }
                const bubbleCap = profile === "power"
                    ? 2 + Math.floor(rng() * 3)
                    : 1 + Math.floor(rng() * 3);
                const strong = profile === "power"
                    ? 2 + Math.floor(rng() * 5)
                    : 1 + Math.floor(rng() * 4);
                role.CanPaopaoLength = Math.max(1, bubbleCap);
                role.PaopaoStrong = Math.max(1, strong);
                role.PaopaoCount = 0;
                role.LastBombAt = 0;
                if (typeof role.SetMoveSpeedPxPerSec === "function") {
                    role.SetMoveSpeedPxPerSec(105 + Math.floor(rng() * 80));
                }
            }

            function pickSpawnPairByShortestPath() {
                const cells = collectWalkableCells();
                if (!cells.length) {
                    return null;
                }
                const buckets = [
                    [Math.max(1, window.__combatEvalEnv.spawnShortestPathMin), Math.min(3, window.__combatEvalEnv.spawnShortestPathMax)],
                    [Math.max(4, window.__combatEvalEnv.spawnShortestPathMin), Math.min(6, window.__combatEvalEnv.spawnShortestPathMax)],
                    [Math.max(7, window.__combatEvalEnv.spawnShortestPathMin), Math.min(10, window.__combatEvalEnv.spawnShortestPathMax)]
                ];
                const targetBucket = buckets[window.__combatEvalEnv.spawnBucketTarget] || buckets[0];
                const tries = Math.max(120, cells.length * 3);
                let best = null;
                let bestScore = -1e9;
                for (let i = 0; i < tries; i++) {
                    const aiCell = cells[Math.floor(rng() * cells.length)];
                    const playerCell = cells[Math.floor(rng() * cells.length)];
                    if (!aiCell || !playerCell) continue;
                    if (aiCell.X === playerCell.X && aiCell.Y === playerCell.Y) continue;
                    const pathDist = bfsDistance(aiCell, playerCell);
                    if (pathDist >= 999) continue;
                    const inTarget = pathDist >= targetBucket[0] && pathDist <= targetBucket[1];
                    const inGlobal = pathDist >= window.__combatEvalEnv.spawnShortestPathMin
                        && pathDist <= window.__combatEvalEnv.spawnShortestPathMax;
                    let score = inTarget ? 100 : (inGlobal ? 40 : -Math.abs(pathDist - targetBucket[1]));
                    score += Math.max(0, 10 - pathDist) * 0.5;
                    score += rng();
                    if (score > bestScore) {
                        bestScore = score;
                        best = { aiCell, playerCell, pathDist };
                    }
                }
                return best;
            }

            function placeRoles() {
                const roles = findPlayerAndAiRoles();
                const player = roles.player;
                const ai = roles.ai;
                if (!player || !ai || typeof player.SetToMap !== "function" || typeof ai.SetToMap !== "function") {
                    return;
                }
                const pair = pickSpawnPairByShortestPath();
                if (!pair || !pair.aiCell || !pair.playerCell) {
                    return;
                }
                ai.SetToMap(pair.aiCell.X, pair.aiCell.Y);
                player.SetToMap(pair.playerCell.X, pair.playerCell.Y);
                tuneRole(ai, rng() < 0.45 ? "power" : "normal");
                tuneRole(player, rng() < 0.45 ? "power" : "normal");
                window.__combatEvalEnv.currentEpisodeMeta = {
                    spawnShortestPathDist: pair.pathDist,
                    spawnShortestPathDistNorm: Math.max(0, Math.min(1, pair.pathDist / 28)),
                    minEnemyDist: pair.pathDist,
                    lastAiThreatTs: 0,
                    myBombThreatScore: 0,
                    closeRangeDuelScore: Math.max(0, Math.min(1, 1 - pair.pathDist / 10)),
                    winningBombSourceRecent: 0,
                };
            }

            function refreshEpisodeMeta() {
                const roles = findPlayerAndAiRoles();
                const player = roles.player;
                const ai = roles.ai;
                const meta = window.__combatEvalEnv.currentEpisodeMeta || {};
                if (!player || !ai || typeof player.CurrentMapID !== "function" || typeof ai.CurrentMapID !== "function") {
                    return meta;
                }
                const pMap = player.CurrentMapID();
                const aiMap = ai.CurrentMapID();
                if (!pMap || !aiMap) {
                    return meta;
                }
                const pathDist = bfsDistance(aiMap, pMap);
                if (Number.isFinite(pathDist) && pathDist < 999) {
                    meta.minEnemyDist = Math.min(Number(meta.minEnemyDist || pathDist), pathDist);
                    meta.closeRangeDuelScore = Math.max(Number(meta.closeRangeDuelScore || 0), Math.max(0, Math.min(1, 1 - pathDist / 10)));
                }
                if (Array.isArray(window.PaopaoArray)) {
                    let threatScore = 0;
                    const dangerMarks = {};
                    for (let y = 0; y < window.PaopaoArray.length; y++) {
                        const row = window.PaopaoArray[y];
                        if (!Array.isArray(row)) continue;
                        for (let x = 0; x < row.length; x++) {
                            const bomb = row[x];
                            if (!bomb || bomb.IsExploded || !bomb.Master || bomb.Master !== ai) continue;
                            const power = Math.max(1, Number(bomb.PaopaoStrong || ai.PaopaoStrong || 1));
                            dangerMarks[`${x}_${y}`] = 1;
                            for (let step = 1; step <= power; step++) {
                                if (x - step >= 0) dangerMarks[`${x - step}_${y}`] = 1;
                                if (x + step <= 14) dangerMarks[`${x + step}_${y}`] = 1;
                                if (y - step >= 0) dangerMarks[`${x}_${y - step}`] = 1;
                                if (y + step <= 12) dangerMarks[`${x}_${y + step}`] = 1;
                            }
                            const dist = Math.abs(pMap.X - x) + Math.abs(pMap.Y - y);
                            if (dist <= Math.max(1, Number(bomb.PaopaoStrong || ai.PaopaoStrong || 1))) {
                                threatScore = Math.max(threatScore, Math.max(0, Math.min(1, 1 - dist / 6)));
                            }
                        }
                    }
                    if (threatScore > 0) {
                        meta.lastAiThreatTs = Date.now();
                        meta.myBombThreatScore = Math.max(Number(meta.myBombThreatScore || 0), threatScore);
                    }
                    meta.dangerCellsCreatedScore = Math.max(
                        Number(meta.dangerCellsCreatedScore || 0),
                        clamp01(Object.keys(dangerMarks).length / 20)
                    );
                }
                const aiKills = ai ? Number(ai.kills || 0) : 0;
                const playerKills = player ? Number(player.kills || 0) : 0;
                const aiDeaths = ai ? Number(ai.deaths || 0) : 0;
                const aiSelfKills = Math.max(0, aiDeaths - playerKills);
                const roundSec = Math.max(10, Number(window.__combatEvalEnv.specialRoundSec || window.__combatEvalEnv.matchDurationSec || 60));
                meta.roundKillCredit = clamp01(aiKills / Math.max(1, roundSec / 8));
                meta.roundSelfKillPenalty = clamp01(aiSelfKills / Math.max(1, roundSec / 10));
                meta.roundNetKdCredit = clamp01((aiKills - aiSelfKills + 6) / 12);
                window.__combatEvalEnv.currentEpisodeMeta = meta;
                return meta;
            }

            function finalizeCurrentMatch(reason) {
                if (window.__combatEvalEnv.matchFinalized) {
                    return window.__combatEvalEnv.lastTerminal;
                }
                const roles = findPlayerAndAiRoles();
                const player = roles.player;
                const ai = roles.ai;
                const meta = refreshEpisodeMeta();
                const aiThreatRecent = Number(meta.lastAiThreatTs || 0) > 0 && (Date.now() - Number(meta.lastAiThreatTs || 0)) <= 2600;
                let terminalReason = String(reason || "stall_abort");
                let result = "draw";
                let aiKills = 0;
                let aiDeaths = 0;
                let playerKills = 0;
                let playerDeaths = 0;
                let aiSelfKills = 0;

                if (window.__combatEvalEnv.specialBombEscape) {
                    aiKills = ai ? Number(ai.kills || 0) : 0;
                    playerKills = player ? Number(player.kills || 0) : 0;
                    aiDeaths = ai ? Number(ai.deaths || 0) : 0;
                    playerDeaths = player ? Number(player.deaths || 0) : 0;
                    aiSelfKills = Math.max(0, aiDeaths - playerKills);
                    terminalReason = "round_end";
                    if (aiKills > playerKills) result = "win";
                    else if (aiKills < playerKills) result = "loss";
                    else result = "draw";
                    const playerPowerSpecial = computeRolePower(player);
                    const aiPowerSpecial = computeRolePower(ai);
                    const itemControlSpecial = aiPowerSpecial / Math.max(1, aiPowerSpecial + playerPowerSpecial);
                    window.__combatEvalEnv.matchFinalized = true;
                    window.__combatEvalEnv.currentEpisodeMeta = meta;
                    window.__combatEvalEnv.lastTerminal = {
                        result,
                        terminal_reason: terminalReason,
                        ai_kills: aiKills,
                        ai_deaths: aiDeaths,
                        ai_self_kills: aiSelfKills,
                        player_kills: playerKills,
                        player_deaths: playerDeaths,
                        ai_power: aiPowerSpecial,
                        player_power: playerPowerSpecial,
                        item_control: itemControlSpecial,
                        spawn_shortest_path_dist: Number(meta.spawnShortestPathDist || null),
                        my_bomb_threat_score: Number(meta.myBombThreatScore || 0),
                        close_range_duel_score: Number(meta.closeRangeDuelScore || 0),
                        winning_bomb_source_recent: Number(meta.winningBombSourceRecent || 0),
                        danger_cells_created_score: Number(meta.dangerCellsCreatedScore || 0),
                        round_kill_credit: Number(meta.roundKillCredit || 0),
                        round_self_kill_penalty: Number(meta.roundSelfKillPenalty || 0),
                        round_net_kd_credit: Number(meta.roundNetKdCredit || 0),
                        ts: Date.now(),
                    };
                    window.gameRunning = false;
                    stopIntervals();
                    return window.__combatEvalEnv.lastTerminal;
                }

                if (terminalReason === "caught_enemy") {
                    if (window.__combatEvalEnv.ignoreEnemySelfKill && !aiThreatRecent && Number(meta.myBombThreatScore || 0) < 0.20) {
                        terminalReason = "enemy_self_kill_discard";
                        result = "draw";
                        playerDeaths = 1;
                    } else {
                        result = "win";
                        aiKills = 1;
                        playerDeaths = 1;
                        meta.winningBombSourceRecent = aiThreatRecent ? 1 : 0;
                    }
                } else if (terminalReason === "caught_self") {
                    result = "loss";
                    aiDeaths = 1;
                    aiSelfKills = 1;
                    playerKills = 1;
                } else {
                    terminalReason = "stall_abort";
                    result = "draw";
                }

                const playerPower = computeRolePower(player);
                const aiPower = computeRolePower(ai);
                const itemControl = aiPower / Math.max(1, aiPower + playerPower);

                window.__combatEvalEnv.matchFinalized = true;
                window.__combatEvalEnv.currentEpisodeMeta = meta;
                window.__combatEvalEnv.lastTerminal = {
                    result,
                    terminal_reason: terminalReason,
                    ai_kills: aiKills,
                    ai_deaths: aiDeaths,
                    ai_self_kills: aiSelfKills,
                    player_kills: playerKills,
                    player_deaths: playerDeaths,
                    ai_power: aiPower,
                    player_power: playerPower,
                    item_control: itemControl,
                    spawn_shortest_path_dist: Number(meta.spawnShortestPathDist || null),
                    my_bomb_threat_score: Number(meta.myBombThreatScore || 0),
                    close_range_duel_score: Number(meta.closeRangeDuelScore || 0),
                    winning_bomb_source_recent: Number(meta.winningBombSourceRecent || 0),
                    ts: Date.now(),
                };
                window.gameRunning = false;
                stopIntervals();
                return window.__combatEvalEnv.lastTerminal;
            }

            function applySpecialBombEscapePatch() {
                if (!window.__combatEvalEnv.specialBombEscape || !window.Role || !window.Role.prototype) {
                    return;
                }
                if (window.__combatEvalEnv.__specialBombPatched) {
                    return;
                }
                const originalBomb = window.Role.prototype.Bomb;
                if (typeof originalBomb !== "function") {
                    return;
                }
                window.Role.prototype.Bomb = function(attacker, forceTrap) {
                    if (this.DismountProtectionUntil > Date.now() || this.ExplosionImmuneUntil > Date.now()) {
                        return;
                    }
                    if (this.IsDeath || this.IsInPaopao) {
                        return;
                    }
                    if (attacker != null) {
                        this.LastAttacker = attacker;
                    }
                    if (typeof this.OnBombed === "function") {
                        this.OnBombed(this, attacker, !!forceTrap);
                    }
                    if (this.MoveHorse != null && this.MoveHorse !== MoveHorseObject.None && typeof this.OutRide === "function") {
                        this.OutRide(false);
                    }
                    if (typeof this.OnDeath === "function") {
                        this.OnDeath(this, this.LastAttacker || attacker || null);
                    }
                };
                window.__combatEvalEnv.__specialBombPatched = true;
            }

            function startOpponentController(mode) {
                if (window.__combatEvalController) {
                    clearInterval(window.__combatEvalController);
                    window.__combatEvalController = null;
                }
                window.__combatEvalController = setInterval(function() {
                    if (!window.singlePlayerState || !window.gameRunning) {
                        return;
                    }
                    const player = window.singlePlayerState.Player;
                    if (!player || player.IsDeath || player.IsInPaopao || typeof player.CurrentMapID !== "function") {
                        return;
                    }
                    if (mode === "stationary_dummy") {
                        player.Stop();
                        return;
                    }
                    const playerMap = player.CurrentMapID();
                    if (!playerMap) {
                        return;
                    }
                    let aiMap = null;
                    const fighters = window.singlePlayerState && Array.isArray(window.singlePlayerState.Fighters)
                        ? window.singlePlayerState.Fighters
                        : [];
                    for (let i = 0; i < fighters.length; i++) {
                        if (fighters[i] && typeof fighters[i].id === "string" && fighters[i].id.indexOf("ai_") === 0) {
                            const role = fighters[i].role;
                            if (role && !role.IsDeath && typeof role.CurrentMapID === "function") {
                                aiMap = role.CurrentMapID();
                                break;
                            }
                        }
                    }
                    const snapshot = (typeof window.BuildThreatSnapshot === "function") ? window.BuildThreatSnapshot() : null;
                    const dirs = [
                        { key: 38, dx: 0, dy: -1 },
                        { key: 40, dx: 0, dy: 1 },
                        { key: 37, dx: -1, dy: 0 },
                        { key: 39, dx: 1, dy: 0 }
                    ];
                    let best = null;
                    let bestScore = -1e9;
                    for (let i = 0; i < dirs.length; i++) {
                        const d = dirs[i];
                        const nx = playerMap.X + d.dx;
                        const ny = playerMap.Y + d.dy;
                        if (!(typeof window.IsAIWalkable === "function" && window.IsAIWalkable(nx, ny))) {
                            continue;
                        }
                        const key = typeof window.MapKey === "function" ? window.MapKey(nx, ny) : (nx + "_" + ny);
                        const isThreat = !!(snapshot && snapshot.threatMap && snapshot.threatMap[key]);
                        const eta = snapshot && snapshot.dangerEtaMap && typeof snapshot.dangerEtaMap[key] === "number"
                            ? snapshot.dangerEtaMap[key]
                            : null;
                        const safeN = typeof window.CountSafeNeighborTiles === "function"
                            ? window.CountSafeNeighborTiles(nx, ny, snapshot)
                            : 0;
                        const distToAi = aiMap ? Math.abs(aiMap.X - nx) + Math.abs(aiMap.Y - ny) : 8;
                        let score = 0;
                        score += safeN * 10;
                        score += isThreat ? -950 : 26;
                        score += typeof eta === "number" ? Math.min(eta, 1200) / 110 : 6;
                        if (mode === "aggressive_trapper") {
                            score += Math.max(0, 7 - distToAi) * 12.0;
                        } else if (mode === "coward_runner") {
                            score += Math.max(0, distToAi - 1) * 8.0;
                        } else if (mode === "item_rusher" && window.townBarrierMap && window.townBarrierMap[ny] && Number(window.townBarrierMap[ny][nx]) > 100) {
                            score += 120;
                        } else if (mode === "randomized_mistake_bot" && rng() < 0.18) {
                            score += rng() * 80;
                        } else if (mode === "heuristic_v2") {
                            score += Math.max(0, 6 - distToAi) * 7.5;
                        } else {
                            score += Math.max(0, distToAi - 2) * 2.2;
                        }
                        score += (rng() - 0.5) * 2.0;
                        if (score > bestScore) {
                            bestScore = score;
                            best = d;
                        }
                    }
                    if (best && typeof window.RoleKeyEvent === "function" && typeof window.RoleKeyEventEnd === "function") {
                        window.RoleKeyEvent(best.key, player);
                        setTimeout(function() {
                            window.RoleKeyEventEnd(best.key, player);
                        }, 70);
                    }
                    if (aiMap && player.CanPaopaoLength > player.PaopaoCount && typeof window.RoleKeyEvent === "function") {
                        const dist = Math.abs(aiMap.X - playerMap.X) + Math.abs(aiMap.Y - playerMap.Y);
                        const bombProb = mode === "aggressive_trapper"
                            ? 0.42
                            : (mode === "randomized_mistake_bot" ? 0.18 : (mode === "heuristic_v2" ? 0.24 : 0.10));
                        const bombRange = mode === "aggressive_trapper" ? 4 : 2;
                        if (dist <= bombRange && rng() < bombProb) {
                            window.RoleKeyEvent(32, player);
                        }
                    }
                }, Math.max(60, runtimeCfg.opponentThinkMs || 95));
            }

            function startMonitor() {
                if (window.__combatEvalMonitor) {
                    clearInterval(window.__combatEvalMonitor);
                }
                window.__combatEvalMonitor = setInterval(function() {
                    if (!window.gameRunning || window.__combatEvalEnv.matchFinalized) {
                        return;
                    }
                    const roles = findPlayerAndAiRoles();
                    const playerTrapped = !!(roles.player && !roles.player.IsDeath && roles.player.IsInPaopao);
                    const aiTrapped = !!(roles.ai && !roles.ai.IsDeath && roles.ai.IsInPaopao);
                    const meta = refreshEpisodeMeta();
                    const playerMap = getRoleMap(roles.player);
                    const aiMap = getRoleMap(roles.ai);
                    const currentSig = JSON.stringify({
                        player: playerMap,
                        ai: aiMap,
                        bombs: typeof window.CountActiveBombs === "function" ? window.CountActiveBombs() : 0,
                        items: countItems(),
                        minEnemyDist: meta && typeof meta.minEnemyDist === "number" ? meta.minEnemyDist : null,
                        threat: meta && typeof meta.myBombThreatScore === "number" ? meta.myBombThreatScore : 0
                    });
                    if (window.__combatEvalEnv.lastProgressSignature !== currentSig) {
                        window.__combatEvalEnv.lastProgressSignature = currentSig;
                        window.__combatEvalEnv.lastProgressAt = Date.now();
                    }
                    const stalled = !!(window.__combatEvalEnv.lastProgressAt
                        && (Date.now() - window.__combatEvalEnv.lastProgressAt >= window.__combatEvalEnv.stallNoProgressMs));
                    const specialRoundEnded = !!(window.__combatEvalEnv.specialBombEscape
                        && window.__combatEvalEnv.matchStartAt
                        && (Date.now() - window.__combatEvalEnv.matchStartAt >= Math.max(1, runtimeCfg.matchDurationSec) * 1000));
                    if (specialRoundEnded) {
                        finalizeCurrentMatch("round_end");
                        return;
                    }
                    if (playerTrapped || aiTrapped || stalled) {
                        finalizeCurrentMatch(playerTrapped ? "caught_enemy" : (aiTrapped ? "caught_self" : "stall_abort"));
                    }
                }, 120);
            }

            if (typeof window.BNBMLRefreshConfig === "function") {
                window.BNBMLRefreshConfig();
            }
            if (typeof window.ApplySelectedGameMap === "function" && runtimeCfg.mapId) {
                window.ApplySelectedGameMap(runtimeCfg.mapId, false);
            }
            if (typeof window.SetAIEnemyCount === "function") {
                window.SetAIEnemyCount(1, false);
            }
            window.roundDurationSeconds = runtimeCfg.suddenDeath ? 300 : runtimeCfg.matchDurationSec;
            if (typeof window.StartSinglePlayerGame === "function") {
                window.StartSinglePlayerGame(1);
            }
            if (window.__combatEvalEnv.specialBombEscape) {
                window.respawnDelayMs = Math.max(0, Number(window.__combatEvalEnv.specialRespawnDelayMs || 0));
                window.respawnInvincibleMs = Math.max(0, Number(window.__combatEvalEnv.specialRespawnInvincibleMs || 300));
                applySpecialBombEscapePatch();
            } else if (window.__combatEvalEnv.disableRevive) {
                window.respawnDelayMs = 999999;
                window.respawnInvincibleMs = 0;
            }
            window.__combatEvalEnv.matchFinalized = false;
            window.__combatEvalEnv.lastTerminal = null;
            window.__combatEvalEnv.lastProgressAt = Date.now();
            window.__combatEvalEnv.lastProgressSignature = "";
            window.__combatEvalEnv.matchStartAt = Date.now();

            setTimeout(function() {
                rollMapItemParams();
                applyPartialClearMap();
                placeRoles();
                trySpawnRandomItems();
                startItemRespawn();
                startOpponentController(runtimeCfg.opponentMode);
                startMonitor();
            }, 180);
        }, {
            seed,
            mapId: cfg.mapId,
            matchDurationSec: cfg.matchDurationSec,
            opponentMode: cfg.opponentMode,
            opponentThinkMs: cfg.opponentThinkMs,
            clearNonRigid: cfg.clearNonRigid,
            randomItemDensity: cfg.randomItemDensity,
            randomItemDensityJitter: cfg.randomItemDensityJitter,
            itemRespawnMs: cfg.itemRespawnMs,
            itemRespawnJitterRatio: cfg.itemRespawnJitterRatio,
            itemSafeRadius: cfg.itemSafeRadius,
            itemSafeRadiusJitter: cfg.itemSafeRadiusJitter,
            itemMax: cfg.itemMax,
            spawnBucketTarget: runIndex % 3,
            suddenDeath: cfg.suddenDeath,
            disableRevive: cfg.disableRevive,
            ignoreEnemySelfKill: cfg.ignoreEnemySelfKill,
            stallNoProgressMs: cfg.stallNoProgressMs,
            partialClearMinRatio: cfg.partialClearMinRatio,
            partialClearMaxRatio: cfg.partialClearMaxRatio,
            spawnShortestPathMin: cfg.spawnShortestPathMin,
            spawnShortestPathMax: cfg.spawnShortestPathMax,
            evalProfile: cfg.evalProfile,
            specialBombEscape: !!cfg.specialBombEscape,
            specialRoundSec: cfg.specialRoundSec,
            specialItemRespawnMs: cfg.specialItemRespawnMs,
            specialRespawnDelayMs: cfg.specialRespawnDelayMs,
            specialRespawnInvincibleMs: cfg.specialRespawnInvincibleMs,
        });

        const timeoutMs = (cfg.matchDurationSec + 8) * 1000;
        const startedAt = Date.now();
        let loopTimedOut = false;
        while (Date.now() - startedAt < timeoutMs) {
            if (timedOut) {
                throw new Error("match_hard_timeout");
            }
            const done = await page.evaluate(() => {
                const env = window.__combatEvalEnv || null;
                return !window.gameRunning || !!(env && env.matchFinalized);
            });
            if (done) {
                break;
            }
            if (previewEnabled && live) {
                const elapsedSec = (Date.now() - startedAt) / 1000;
                live.touchPreview(runNo, elapsedSec);
                if (live.framePath && Date.now() - lastFrameAt >= cfg.liveFrameMs) {
                    try {
                        await page.screenshot({ path: live.framePath, fullPage: true });
                    } catch (err) {
                        // ignore transient screenshot failures during page transitions
                    }
                    lastFrameAt = Date.now();
                }
            }
            await page.waitForTimeout(250);
        }
        if (Date.now() - startedAt >= timeoutMs) {
            loopTimedOut = true;
        }

        const match = await page.evaluate((payload) => {
            const runNo = payload && typeof payload.runNo === "number" ? payload.runNo : 0;
            const timedOutByLoop = !!(payload && payload.timedOutByLoop);
            if (window.__combatEvalController) {
                clearInterval(window.__combatEvalController);
                window.__combatEvalController = null;
            }
            if (window.__combatEvalMonitor) {
                clearInterval(window.__combatEvalMonitor);
                window.__combatEvalMonitor = null;
            }
            if (window.__combatEvalItemRespawnTicker) {
                clearInterval(window.__combatEvalItemRespawnTicker);
                window.__combatEvalItemRespawnTicker = null;
            }

            const state = window.singlePlayerState || null;
            const env = window.__combatEvalEnv || null;
            const runtime = window.BNBMLRuntimeState || null;
            const terminal = env && env.lastTerminal
                ? env.lastTerminal
                : {
                    result: timedOutByLoop ? "draw" : "error",
                    terminal_reason: timedOutByLoop ? "stall_abort" : "stall_abort",
                    ai_kills: 0,
                    ai_deaths: 0,
                    ai_self_kills: 0,
                    player_kills: 0,
                    player_deaths: 0,
                    ai_power: 0,
                    player_power: 0,
                    item_control: 0.5,
                    spawn_shortest_path_dist: null,
                    my_bomb_threat_score: 0,
                    close_range_duel_score: 0,
                    winning_bomb_source_recent: 0,
                };

            return {
                run_index: runNo,
                result: terminal.result,
                player_kills: terminal.player_kills || 0,
                player_deaths: terminal.player_deaths || 0,
                ai_kills: terminal.ai_kills || 0,
                ai_deaths: terminal.ai_deaths || 0,
                ai_self_kills: terminal.ai_self_kills || 0,
                ai_power: terminal.ai_power || 0,
                player_power: terminal.player_power || 0,
                item_control: Number(terminal.item_control || 0.5),
                terminal_reason: terminal.terminal_reason || "stall_abort",
                spawn_shortest_path_dist: terminal.spawn_shortest_path_dist,
                my_bomb_threat_score: Number(terminal.my_bomb_threat_score || 0),
                close_range_duel_score: Number(terminal.close_range_duel_score || 0),
                winning_bomb_source_recent: Number(terminal.winning_bomb_source_recent || 0),
                danger_cells_created_score: Number(terminal.danger_cells_created_score || 0),
                round_kill_credit: Number(terminal.round_kill_credit || 0),
                round_self_kill_penalty: Number(terminal.round_self_kill_penalty || 0),
                round_net_kd_credit: Number(terminal.round_net_kd_credit || 0),
                runtime_enabled: !!(runtime && runtime.enabled),
                runtime_loaded: !!(runtime && runtime.loaded),
                runtime_loading: !!(runtime && runtime.loading),
                runtime_inference_count: Number(runtime && runtime.inference_count || 0),
                runtime_error: String(runtime && runtime.error || ""),
                runtime_sequence_path_hits: Number(runtime && runtime.sequence_path_hits || 0),
                runtime_single_path_hits: Number(runtime && runtime.single_path_hits || 0),
                runtime_sequence_path_hit_rate: Number(runtime && runtime.sequence_path_hit_rate || 0),
                runtime_model_input_mode: String(runtime && runtime.model_input_mode || ""),
                runtime_model_sequence_len: Number(runtime && runtime.model_sequence_len || 1),
                remaining_seconds: state && typeof state.RemainingSeconds === "number" ? state.RemainingSeconds : null,
                game_running: !!window.gameRunning,
                loop_timeout: !!timedOutByLoop
            };
        }, { runNo: runIndex + 1, timedOutByLoop: loopTimedOut });

        if (cfg.saveScreenshots) {
            const screenshotPath = path.join(OUT_DIR, `combat_1v1_eval_match_${String(runIndex + 1).padStart(3, "0")}.png`);
            await page.screenshot({ path: screenshotPath, fullPage: true });
            match.screenshot = screenshotPath;
        }

        return match;
    } catch (err) {
        return {
            run_index: runNo,
            result: "error",
            player_kills: 0,
            player_deaths: 0,
            ai_kills: 0,
            ai_deaths: 0,
            ai_self_kills: 0,
            ai_power: 0,
            player_power: 0,
            item_control: 0.5,
            remaining_seconds: null,
            game_running: false,
            error: String(err && err.message ? err.message : err),
        };
    } finally {
        if (previewEnabled && live) {
            live.releasePreview(runNo);
        }
        clearTimeout(hardTimer);
        if (browser) {
            await browser.close().catch(() => {});
        }
    }
}

function estimateEloDelta(score) {
    const s = Math.min(0.99, Math.max(0.01, score));
    const delta = 400 * Math.log10(s / (1 - s));
    return Math.max(-800, Math.min(800, delta));
}

function summarize(matches) {
    let win = 0;
    let loss = 0;
    let draw = 0;
    let error = 0;
    let aiKills = 0;
    let aiDeaths = 0;
    let aiSelfKills = 0;
    let itemControlSum = 0;
    let stallDraw = 0;
    let opponentSelfKill = 0;
    let closeRangeWins = 0;
    let closeRangeGames = 0;
    let threatenedBombFinishes = 0;
    let threatenedBombOpportunities = 0;
    let dangerCellsScoreSum = 0;
    let runtimeSequenceHits = 0;
    let runtimeSingleHits = 0;
    const terminalReasonHist = { caught_enemy: 0, caught_self: 0, enemy_self_kill_discard: 0, stall_abort: 0, round_end: 0 };
    const spawnDistHist = { "1_3": 0, "4_6": 0, "7_10": 0, other: 0 };

    for (const m of matches) {
        if (m.result === "win") win += 1;
        else if (m.result === "loss") loss += 1;
        else if (m.result === "draw") draw += 1;
        else error += 1;

        aiKills += m.ai_kills || 0;
        aiDeaths += m.ai_deaths || 0;
        aiSelfKills += m.ai_self_kills || 0;
        itemControlSum += Number(m.item_control || 0);
        if (m.terminal_reason && terminalReasonHist[m.terminal_reason] != null) {
            terminalReasonHist[m.terminal_reason] += 1;
        }
        if (m.terminal_reason === "stall_abort") {
            stallDraw += 1;
        }
        if (m.terminal_reason === "enemy_self_kill_discard") {
            opponentSelfKill += 1;
        }
        const spawnDist = Number(m.spawn_shortest_path_dist);
        if (Number.isFinite(spawnDist)) {
            if (spawnDist >= 1 && spawnDist <= 3) spawnDistHist["1_3"] += 1;
            else if (spawnDist >= 4 && spawnDist <= 6) spawnDistHist["4_6"] += 1;
            else if (spawnDist >= 7 && spawnDist <= 10) spawnDistHist["7_10"] += 1;
            else spawnDistHist.other += 1;
            if (spawnDist <= 3) {
                closeRangeGames += 1;
                if (m.result === "win") {
                    closeRangeWins += 1;
                }
            }
        }
        if (Number(m.my_bomb_threat_score || 0) > 0.25) {
            threatenedBombOpportunities += 1;
            if (m.result === "win" && Number(m.winning_bomb_source_recent || 0) > 0.5) {
                threatenedBombFinishes += 1;
            }
        }
        dangerCellsScoreSum += Number(m.danger_cells_created_score || 0);
        runtimeSequenceHits += Number(m.runtime_sequence_path_hits || 0);
        runtimeSingleHits += Number(m.runtime_single_path_hits || 0);
    }

    const total = Math.max(1, matches.length);
    const score = (win + 0.5 * draw) / total;
    const eloDelta = estimateEloDelta(score);

    return {
        matches: matches.length,
        win_count: win,
        loss_count: loss,
        draw_count: draw,
        error_count: error,
        win_rate: win / total,
        loss_rate: loss / total,
        draw_rate: draw / total,
        stall_draw_rate: stallDraw / total,
        error_rate: error / total,
        ai_kills_total: aiKills,
        ai_deaths_total: aiDeaths,
        ai_self_kills_total: aiSelfKills,
        opponent_self_kill_rate: opponentSelfKill / total,
        self_kill_rate: aiSelfKills / Math.max(1, aiDeaths),
        ai_kd: aiKills / Math.max(1, aiDeaths),
        item_control: itemControlSum / total,
        terminal_reason_hist: terminalReasonHist,
        spawn_dist_hist: spawnDistHist,
        close_range_win_rate: closeRangeWins / Math.max(1, closeRangeGames),
        threatened_bomb_finish_rate: threatenedBombFinishes / Math.max(1, threatenedBombOpportunities),
        kills_per_min: aiKills / Math.max(1, total),
        self_kills_per_min: aiSelfKills / Math.max(1, total),
        net_kd: (aiKills - aiSelfKills) / Math.max(1, total),
        danger_cells_per_min: dangerCellsScoreSum / Math.max(1, total),
        sequence_path_hit_rate: runtimeSequenceHits / Math.max(1, runtimeSequenceHits + runtimeSingleHits),
        sequence_path_hits: runtimeSequenceHits,
        single_path_hits: runtimeSingleHits,
        elo_result: {
            ai_baseline: 1000,
            opponent_baseline: 1000,
            ai_estimated: 1000 + eloDelta,
            opponent_estimated: 1000 - eloDelta,
            ai_delta: eloDelta,
            score
        }
    };
}

function createLiveBridge(cfg) {
    const enabled = !!(cfg && cfg.enabled);
    const statusPath = cfg ? cfg.statusPath : null;
    const framePath = cfg ? cfg.framePath : null;
    const viewerUrl = cfg ? cfg.viewerUrl : "";
    const state = {
        ts: Date.now(),
        heartbeat: Date.now(),
        active: false,
        error: null,
        report_path: null,
        map_id: null,
        opponent_mode: null,
        model_url: null,
        progress: {
            total: 0,
            completed: 0,
            parallel: 0
        },
        current_run_index: null,
        current_match_elapsed_sec: null,
        summary: {
            win_rate: 0,
            self_kill_rate: 0,
            draw_rate: 0,
            item_control: 0,
            elo_delta: 0
        },
        last_matches: [],
        viewer_url: viewerUrl
    };
    let previewOwner = null;
    let lastWrite = 0;

    function write(force) {
        if (!enabled || !statusPath) {
            return;
        }
        const now = Date.now();
        if (!force && now - lastWrite < 120) {
            return;
        }
        state.ts = now;
        state.heartbeat = now;
        fs.writeFileSync(statusPath, JSON.stringify(state, null, 2));
        lastWrite = now;
    }

    function updateSummary(summary) {
        const elo = summary && summary.elo_result ? summary.elo_result.ai_delta : 0;
        state.summary = {
            win_rate: Number(summary && summary.win_rate || 0),
            self_kill_rate: Number(summary && summary.self_kill_rate || 0),
            draw_rate: Number(summary && summary.draw_rate || 0),
            item_control: Number(summary && summary.item_control || 0),
            elo_delta: Number(elo || 0)
        };
    }

    return {
        enabled,
        framePath,
        start(meta) {
            if (!enabled) return;
            state.active = true;
            state.error = null;
            state.report_path = null;
            state.map_id = meta.mapId;
            state.opponent_mode = meta.opponentMode;
            state.model_url = meta.modelUrl;
            state.progress.total = meta.runs || 0;
            state.progress.completed = 0;
            state.progress.parallel = meta.parallel || 1;
            state.current_run_index = null;
            state.current_match_elapsed_sec = null;
            state.last_matches = [];
            updateSummary({});
            write(true);
        },
        tryClaimPreview(runNo) {
            if (!enabled) return false;
            if (previewOwner == null) {
                previewOwner = runNo;
                state.current_run_index = runNo;
                state.current_match_elapsed_sec = 0;
                write();
                return true;
            }
            return previewOwner === runNo;
        },
        touchPreview(runNo, elapsedSec) {
            if (!enabled) return;
            if (previewOwner !== runNo) return;
            state.current_run_index = runNo;
            state.current_match_elapsed_sec = Number(elapsedSec || 0);
            write();
        },
        releasePreview(runNo) {
            if (!enabled) return;
            if (previewOwner === runNo) {
                previewOwner = null;
                state.current_match_elapsed_sec = null;
                write();
            }
        },
        updateProgress(completed, total, parallel, summary) {
            if (!enabled) return;
            state.progress.completed = completed;
            state.progress.total = total;
            state.progress.parallel = parallel;
            updateSummary(summary);
            write();
        },
        pushMatch(match) {
            if (!enabled) return;
            state.last_matches.push(match);
            if (state.last_matches.length > 24) {
                state.last_matches = state.last_matches.slice(-24);
            }
            if (previewOwner === match.run_index) {
                previewOwner = null;
                state.current_match_elapsed_sec = null;
            }
            write();
        },
        done(summary, reportPath) {
            if (!enabled) return;
            state.active = false;
            state.report_path = reportPath;
            updateSummary(summary);
            write(true);
        },
        fail(err) {
            if (!enabled) return;
            state.active = false;
            state.error = String(err && err.message ? err.message : err);
            write(true);
        }
    };
}

async function main() {
    const evalProfileRaw = String(getArg("eval-profile", "standard")).trim().toLowerCase();
    const evalProfile = evalProfileRaw === "special_respawn_duel" ? "special_respawn_duel" : "standard";
    const isSpecialProfile = evalProfile === "special_respawn_duel";
    const runs = asPositiveInt(getArg("runs", getArg("matches", "50")), 50);
    const matchDurationSec = asPositiveInt(getArg("match-duration-sec", isSpecialProfile ? "60" : "45"), isSpecialProfile ? 60 : 45);
    const parallel = asPositiveInt(getArg("parallel", "8"), 8);
    const seedBase = asInt(getArg("seed-base", "20260419"), 20260419);
    const mlEnabled = asBool(getArg("ml-enabled", "1"), true);
    const modelUrl = getArg("model-url", "/output/ml/models/combat_phase0_iql_v1.onnx");
    const reportPath = path.resolve(getArg("report-path", REPORT_PATH_DEFAULT));
    const mapId = getArg("map", getArg("map-id", "windmill-heart"));
    const policyMode = getArg("policy-mode", "pure");
    const opponentMode = normalizeOpponentMode(getArg("opponent", getArg("opponent-mode", isSpecialProfile ? "stationary_dummy" : "heuristic_v2")));
    const opponentThinkMs = asPositiveInt(getArg("opponent-think-ms", "120"), 120);
    const saveScreenshots = getArg("save-screenshots", "0") === "1";
    const clearNonRigid = getArg("clear-nonrigid", "1") !== "0";
    const randomItemDensity = Math.max(0, Math.min(0.4, Number(getArg("random-item-density", "0.12")) || 0.12));
    const randomItemDensityJitter = Math.max(0, Math.min(0.2, Number(getArg("random-item-density-jitter", isSpecialProfile ? "0.0" : "0.04")) || (isSpecialProfile ? 0 : 0.04)));
    const itemRespawnMs = asPositiveInt(getArg("item-respawn-ms", isSpecialProfile ? "2000" : "1000"), isSpecialProfile ? 2000 : 1000);
    const itemRespawnJitterRatio = Math.max(0, Math.min(0.8, Number(getArg("item-respawn-jitter-ratio", isSpecialProfile ? "0.0" : "0.3")) || (isSpecialProfile ? 0 : 0.3)));
    const itemSafeRadius = Math.max(0, Math.min(8, asPositiveInt(getArg("item-safe-radius", "2"), 2)));
    const itemSafeRadiusJitter = Math.max(0, Math.min(3, Number(getArg("item-safe-radius-jitter", "1")) || 1));
    const itemMax = Math.max(0, asPositiveInt(getArg("item-max", "22"), 22));
    const suddenDeath = asBool(getArg("sudden-death", isSpecialProfile ? "0" : "1"), !isSpecialProfile);
    const disableRevive = asBool(getArg("disable-revive", isSpecialProfile ? "0" : "1"), !isSpecialProfile);
    const ignoreEnemySelfKill = asBool(getArg("ignore-enemy-self-kill", "1"), true);
    const stallNoProgressMs = asPositiveInt(getArg("stall-no-progress-ms", "12000"), 12000);
    const partialClearMinRatio = Math.max(0, Math.min(1, Number(getArg("partial-clear-min-ratio", "0.35")) || 0.35));
    const partialClearMaxRatio = Math.max(partialClearMinRatio, Math.min(1, Number(getArg("partial-clear-max-ratio", "0.75")) || 0.75));
    const spawnShortestPathMin = Math.max(1, asPositiveInt(getArg("spawn-shortest-path-min", "1"), 1));
    const spawnShortestPathMax = Math.max(spawnShortestPathMin, asPositiveInt(getArg("spawn-shortest-path-max", "10"), 10));
    const specialBombEscape = asBool(getArg("special-bomb-escape", isSpecialProfile ? "1" : "0"), isSpecialProfile);
    const specialRoundSec = asPositiveInt(getArg("special-round-sec", "60"), 60);
    const specialItemRespawnMs = asPositiveInt(getArg("special-item-respawn-ms", "2000"), 2000);
    const specialRespawnDelayMs = asInt(getArg("respawn-delay-ms", "0"), 0);
    const specialRespawnInvincibleMs = asInt(getArg("respawn-invincible-ms", "300"), 300);
    const liveView = asBool(getArg("live-view", "1"), true);
    const liveStatusPath = path.resolve(getArg("live-status-path", LIVE_STATUS_PATH_DEFAULT));
    const liveFramePath = path.resolve(getArg("live-frame-path", LIVE_FRAME_PATH_DEFAULT));
    const liveFrameMs = asPositiveInt(getArg("live-frame-ms", "250"), 250);
    const viewerUrl = getArg("viewer-url", "http://127.0.0.1:4000/eval-viewer");

    const cfg = {
        mlEnabled: mlEnabled,
        modelUrl,
        mapId,
        policyMode,
        matchDurationSec,
        seedBase,
        opponentMode,
        opponentThinkMs,
        saveScreenshots,
        clearNonRigid,
        randomItemDensity,
        randomItemDensityJitter,
        itemRespawnMs,
        itemRespawnJitterRatio,
        itemSafeRadius,
        itemSafeRadiusJitter,
        itemMax,
        suddenDeath,
        disableRevive,
        ignoreEnemySelfKill,
        stallNoProgressMs,
        partialClearMinRatio,
        partialClearMaxRatio,
        spawnShortestPathMin,
        spawnShortestPathMax,
        evalProfile,
        specialBombEscape,
        specialRoundSec,
        specialItemRespawnMs,
        specialRespawnDelayMs,
        specialRespawnInvincibleMs,
        liveFrameMs,
        mlConf: asFloatOrNull(getArg("ml-conf", "")),
        mlMoveConf: asFloatOrNull(getArg("ml-move-conf", "")),
        mlMargin: asFloatOrNull(getArg("ml-margin", "")),
        mlForceMoveEta: asFloatOrNull(getArg("ml-force-move-eta", "")),
        mlWaitBlockEta: asFloatOrNull(getArg("ml-wait-block-eta", "")),
        mlMoveThreatMs: asFloatOrNull(getArg("ml-move-threat-ms", ""))
    };

    fs.mkdirSync(path.dirname(reportPath), { recursive: true });
    fs.mkdirSync(OUT_DIR, { recursive: true });
    fs.mkdirSync(path.dirname(liveStatusPath), { recursive: true });
    fs.mkdirSync(path.dirname(liveFramePath), { recursive: true });

    const live = createLiveBridge({
        enabled: liveView,
        statusPath: liveStatusPath,
        framePath: liveFramePath,
        viewerUrl
    });
    live.start({
        runs,
        parallel,
        mapId,
        opponentMode,
        modelUrl
    });
    console.log("[COMBAT-EVAL-LIVE]", JSON.stringify({
        viewer_url: viewerUrl,
        status_url: "http://127.0.0.1:4000/api/eval/status",
        frame_url: "http://127.0.0.1:4000/api/eval/frame",
        live_status_path: liveStatusPath,
        live_frame_path: liveFramePath
    }));

    const startedAt = Date.now();
    const allMatches = [];
    let completed = 0;
    let nextIndex = 0;

    async function worker() {
        while (true) {
            const i = nextIndex;
            nextIndex += 1;
            if (i >= runs) {
                return;
            }
            const m = await runMatch(i, cfg, live);
            allMatches.push(m);
            live.pushMatch(m);
            completed += 1;
            const s = summarize(allMatches);
            live.updateProgress(completed, runs, workerCount, s);
            if (completed % 10 === 0 || completed === runs) {
                console.log(
                    "[COMBAT-EVAL]",
                    `progress=${completed}/${runs}`,
                    `win_rate=${s.win_rate.toFixed(4)}`,
                    `self_kill_rate=${s.self_kill_rate.toFixed(4)}`,
                    `draw_rate=${s.draw_rate.toFixed(4)}`,
                    `item_control=${s.item_control.toFixed(4)}`,
                    `elo_delta=${s.elo_result.ai_delta.toFixed(2)}`
                );
            }
        }
    }

    const workerCount = Math.max(1, Math.min(parallel, runs));
    await Promise.all(new Array(workerCount).fill(0).map(() => worker()));

    const summary = summarize(allMatches);
    const report = {
        ts: Date.now(),
        duration_sec: (Date.now() - startedAt) / 1000,
        protocol: {
            mode: "combat_1v1",
            eval_profile: evalProfile,
            fixed_map_id: mapId,
            fixed_opponent_mode: opponentMode,
            fixed_seed_base: seedBase,
            runs,
            match_duration_sec: matchDurationSec,
            opponent_think_ms: opponentThinkMs,
            clear_nonrigid: clearNonRigid,
            random_item_density: randomItemDensity,
            random_item_density_jitter: randomItemDensityJitter,
            item_respawn_ms: itemRespawnMs,
            item_respawn_jitter_ratio: itemRespawnJitterRatio,
            item_safe_radius: itemSafeRadius,
            item_safe_radius_jitter: itemSafeRadiusJitter,
            item_max: itemMax,
            sudden_death: suddenDeath,
            disable_revive: disableRevive,
            ignore_enemy_self_kill: ignoreEnemySelfKill,
            stall_no_progress_ms: stallNoProgressMs,
            partial_clear_min_ratio: partialClearMinRatio,
            partial_clear_max_ratio: partialClearMaxRatio,
            spawn_shortest_path_min: spawnShortestPathMin,
            spawn_shortest_path_max: spawnShortestPathMax,
            special_bomb_escape: specialBombEscape,
            special_round_sec: specialRoundSec,
            special_item_respawn_ms: specialItemRespawnMs,
            respawn_delay_ms: specialRespawnDelayMs,
            respawn_invincible_ms: specialRespawnInvincibleMs,
        },
        ml_cfg: {
            ml_enabled: !!cfg.mlEnabled,
            model_url: modelUrl,
            policy_mode: policyMode,
            ml_conf: cfg.mlConf,
            ml_move_conf: cfg.mlMoveConf,
            ml_margin: cfg.mlMargin,
            ml_force_move_eta: cfg.mlForceMoveEta,
            ml_wait_block_eta: cfg.mlWaitBlockEta,
            ml_move_threat_ms: cfg.mlMoveThreatMs
        },
        summary,
        matches: allMatches
    };

    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
    live.done(summary, reportPath);
    console.log("[COMBAT-EVAL-DONE]", JSON.stringify({ report_path: reportPath, summary }));
}

main().catch((err) => {
    try {
        const liveStatusPath = path.resolve(getArg("live-status-path", LIVE_STATUS_PATH_DEFAULT));
        fs.mkdirSync(path.dirname(liveStatusPath), { recursive: true });
        fs.writeFileSync(liveStatusPath, JSON.stringify({
            ts: Date.now(),
            heartbeat: Date.now(),
            active: false,
            error: String(err && err.message ? err.message : err)
        }, null, 2));
    } catch (_) {}
    console.error(err);
    process.exit(1);
});
