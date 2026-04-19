const crypto = require("crypto");
const fs = require("fs");
const path = require("path");

const skillPlaywrightPath = path.join(
    process.env.HOME || "",
    ".codex/skills/develop-web-game/node_modules/playwright"
);
const { chromium } = require(skillPlaywrightPath);

const OUT_ROOT = path.resolve(__dirname, "../output/ml");
const DATASET_PATH_DEFAULT = path.join(OUT_ROOT, "datasets", "combat_phase0_v1.jsonl");
const REPORT_PATH_DEFAULT = path.join(OUT_ROOT, "reports", `combat_phase0_collect_${Date.now()}.json`);
const SCREENSHOT_PATH_DEFAULT = path.join(OUT_ROOT, "reports", "combat_phase0_collect_final.png");
const URL_BASE = "http://127.0.0.1:4000/?autostart=0&ml=0&ml_collect=1&ml_freeze=1&ml_policy_mode=pure&ml_iql_mix=1";

const TERMINAL_OUTCOME_TAGS = new Set(["win", "loss", "draw", "self_kill"]);
const ALL_OUTCOME_TAGS = new Set(["ongoing", ...TERMINAL_OUTCOME_TAGS]);

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

function asPositiveInt(v, fallback) {
    const n = parseInt(v, 10);
    return Number.isFinite(n) && n > 0 ? n : fallback;
}

function asFloat(v, fallback) {
    const n = Number(v);
    return Number.isFinite(n) ? n : fallback;
}

function ensureParent(filePath) {
    fs.mkdirSync(path.dirname(filePath), { recursive: true });
}

function normalizeAction(v) {
    const a = parseInt(v, 10);
    return Number.isFinite(a) && a >= 0 && a <= 5 ? a : 0;
}

function canonicalizeOutcomeTag(row) {
    const done = !!row.done;
    const preDeath = !!row.pre_death;
    let tag = typeof row.outcome_tag === "string" ? row.outcome_tag.trim().toLowerCase() : "";

    if (tag === "death") {
        tag = preDeath ? "self_kill" : "loss";
    }
    if (tag === "done") {
        tag = done ? "draw" : "ongoing";
    }

    if (!ALL_OUTCOME_TAGS.has(tag)) {
        if (done && preDeath) {
            tag = "self_kill";
        } else if (done) {
            tag = "draw";
        } else {
            tag = "ongoing";
        }
    }

    if (TERMINAL_OUTCOME_TAGS.has(tag)) {
        row.done = true;
        if (tag === "self_kill") {
            row.pre_death = true;
        }
    }

    row.outcome_tag = tag;
    return tag;
}

function canonicalizeSample(raw) {
    const row = raw && typeof raw === "object" ? raw : null;
    if (!row) {
        return null;
    }

    row.action = normalizeAction(row.action);
    row.done = !!row.done;
    row.pre_death = !!row.pre_death;
    canonicalizeOutcomeTag(row);

    if (typeof row.risk_label !== "number") {
        row.risk_label = row.pre_death || row.done ? 1 : 0;
    }

    return row;
}

function validateSample(row) {
    if (!row || typeof row !== "object") {
        return { ok: false, reason: "row_not_object" };
    }

    const state = row.state || {};
    const stateMap = state.state_map;
    const stateVec = state.state_vector;

    if (!Array.isArray(stateMap) || stateMap.length !== 13) {
        return { ok: false, reason: "bad_state_map_rows" };
    }
    for (let y = 0; y < 13; y++) {
        if (!Array.isArray(stateMap[y]) || stateMap[y].length !== 15) {
            return { ok: false, reason: "bad_state_map_cols" };
        }
        for (let x = 0; x < 15; x++) {
            if (!Array.isArray(stateMap[y][x]) || stateMap[y][x].length !== 10) {
                return { ok: false, reason: "bad_state_map_channels" };
            }
        }
    }

    if (!Array.isArray(stateVec) || stateVec.length < 16) {
        return { ok: false, reason: "bad_state_vector_dim" };
    }

    const action = normalizeAction(row.action);
    const actionMask = row.action_mask;
    if (!Array.isArray(actionMask) || actionMask.length !== 6) {
        return { ok: false, reason: "bad_action_mask_dim" };
    }
    for (let i = 0; i < actionMask.length; i++) {
        const v = Number(actionMask[i]);
        if (!(v === 0 || v === 1)) {
            return { ok: false, reason: "bad_action_mask_value" };
        }
    }
    if (actionMask[action] !== 1) {
        return { ok: false, reason: "action_not_legal_by_mask" };
    }

    if (typeof row.episode_id !== "string" || !row.episode_id) {
        return { ok: false, reason: "missing_episode_id" };
    }
    if (typeof row.agent_id !== "string" || !row.agent_id) {
        return { ok: false, reason: "missing_agent_id" };
    }
    if (typeof row.opponent_id !== "string" || !row.opponent_id) {
        return { ok: false, reason: "missing_opponent_id" };
    }
    if (typeof row.outcome_tag !== "string" || !ALL_OUTCOME_TAGS.has(row.outcome_tag)) {
        return { ok: false, reason: "bad_outcome_tag" };
    }

    return { ok: true };
}

function compactNumber(n) {
    if (!Number.isFinite(n)) {
        return 0;
    }
    return Math.round(n * 1000) / 1000;
}

function compactVector(vec) {
    if (!Array.isArray(vec)) {
        return [];
    }
    const out = new Array(vec.length);
    for (let i = 0; i < vec.length; i++) {
        out[i] = compactNumber(Number(vec[i]));
    }
    return out;
}

function compactStateMap(map) {
    if (!Array.isArray(map)) {
        return [];
    }
    const out = new Array(map.length);
    for (let y = 0; y < map.length; y++) {
        const row = Array.isArray(map[y]) ? map[y] : [];
        out[y] = new Array(row.length);
        for (let x = 0; x < row.length; x++) {
            const cell = Array.isArray(row[x]) ? row[x] : [];
            out[y][x] = new Array(cell.length);
            for (let c = 0; c < cell.length; c++) {
                out[y][x][c] = compactNumber(Number(cell[c]));
            }
        }
    }
    return out;
}

function buildTransitionSignature(row) {
    const state = row.state || {};
    const nextState = row.next_state || null;
    const core = {
        sm: compactStateMap(state.state_map),
        sv: compactVector(state.state_vector),
        a: normalizeAction(row.action),
        m: Array.isArray(row.action_mask) ? row.action_mask.map((v) => (Number(v) > 0.5 ? 1 : 0)) : [1, 1, 1, 1, 1, 1],
        done: row.done ? 1 : 0,
        pre_death: row.pre_death ? 1 : 0,
        outcome_tag: row.outcome_tag || "ongoing",
        nsm: nextState ? compactStateMap(nextState.state_map) : null,
        nsv: nextState ? compactVector(nextState.state_vector) : null,
    };
    return crypto.createHash("sha1").update(JSON.stringify(core)).digest("hex");
}

async function ensureServerReady(url) {
    for (let i = 0; i < 40; i++) {
        try {
            const res = await fetch(url);
            if (res.ok || res.status === 302) {
                return;
            }
        } catch (err) {}
        await new Promise((r) => setTimeout(r, 500));
    }
    throw new Error("server_not_ready");
}

async function launchBrowser() {
    try {
        return await chromium.launch({
            headless: true,
            args: ["--use-gl=angle", "--use-angle=swiftshader"]
        });
    } catch (err) {
        const chromePath = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
        if (!fs.existsSync(chromePath)) {
            throw err;
        }
        return chromium.launch({
            headless: true,
            executablePath: chromePath,
            args: ["--use-gl=angle", "--use-angle=swiftshader"]
        });
    }
}

async function main() {
    const targetFrames = asPositiveInt(getArg("target-frames", "200000"), 200000);
    const pollMs = asPositiveInt(getArg("poll-ms", "400"), 400);
    const batchSize = asPositiveInt(getArg("batch-size", "2048"), 2048);
    const thinkMs = asPositiveInt(getArg("think-interval-ms", "8"), 8);
    const matchDurationSec = asPositiveInt(getArg("match-duration-sec", "45"), 45);
    const seedBase = asInt(getArg("seed-base", "20260419"), 20260419) >>> 0;
    const mapId = getArg("map", "windmill-heart");
    const fresh = getArg("fresh", "1") !== "0";
    const maxWallSec = asPositiveInt(getArg("max-wall-sec", "0"), 0);

    const clearNonRigid = getArg("clear-nonrigid", "1") !== "0";
    const randomItemDensity = Math.max(0, Math.min(0.4, asFloat(getArg("random-item-density", "0.12"), 0.12)));
    const itemRespawnMs = asPositiveInt(getArg("item-respawn-ms", "1200"), 1200);
    const itemSafeRadius = Math.max(0, Math.min(8, asPositiveInt(getArg("item-safe-radius", "2"), 2)));
    const itemMax = Math.max(0, asPositiveInt(getArg("item-max", "22"), 22));

    const arena = getArg("arena", "1v1");
    const actionSpace = getArg("action-space", "discrete6");
    if (arena !== "1v1") {
        throw new Error("only --arena=1v1 is supported for now");
    }
    if (actionSpace !== "discrete6") {
        throw new Error("only --action-space=discrete6 is supported for now");
    }

    const opponentPool = getArg("opponent-pool", "heuristic_v1,heuristic_v2")
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean);
    if (opponentPool.length === 0) {
        opponentPool.push("heuristic_v1");
    }

    const datasetPath = path.resolve(getArg("dataset-path", DATASET_PATH_DEFAULT));
    const reportPath = path.resolve(getArg("report-path", REPORT_PATH_DEFAULT));
    const screenshotPath = path.resolve(getArg("screenshot-path", SCREENSHOT_PATH_DEFAULT));

    ensureParent(datasetPath);
    ensureParent(reportPath);
    ensureParent(screenshotPath);
    if (fresh && fs.existsSync(datasetPath)) {
        fs.unlinkSync(datasetPath);
    }

    await ensureServerReady("http://127.0.0.1:4000/");

    const stream = fs.createWriteStream(datasetPath, { flags: "a" });
    const browser = await launchBrowser();
    const page = await browser.newPage({ viewport: { width: 1460, height: 940 } });

    page.on("pageerror", (err) => console.log("[PAGEERROR]", String(err)));
    page.on("console", (msg) => {
        if (msg.type() === "error") {
            console.log("[CONSOLE.ERROR]", msg.text());
        }
    });

    let wrote = 0;
    let dropped = 0;
    let duplicateTransitions = 0;
    let repairedIllegalActionRows = 0;
    const seenSigs = new Set();
    const actionHist = { "0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0 };
    const doneHist = { "0": 0, "1": 0 };
    const preDeathHist = { "0": 0, "1": 0 };
    const outcomeHist = { ongoing: 0, win: 0, loss: 0, draw: 0, self_kill: 0 };
    const dropReasons = {};
    const terminalOutcomeEvents = { win: 0, loss: 0, draw: 0, self_kill: 0 };
    let lastRuntimeState = null;
    let terminalFinalizeCalls = 0;
    let terminalCollectorFinalizeOk = 0;
    let terminalCollectorFinalizeMiss = 0;
    const startedAt = Date.now();

    function markDrop(reason) {
        dropReasons[reason] = (dropReasons[reason] || 0) + 1;
    }

    function writeUniqueRow(rawInput) {
        if (wrote >= targetFrames) {
            return;
        }
        const raw = canonicalizeSample(rawInput);
        if (!raw) {
            dropped += 1;
            markDrop("row_not_object");
            return;
        }

        const check = validateSample(raw);
        if (!check.ok && check.reason === "action_not_legal_by_mask" && Array.isArray(raw.action_mask)) {
            const firstLegal = raw.action_mask.findIndex((v) => Number(v) === 1);
            if (firstLegal >= 0 && firstLegal < 6) {
                raw.action = firstLegal;
                raw.meta = raw.meta || {};
                raw.meta.action_repaired = 1;
                const repairedCheck = validateSample(raw);
                if (repairedCheck.ok) {
                    repairedIllegalActionRows += 1;
                } else {
                    dropped += 1;
                    markDrop(repairedCheck.reason);
                    return;
                }
            } else {
                dropped += 1;
                markDrop(check.reason);
                return;
            }
        } else if (!check.ok) {
            dropped += 1;
            markDrop(check.reason);
            return;
        }

        const sig = buildTransitionSignature(raw);
        if (seenSigs.has(sig)) {
            duplicateTransitions += 1;
            markDrop("duplicate_transition");
            return;
        }
        seenSigs.add(sig);

        stream.write(JSON.stringify(raw) + "\n");
        wrote += 1;

        const action = normalizeAction(raw.action);
        actionHist[String(action)] = (actionHist[String(action)] || 0) + 1;
        doneHist[raw.done ? "1" : "0"] += 1;
        preDeathHist[raw.pre_death ? "1" : "0"] += 1;
        outcomeHist[raw.outcome_tag] = (outcomeHist[raw.outcome_tag] || 0) + 1;
    }

    try {
        await page.goto(URL_BASE, { waitUntil: "domcontentloaded" });
        await page.waitForTimeout(250);

        await page.evaluate((cfg) => {
            function mulberry32(a) {
                let t = a >>> 0;
                return function() {
                    t += 0x6D2B79F5;
                    let z = Math.imul(t ^ (t >>> 15), 1 | t);
                    z ^= z + Math.imul(z ^ (z >>> 7), 61 | z);
                    return ((z ^ (z >>> 14)) >>> 0) / 4294967296;
                };
            }

            const rng = mulberry32(cfg.seed >>> 0);
            window.Math.random = function() {
                return rng();
            };

            window.__combatCollect = {
                opponentPool: cfg.opponentPool,
                mapId: cfg.mapId,
                thinkMs: cfg.thinkMs,
                matchDurationSec: cfg.matchDurationSec,
                matchIndex: 0,
                playerCtrl: null,
                itemRespawnTicker: null,
                aiRoleNumber: null,
                lastStartAt: 0,
                currentOpponent: cfg.opponentPool[0] || "heuristic_v1",
                matchFinalized: false,
                clearNonRigid: !!cfg.clearNonRigid,
                randomItemDensity: cfg.randomItemDensity,
                itemRespawnMs: cfg.itemRespawnMs,
                itemSafeRadius: cfg.itemSafeRadius,
                itemMax: cfg.itemMax,
                lastTerminal: null,
                terminalEvents: 0,
                mapRows: 13,
                mapCols: 15,
            };

            function findAiRoleNumber() {
                const s = window.singlePlayerState;
                const fighters = s && Array.isArray(s.Fighters) ? s.Fighters : [];
                for (let i = 0; i < fighters.length; i++) {
                    if (fighters[i] && typeof fighters[i].id === "string" && fighters[i].id.indexOf("ai_") === 0) {
                        const role = fighters[i].role;
                        if (role && typeof role.RoleNumber === "number") {
                            return role.RoleNumber;
                        }
                    }
                }
                return null;
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

            function patchTrainerTarget(aiRoleNumber) {
                if (!window.AIDodgeTrainer || aiRoleNumber == null) {
                    return;
                }
                if (window.AIDodgeTrainer.Config) {
                    window.AIDodgeTrainer.Config.trainingThinkIntervalMs = window.__combatCollect.thinkMs;
                }
                window.AIDodgeTrainer.IsRunning = true;
                window.AIDodgeTrainer.IsMonsterTraining = function(monster) {
                    return !!monster
                        && !!monster.Role
                        && !monster.Role.IsDeath
                        && monster.Role.RoleNumber === aiRoleNumber;
                };
            }

            function pickOpponentMode(matchIndex) {
                const pool = window.__combatCollect.opponentPool;
                if (!pool || pool.length === 0) {
                    return "heuristic_v1";
                }
                return pool[matchIndex % pool.length] || "heuristic_v1";
            }

            function stopPlayerController() {
                if (window.__combatCollect.playerCtrl) {
                    clearInterval(window.__combatCollect.playerCtrl);
                    window.__combatCollect.playerCtrl = null;
                }
            }

            function stopItemRespawn() {
                if (window.__combatCollect.itemRespawnTicker) {
                    clearInterval(window.__combatCollect.itemRespawnTicker);
                    window.__combatCollect.itemRespawnTicker = null;
                }
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

            function collectSpawnAnchors() {
                const anchors = [];
                const roles = findPlayerAndAiRoles();
                const pMap = getRoleMap(roles.player);
                const aiMap = getRoleMap(roles.ai);
                if (pMap) {
                    anchors.push({ X: pMap.X, Y: pMap.Y });
                }
                if (aiMap) {
                    anchors.push({ X: aiMap.X, Y: aiMap.Y });
                }
                if (anchors.length === 0 && typeof window.GetCurrentGameMapSpawn === "function") {
                    const spawn = window.GetCurrentGameMapSpawn();
                    if (spawn) {
                        anchors.push({ X: spawn.X, Y: spawn.Y });
                    }
                }
                return anchors;
            }

            function isNearAnchors(x, y, anchors, minRadius) {
                for (let i = 0; i < anchors.length; i++) {
                    const a = anchors[i];
                    const d = Math.abs(a.X - x) + Math.abs(a.Y - y);
                    if (d < minRadius) {
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

            function clearNonRigidBarriersLive() {
                if (!window.__combatCollect.clearNonRigid) {
                    return;
                }
                if (typeof window.StripNonRigidBarriersFromMap === "function") {
                    window.StripNonRigidBarriersFromMap();
                }
                if (typeof window.Barrier === "undefined" || !window.Barrier.Storage) {
                    return;
                }
                for (let y = 0; y < window.Barrier.Storage.length; y++) {
                    if (!window.Barrier.Storage[y]) {
                        continue;
                    }
                    for (let x = 0; x < window.Barrier.Storage[y].length; x++) {
                        const cell = window.Barrier.Storage[y][x];
                        if (!cell) {
                            continue;
                        }
                        const no = Number(cell.No || 0);
                        const isRigid = typeof window.IsRigidBarrierNo === "function"
                            ? window.IsRigidBarrierNo(no)
                            : (no > 0 && no < 100 && no !== 3 && no !== 8);
                        if (!isRigid) {
                            if (cell.Object && typeof cell.Object.Dispose === "function") {
                                cell.Object.Dispose();
                            }
                            window.Barrier.Storage[y][x] = null;
                            if (window.townBarrierMap
                                && window.townBarrierMap[y]
                                && typeof window.townBarrierMap[y][x] !== "undefined") {
                                window.townBarrierMap[y][x] = 0;
                            }
                        }
                    }
                }
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
                const densityTarget = Math.floor(rows * cols * Math.max(0, Math.min(0.4, window.__combatCollect.randomItemDensity || 0)));
                const hardCap = Math.max(0, window.__combatCollect.itemMax || 0);
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
                    if (window.townBarrierMap[y][x] !== 0) {
                        continue;
                    }
                    if (typeof window.IsAIWalkable === "function" && !window.IsAIWalkable(x, y)) {
                        continue;
                    }
                    if (isNearAnchors(x, y, anchors, window.__combatCollect.itemSafeRadius || 0)) {
                        continue;
                    }
                    if (isOccupiedByRole(x, y)) {
                        continue;
                    }
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
                if ((window.__combatCollect.itemRespawnMs || 0) <= 0) {
                    return;
                }
                window.__combatCollect.itemRespawnTicker = setInterval(function() {
                    if (!window.gameRunning) {
                        return;
                    }
                    trySpawnRandomItems();
                }, window.__combatCollect.itemRespawnMs);
            }

            function startPlayerController(mode) {
                stopPlayerController();
                window.__combatCollect.currentOpponent = mode;
                window.__combatCollect.playerCtrl = setInterval(function() {
                    if (!window.gameRunning || !window.singlePlayerState) {
                        return;
                    }
                    const player = window.singlePlayerState.Player;
                    if (!player || player.IsDeath || player.IsInPaopao || typeof player.CurrentMapID !== "function") {
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

                        if (mode === "heuristic_v2") {
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
                        const bombProb = mode === "heuristic_v2" ? 0.24 : 0.10;
                        if (dist <= 2 && rng() < bombProb) {
                            window.RoleKeyEvent(32, player);
                        }
                    }
                }, 95);
            }

            function classifyOutcome() {
                const roles = findPlayerAndAiRoles();
                const player = roles.player;
                const ai = roles.ai;
                const playerKills = player ? Number(player.kills || 0) : 0;
                const aiKills = ai ? Number(ai.kills || 0) : 0;
                const aiDeaths = ai ? Number(ai.deaths || 0) : 0;
                const aiSelfKills = Math.max(0, aiDeaths - playerKills);

                let outcomeTag = "draw";
                if (aiSelfKills > 0) {
                    outcomeTag = "self_kill";
                } else if (aiKills > playerKills) {
                    outcomeTag = "win";
                } else if (aiKills < playerKills) {
                    outcomeTag = "loss";
                }

                return {
                    outcome_tag: outcomeTag,
                    ai_kills: aiKills,
                    player_kills: playerKills,
                    ai_deaths: aiDeaths,
                    ai_self_kills: aiSelfKills,
                    ts: Date.now(),
                };
            }

            function finalizeCurrentMatch() {
                if (window.__combatCollect.matchFinalized) {
                    return null;
                }
                const terminal = classifyOutcome();
                let collectorFinalized = false;
                if (typeof window.BNBMLCollectorFinalizeEpisode === "function") {
                    collectorFinalized = !!window.BNBMLCollectorFinalizeEpisode(
                        terminal.outcome_tag,
                        {
                            done: true,
                            preDeath: terminal.outcome_tag === "self_kill",
                            reward: terminal.outcome_tag === "win"
                                ? 1.5
                                : (terminal.outcome_tag === "self_kill" ? -2.0 : (terminal.outcome_tag === "loss" ? -1.2 : -0.2)),
                            forceFlush: true,
                        }
                    );
                } else {
                    const collector = window.BNBMLDatasetCollector;
                    if (collector && typeof collector.FinalizeEpisode === "function") {
                        collectorFinalized = !!collector.FinalizeEpisode(
                            terminal.outcome_tag,
                            {
                                done: true,
                                preDeath: terminal.outcome_tag === "self_kill",
                                reward: terminal.outcome_tag === "win"
                                    ? 1.5
                                    : (terminal.outcome_tag === "self_kill" ? -2.0 : (terminal.outcome_tag === "loss" ? -1.2 : -0.2)),
                                forceFlush: true,
                            }
                        );
                    }
                }
                terminal.collector_finalized = collectorFinalized;
                window.__combatCollect.matchFinalized = true;
                window.__combatCollect.lastTerminal = terminal;
                window.__combatCollect.terminalEvents += 1;
                return terminal;
            }

            function startMatch() {
                stopItemRespawn();
                if (typeof window.BNBMLRefreshConfig === "function") {
                    window.BNBMLFreezeExpertPolicy = true;
                    window.BNBMLCollectWaitKeepProb = 1;
                    window.BNBMLRefreshConfig();
                }
                if (typeof window.ApplySelectedGameMap === "function" && window.__combatCollect.mapId) {
                    window.ApplySelectedGameMap(window.__combatCollect.mapId, false);
                }
                if (typeof window.SetAIEnemyCount === "function") {
                    window.SetAIEnemyCount(1, false);
                }
                window.roundDurationSeconds = window.__combatCollect.matchDurationSec;
                if (typeof window.StartSinglePlayerGame === "function") {
                    window.StartSinglePlayerGame(1);
                }

                window.__combatCollect.matchIndex += 1;
                window.__combatCollect.matchFinalized = false;
                window.__combatCollect.lastStartAt = Date.now();
                startPlayerController(pickOpponentMode(window.__combatCollect.matchIndex));

                setTimeout(function() {
                    clearNonRigidBarriersLive();
                    trySpawnRandomItems();
                    startItemRespawn();
                }, 180);

                setTimeout(function() {
                    const aiRoleNo = findAiRoleNumber();
                    window.__combatCollect.aiRoleNumber = aiRoleNo;
                    patchTrainerTarget(aiRoleNo);
                }, 250);
            }

            window.__combatCollectStartMatch = startMatch;
            window.__combatCollectFinalizeCurrentMatch = finalizeCurrentMatch;
            startMatch();
        }, {
            seed: seedBase,
            opponentPool,
            mapId,
            thinkMs,
            matchDurationSec,
            clearNonRigid,
            randomItemDensity,
            itemRespawnMs,
            itemSafeRadius,
            itemMax,
        });

        let lastLogTs = 0;

        while (wrote < targetFrames) {
            const elapsedSec = (Date.now() - startedAt) / 1000;
            if (maxWallSec > 0 && elapsedSec >= maxWallSec) {
                console.log("[COMBAT-COLLECT] reached max-wall-sec, stop early");
                break;
            }

            const batchResult = await page.evaluate((maxRows) => {
                const rows = typeof window.BNBMLCollectorDrain === "function"
                    ? (window.BNBMLCollectorDrain(maxRows) || [])
                    : [];
                const runtime = window.BNBTrainingRuntimeState || null;
                const collector = window.BNBMLDatasetCollectorState || null;
                const collectState = window.__combatCollect || null;
                const running = !!window.gameRunning;
                let terminal = null;

                if (!running) {
                    if (typeof window.__combatCollectFinalizeCurrentMatch === "function") {
                        terminal = window.__combatCollectFinalizeCurrentMatch();
                    }
                    if (typeof window.BNBMLCollectorDrainAll === "function") {
                        const terminalRows = window.BNBMLCollectorDrainAll() || [];
                        if (Array.isArray(terminalRows) && terminalRows.length > 0) {
                            for (let i = 0; i < terminalRows.length; i++) {
                                rows.push(terminalRows[i]);
                            }
                        }
                    }
                    if (typeof window.__combatCollectStartMatch === "function") {
                        window.__combatCollectStartMatch();
                    }
                }

                return {
                    rows,
                    runtime,
                    collector,
                    running,
                    terminal,
                    matchIndex: collectState ? collectState.matchIndex : 0,
                    aiRoleNumber: collectState ? collectState.aiRoleNumber : null,
                    opponentMode: collectState ? collectState.currentOpponent : "heuristic_v1",
                    terminalEvents: collectState ? collectState.terminalEvents : 0,
                };
            }, batchSize);

            lastRuntimeState = batchResult.runtime || null;

            if (batchResult.terminal && TERMINAL_OUTCOME_TAGS.has(batchResult.terminal.outcome_tag)) {
                terminalOutcomeEvents[batchResult.terminal.outcome_tag] += 1;
                if (batchResult.terminal.collector_finalized) {
                    terminalCollectorFinalizeOk += 1;
                } else {
                    terminalCollectorFinalizeMiss += 1;
                }
            }
            terminalFinalizeCalls = Math.max(terminalFinalizeCalls, Number(batchResult.terminalEvents || 0));

            if (Array.isArray(batchResult.rows) && batchResult.rows.length > 0) {
                for (const raw of batchResult.rows) {
                    writeUniqueRow(raw);
                    if (wrote >= targetFrames) {
                        break;
                    }
                }
            }

            const now = Date.now();
            if (now - lastLogTs > 2000) {
                lastLogTs = now;
                const c = batchResult.collector || {};
                console.log(
                    "[COMBAT-COLLECT]",
                    `rows=${wrote}/${targetFrames}`,
                    `ready=${c.rows_ready || 0}`,
                    `samples=${c.samples_finalized || 0}`,
                    `match=${batchResult.matchIndex || 0}`,
                    `opponent=${batchResult.opponentMode || "heuristic_v1"}`,
                    `ai_role=${batchResult.aiRoleNumber == null ? "na" : batchResult.aiRoleNumber}`,
                    `duplicates=${duplicateTransitions}`,
                    `dropped=${dropped}`
                );
            }

            await page.waitForTimeout(pollMs);
        }

        const tail = await page.evaluate(() => {
            const rows = typeof window.BNBMLCollectorDrainAll === "function"
                ? (window.BNBMLCollectorDrainAll() || [])
                : [];
            return rows;
        });

        for (const raw of tail) {
            if (wrote >= targetFrames) {
                break;
            }
            writeUniqueRow(raw);
        }

        await page.screenshot({ path: screenshotPath, fullPage: true });
    } finally {
        stream.end();
        await browser.close();
    }

    const durationSec = (Date.now() - startedAt) / 1000;
    const report = {
        ts: Date.now(),
        dataset_path: datasetPath,
        report_path: reportPath,
        target_frames: targetFrames,
        rows_written: wrote,
        rows_dropped: dropped,
        repaired_illegal_action_rows: repairedIllegalActionRows,
        duplicate_transition_rows: duplicateTransitions,
        unique_transition_rows: seenSigs.size,
        unique_transition_rate: wrote / Math.max(1, wrote + duplicateTransitions),
        drop_reasons: dropReasons,
        rows_per_sec: durationSec > 0 ? wrote / durationSec : 0,
        duration_sec: durationSec,
        action_hist_written: actionHist,
        done_hist_written: doneHist,
        pre_death_hist_written: preDeathHist,
        outcome_hist_written: outcomeHist,
        terminal_outcome_events: terminalOutcomeEvents,
        terminal_finalize_calls: terminalFinalizeCalls,
        terminal_collector_finalize_ok: terminalCollectorFinalizeOk,
        terminal_collector_finalize_miss: terminalCollectorFinalizeMiss,
        arena,
        action_space: actionSpace,
        map_id: mapId,
        opponent_pool: opponentPool,
        clear_nonrigid: clearNonRigid,
        random_item_density: randomItemDensity,
        item_respawn_ms: itemRespawnMs,
        item_safe_radius: itemSafeRadius,
        item_max: itemMax,
        latest_training_runtime_state: lastRuntimeState,
        screenshot: screenshotPath
    };

    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
    console.log("[DONE]", JSON.stringify(report));
}

main().catch((err) => {
    console.error(err);
    process.exit(1);
});
