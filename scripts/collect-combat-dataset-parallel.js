#!/usr/bin/env node
const crypto = require("crypto");
const fs = require("fs");
const os = require("os");
const path = require("path");
const readline = require("readline");
const { spawn, spawnSync } = require("child_process");

const ROOT = path.resolve(__dirname, "..");
const COLLECT_SCRIPT = path.resolve(__dirname, "collect-combat-dataset.js");
const OFFLINE_MICRO_SCRIPT = path.resolve(__dirname, "collect-combat-micro-offline.js");

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

function ensureParent(p) {
    fs.mkdirSync(path.dirname(p), { recursive: true });
}

function compactNumber(n) {
    const x = Number(n);
    if (!Number.isFinite(x)) return 0;
    return Math.round(x * 1000) / 1000;
}

function compactVector(vec) {
    if (!Array.isArray(vec)) return [];
    const out = new Array(vec.length);
    for (let i = 0; i < vec.length; i++) {
        out[i] = compactNumber(vec[i]);
    }
    return out;
}

function compactStateMap(map) {
    if (!Array.isArray(map)) return [];
    const out = new Array(map.length);
    for (let y = 0; y < map.length; y++) {
        const row = Array.isArray(map[y]) ? map[y] : [];
        out[y] = new Array(row.length);
        for (let x = 0; x < row.length; x++) {
            const cell = Array.isArray(row[x]) ? row[x] : [];
            out[y][x] = new Array(cell.length);
            for (let c = 0; c < cell.length; c++) {
                out[y][x][c] = compactNumber(cell[c]);
            }
        }
    }
    return out;
}

function normalizeAction(v) {
    const a = parseInt(v, 10);
    return Number.isFinite(a) && a >= 0 && a <= 5 ? a : 0;
}

const TERMINAL_OUTCOME_TAGS = new Set(["win", "loss", "draw", "self_kill"]);
const SAMPLE_BUCKETS = ["ongoing", "pre_death", "drop_bomb_safe", "drop_bomb_bad", "terminal"];
const SCORE_TIERS = ["high", "mid", "low"];
const DEFAULT_BUCKET_TARGETS = {
    ongoing: 0.60,
    pre_death: 0.15,
    drop_bomb_safe: 0.10,
    drop_bomb_bad: 0.10,
    terminal: 0.05,
};
const BEHAVIOR_BREAKDOWN_KEYS = [
    "item_control",
    "bomb_space_reduction",
    "post_bomb_escape",
    "obstacle_escape",
    "move_to_safe",
    "close_chase",
    "choke_control",
    "deny_space",
    "power_advantage_attack",
    "tactical_wait",
    "terminal_reference",
    "survival_failure_reference",
];

function parseBucketTargets(raw) {
    const out = Object.assign({}, DEFAULT_BUCKET_TARGETS);
    String(raw || "")
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean)
        .forEach((part) => {
            const bits = part.split(":");
            if (bits.length !== 2) return;
            const k = bits[0].trim();
            const v = Number(bits[1]);
            if (SAMPLE_BUCKETS.includes(k) && Number.isFinite(v) && v >= 0) {
                out[k] = v;
            }
        });
    const sum = SAMPLE_BUCKETS.reduce((acc, k) => acc + Number(out[k] || 0), 0);
    if (sum <= 0) return Object.assign({}, DEFAULT_BUCKET_TARGETS);
    for (const k of SAMPLE_BUCKETS) {
        out[k] = Number(out[k] || 0) / sum;
    }
    return out;
}

function clamp01(v) {
    const n = Number(v);
    if (!Number.isFinite(n)) return 0;
    return Math.max(0, Math.min(1, n));
}

function getStateVector(row) {
    const state = row && row.state ? row.state : {};
    const vec = Array.isArray(state.state_vector) ? state.state_vector : row.state_vector;
    return Array.isArray(vec) ? vec : [];
}

function getNextStateVector(row) {
    const state = row && row.next_state ? row.next_state : {};
    const vec = Array.isArray(state.state_vector) ? state.state_vector : row.next_state_vector;
    return Array.isArray(vec) ? vec : [];
}

function clampRange(v, lo, hi) {
    const n = Number(v);
    if (!Number.isFinite(n)) return lo;
    return Math.max(lo, Math.min(hi, n));
}

function behaviorScoreBand(score) {
    const s = Number(score) || 0;
    if (s >= 0.70) return "very_high";
    if (s >= 0.45) return "high";
    if (s >= 0.18) return "mid";
    if (s > 0.000001) return "low";
    return "zero";
}

function scoreTier(score) {
    const s = Number(score) || 0;
    if (s >= 0.45) return "high";
    if (s >= 0.18) return "mid";
    return "low";
}

function computeBehaviorScoreBreakdown(row, labels) {
    const vec = getStateVector(row);
    const nextVec = getNextStateVector(row);
    const action = normalizeAction(row.action);
    const meta = row.meta || {};
    const scenario = String(meta.scenarioName || "");
    const done = !!row.done;
    const preDeath = !!row.pre_death;
    const outcome = String(row.outcome_tag || "ongoing");
    const minEscapeEta = clamp01(vec[17]);
    const nextMinEscapeEta = nextVec.length > 17 ? clamp01(nextVec[17]) : minEscapeEta;
    const deadendDepth = clamp01(vec[18]);
    const blastOverlap = clamp01(vec[19]);
    const nextBlastOverlap = nextVec.length > 19 ? clamp01(nextVec[19]) : blastOverlap;
    const enemyEscapeOptions = clamp01(vec[20]);
    const trapClosure = clamp01(vec[21]);
    const itemRaceDelta = Math.abs(clamp01(vec[22]) - 0.5) * 2;
    const enemyDistNorm = vec.length > 30 ? clamp01(vec[30]) : clamp01(vec[13]);
    const nextEnemyDistNorm = nextVec.length > 30 ? clamp01(nextVec[30]) : enemyDistNorm;
    const closeRange = clamp01(Number(meta.closeRangeDuelScore) || (1 - enemyDistNorm));
    const bombThreat = clamp01(Number(meta.myBombThreatScore) || Number(labels.enemy_trap_after_bomb || 0));
    const selfCap = vec.length > 24 ? clamp01(vec[24]) : clamp01(vec[10]);
    const selfPower = clamp01(vec[11]);
    const selfSpeed = clamp01(vec[12]);
    const enemyCap = vec.length > 26 ? clamp01(vec[26]) : 0.4;
    const enemyPower = vec.length > 28 ? clamp01(vec[28]) : 0.4;
    const enemySpeed = vec.length > 29 ? clamp01(vec[29]) : 0.4;
    const powerGap = clampRange((selfCap + selfPower + selfSpeed - enemyCap - enemyPower - enemySpeed + 3) / 6, 0, 1);
    const breakdown = {};

    if (scenario === "item_race" || (action >= 1 && action <= 4 && itemRaceDelta >= 0.35)) {
        breakdown.item_control = clampRange(0.10 + itemRaceDelta * 0.20 + (action >= 1 && action <= 4 ? 0.05 : 0), 0.10, 0.35);
    }
    if (action === 5) {
        breakdown.bomb_space_reduction = clampRange(0.20 + (1 - enemyEscapeOptions) * 0.16 + trapClosure * 0.16 + bombThreat * 0.18, 0.20, 0.55);
        if (Number(labels.bomb_self_trap_risk || 0) < 0.55) {
            breakdown.post_bomb_escape = clampRange(0.15 + Number(labels.bomb_escape_success_label || 0) * 0.10 + minEscapeEta * 0.10 + (1 - blastOverlap) * 0.05, 0.15, 0.40);
        }
    }
    if (action !== 5 && (blastOverlap >= 0.25 || nextBlastOverlap + 0.05 < blastOverlap)) {
        breakdown.obstacle_escape = clampRange(0.15 + Math.max(0, blastOverlap - nextBlastOverlap) * 0.24 + deadendDepth * 0.08, 0.15, 0.35);
    }
    if (action >= 1 && action <= 4 && (nextMinEscapeEta > minEscapeEta + 0.04 || nextBlastOverlap + 0.04 < blastOverlap)) {
        breakdown.move_to_safe = clampRange(0.10 + Math.max(0, nextMinEscapeEta - minEscapeEta) * 0.45 + Math.max(0, blastOverlap - nextBlastOverlap) * 0.30, 0.10, 0.30);
    }
    if (action !== 0 && (nextEnemyDistNorm + 0.03 < enemyDistNorm || (["deadend_chase", "enemy_choke"].includes(scenario) && closeRange >= 0.55))) {
        breakdown.close_chase = clampRange(0.10 + closeRange * 0.12 + Math.max(0, enemyDistNorm - nextEnemyDistNorm) * 0.55, 0.10, 0.30);
    }
    if ((action === 5 || ["enemy_choke", "deadend_chase"].includes(scenario) || bombThreat >= 0.20) && (trapClosure >= 0.45 || (enemyEscapeOptions <= 0.38 && deadendDepth >= 0.30))) {
        breakdown.choke_control = clampRange(0.20 + trapClosure * 0.18 + (1 - enemyEscapeOptions) * 0.12 + deadendDepth * 0.10, 0.20, 0.50);
    }
    if ((action === 5 || bombThreat >= 0.20) && enemyEscapeOptions <= 0.55) {
        breakdown.deny_space = clampRange(0.10 + (1 - enemyEscapeOptions) * 0.16 + bombThreat * 0.10, 0.10, 0.30);
    }
    if (powerGap >= 0.55 && (action === 5 || (action >= 1 && action <= 4 && closeRange >= 0.35))) {
        breakdown.power_advantage_attack = clampRange(0.10 + (powerGap - 0.50) * 0.30 + closeRange * 0.08, 0.10, 0.25);
    }
    if (action === 0 && !preDeath && blastOverlap <= 0.45 && (bombThreat >= 0.20 || closeRange >= 0.55 || minEscapeEta >= 0.45)) {
        breakdown.tactical_wait = clampRange(0.05 + Math.max(bombThreat, closeRange, minEscapeEta) * 0.12, 0.05, 0.20);
    }
    if (done || TERMINAL_OUTCOME_TAGS.has(outcome)) {
        breakdown.terminal_reference = outcome === "win" ? 0.25 : (outcome === "self_kill" || outcome === "loss" ? 0.18 : 0.08);
    }
    if (preDeath || outcome === "self_kill" || Number(labels.bomb_self_trap_risk || 0) >= 0.55) {
        breakdown.survival_failure_reference = clampRange(0.18 + Number(labels.bomb_self_trap_risk || 0) * 0.18, 0.18, 0.36);
    }

    let score = 0;
    for (const key of Object.keys(breakdown)) score += Number(breakdown[key]) || 0;
    score = score * 0.70;
    if (action === 5) score = Math.max(score, 0.30);
    if (preDeath || done) score = Math.max(score, 0.22);
    return { score: clamp01(score), breakdown };
}

function buildAuxLabels(row) {
    const vec = getStateVector(row);
    const action = normalizeAction(row.action);
    const done = !!row.done;
    const preDeath = !!row.pre_death;
    const postEscape = clamp01(vec[16]);
    const minEscapeEta = clamp01(vec[17]);
    const deadendDepth = clamp01(vec[18]);
    const blastOverlap = clamp01(vec[19]);
    const enemyEscapeOptions = clamp01(vec[20]);
    const trapClosure = clamp01(vec[21]);
    const contextualRisk = clamp01(
        (preDeath ? 1 : 0)
        + (Number(row.risk_label || 0) > 0.5 ? 0.20 : 0)
        + (minEscapeEta < 0.20 ? 0.25 : 0)
        + (blastOverlap > 0.65 ? 0.30 : 0)
        + deadendDepth * 0.25
    );
    const bombSelfTrapRisk = action === 5
        ? clamp01(
            (postEscape > 0.70 ? 0.45 : 0)
            + (minEscapeEta < 0.20 ? 0.25 : 0)
            + (blastOverlap > 0.60 ? 0.20 : 0)
            + deadendDepth * 0.25
        )
        : contextualRisk;
    const myBombThreatScore = clamp01(Number(row.meta && row.meta.myBombThreatScore) || 0);
    const dangerCellsCreatedScore = clamp01(
        Number(row.meta && row.meta.dangerCellsCreatedScore)
        || (action === 5 ? Math.max(0.15, myBombThreatScore) : 0)
    );
    const roundKillCredit = clamp01(
        Number(row.meta && row.meta.roundKillCredit)
        || ((row.outcome_tag === "win" || row.terminal_reason === "caught_enemy") ? 0.35 : 0)
    );
    const roundSelfKillPenalty = clamp01(
        Number(row.meta && row.meta.roundSelfKillPenalty)
        || ((row.outcome_tag === "self_kill" || preDeath) ? Math.max(0.35, bombSelfTrapRisk) : (action === 5 ? bombSelfTrapRisk * 0.6 : 0))
    );
    const roundNetKdCredit = clamp01(
        Number(row.meta && row.meta.roundNetKdCredit)
        || clamp01((roundKillCredit - roundSelfKillPenalty + 1) / 2)
    );
    const baseLabels = {
        bomb_escape_success_label: action === 5 && !preDeath && !done && bombSelfTrapRisk < 0.45 && postEscape <= 0.60 ? 1 : 0,
        bomb_self_trap_risk: bombSelfTrapRisk,
        enemy_trap_after_bomb: action === 5 ? clamp01(0.65 * trapClosure + 0.35 * (1 - enemyEscapeOptions)) : 0,
        nearest_safe_tile_eta: minEscapeEta,
        commitment_depth: deadendDepth,
        terminal_credit_action: done || row.outcome_tag !== "ongoing" || preDeath ? 1 : 0,
        danger_cells_created_score: dangerCellsCreatedScore,
        round_kill_credit: roundKillCredit,
        round_self_kill_penalty: roundSelfKillPenalty,
        round_net_kd_credit: roundNetKdCredit,
    };
    const behavior = computeBehaviorScoreBreakdown(row, baseLabels);
    return Object.assign({}, row.aux_labels || {}, baseLabels, {
        behavior_score: behavior.score,
        behavior_score_band: behaviorScoreBand(behavior.score),
        behavior_high_value: behavior.score >= 0.35 ? 1 : 0,
        behavior_score_breakdown: behavior.breakdown,
    });
}

function classifySampleBucket(row) {
    if (SAMPLE_BUCKETS.includes(row.sample_bucket)) {
        return row.sample_bucket;
    }
    const aux = row.aux_labels || buildAuxLabels(row);
    const action = normalizeAction(row.action);
    const outcome = typeof row.outcome_tag === "string" ? row.outcome_tag : "ongoing";
    if (row.done || TERMINAL_OUTCOME_TAGS.has(outcome)) return "terminal";
    if (action === 5 && Number(aux.bomb_self_trap_risk || 0) >= 0.55) return "drop_bomb_bad";
    if (action === 5) return "drop_bomb_safe";
    if (row.pre_death || Number(aux.bomb_self_trap_risk || 0) >= 0.55 || Number(aux.commitment_depth || 0) >= 0.70) return "pre_death";
    return "ongoing";
}

function nearConfigForBucket(bucket, baseWindowMs, baseBurstLimit, balanced) {
    if (!balanced) return { windowMs: baseWindowMs, burstLimit: baseBurstLimit };
    if (bucket === "ongoing") return { windowMs: Math.max(baseWindowMs, 900), burstLimit: Math.max(2, Math.min(baseBurstLimit, 4)) };
    if (bucket === "terminal" || bucket === "pre_death") return { windowMs: Math.min(baseWindowMs, 350), burstLimit: Math.max(baseBurstLimit, 40) };
    return { windowMs: Math.min(baseWindowMs, 500), burstLimit: Math.max(baseBurstLimit, 24) };
}

function buildExactSignature(row) {
    const state = row.state || {};
    const nextState = row.next_state || null;
    const core = {
        sm: compactStateMap(state.state_map),
        sv: compactVector(state.state_vector),
        a: normalizeAction(row.action),
        m: Array.isArray(row.action_mask)
            ? row.action_mask.map((v) => (Number(v) > 0.5 ? 1 : 0))
            : [1, 1, 1, 1, 1, 1],
        done: row.done ? 1 : 0,
        pre_death: row.pre_death ? 1 : 0,
        outcome_tag: row.outcome_tag || "ongoing",
        nsm: nextState ? compactStateMap(nextState.state_map) : null,
        nsv: nextState ? compactVector(nextState.state_vector) : null,
    };
    return crypto.createHash("sha1").update(JSON.stringify(core)).digest("hex");
}

function findPeakPos(channelMap, threshold) {
    if (!Array.isArray(channelMap) || channelMap.length === 0) {
        return "-1:-1";
    }
    let best = -Infinity;
    let bx = -1;
    let by = -1;
    for (let y = 0; y < channelMap.length; y++) {
        const row = channelMap[y];
        if (!Array.isArray(row)) continue;
        for (let x = 0; x < row.length; x++) {
            const v = Number(row[x]) || 0;
            if (v > best) {
                best = v;
                bx = x;
                by = y;
            }
        }
    }
    if (best < threshold) {
        return "-1:-1";
    }
    return `${bx}:${by}`;
}

function quantizeVector(vec, scale) {
    if (!Array.isArray(vec)) return "";
    const s = Math.max(1, Number(scale) || 10);
    return vec
        .slice(0, 24)
        .map((v) => {
            const n = Number(v) || 0;
            return String(Math.round(n * s));
        })
        .join(",");
}

function buildNearDuplicateSignature(row) {
    const state = row.state || {};
    const stateMap = state.state_map || [];
    const selfLayer = stateMap.map((line) => (Array.isArray(line)
        ? line.map((cell) => (Array.isArray(cell) ? (Number(cell[3]) || 0) : 0))
        : []));
    const enemyLayer = stateMap.map((line) => (Array.isArray(line)
        ? line.map((cell) => (Array.isArray(cell) ? (Number(cell[8]) || 0) : 0))
        : []));
    const selfPos = findPeakPos(selfLayer, 0.5);
    const enemyPos = findPeakPos(enemyLayer, 0.2);
    const vec = (state && state.state_vector) || [];
    const action = normalizeAction(row.action);
    const outcome = String(row.outcome_tag || "ongoing");

    let localPatch = "";
    const pos = selfPos.split(":");
    const sx = parseInt(pos[0], 10);
    const sy = parseInt(pos[1], 10);
    if (Array.isArray(stateMap) && Number.isFinite(sx) && Number.isFinite(sy) && sx >= 0 && sy >= 0) {
        const patch = [];
        for (let dy = -1; dy <= 1; dy++) {
            for (let dx = -1; dx <= 1; dx++) {
                const y = sy + dy;
                const x = sx + dx;
                const cell = stateMap[y] && stateMap[y][x];
                if (!Array.isArray(cell)) {
                    patch.push("x");
                    continue;
                }
                patch.push(
                    `${Math.round((Number(cell[0]) || 0) * 4)}`
                    + `${Math.round((Number(cell[2]) || 0) * 4)}`
                    + `${Math.round((Number(cell[5]) || 0) * 4)}`
                    + `${Math.round((Number(cell[9]) || 0) * 4)}`
                );
            }
        }
        localPatch = patch.join(".");
    }
    return `${action}|${selfPos}|${enemyPos}|${quantizeVector(vec, 10)}|${localPatch}|${outcome}`;
}

function signatureHash32Hex(sig) {
    return crypto.createHash("sha1").update(sig).digest("hex").slice(0, 8);
}

function updateHist(row, stats) {
    const a = String(normalizeAction(row.action));
    const done = row.done ? "1" : "0";
    const preDeath = row.pre_death ? "1" : "0";
    const tag = String(row.outcome_tag || "ongoing").toLowerCase();
    stats.action_hist[a] = (stats.action_hist[a] || 0) + 1;
    stats.done_hist[done] = (stats.done_hist[done] || 0) + 1;
    if (stats.pre_death_hist) {
        stats.pre_death_hist[preDeath] = (stats.pre_death_hist[preDeath] || 0) + 1;
    }
    stats.outcome_hist[tag] = (stats.outcome_hist[tag] || 0) + 1;
}

async function readLines(filePath, onLine) {
    if (!fs.existsSync(filePath)) {
        return;
    }
    const rl = readline.createInterface({
        input: fs.createReadStream(filePath),
        crlfDelay: Infinity,
    });
    for await (const line of rl) {
        if (!line || !line.trim()) {
            continue;
        }
        await onLine(line);
    }
}

function countLinesSync(filePath) {
    if (!fs.existsSync(filePath)) return 0;
    const fd = fs.openSync(filePath, "r");
    const buffer = Buffer.allocUnsafe(1024 * 1024);
    let count = 0;
    try {
        let bytes = 0;
        do {
            bytes = fs.readSync(fd, buffer, 0, buffer.length, null);
            for (let i = 0; i < bytes; i++) {
                if (buffer[i] === 10) count += 1;
            }
        } while (bytes > 0);
    } finally {
        fs.closeSync(fd);
    }
    return count;
}

function getCpuTotalsSnapshot() {
    const cpus = os.cpus();
    let idle = 0;
    let total = 0;
    for (const cpu of cpus) {
        const times = cpu && cpu.times ? cpu.times : {};
        const user = Number(times.user) || 0;
        const nice = Number(times.nice) || 0;
        const sys = Number(times.sys) || 0;
        const irq = Number(times.irq) || 0;
        const cpuIdle = Number(times.idle) || 0;
        idle += cpuIdle;
        total += user + nice + sys + irq + cpuIdle;
    }
    return { idle, total };
}

function createCpuUsageSampler() {
    let prev = getCpuTotalsSnapshot();
    return function sampleCpuPercent() {
        const cur = getCpuTotalsSnapshot();
        const totalDiff = cur.total - prev.total;
        const idleDiff = cur.idle - prev.idle;
        prev = cur;
        if (!Number.isFinite(totalDiff) || totalDiff <= 0) return null;
        const usage = (1 - (idleDiff / totalDiff)) * 100;
        return Number.isFinite(usage) ? Math.max(0, Math.min(100, usage)) : null;
    };
}

function setWindowsProcessSuspended(pid, suspended) {
    if (process.platform !== "win32") return false;
    const n = Number(pid);
    if (!Number.isFinite(n) || n <= 0) return false;
    const cmd = suspended
        ? `Suspend-Process -Id ${n} -ErrorAction Stop`
        : `Resume-Process -Id ${n} -ErrorAction Stop`;
    const result = spawnSync("powershell", ["-NoProfile", "-Command", cmd], {
        cwd: ROOT,
        stdio: "ignore",
    });
    return result && result.status === 0;
}

function spawnWorker(workerIndex, args, logPrefix) {
    return new Promise((resolve) => {
        const child = spawn(process.execPath, args, {
            cwd: ROOT,
            stdio: "inherit",
        });
        resolve({ child, workerIndex, logPrefix });
    });
}

async function waitChild(childInfo) {
    const { child, workerIndex, logPrefix } = childInfo;
    return new Promise((resolve) => {
        child.on("exit", (code, signal) => {
            resolve({
                workerIndex,
                code: Number.isFinite(code) ? code : null,
                signal: signal || null,
                logPrefix,
            });
        });
    });
}

async function mergeAndDedupe(opts) {
    const {
        partDatasets,
        datasetPath,
        targetFrames,
        tempPath,
        nearDupWindowMs,
        nearDupBurstLimit,
        balanced,
        bucketTargets,
        noGlobalDedupe,
    } = opts;

    ensureParent(datasetPath);
    ensureParent(tempPath);
    if (fs.existsSync(tempPath)) fs.unlinkSync(tempPath);
    if (fs.existsSync(datasetPath)) fs.unlinkSync(datasetPath);

    const bucketPaths = {};
    const bucketStreams = {};
    const bucketCounts = {};
    const tierPaths = {};
    const tierStreams = {};
    const tierCounts = { high: 0, mid: 0, low: 0 };
    for (const bucket of SAMPLE_BUCKETS) {
        bucketPaths[bucket] = `${tempPath}.${bucket}.jsonl`;
        if (fs.existsSync(bucketPaths[bucket])) fs.unlinkSync(bucketPaths[bucket]);
        bucketStreams[bucket] = fs.createWriteStream(bucketPaths[bucket], { flags: "w" });
        bucketCounts[bucket] = 0;
    }
    for (const tier of SCORE_TIERS) {
        tierPaths[tier] = `${tempPath}.score_${tier}.jsonl`;
        if (fs.existsSync(tierPaths[tier])) fs.unlinkSync(tierPaths[tier]);
        tierStreams[tier] = fs.createWriteStream(tierPaths[tier], { flags: "w" });
    }

    let scannedRows = 0;
    let invalidRows = 0;
    let uniqueRows = 0;

    for (const partPath of partDatasets) {
        await readLines(partPath, async (line) => {
            scannedRows += 1;
            let row = null;
            try {
                row = JSON.parse(line);
            } catch (err) {
                invalidRows += 1;
                return;
            }
            row.aux_labels = buildAuxLabels(row);
            row.sample_bucket = classifySampleBucket(row);
            const bucket = SAMPLE_BUCKETS.includes(row.sample_bucket) ? row.sample_bucket : "ongoing";
            bucketStreams[bucket].write(JSON.stringify(row) + "\n");
            const tier = scoreTier(row.aux_labels && row.aux_labels.behavior_score);
            tierStreams[tier].write(JSON.stringify(row) + "\n");
            tierCounts[tier] += 1;
            bucketCounts[bucket] += 1;
            uniqueRows += 1;
        });
    }

    await Promise.all(SAMPLE_BUCKETS.map((bucket) => new Promise((resolve) => bucketStreams[bucket].end(resolve))));
    await Promise.all(SCORE_TIERS.map((tier) => new Promise((resolve) => tierStreams[tier].end(resolve))));

    const finalStats = {
        rows_written: 0,
        action_hist: { "0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0 },
        done_hist: { "0": 0, "1": 0 },
        pre_death_hist: { "0": 0, "1": 0 },
        outcome_hist: { ongoing: 0, win: 0, loss: 0, draw: 0, self_kill: 0 },
        bucket_hist: { ongoing: 0, pre_death: 0, drop_bomb_safe: 0, drop_bomb_bad: 0, terminal: 0 },
        terminal_reason_hist: { caught_enemy: 0, caught_self: 0, enemy_self_kill_discard: 0, stall_abort: 0, round_end: 0 },
        spawn_dist_hist: { "1_3": 0, "4_6": 0, "7_10": 0, other: 0 },
        property_bucket_hist: { low: 0, medium: 0, high: 0 },
        aux_label_hist: {
            bomb_escape_success_label: 0,
            bomb_self_trap_risk_high: 0,
            enemy_trap_after_bomb_high: 0,
            terminal_credit_action: 0,
            danger_cells_created_score_nonzero: 0,
            round_kill_credit_nonzero: 0,
            round_self_kill_penalty_nonzero: 0,
        },
        behavior_score_hist: { zero: 0, low: 0, mid: 0, high: 0, very_high: 0 },
        behavior_score_breakdown_hist: BEHAVIOR_BREAKDOWN_KEYS.reduce((acc, key) => {
            acc[key] = 0;
            return acc;
        }, {}),
        high_value_behavior_rows: 0,
        behavior_score_sum: 0,
    };

    function updateFinalStats(row) {
        updateHist(row, finalStats);
        const bucket = SAMPLE_BUCKETS.includes(row.sample_bucket) ? row.sample_bucket : classifySampleBucket(row);
        finalStats.bucket_hist[bucket] = (finalStats.bucket_hist[bucket] || 0) + 1;
        const reason = String(row.terminal_reason || "").trim().toLowerCase();
        if (finalStats.terminal_reason_hist[reason] != null) {
            finalStats.terminal_reason_hist[reason] += 1;
        }
        const spawnDist = Number(row.meta && row.meta.spawnShortestPathDist);
        if (Number.isFinite(spawnDist)) {
            if (spawnDist >= 1 && spawnDist <= 3) finalStats.spawn_dist_hist["1_3"] += 1;
            else if (spawnDist >= 4 && spawnDist <= 6) finalStats.spawn_dist_hist["4_6"] += 1;
            else if (spawnDist >= 7 && spawnDist <= 10) finalStats.spawn_dist_hist["7_10"] += 1;
            else finalStats.spawn_dist_hist.other += 1;
        }
        const propertyScore = Number(row.meta && row.meta.property_bucket_score);
        if (Number.isFinite(propertyScore)) {
            if (propertyScore < 0.34) finalStats.property_bucket_hist.low += 1;
            else if (propertyScore < 0.67) finalStats.property_bucket_hist.medium += 1;
            else finalStats.property_bucket_hist.high += 1;
        }
        const aux = row.aux_labels || {};
        if (Number(aux.bomb_escape_success_label || 0) > 0.5) finalStats.aux_label_hist.bomb_escape_success_label += 1;
        if (Number(aux.bomb_self_trap_risk || 0) >= 0.55) finalStats.aux_label_hist.bomb_self_trap_risk_high += 1;
        if (Number(aux.enemy_trap_after_bomb || 0) >= 0.55) finalStats.aux_label_hist.enemy_trap_after_bomb_high += 1;
        if (Number(aux.terminal_credit_action || 0) > 0.5) finalStats.aux_label_hist.terminal_credit_action += 1;
        if (Number(aux.danger_cells_created_score || 0) > 0) finalStats.aux_label_hist.danger_cells_created_score_nonzero += 1;
        if (Number(aux.round_kill_credit || 0) > 0) finalStats.aux_label_hist.round_kill_credit_nonzero += 1;
        if (Number(aux.round_self_kill_penalty || 0) > 0) finalStats.aux_label_hist.round_self_kill_penalty_nonzero += 1;
        const score = clamp01(aux.behavior_score);
        finalStats.behavior_score_sum += score;
        finalStats.behavior_score_hist[behaviorScoreBand(score)] = (finalStats.behavior_score_hist[behaviorScoreBand(score)] || 0) + 1;
        if (score >= 0.35) finalStats.high_value_behavior_rows += 1;
        const breakdown = aux.behavior_score_breakdown || {};
        for (const key of BEHAVIOR_BREAKDOWN_KEYS) {
            if (Number(breakdown[key] || 0) > 0) {
                finalStats.behavior_score_breakdown_hist[key] = (finalStats.behavior_score_breakdown_hist[key] || 0) + 1;
            }
        }
    }

    async function writeSelected(filePath, available, keepCount, out) {
        if (keepCount <= 0 || available <= 0) return;
        let keep = null;
        if (available > keepCount) {
            const scored = [];
            let idx = 0;
            await readLines(filePath, async (line) => {
                let score = 0;
                try {
                    const row = JSON.parse(line);
                    score = clamp01(row.aux_labels && row.aux_labels.behavior_score);
                } catch (err) {}
                scored.push({ score, h: signatureHash32Hex(line), idx });
                idx += 1;
            });
            scored.sort((a, b) => {
                if (b.score !== a.score) return b.score - a.score;
                if (a.h < b.h) return -1;
                if (a.h > b.h) return 1;
                return a.idx - b.idx;
            });
            keep = new Set(scored.slice(0, keepCount).map((x) => x.idx));
        }
        let idx = 0;
        await readLines(filePath, async (line) => {
            if (keep && !keep.has(idx)) {
                idx += 1;
                return;
            }
            out.write(line.trim() + "\n");
            finalStats.rows_written += 1;
            try {
                updateFinalStats(JSON.parse(line));
            } catch (err) {}
            idx += 1;
        });
    }

    let keepByBucket = {};
    let keepByTier = { high: 0, mid: 0, low: 0 };
    if (uniqueRows > targetFrames) {
        let remaining = targetFrames;
        if (noGlobalDedupe) {
            const totalTier = Math.max(1, (tierCounts.high || 0) + (tierCounts.mid || 0) + (tierCounts.low || 0));
            keepByTier.high = Math.min(tierCounts.high || 0, Math.round(targetFrames * (tierCounts.high || 0) / totalTier));
            keepByTier.mid = Math.min(tierCounts.mid || 0, Math.round(targetFrames * (tierCounts.mid || 0) / totalTier));
            if (keepByTier.high + keepByTier.mid > targetFrames) {
                const overflow = keepByTier.high + keepByTier.mid - targetFrames;
                keepByTier.mid = Math.max(0, keepByTier.mid - overflow);
            }
            keepByTier.low = Math.max(0, Math.min(tierCounts.low || 0, targetFrames - keepByTier.high - keepByTier.mid));
            const used = keepByTier.high + keepByTier.mid + keepByTier.low;
            if (used < targetFrames) {
                const spill = targetFrames - used;
                keepByTier.high = Math.min(tierCounts.high || 0, keepByTier.high + spill);
            }
        } else {
            keepByTier.high = Math.min(tierCounts.high || 0, remaining);
            remaining -= keepByTier.high;
            keepByTier.mid = Math.min(tierCounts.mid || 0, remaining);
            remaining -= keepByTier.mid;
            keepByTier.low = Math.min(tierCounts.low || 0, remaining);
        }
        for (const bucket of SAMPLE_BUCKETS) {
            keepByBucket[bucket] = bucketCounts[bucket] || 0;
        }
    } else {
        for (const bucket of SAMPLE_BUCKETS) {
            keepByBucket[bucket] = bucketCounts[bucket] || 0;
        }
        keepByTier = Object.assign({}, tierCounts);
    }

    const out = fs.createWriteStream(datasetPath, { flags: "w" });
    for (const tier of SCORE_TIERS) {
        await writeSelected(tierPaths[tier], tierCounts[tier] || 0, keepByTier[tier] || 0, out);
    }
    await new Promise((resolve) => out.end(resolve));

    for (const bucket of SAMPLE_BUCKETS) {
        if (fs.existsSync(bucketPaths[bucket])) {
            fs.unlinkSync(bucketPaths[bucket]);
        }
    }
    for (const tier of SCORE_TIERS) {
        if (fs.existsSync(tierPaths[tier])) {
            fs.unlinkSync(tierPaths[tier]);
        }
    }

    return {
        scanned_rows: scannedRows,
        invalid_rows: invalidRows,
        exact_duplicate_rows: 0,
        near_duplicate_rows: 0,
        bucket_near_duplicate_rows: { ongoing: 0, pre_death: 0, drop_bomb_safe: 0, drop_bomb_bad: 0, terminal: 0 },
        unique_rows_before_cap: uniqueRows,
        rows_written: finalStats.rows_written,
        action_hist_written: finalStats.action_hist,
        done_hist_written: finalStats.done_hist,
        pre_death_hist_written: finalStats.pre_death_hist,
        outcome_hist_written: finalStats.outcome_hist,
        bucket_hist_before_cap: bucketCounts,
        bucket_keep_target: keepByBucket,
        score_tier_hist_before_cap: tierCounts,
        score_tier_keep_target: keepByTier,
        bucket_hist_written: finalStats.bucket_hist,
        terminal_reason_hist_written: finalStats.terminal_reason_hist,
        spawn_dist_hist_written: finalStats.spawn_dist_hist,
        property_bucket_hist_written: finalStats.property_bucket_hist,
        aux_label_hist_written: finalStats.aux_label_hist,
        behavior_score_hist_written: finalStats.behavior_score_hist,
        behavior_score_mean: finalStats.rows_written > 0 ? finalStats.behavior_score_sum / finalStats.rows_written : 0,
        behavior_score_breakdown_hist_written: finalStats.behavior_score_breakdown_hist,
        high_value_behavior_rows: finalStats.high_value_behavior_rows,
        high_value_ratio: finalStats.rows_written > 0 ? finalStats.high_value_behavior_rows / finalStats.rows_written : 0,
    };
}

function readJsonMaybe(filePath) {
    if (!fs.existsSync(filePath)) return null;
    try {
        return JSON.parse(fs.readFileSync(filePath, "utf8"));
    } catch (err) {
        return null;
    }
}

async function main() {
    const startedAt = Date.now();
    const workers = asPositiveInt(getArg("workers", "1"), 1);
    const targetFrames = asPositiveInt(getArg("target-frames", "200000"), 200000);
    const oversampleRatio = Math.max(1.0, asFloat(getArg("oversample-ratio", "1.35"), 1.35));
    const maxWallSec = asPositiveInt(getArg("max-wall-sec", "7200"), 7200);
    const seedBase = asInt(getArg("seed-base", "20260419"), 20260419) >>> 0;
    const balanced = getArg("balanced", "0") === "1";
    const bucketTargets = parseBucketTargets(getArg("bucket-targets", "ongoing:0.60,pre_death:0.15,drop_bomb_safe:0.10,drop_bomb_bad:0.10,terminal:0.05"));
    const scenarioBuckets = getArg("scenario-buckets", "open_random,escape_after_bomb,enemy_choke,item_race,deadend_chase");
    const softObstacleReinjectRatio = getArg("soft-obstacle-reinject-ratio", balanced ? "0.08" : "0");
    const spawnRandomize = getArg("spawn-randomize", balanced ? "1" : "0");
    const terminalTailMs = getArg("terminal-tail-ms", "3000");
    const matchDurationSec = getArg("match-duration-sec", balanced ? "18" : "45");
    const suddenDeath = getArg("sudden-death", "1");
    const disableRevive = getArg("disable-revive", "1");
    const ignoreEnemySelfKill = getArg("ignore-enemy-self-kill", "1");
    const stallNoProgressMs = getArg("stall-no-progress-ms", "12000");
    const maxEpisodeMs = getArg("max-episode-ms", "18000");
    const mirrorSampling = getArg("mirror-sampling", balanced ? "1" : "0");
    const earlyCommitHighValue = getArg("early-commit-high-value", balanced ? "1" : "0");
    const behaviorScoring = getArg("behavior-scoring", balanced ? "1" : "0");
    const behaviorScoreThreshold = getArg("behavior-score-threshold", "0.12");
    const highValueBehaviorThreshold = getArg("high-value-behavior-threshold", "0.35");
    const behaviorCreditWindowMs = getArg("behavior-credit-window-ms", "6000");
    const offlineMicro = getArg("offline-micro", balanced ? "1" : "0") !== "0";
    const offlineMicroRatio = Math.max(0, Math.min(0.99, asFloat(getArg("offline-micro-ratio", balanced ? "0.98" : "0"), balanced ? 0.98 : 0)));
    const offlineMicroFramesArg = asInt(getArg("offline-micro-frames", "-1"), -1);
    const softObstacleKeepMin = getArg("soft-obstacle-keep-min", "0");
    const softObstacleKeepMax = getArg("soft-obstacle-keep-max", "20");
    const microScenario = getArg("micro-scenario", offlineMicro ? "0" : (balanced ? "1" : "0"));
    const microSamplesPerEpisode = getArg("micro-samples-per-episode", balanced ? "16" : "0");
    const pollMs = getArg("poll-ms", "400");
    const batchSize = getArg("batch-size", "2048");
    const thinkIntervalMs = getArg("think-interval-ms", "8");
    const partialClearMinRatio = getArg("partial-clear-min-ratio", "0.35");
    const partialClearMaxRatio = getArg("partial-clear-max-ratio", "0.75");
    const spawnShortestPathMin = getArg("spawn-shortest-path-min", "1");
    const spawnShortestPathMax = getArg("spawn-shortest-path-max", "10");
    const burstCapOngoing = getArg("burst-cap-ongoing", "6");
    const noGlobalDedupe = getArg("no-global-dedupe", "1") !== "0";
    const specialBombEscape = getArg("special-bomb-escape", "0");
    const specialRoundSec = getArg("special-round-sec", "60");
    const specialItemRespawnMs = getArg("special-item-respawn-ms", "2000");
    const respawnDelayMs = getArg("respawn-delay-ms", "0");
    const respawnInvincibleMs = getArg("respawn-invincible-ms", "300");
    const dangerScoreWeight = getArg("danger-score-weight", "1.0");
    const killScoreWeight = getArg("kill-score-weight", "1.0");
    const selfKillPenaltyWeight = getArg("self-kill-penalty-weight", "2.0");
    const screenshot = getArg("screenshot", "0");

    const mapId = getArg("map", "windmill-heart");
    const arena = getArg("arena", "1v1");
    const actionSpace = getArg("action-space", "discrete6");

    const clearNonRigid = getArg("clear-nonrigid", "1");
    const randomItemDensity = getArg("random-item-density", "0.12");
    const randomItemDensityJitter = getArg("random-item-density-jitter", "0.06");
    const itemRespawnMs = getArg("item-respawn-ms", "1000");
    const itemRespawnJitterRatio = getArg("item-respawn-jitter-ratio", "0.35");
    const itemSafeRadius = getArg("item-safe-radius", "2");
    const itemSafeRadiusJitter = getArg("item-safe-radius-jitter", "1");
    const itemMax = getArg("item-max", "24");
    const opponentThinkJitterRatio = getArg("opponent-think-jitter-ratio", "0.3");
    const mlEnabled = getArg("ml-enabled", "0");
    const mlCollect = getArg("ml-collect", "1");
    const mlFreeze = getArg("ml-freeze", "1");
    const mlPolicyMode = getArg("ml-policy-mode", "pure");
    const mlIqlMix = getArg("ml-iql-mix", "1");
    const mlModelUrl = getArg("ml-model-url", "");
    const mlConf = getArg("ml-conf", "");
    const mlMoveConf = getArg("ml-move-conf", "");
    const mlMargin = getArg("ml-margin", "");
    const mlForceMoveEta = getArg("ml-force-move-eta", "");
    const mlWaitBlockEta = getArg("ml-wait-block-eta", "");
    const mlMoveThreatMs = getArg("ml-move-threat-ms", "");
    const opponentPool = getArg(
        "opponent-pool",
        getArg("opponents", "heuristic_v1,heuristic_v2,aggressive_trapper,coward_runner,item_rusher,randomized_mistake_bot")
    );
    const agentPool = getArg("agent-pool", "heuristic_v2,aggressive_trapper");
    const agentExpertDuel = getArg("agent-expert-duel", "1");

    const nearDupWindowMs = asPositiveInt(getArg("near-dup-window-ms", "700"), 700);
    const nearDupBurstLimit = asPositiveInt(getArg("near-dup-burst-limit", "8"), 8);
    const minFinalRatio = Math.max(0, Math.min(1, asFloat(getArg("min-final-ratio", "0.98"), 0.98)));
    const cpuCapPercent = clampRange(asFloat(getArg("cpu-cap-percent", "70"), 70), 5, 100);
    const cpuControlMs = asPositiveInt(getArg("cpu-control-ms", "2500"), 2500);
    const cpuResumeHysteresis = clampRange(asFloat(getArg("cpu-resume-hysteresis", "8"), 8), 1, 30);
    const cpuControlEnabled = process.platform === "win32" && cpuCapPercent < 99.9;

    const fresh = getArg("fresh", "1") !== "0";

    const datasetPath = path.resolve(getArg("dataset-path", "output/ml/datasets/combat_phase0_v2_features24.jsonl"));
    const reportPath = path.resolve(getArg("report-path", `output/ml/reports/combat_phase0_collect_parallel_${Date.now()}.json`));
    const partsDirBase = path.resolve(getArg("parts-dir", "output/ml/datasets/parts"));
    const defaultRunId = path.basename(datasetPath, path.extname(datasetPath)).replace(/[^a-zA-Z0-9_.-]+/g, "_") + "_" + startedAt;
    const partsRunId = getArg("parts-run-id", defaultRunId).replace(/[^a-zA-Z0-9_.-]+/g, "_");
    const partsDir = path.join(partsDirBase, partsRunId);

    ensureParent(datasetPath);
    ensureParent(reportPath);
    fs.rmSync(partsDir, { recursive: true, force: true });
    fs.mkdirSync(partsDir, { recursive: true });

    if (fresh && fs.existsSync(datasetPath)) {
        fs.unlinkSync(datasetPath);
    }

    const partDatasets = [];
    const partReports = [];
    const workerTasks = [];
    const workerProcs = [];
    const collectorControls = [];

    const offlineMicroTarget = offlineMicro
        ? Math.max(0, Math.min(
            targetFrames,
            offlineMicroFramesArg >= 0 ? offlineMicroFramesArg : Math.floor(targetFrames * offlineMicroRatio)
        ))
        : 0;
    const onlineTargetFrames = Math.max(0, targetFrames - offlineMicroTarget);
    const activeWorkers = onlineTargetFrames > 0 ? workers : 0;
    const perWorkerTarget = activeWorkers > 0
        ? Math.max(500, Math.ceil((onlineTargetFrames * oversampleRatio) / activeWorkers))
        : 0;

    if (offlineMicroTarget > 0) {
        const offlineDataset = path.join(partsDir, "combat_phase0_offline_micro.jsonl");
        const offlineReport = path.join(partsDir, "combat_phase0_offline_micro.json");
        partDatasets.push(offlineDataset);
        partReports.push(offlineReport);
        const args = [
            OFFLINE_MICRO_SCRIPT,
            `--target-frames=${offlineMicroTarget}`,
            `--seed-base=${(seedBase + 0x5eed123) >>> 0}`,
            `--dataset-path=${offlineDataset}`,
            `--report-path=${offlineReport}`,
            `--scenario-buckets=${scenarioBuckets}`,
            `--random-item-density=${randomItemDensity}`,
            `--soft-obstacle-keep-min=${softObstacleKeepMin}`,
            `--soft-obstacle-keep-max=${softObstacleKeepMax}`,
            `--spawn-shortest-path-min=${spawnShortestPathMin}`,
            `--spawn-shortest-path-max=${spawnShortestPathMax}`,
        ];
        const info = await spawnWorker(-1, args, "[offline-micro]");
        workerProcs.push(info.child);
        workerTasks.push(waitChild(info));
    }

    for (let i = 0; i < activeWorkers; i++) {
        const partDataset = path.join(partsDir, `combat_phase0_v2_features24_worker_${i}.jsonl`);
        const partReport = path.join(partsDir, `combat_phase0_v2_features24_worker_${i}.json`);
        const workerSeed = (seedBase + i * 100003) >>> 0;

        if (fresh && fs.existsSync(partDataset)) fs.unlinkSync(partDataset);
        if (fresh && fs.existsSync(partReport)) fs.unlinkSync(partReport);

        partDatasets.push(partDataset);
        partReports.push(partReport);

        const args = [
            COLLECT_SCRIPT,
            `--target-frames=${perWorkerTarget}`,
            `--max-wall-sec=${maxWallSec}`,
            `--seed-base=${workerSeed}`,
            `--arena=${arena}`,
            `--map=${mapId}`,
            `--action-space=${actionSpace}`,
            "--fresh=1",
            "--dedupe-scope=off",
            `--dataset-path=${partDataset}`,
            `--report-path=${partReport}`,
            `--clear-nonrigid=${clearNonRigid}`,
            `--random-item-density=${randomItemDensity}`,
            `--random-item-density-jitter=${randomItemDensityJitter}`,
            `--item-respawn-ms=${itemRespawnMs}`,
            `--item-respawn-jitter-ratio=${itemRespawnJitterRatio}`,
            `--item-safe-radius=${itemSafeRadius}`,
            `--item-safe-radius-jitter=${itemSafeRadiusJitter}`,
            `--item-max=${itemMax}`,
            `--ml-enabled=${mlEnabled}`,
            `--ml-collect=${mlCollect}`,
            `--ml-freeze=${mlFreeze}`,
            `--ml-policy-mode=${mlPolicyMode}`,
            `--ml-iql-mix=${mlIqlMix}`,
            `--ml-model-url=${mlModelUrl}`,
            `--ml-conf=${mlConf}`,
            `--ml-move-conf=${mlMoveConf}`,
            `--ml-margin=${mlMargin}`,
            `--ml-force-move-eta=${mlForceMoveEta}`,
            `--ml-wait-block-eta=${mlWaitBlockEta}`,
            `--ml-move-threat-ms=${mlMoveThreatMs}`,
            `--opponent-think-jitter-ratio=${opponentThinkJitterRatio}`,
            `--opponent-pool=${opponentPool}`,
            `--agent-pool=${agentPool}`,
            `--agent-expert-duel=${agentExpertDuel}`,
            `--match-duration-sec=${matchDurationSec}`,
            `--balanced=${balanced ? 1 : 0}`,
            `--scenario-buckets=${scenarioBuckets}`,
            `--soft-obstacle-reinject-ratio=${softObstacleReinjectRatio}`,
            `--spawn-randomize=${spawnRandomize}`,
            `--terminal-tail-ms=${terminalTailMs}`,
            `--sudden-death=${suddenDeath}`,
            `--disable-revive=${disableRevive}`,
            `--ignore-enemy-self-kill=${ignoreEnemySelfKill}`,
            `--stall-no-progress-ms=${stallNoProgressMs}`,
            `--max-episode-ms=${maxEpisodeMs}`,
            `--mirror-sampling=${mirrorSampling}`,
            `--early-commit-high-value=${earlyCommitHighValue}`,
            `--behavior-scoring=${behaviorScoring}`,
            `--behavior-score-threshold=${behaviorScoreThreshold}`,
            `--high-value-behavior-threshold=${highValueBehaviorThreshold}`,
            `--behavior-credit-window-ms=${behaviorCreditWindowMs}`,
            `--micro-scenario=${microScenario}`,
            `--micro-samples-per-episode=${microSamplesPerEpisode}`,
            `--poll-ms=${pollMs}`,
            `--batch-size=${batchSize}`,
            `--think-interval-ms=${thinkIntervalMs}`,
            `--partial-clear-min-ratio=${partialClearMinRatio}`,
            `--partial-clear-max-ratio=${partialClearMaxRatio}`,
            `--soft-obstacle-keep-min=${softObstacleKeepMin}`,
            `--soft-obstacle-keep-max=${softObstacleKeepMax}`,
            `--spawn-shortest-path-min=${spawnShortestPathMin}`,
            `--spawn-shortest-path-max=${spawnShortestPathMax}`,
            `--burst-cap-ongoing=${burstCapOngoing}`,
            `--no-global-dedupe=${noGlobalDedupe ? 1 : 0}`,
            `--special-bomb-escape=${specialBombEscape}`,
            `--special-round-sec=${specialRoundSec}`,
            `--special-item-respawn-ms=${specialItemRespawnMs}`,
            `--respawn-delay-ms=${respawnDelayMs}`,
            `--respawn-invincible-ms=${respawnInvincibleMs}`,
            `--danger-score-weight=${dangerScoreWeight}`,
            `--kill-score-weight=${killScoreWeight}`,
            `--self-kill-penalty-weight=${selfKillPenaltyWeight}`,
            `--screenshot=${screenshot}`,
        ];

        const info = await spawnWorker(i, args, `[collector-${i}]`);
        workerProcs.push(info.child);
        collectorControls.push({
            workerIndex: i,
            child: info.child,
            paused: false,
        });
        workerTasks.push(waitChild(info));
    }

    const hardStopMs = Math.max(0, maxWallSec * 1000 + 120000);
    let hardStopped = false;
    let targetStopTriggered = false;
    let hardStopTimer = null;
    let targetStopTimer = null;
    let cpuControlTimer = null;
    let killEscalationTimer = null;
    let cpuUsageSampleCount = 0;
    let cpuUsageSampleSum = 0;
    let cpuUsageMax = 0;
    let cpuThrottleEvents = 0;
    let cpuResumeEvents = 0;
    let cpuControlErrors = 0;
    const sampleCpuUsage = createCpuUsageSampler();

    function resumeAllPausedCollectors() {
        for (const ctl of collectorControls) {
            if (!ctl || !ctl.paused || !ctl.child || ctl.child.exitCode != null) {
                continue;
            }
            if (setWindowsProcessSuspended(ctl.child.pid, false)) {
                ctl.paused = false;
            } else {
                cpuControlErrors += 1;
            }
        }
    }

    function stopWorkers(reason) {
        if (reason === "target") {
            targetStopTriggered = true;
        } else {
            hardStopped = true;
        }
        resumeAllPausedCollectors();
        for (const p of workerProcs) {
            if (p && p.exitCode == null && !p.killed) {
                try {
                    p.kill("SIGTERM");
                } catch (err) {}
            }
        }
        if (!killEscalationTimer) {
            killEscalationTimer = setTimeout(() => {
                for (const p of workerProcs) {
                    if (p && p.exitCode == null) {
                        try {
                            p.kill("SIGKILL");
                        } catch (err) {}
                    }
                }
            }, 15000);
        }
    }
    if (hardStopMs > 0) {
        hardStopTimer = setTimeout(() => {
            stopWorkers("hard_stop");
        }, hardStopMs);
    }
    targetStopTimer = setInterval(() => {
        const rowsSoFar = partDatasets
            .map((p) => countLinesSync(p))
            .reduce((a, b) => a + b, 0);
        if (rowsSoFar >= targetFrames) {
            stopWorkers("target");
            clearInterval(targetStopTimer);
            targetStopTimer = null;
        }
    }, 2500);

    if (cpuControlEnabled && collectorControls.length > 0) {
        cpuControlTimer = setInterval(() => {
            const usage = sampleCpuUsage();
            if (!Number.isFinite(usage)) {
                return;
            }
            cpuUsageSampleCount += 1;
            cpuUsageSampleSum += usage;
            cpuUsageMax = Math.max(cpuUsageMax, usage);

            const runningCollectors = collectorControls.filter((ctl) => ctl && ctl.child && ctl.child.exitCode == null);
            if (runningCollectors.length <= 0) {
                return;
            }
            const activeCollectors = runningCollectors.filter((ctl) => !ctl.paused);
            const pausedCollectors = runningCollectors.filter((ctl) => !!ctl.paused);
            const resumeThreshold = Math.max(0, cpuCapPercent - cpuResumeHysteresis);

            if (usage > cpuCapPercent && activeCollectors.length > 0) {
                const target = activeCollectors[activeCollectors.length - 1];
                if (setWindowsProcessSuspended(target.child.pid, true)) {
                    target.paused = true;
                    cpuThrottleEvents += 1;
                } else {
                    cpuControlErrors += 1;
                }
                return;
            }

            if (usage <= resumeThreshold && pausedCollectors.length > 0) {
                const target = pausedCollectors[0];
                if (setWindowsProcessSuspended(target.child.pid, false)) {
                    target.paused = false;
                    cpuResumeEvents += 1;
                } else {
                    cpuControlErrors += 1;
                }
            }
        }, cpuControlMs);
    }

    const workerExit = await Promise.all(workerTasks);
    if (hardStopTimer) {
        clearTimeout(hardStopTimer);
    }
    if (targetStopTimer) {
        clearInterval(targetStopTimer);
    }
    if (cpuControlTimer) {
        clearInterval(cpuControlTimer);
    }
    resumeAllPausedCollectors();
    if (killEscalationTimer) {
        clearTimeout(killEscalationTimer);
    }

    const tmpUniquePath = path.resolve(path.join(partsDir, `combat_phase0_v2_features24_unique_tmp_${Date.now()}.jsonl`));
    const mergeStats = await mergeAndDedupe({
        partDatasets,
        datasetPath,
        targetFrames,
        tempPath: tmpUniquePath,
        nearDupWindowMs,
        nearDupBurstLimit,
        balanced,
        bucketTargets,
        noGlobalDedupe,
    });

    const workerReports = partReports.map((p) => ({ path: p, report: readJsonMaybe(p) }));

    const aggregateRowsWrittenByWorkers = workerReports
        .map((x) => (x.report && Number(x.report.rows_written)) || 0)
        .reduce((a, b) => a + b, 0);
    const aggregateEarlyCommittedRows = workerReports
        .map((x) => (x.report && Number(x.report.early_committed_rows)) || 0)
        .reduce((a, b) => a + b, 0);
    const aggregateMirrorRowsSeen = workerReports
        .map((x) => (x.report && Number(x.report.mirror_rows_seen)) || 0)
        .reduce((a, b) => a + b, 0);
    const aggregateMirrorEpisodesCommitted = workerReports
        .map((x) => (x.report && Number(x.report.mirror_episodes_committed)) || 0)
        .reduce((a, b) => a + b, 0);
    const aggregateMicroRowsSeen = workerReports
        .map((x) => (x.report && Number(x.report.micro_rows_seen)) || 0)
        .reduce((a, b) => a + b, 0);
    const aggregateStreamingWrittenRows = workerReports
        .map((x) => (x.report && Number(x.report.streaming_written_rows)) || 0)
        .reduce((a, b) => a + b, 0);
    const aggregateEpisodeBufferWrittenRows = workerReports
        .map((x) => (x.report && Number(x.report.episode_buffer_written_rows)) || 0)
        .reduce((a, b) => a + b, 0);
    const aggregateTerminalCreditAppliedRows = workerReports
        .map((x) => (x.report && Number(x.report.terminal_credit_applied_rows)) || 0)
        .reduce((a, b) => a + b, 0);
    const aggregateStallKeptRows = workerReports
        .map((x) => (x.report && Number(x.report.stall_kept_rows)) || 0)
        .reduce((a, b) => a + b, 0);

    const doneWritten = mergeStats.done_hist_written["1"] || 0;
    const rowsWritten = mergeStats.rows_written || 0;
    const doneRatio = rowsWritten > 0 ? doneWritten / rowsWritten : 0;
    const minRequiredRows = Math.max(1, Math.floor(targetFrames * minFinalRatio));
    const isExpectedStopSignal = (sig) => sig === "SIGTERM" || sig === "SIGKILL";
    const workerFailureCount = workerExit.filter((x) => {
        if (!x) return true;
        if (x.code === 0 && !x.signal) return false;
        if (targetStopTriggered && isExpectedStopSignal(x.signal)) return false;
        return true;
    }).length;
    const collectionComplete = rowsWritten >= minRequiredRows;

    const report = {
        ts: Date.now(),
        dataset_path: datasetPath,
        report_path: reportPath,
        mode: "collect_parallel",
        map_id: mapId,
        arena,
        action_space: actionSpace,
        workers,
        active_workers: activeWorkers,
        target_frames: targetFrames,
        per_worker_target: perWorkerTarget,
        oversample_ratio: oversampleRatio,
        max_wall_sec: maxWallSec,
        hard_stopped: hardStopped,
        target_stop_triggered: targetStopTriggered,
        parts_dir: partsDir,
        parts_run_id: partsRunId,
        offline_micro: offlineMicro,
        offline_micro_target: offlineMicroTarget,
        offline_micro_ratio: offlineMicroRatio,
        online_target_frames: onlineTargetFrames,
        balanced,
        bucket_targets: bucketTargets,
        scenario_buckets: String(scenarioBuckets).split(",").map((s) => s.trim()).filter(Boolean),
        near_dup_window_ms: nearDupWindowMs,
        near_dup_burst_limit: nearDupBurstLimit,
        min_final_ratio: minFinalRatio,
        min_required_rows: minRequiredRows,
        cpu_cap_percent: Number(cpuCapPercent),
        cpu_control_ms: Number(cpuControlMs),
        cpu_resume_hysteresis: Number(cpuResumeHysteresis),
        cpu_control_enabled: cpuControlEnabled,
        cpu_usage_avg_percent: cpuUsageSampleCount > 0 ? (cpuUsageSampleSum / cpuUsageSampleCount) : null,
        cpu_usage_max_percent: cpuUsageSampleCount > 0 ? cpuUsageMax : null,
        cpu_usage_sample_count: cpuUsageSampleCount,
        cpu_throttle_events: cpuThrottleEvents,
        cpu_resume_events: cpuResumeEvents,
        cpu_control_errors: cpuControlErrors,
        collection_complete: collectionComplete,
        worker_failure_count: workerFailureCount,
        soft_obstacle_reinject_ratio: Number(softObstacleReinjectRatio),
        spawn_randomize: String(spawnRandomize) !== "0",
        terminal_tail_ms: Number(terminalTailMs),
        match_duration_sec: Number(matchDurationSec),
        sudden_death: String(suddenDeath) !== "0",
        disable_revive: String(disableRevive) !== "0",
        ignore_enemy_self_kill: String(ignoreEnemySelfKill) !== "0",
        stall_no_progress_ms: Number(stallNoProgressMs),
        max_episode_ms: Number(maxEpisodeMs),
        mirror_sampling: String(mirrorSampling) !== "0",
        early_commit_high_value: String(earlyCommitHighValue) !== "0",
        behavior_scoring: String(behaviorScoring) !== "0",
        behavior_score_threshold: Number(behaviorScoreThreshold),
        high_value_behavior_threshold: Number(highValueBehaviorThreshold),
        behavior_credit_window_ms: Number(behaviorCreditWindowMs),
        micro_scenario: String(microScenario) !== "0",
        micro_samples_per_episode: Number(microSamplesPerEpisode),
        poll_ms: Number(pollMs),
        batch_size: Number(batchSize),
        think_interval_ms: Number(thinkIntervalMs),
        partial_clear_min_ratio: Number(partialClearMinRatio),
        partial_clear_max_ratio: Number(partialClearMaxRatio),
        soft_obstacle_keep_min: Number(softObstacleKeepMin),
        soft_obstacle_keep_max: Number(softObstacleKeepMax),
        spawn_shortest_path_min: Number(spawnShortestPathMin),
        spawn_shortest_path_max: Number(spawnShortestPathMax),
        burst_cap_ongoing: Number(burstCapOngoing),
        no_global_dedupe: noGlobalDedupe,
        special_bomb_escape: String(specialBombEscape) !== "0",
        special_round_sec: Number(specialRoundSec),
        special_item_respawn_ms: Number(specialItemRespawnMs),
        respawn_delay_ms: Number(respawnDelayMs),
        respawn_invincible_ms: Number(respawnInvincibleMs),
        danger_score_weight: Number(dangerScoreWeight),
        kill_score_weight: Number(killScoreWeight),
        self_kill_penalty_weight: Number(selfKillPenaltyWeight),
        screenshot: String(screenshot) !== "0",
        clear_nonrigid: clearNonRigid !== "0",
        random_item_density: Number(randomItemDensity),
        random_item_density_jitter: Number(randomItemDensityJitter),
        item_respawn_ms: Number(itemRespawnMs),
        item_respawn_jitter_ratio: Number(itemRespawnJitterRatio),
        item_safe_radius: Number(itemSafeRadius),
        item_safe_radius_jitter: Number(itemSafeRadiusJitter),
        item_max: Number(itemMax),
        ml_enabled: String(mlEnabled) !== "0",
        ml_collect: String(mlCollect) !== "0",
        ml_freeze: String(mlFreeze) !== "0",
        ml_policy_mode: String(mlPolicyMode || "pure"),
        ml_iql_mix: String(mlIqlMix) !== "0",
        ml_model_url: String(mlModelUrl || ""),
        ml_conf: mlConf === "" ? null : Number(mlConf),
        ml_move_conf: mlMoveConf === "" ? null : Number(mlMoveConf),
        ml_margin: mlMargin === "" ? null : Number(mlMargin),
        ml_force_move_eta: mlForceMoveEta === "" ? null : Number(mlForceMoveEta),
        ml_wait_block_eta: mlWaitBlockEta === "" ? null : Number(mlWaitBlockEta),
        ml_move_threat_ms: mlMoveThreatMs === "" ? null : Number(mlMoveThreatMs),
        opponent_think_jitter_ratio: Number(opponentThinkJitterRatio),
        opponent_pool: String(opponentPool).split(",").map((s) => s.trim()).filter(Boolean),
        agent_pool: String(agentPool).split(",").map((s) => s.trim()).filter(Boolean),
        agent_expert_duel: agentExpertDuel !== "0",
        rows_written_by_workers: aggregateRowsWrittenByWorkers,
        early_committed_rows_by_workers: aggregateEarlyCommittedRows,
        mirror_rows_seen_by_workers: aggregateMirrorRowsSeen,
        mirror_episodes_committed_by_workers: aggregateMirrorEpisodesCommitted,
        micro_rows_seen_by_workers: aggregateMicroRowsSeen,
        streaming_written_rows_by_workers: aggregateStreamingWrittenRows,
        episode_buffer_written_rows_by_workers: aggregateEpisodeBufferWrittenRows,
        terminal_credit_applied_rows_by_workers: aggregateTerminalCreditAppliedRows,
        stall_kept_rows_by_workers: aggregateStallKeptRows,
        merge: {
            scanned_rows: mergeStats.scanned_rows,
            invalid_rows: mergeStats.invalid_rows,
            exact_duplicate_rows: mergeStats.exact_duplicate_rows,
            near_duplicate_rows: mergeStats.near_duplicate_rows,
            unique_rows_before_cap: mergeStats.unique_rows_before_cap,
            rows_written: mergeStats.rows_written,
            done_ratio: doneRatio,
            action_hist_written: mergeStats.action_hist_written,
            done_hist_written: mergeStats.done_hist_written,
            pre_death_hist_written: mergeStats.pre_death_hist_written,
            outcome_hist_written: mergeStats.outcome_hist_written,
            bucket_hist_before_cap: mergeStats.bucket_hist_before_cap,
            bucket_keep_target: mergeStats.bucket_keep_target,
            score_tier_hist_before_cap: mergeStats.score_tier_hist_before_cap,
            score_tier_keep_target: mergeStats.score_tier_keep_target,
            bucket_hist_written: mergeStats.bucket_hist_written,
            terminal_reason_hist_written: mergeStats.terminal_reason_hist_written,
            spawn_dist_hist_written: mergeStats.spawn_dist_hist_written,
            property_bucket_hist_written: mergeStats.property_bucket_hist_written,
            aux_label_hist_written: mergeStats.aux_label_hist_written,
            behavior_score_hist_written: mergeStats.behavior_score_hist_written,
            behavior_score_mean: mergeStats.behavior_score_mean,
            behavior_score_breakdown_hist_written: mergeStats.behavior_score_breakdown_hist_written,
            high_value_behavior_rows: mergeStats.high_value_behavior_rows,
            high_value_ratio: mergeStats.high_value_ratio,
            bucket_near_duplicate_rows: mergeStats.bucket_near_duplicate_rows,
        },
        worker_exit: workerExit,
        worker_reports: workerReports,
        duration_sec: (Date.now() - startedAt) / 1000,
        rows_per_sec: ((mergeStats.rows_written || 0) / Math.max(1, (Date.now() - startedAt) / 1000)),
    };

    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
    if (!collectionComplete) {
        console.error("[ERROR]", JSON.stringify({
            reason: "insufficient_rows_after_merge",
            rows_written: rowsWritten,
            min_required_rows: minRequiredRows,
            worker_failure_count: workerFailureCount,
            report_path: reportPath,
        }));
        process.exitCode = 2;
    }
    console.log("[DONE]", JSON.stringify({
        dataset_path: datasetPath,
        report_path: reportPath,
        rows_written: mergeStats.rows_written,
        collection_complete: collectionComplete,
        worker_failure_count: workerFailureCount,
        unique_rows_before_cap: mergeStats.unique_rows_before_cap,
        exact_duplicate_rows: mergeStats.exact_duplicate_rows,
        near_duplicate_rows: mergeStats.near_duplicate_rows,
        high_value_ratio: mergeStats.high_value_ratio,
        duration_sec: report.duration_sec,
    }));
}

main().catch((err) => {
    console.error(err);
    process.exit(1);
});
