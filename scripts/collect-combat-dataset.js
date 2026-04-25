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
const SAMPLE_BUCKETS = ["ongoing", "pre_death", "drop_bomb_safe", "drop_bomb_bad", "terminal"];
const TERMINAL_REASON_TAGS = new Set(["caught_enemy", "caught_self", "enemy_self_kill_discard", "stall_abort", "round_end"]);
const HIGH_VALUE_BUCKETS = new Set(["pre_death", "drop_bomb_safe", "drop_bomb_bad"]);
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
    "danger_cells_created",
    "kill_conversion",
    "self_kill_penalty",
];

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

function asNonNegativeInt(v, fallback) {
    const n = parseInt(v, 10);
    return Number.isFinite(n) && n >= 0 ? n : fallback;
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
    const dangerCellsCreatedScore = clamp01(Number(meta.dangerCellsCreatedScore || 0));
    const roundKillCredit = clamp01(Number(meta.roundKillCredit || 0));
    const roundSelfKillPenalty = clamp01(Number(meta.roundSelfKillPenalty || 0));
    const roundNetKdCredit = clamp01(Number(meta.roundNetKdCredit || 0));
    const dangerScoreWeight = clampRange(Number(meta.dangerScoreWeight || 1), 0.2, 3.0);
    const killScoreWeight = clampRange(Number(meta.killScoreWeight || 1), 0.2, 3.0);
    const selfKillPenaltyWeight = clampRange(Number(meta.selfKillPenaltyWeight || 1), 0.2, 4.0);
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
        breakdown.bomb_space_reduction = clampRange(
            0.20 + (1 - enemyEscapeOptions) * 0.16 + trapClosure * 0.16 + bombThreat * 0.18,
            0.20,
            0.55
        );
        if (Number(labels.bomb_self_trap_risk || 0) < 0.55) {
            breakdown.post_bomb_escape = clampRange(
                0.15 + Number(labels.bomb_escape_success_label || 0) * 0.10 + minEscapeEta * 0.10 + (1 - blastOverlap) * 0.05,
                0.15,
                0.40
            );
        }
    }
    if (action !== 5 && (blastOverlap >= 0.25 || nextBlastOverlap + 0.05 < blastOverlap)) {
        breakdown.obstacle_escape = clampRange(0.15 + Math.max(0, blastOverlap - nextBlastOverlap) * 0.24 + deadendDepth * 0.08, 0.15, 0.35);
    }
    if (action >= 1 && action <= 4 && (nextMinEscapeEta > minEscapeEta + 0.04 || nextBlastOverlap + 0.04 < blastOverlap)) {
        breakdown.move_to_safe = clampRange(
            0.10 + Math.max(0, nextMinEscapeEta - minEscapeEta) * 0.45 + Math.max(0, blastOverlap - nextBlastOverlap) * 0.30,
            0.10,
            0.30
        );
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
    if (dangerCellsCreatedScore > 0.001) {
        breakdown.danger_cells_created = clampRange(0.08 + dangerCellsCreatedScore * 0.30 * dangerScoreWeight, 0.08, 0.42);
    }
    if (roundKillCredit > 0.001 || roundNetKdCredit > 0.001) {
        breakdown.kill_conversion = clampRange(
            0.06 + (roundKillCredit * 0.20 + roundNetKdCredit * 0.20) * killScoreWeight,
            0.06,
            0.50
        );
    }
    if (roundSelfKillPenalty > 0.001) {
        breakdown.self_kill_penalty = clampRange(0.10 + roundSelfKillPenalty * 0.35 * selfKillPenaltyWeight, 0.10, 0.70);
    }

    let score = 0;
    for (const key of Object.keys(breakdown)) {
        score += Number(breakdown[key]) || 0;
    }
    score -= (Number(breakdown.self_kill_penalty || 0) * 0.55);
    score = score * 0.70;
    if (action === 5) {
        score = Math.max(score, 0.30);
    }
    if (preDeath || done) {
        score = Math.max(score, 0.22);
    }
    return {
        score: clamp01(score),
        breakdown,
    };
}

function buildAuxLabels(row) {
    const vec = getStateVector(row);
    const action = normalizeAction(row.action);
    const done = !!row.done;
    const preDeath = !!row.pre_death;
    const outcome = row.outcome_tag || "ongoing";
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
    const escapeSuccess = action === 5 && !preDeath && !done && bombSelfTrapRisk < 0.45 && postEscape <= 0.60 ? 1 : 0;
    const enemyTrap = action === 5 ? clamp01(0.65 * trapClosure + 0.35 * (1 - enemyEscapeOptions)) : 0;
    const terminalCredit = done || TERMINAL_OUTCOME_TAGS.has(outcome) || preDeath ? 1 : 0;
    const terminalReason = String(row.terminal_reason || row.outcome_tag || "ongoing").trim().toLowerCase();
    const reasonCode = terminalReason === "caught_enemy"
        ? 0.25
        : (terminalReason === "caught_self" ? 0.5 : (terminalReason === "enemy_self_kill_discard" ? 0.75 : (terminalReason === "stall_abort" ? 1.0 : 0.0)));
    const myBombThreatScore = clamp01(Number(row.meta && row.meta.myBombThreatScore) || enemyTrap);
    const closeRangeDuelScore = clamp01(Number(row.meta && row.meta.closeRangeDuelScore) || (1 - clamp01(Number(row.meta && row.meta.spawnShortestPathDistNorm) || 1)));
    const winningBombSourceRecent = clamp01(Number(row.meta && row.meta.winningBombSourceRecent) || 0);
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
        || ((row.outcome_tag === "self_kill" || row.pre_death) ? Math.max(0.35, bombSelfTrapRisk) : (action === 5 ? bombSelfTrapRisk * 0.6 : 0))
    );
    const roundNetKdCredit = clamp01(
        Number(row.meta && row.meta.roundNetKdCredit)
        || clamp01((roundKillCredit - roundSelfKillPenalty + 1) / 2)
    );

    const baseLabels = {
        bomb_escape_success_label: escapeSuccess,
        bomb_self_trap_risk: bombSelfTrapRisk,
        enemy_trap_after_bomb: enemyTrap,
        nearest_safe_tile_eta: minEscapeEta,
        commitment_depth: deadendDepth,
        terminal_credit_action: terminalCredit,
        terminal_reason_code: reasonCode,
        my_bomb_threat_score: myBombThreatScore,
        close_range_duel_score: closeRangeDuelScore,
        enemy_self_kill_episode: terminalReason === "enemy_self_kill_discard" ? 1 : 0,
        stall_abort_episode: terminalReason === "stall_abort" ? 1 : 0,
        winning_bomb_source_recent: winningBombSourceRecent,
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

function relabelBalancedActionIfUseful(row) {
    if (!row || row.done || row.pre_death || normalizeAction(row.action) === 5) {
        return false;
    }
    if (!Array.isArray(row.action_mask) || Number(row.action_mask[5] || 0) <= 0) {
        return false;
    }
    const vec = getStateVector(row);
    const minEscapeEta = clamp01(vec[17]);
    const deadendDepth = clamp01(vec[18]);
    const blastOverlap = clamp01(vec[19]);
    const enemyEscapeOptions = clamp01(vec[20]);
    const trapClosure = clamp01(vec[21]);
    const itemRaceDelta = Math.abs(clamp01(vec[22]) - 0.5) * 2;
    const meta = row.meta || {};
    const spawnDist = Number(meta.spawnShortestPathDist || 99);
    const spawnDistNorm = clamp01(Number(meta.spawnShortestPathDistNorm) || (Number.isFinite(spawnDist) ? (spawnDist / 28) : 1));
    const closeRangeScore = clamp01(Number(meta.closeRangeDuelScore) || Math.max(0, 1 - spawnDistNorm));
    const threatScore = clamp01(Number(meta.myBombThreatScore) || 0);
    const legalBombBias = Number(meta.sample_bucket === "drop_bomb_safe" || meta.sample_bucket === "drop_bomb_bad" ? 1 : 0);

    const highTrapValue = trapClosure >= 0.28 || enemyEscapeOptions <= 0.58 || threatScore >= 0.16;
    const highRiskValue = deadendDepth >= 0.48 || blastOverlap >= 0.48 || minEscapeEta <= 0.30;
    const raceValue = itemRaceDelta >= 0.38;
    const closeRangeValue = closeRangeScore >= 0.45 || spawnDist <= 6;
    let relabelProb = 0.14;
    relabelProb += closeRangeValue ? 0.30 : 0;
    relabelProb += highTrapValue ? 0.38 : 0;
    relabelProb += highRiskValue ? 0.14 : 0;
    relabelProb += raceValue ? 0.12 : 0;
    relabelProb += threatScore * 0.25;
    relabelProb += legalBombBias ? 0.08 : 0;
    relabelProb = Math.max(0.18, Math.min(0.97, relabelProb));
    if (Math.random() > relabelProb) {
        return false;
    }
    row.meta = row.meta || {};
    row.meta.original_action = normalizeAction(row.action);
    row.meta.balanced_relabel_action5 = 1;
    row.meta.balanced_relabel_prob = relabelProb;
    row.action = 5;
    return true;
}

function classifySampleBucket(row, auxLabels) {
    const action = normalizeAction(row.action);
    const outcome = row.outcome_tag || "ongoing";
    const labels = auxLabels || buildAuxLabels(row);
    if (row.done || TERMINAL_OUTCOME_TAGS.has(outcome)) {
        return "terminal";
    }
    if (action === 5 && Number(labels.bomb_self_trap_risk || 0) >= 0.55) {
        return "drop_bomb_bad";
    }
    if (action === 5) {
        return "drop_bomb_safe";
    }
    if (row.pre_death || Number(labels.bomb_self_trap_risk || 0) >= 0.55 || Number(labels.commitment_depth || 0) >= 0.70) {
        return "pre_death";
    }
    return "ongoing";
}

function enrichBalancedSample(row) {
    const auxLabels = Object.assign({}, row.aux_labels || {}, buildAuxLabels(row));
    const bucket = SAMPLE_BUCKETS.includes(row.sample_bucket)
        ? row.sample_bucket
        : classifySampleBucket(row, auxLabels);
    row.aux_labels = auxLabels;
    row.sample_bucket = bucket;
    row.meta = row.meta || {};
    row.meta.sample_bucket = bucket;
    return row;
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

    if (!Array.isArray(stateVec) || stateVec.length < 32) {
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

function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
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
                await sleep(400 * attempt);
            }
        }
    }
    throw lastErr || new Error("playwright_launch_failed");
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
    const balanced = getArg("balanced", "0") === "1";
    const scenarioBuckets = getArg("scenario-buckets", "open_random,escape_after_bomb,enemy_choke,item_race,deadend_chase");
    const softObstacleReinjectRatio = Math.max(0, Math.min(0.35, asFloat(getArg("soft-obstacle-reinject-ratio", "0.08"), 0.08)));
    const spawnRandomize = getArg("spawn-randomize", "0") === "1";
    const terminalTailMs = asPositiveInt(getArg("terminal-tail-ms", "3000"), 3000);
    const suddenDeath = getArg("sudden-death", "1") !== "0";
    const disableRevive = getArg("disable-revive", suddenDeath ? "1" : "0") !== "0";
    const ignoreEnemySelfKill = getArg("ignore-enemy-self-kill", "1") !== "0";
    const stallNoProgressMs = asPositiveInt(getArg("stall-no-progress-ms", "12000"), 12000);
    const maxEpisodeMs = asPositiveInt(getArg("max-episode-ms", suddenDeath ? "18000" : "0"), suddenDeath ? 18000 : 0);
    const mirrorSampling = getArg("mirror-sampling", balanced ? "1" : "0") !== "0";
    const earlyCommitHighValue = getArg("early-commit-high-value", balanced ? "1" : "0") !== "0";
    const behaviorScoring = getArg("behavior-scoring", balanced ? "1" : "0") !== "0";
    const behaviorScoreThreshold = Math.max(0, Math.min(1, asFloat(getArg("behavior-score-threshold", "0.12"), 0.12)));
    const highValueBehaviorThreshold = Math.max(behaviorScoreThreshold, Math.min(1, asFloat(getArg("high-value-behavior-threshold", "0.35"), 0.35)));
    const behaviorCreditWindowMs = Math.max(1000, asPositiveInt(getArg("behavior-credit-window-ms", "6000"), 6000));
    const microScenario = getArg("micro-scenario", balanced ? "1" : "0") !== "0";
    const microSamplesPerEpisode = Math.max(0, asNonNegativeInt(getArg("micro-samples-per-episode", balanced ? "16" : "0"), balanced ? 16 : 0));
    const partialClearMinRatio = Math.max(0, Math.min(1, asFloat(getArg("partial-clear-min-ratio", "0.35"), 0.35)));
    const partialClearMaxRatio = Math.max(partialClearMinRatio, Math.min(1, asFloat(getArg("partial-clear-max-ratio", "0.75"), 0.75)));
    const softObstacleKeepMin = Math.max(0, asInt(getArg("soft-obstacle-keep-min", "0"), 0));
    const softObstacleKeepMax = Math.max(softObstacleKeepMin, asInt(getArg("soft-obstacle-keep-max", "20"), 20));
    const spawnShortestPathMin = Math.max(0, asPositiveInt(getArg("spawn-shortest-path-min", "1"), 1));
    const spawnShortestPathMax = Math.max(spawnShortestPathMin, asPositiveInt(getArg("spawn-shortest-path-max", "10"), 10));
    const burstCapOngoing = Math.max(1, asPositiveInt(getArg("burst-cap-ongoing", "6"), 6));
    const specialBombEscape = getArg("special-bomb-escape", "0") !== "0";
    const specialRoundSec = asPositiveInt(getArg("special-round-sec", "60"), 60);
    const specialItemRespawnMs = asPositiveInt(getArg("special-item-respawn-ms", "2000"), 2000);
    const specialRespawnDelayMs = asNonNegativeInt(getArg("respawn-delay-ms", "0"), 0);
    const specialRespawnInvincibleMs = asNonNegativeInt(getArg("respawn-invincible-ms", "300"), 300);
    const noGlobalDedupe = getArg("no-global-dedupe", "1") !== "0";
    const dangerScoreWeight = Math.max(0.2, Math.min(3.0, asFloat(getArg("danger-score-weight", "1.0"), 1.0)));
    const killScoreWeight = Math.max(0.2, Math.min(3.0, asFloat(getArg("kill-score-weight", "1.0"), 1.0)));
    const selfKillPenaltyWeight = Math.max(0.2, Math.min(4.0, asFloat(getArg("self-kill-penalty-weight", "2.0"), 2.0)));

    const clearNonRigid = getArg("clear-nonrigid", "1") !== "0";
    const randomItemDensity = Math.max(0, Math.min(0.4, asFloat(getArg("random-item-density", "0.12"), 0.12)));
    const randomItemDensityJitter = Math.max(0, Math.min(0.2, asFloat(getArg("random-item-density-jitter", "0.04"), 0.04)));
    const itemRespawnMs = asPositiveInt(getArg("item-respawn-ms", "1200"), 1200);
    const itemRespawnJitterRatio = Math.max(0, Math.min(0.8, asFloat(getArg("item-respawn-jitter-ratio", "0.3"), 0.3)));
    const itemSafeRadius = Math.max(0, Math.min(8, asPositiveInt(getArg("item-safe-radius", "2"), 2)));
    const itemSafeRadiusJitter = Math.max(0, Math.min(3, asFloat(getArg("item-safe-radius-jitter", "1"), 1)));
    const itemMax = Math.max(0, asPositiveInt(getArg("item-max", "22"), 22));
    const opponentThinkJitterRatio = Math.max(0, Math.min(0.8, asFloat(getArg("opponent-think-jitter-ratio", "0.3"), 0.3)));

    const runtimeMatchDurationSec = specialBombEscape ? specialRoundSec : matchDurationSec;
    const runtimeSuddenDeath = specialBombEscape ? false : suddenDeath;
    const runtimeDisableRevive = specialBombEscape ? false : disableRevive;
    const runtimeIgnoreEnemySelfKill = specialBombEscape ? false : ignoreEnemySelfKill;
    const runtimeClearNonRigid = specialBombEscape ? true : clearNonRigid;
    const runtimeRandomItemDensityJitter = specialBombEscape ? 0 : randomItemDensityJitter;
    const runtimeItemRespawnMs = specialBombEscape ? specialItemRespawnMs : itemRespawnMs;
    const runtimeItemRespawnJitterRatio = specialBombEscape ? 0 : itemRespawnJitterRatio;
    const runtimeSoftObstacleKeepMin = specialBombEscape ? 0 : softObstacleKeepMin;
    const runtimeSoftObstacleKeepMax = specialBombEscape ? 0 : softObstacleKeepMax;

    const arena = getArg("arena", "1v1");
    const actionSpace = getArg("action-space", "discrete6");
    if (arena !== "1v1") {
        throw new Error("only --arena=1v1 is supported for now");
    }
    if (actionSpace !== "discrete6") {
        throw new Error("only --action-space=discrete6 is supported for now");
    }

    const opponentPoolDefault = "heuristic_v1,heuristic_v2,aggressive_trapper,coward_runner,item_rusher,randomized_mistake_bot";
    const opponentPool = getArg("opponent-pool", getArg("opponents", opponentPoolDefault))
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean);
    if (opponentPool.length === 0) {
        opponentPool.push("heuristic_v1");
    }
    const agentPoolDefault = "heuristic_v2,aggressive_trapper";
    const agentPool = getArg("agent-pool", agentPoolDefault)
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean);
    if (agentPool.length === 0) {
        agentPool.push("heuristic_v2");
    }
    const agentExpertDuel = getArg("agent-expert-duel", "1") !== "0";

    const datasetPath = path.resolve(getArg("dataset-path", DATASET_PATH_DEFAULT));
    const reportPath = path.resolve(getArg("report-path", REPORT_PATH_DEFAULT));
    const screenshotPath = path.resolve(getArg("screenshot-path", SCREENSHOT_PATH_DEFAULT));
    const screenshotEnabled = getArg("screenshot", "1") !== "0";

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
    let burstCapDropped = 0;
    let repairedIllegalActionRows = 0;
    let uniqueTransitions = 0;
    const episodeBuffers = new Map();
    const episodeBurstState = new Map();
    let validEpisodeCount = 0;
    let discardedEpisodeCount = 0;
    const actionHist = { "0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0 };
    const doneHist = { "0": 0, "1": 0 };
    const preDeathHist = { "0": 0, "1": 0 };
    const outcomeHist = { ongoing: 0, win: 0, loss: 0, draw: 0, self_kill: 0 };
    const bucketHist = { ongoing: 0, pre_death: 0, drop_bomb_safe: 0, drop_bomb_bad: 0, terminal: 0 };
    const terminalReasonHist = { caught_enemy: 0, caught_self: 0, enemy_self_kill_discard: 0, stall_abort: 0, round_end: 0 };
    const spawnDistHist = { "1_3": 0, "4_6": 0, "7_10": 0, other: 0 };
    const propertyBucketHist = { low: 0, medium: 0, high: 0 };
    const auxLabelHist = {
        bomb_escape_success_label: 0,
        bomb_self_trap_risk_high: 0,
        enemy_trap_after_bomb_high: 0,
        terminal_credit_action: 0,
        danger_cells_created_score_nonzero: 0,
        round_kill_credit_nonzero: 0,
        round_self_kill_penalty_nonzero: 0,
    };
    const behaviorScoreHist = { zero: 0, low: 0, mid: 0, high: 0, very_high: 0 };
    const behaviorBreakdownHist = BEHAVIOR_BREAKDOWN_KEYS.reduce((acc, key) => {
        acc[key] = 0;
        return acc;
    }, {});
    const dropReasons = {};
    const terminalOutcomeEvents = { win: 0, loss: 0, draw: 0, self_kill: 0 };
    let lastRuntimeState = null;
    let terminalFinalizeCalls = 0;
    let terminalCollectorFinalizeOk = 0;
    let terminalCollectorFinalizeMiss = 0;
    let earlyCommittedRows = 0;
    let streamingWrittenRows = 0;
    let episodeBufferWrittenRows = 0;
    let terminalCreditAppliedRows = 0;
    let stallKeptRows = 0;
    let highValueBehaviorRows = 0;
    let behaviorScoreSum = 0;
    let mirrorRowsSeen = 0;
    let mirrorEpisodesCommitted = 0;
    let microRowsSeen = 0;
    const startedAt = Date.now();

    function markDrop(reason) {
        dropReasons[reason] = (dropReasons[reason] || 0) + 1;
    }

    function updateBehaviorStats(raw) {
        const aux = raw.aux_labels || {};
        const score = clamp01(aux.behavior_score);
        behaviorScoreSum += score;
        behaviorScoreHist[behaviorScoreBand(score)] = (behaviorScoreHist[behaviorScoreBand(score)] || 0) + 1;
        if (score >= highValueBehaviorThreshold) {
            highValueBehaviorRows += 1;
        }
        const breakdown = aux.behavior_score_breakdown || {};
        for (const key of BEHAVIOR_BREAKDOWN_KEYS) {
            if (Number(breakdown[key] || 0) > 0) {
                behaviorBreakdownHist[key] = (behaviorBreakdownHist[key] || 0) + 1;
            }
        }
    }

    function writeCommittedRow(raw, source) {
        if (wrote >= targetFrames) {
            return;
        }
        raw.meta = raw.meta || {};
        if (source) {
            raw.meta.write_source = source;
            if (source === "streaming") streamingWrittenRows += 1;
            else if (source === "terminal_credit") terminalCreditAppliedRows += 1;
            else if (source === "episode_buffer") episodeBufferWrittenRows += 1;
            if (source === "stall_kept") stallKeptRows += 1;
        }
        stream.write(JSON.stringify(raw) + "\n");
        wrote += 1;
        uniqueTransitions += 1;

        const action = normalizeAction(raw.action);
        actionHist[String(action)] = (actionHist[String(action)] || 0) + 1;
        doneHist[raw.done ? "1" : "0"] += 1;
        preDeathHist[raw.pre_death ? "1" : "0"] += 1;
        outcomeHist[raw.outcome_tag] = (outcomeHist[raw.outcome_tag] || 0) + 1;
        bucketHist[raw.sample_bucket] = (bucketHist[raw.sample_bucket] || 0) + 1;
        const aux = raw.aux_labels || {};
        if (Number(aux.bomb_escape_success_label || 0) > 0.5) auxLabelHist.bomb_escape_success_label += 1;
        if (Number(aux.bomb_self_trap_risk || 0) >= 0.55) auxLabelHist.bomb_self_trap_risk_high += 1;
        if (Number(aux.enemy_trap_after_bomb || 0) >= 0.55) auxLabelHist.enemy_trap_after_bomb_high += 1;
        if (Number(aux.terminal_credit_action || 0) > 0.5) auxLabelHist.terminal_credit_action += 1;
        if (Number(aux.danger_cells_created_score || 0) > 0) auxLabelHist.danger_cells_created_score_nonzero += 1;
        if (Number(aux.round_kill_credit || 0) > 0) auxLabelHist.round_kill_credit_nonzero += 1;
        if (Number(aux.round_self_kill_penalty || 0) > 0) auxLabelHist.round_self_kill_penalty_nonzero += 1;
        updateBehaviorStats(raw);
    }

    function pushEpisodeBuffer(raw) {
        const episodeId = String(raw.episode_id || "runtime");
        const list = episodeBuffers.get(episodeId) || [];
        const now = Number(raw.meta && raw.meta.collect_ts) || Date.now();
        list.push(raw);
        while (list.length > 0) {
            const ts = Number(list[0].meta && list[0].meta.collect_ts) || now;
            if (now - ts <= behaviorCreditWindowMs && list.length <= 256) break;
            list.shift();
        }
        episodeBuffers.set(episodeId, list);
    }

    function shouldStreamByBehavior(raw) {
        if (!behaviorScoring) return false;
        const aux = raw.aux_labels || {};
        const score = clamp01(aux.behavior_score);
        if (score >= behaviorScoreThreshold) return true;
        if (raw.done || raw.pre_death) return true;
        return false;
    }

    function applyTerminalCredit(row, meta) {
        row.meta = row.meta || {};
        row.aux_labels = row.aux_labels || {};
        const rowTs = Number(row.meta.collect_ts || 0);
        const terminalTs = Number(meta.ts || 0);
        const ageMs = rowTs > 0 && terminalTs > 0 ? Math.max(0, terminalTs - rowTs) : behaviorCreditWindowMs;
        const outcome = String(meta.outcome_tag || row.outcome_tag || "ongoing");
        const reason = String(meta.terminal_reason || row.terminal_reason || "");
        let multiplier = 1.0;
        if (outcome === "win") {
            if (ageMs <= 1000) multiplier = 1.8;
            else if (ageMs <= 3000) multiplier = 1.4;
            else if (ageMs <= behaviorCreditWindowMs) multiplier = 1.15;
        } else if (outcome === "self_kill" || reason === "caught_self") {
            if (ageMs <= 3000) multiplier = 1.25;
        }
        if (multiplier > 1.0) {
            const oldScore = clamp01(row.aux_labels.behavior_score);
            row.aux_labels.behavior_score = clamp01(oldScore * multiplier);
            row.aux_labels.behavior_score_band = behaviorScoreBand(row.aux_labels.behavior_score);
            row.aux_labels.terminal_credit_action = 1;
            row.meta.terminal_credit_multiplier = multiplier;
            row.meta.terminal_credit_age_ms = ageMs;
            if (outcome === "win") {
                row.reward = Math.max(Number(row.reward || 0), 0.05 + row.aux_labels.behavior_score * 0.25);
            } else if (outcome === "self_kill" || reason === "caught_self") {
                row.reward = Math.min(Number(row.reward || 0), -0.25 - clamp01(row.aux_labels.bomb_self_trap_risk) * 0.45);
            }
        }
        return multiplier > 1.0;
    }

    function bufferEpisodeRow(rawInput) {
        const raw = canonicalizeSample(rawInput);
        if (!raw) {
            dropped += 1;
            markDrop("row_not_object");
            return;
        }
        if (balanced) {
            relabelBalancedActionIfUseful(raw);
        }
        enrichBalancedSample(raw);
        raw.meta = raw.meta || {};
        raw.meta.collect_ts = Number(raw.meta.collect_ts || Date.now());

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

        if (shouldStreamByBehavior(raw)) {
            raw.meta.streaming_behavior_score = clamp01(raw.aux_labels && raw.aux_labels.behavior_score);
            writeCommittedRow(raw, "streaming");
            pushEpisodeBuffer(Object.assign({}, raw, {
                meta: Object.assign({}, raw.meta, { terminal_credit_shadow: 1 }),
                aux_labels: Object.assign({}, raw.aux_labels),
            }));
            return;
        }

        if (earlyCommitHighValue && shouldEarlyCommitRow(raw)) {
            raw.meta = raw.meta || {};
            raw.meta.early_commit = 1;
            earlyCommittedRows += 1;
            writeCommittedRow(raw, "streaming");
            pushEpisodeBuffer(Object.assign({}, raw, {
                meta: Object.assign({}, raw.meta, { terminal_credit_shadow: 1 }),
                aux_labels: Object.assign({}, raw.aux_labels),
            }));
            return;
        }

        pushEpisodeBuffer(raw);
    }

    function shouldEarlyCommitRow(raw) {
        if (!raw || raw.done) {
            return false;
        }
        const action = normalizeAction(raw.action);
        const bucket = String(raw.sample_bucket || "");
        const meta = raw.meta || {};
        if (HIGH_VALUE_BUCKETS.has(bucket)) {
            return true;
        }
        if (action === 5) {
            return true;
        }
        if (Number(meta.myBombThreatScore || 0) >= 0.18) {
            return true;
        }
        if (Number(meta.closeRangeDuelScore || 0) >= 0.55 && action !== 0) {
            return true;
        }
        if (["escape_after_bomb", "enemy_choke", "item_race", "deadend_chase"].includes(String(meta.scenarioName || "")) && action !== 0) {
            return true;
        }
        return false;
    }

    function commitEpisodeRows(episodeId, terminalMeta) {
        const key = String(episodeId || "runtime");
        const rows = episodeBuffers.get(key) || [];
        episodeBuffers.delete(key);
        episodeBurstState.delete(key);
        if (!rows.length) {
            return;
        }

        const meta = terminalMeta || {};
        const terminalReason = String(meta.terminal_reason || "").trim().toLowerCase();
        const discardEpisode = !!meta.discard_episode;
        if (TERMINAL_REASON_TAGS.has(terminalReason)) {
            terminalReasonHist[terminalReason] = (terminalReasonHist[terminalReason] || 0) + 1;
        }
        const spawnDist = Number(meta.spawn_shortest_path_dist);
        if (Number.isFinite(spawnDist)) {
            if (spawnDist >= 1 && spawnDist <= 3) spawnDistHist["1_3"] += 1;
            else if (spawnDist >= 4 && spawnDist <= 6) spawnDistHist["4_6"] += 1;
            else if (spawnDist >= 7 && spawnDist <= 10) spawnDistHist["7_10"] += 1;
            else spawnDistHist.other += 1;
        }
        const propScore = Number(meta.property_bucket_score || 0);
        if (propScore < 0.34) propertyBucketHist.low += 1;
        else if (propScore < 0.67) propertyBucketHist.medium += 1;
        else propertyBucketHist.high += 1;

        if (discardEpisode) {
            discardedEpisodeCount += 1;
            markDrop("discard_episode_" + (terminalReason || "unknown"));
        }

        if (!discardEpisode) {
            validEpisodeCount += 1;
        }
        if (String(meta.perspective || "") === "opponent") {
            mirrorEpisodesCommitted += 1;
        }
        let burst = episodeBurstState.get(key) || { sig: "", count: 0 };
        for (const row of rows) {
            if (wrote >= targetFrames) break;
            const sig = buildTransitionSignature(row);
            const limit = row.done ? Number.MAX_SAFE_INTEGER : burstCapOngoing;
            if (burst.sig === sig) {
                burst.count += 1;
            } else {
                burst.sig = sig;
                burst.count = 1;
            }
            const credited = applyTerminalCredit(row, meta);
            const behaviorScore = clamp01(row.aux_labels && row.aux_labels.behavior_score);
            const retainByScore = !behaviorScoring || behaviorScore >= behaviorScoreThreshold || row.done || row.pre_death;
            if (!retainByScore) {
                dropped += 1;
                markDrop("low_behavior_score");
                continue;
            }
            if (burst.count > limit) {
                burstCapDropped += 1;
                dropped += 1;
                markDrop("burst_cap");
                continue;
            }
            const source = terminalReason === "stall_abort" ? "stall_kept" : (credited ? "terminal_credit" : "episode_buffer");
            writeCommittedRow(row, source);
        }
        episodeBurstState.set(key, burst);
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
            function clamp01(v) {
                const n = Number(v);
                if (!isFinite(n)) return 0;
                return Math.max(0, Math.min(1, n));
            }

            window.__combatCollect = {
                agentPool: cfg.agentPool,
                agentExpertDuel: !!cfg.agentExpertDuel,
                opponentPool: cfg.opponentPool,
                mapId: cfg.mapId,
                thinkMs: cfg.thinkMs,
                matchDurationSec: cfg.matchDurationSec,
                matchIndex: 0,
                playerCtrl: null,
                itemRespawnTicker: null,
                aiRoleNumber: null,
                lastStartAt: 0,
                currentAgentExpert: cfg.agentPool[0] || "heuristic_v2",
                currentOpponent: cfg.opponentPool[0] || "heuristic_v1",
                matchFinalized: false,
                clearNonRigid: !!cfg.clearNonRigid,
                randomItemDensity: cfg.randomItemDensity,
                randomItemDensityJitter: cfg.randomItemDensityJitter,
                itemRespawnMs: cfg.itemRespawnMs,
                itemRespawnJitterRatio: cfg.itemRespawnJitterRatio,
                itemSafeRadius: cfg.itemSafeRadius,
                itemSafeRadiusJitter: cfg.itemSafeRadiusJitter,
                itemMax: cfg.itemMax,
                opponentThinkJitterRatio: cfg.opponentThinkJitterRatio,
                balanced: !!cfg.balanced,
                scenarioBuckets: String(cfg.scenarioBuckets || "")
                    .split(",")
                    .map(function(s) { return s.trim(); })
                    .filter(Boolean),
                softObstacleReinjectRatio: Number(cfg.softObstacleReinjectRatio || 0),
                spawnRandomize: !!cfg.spawnRandomize,
                terminalTailMs: Number(cfg.terminalTailMs || 3000),
                suddenDeath: !!cfg.suddenDeath,
                disableRevive: !!cfg.disableRevive,
                ignoreEnemySelfKill: !!cfg.ignoreEnemySelfKill,
                stallNoProgressMs: Number(cfg.stallNoProgressMs || 12000),
                maxEpisodeMs: Number(cfg.maxEpisodeMs || 0),
                mirrorSampling: !!cfg.mirrorSampling,
                earlyCommitHighValue: !!cfg.earlyCommitHighValue,
                microScenario: !!cfg.microScenario,
                microSamplesPerEpisode: Number(cfg.microSamplesPerEpisode || 0),
                partialClearMinRatio: Number(cfg.partialClearMinRatio || 0.35),
                partialClearMaxRatio: Number(cfg.partialClearMaxRatio || 0.75),
                softObstacleKeepMin: Math.max(0, Number(cfg.softObstacleKeepMin || 0)),
                softObstacleKeepMax: Math.max(0, Number(cfg.softObstacleKeepMax || 20)),
                spawnShortestPathMin: Number(cfg.spawnShortestPathMin || 1),
                spawnShortestPathMax: Number(cfg.spawnShortestPathMax || 10),
                specialBombEscape: !!cfg.specialBombEscape,
                specialRoundSec: Number(cfg.specialRoundSec || 60),
                specialItemRespawnMs: Number(cfg.specialItemRespawnMs || 2000),
                specialRespawnDelayMs: Number(cfg.specialRespawnDelayMs || 0),
                specialRespawnInvincibleMs: Number(cfg.specialRespawnInvincibleMs || 300),
                dangerScoreWeight: Number(cfg.dangerScoreWeight || 1),
                killScoreWeight: Number(cfg.killScoreWeight || 1),
                selfKillPenaltyWeight: Number(cfg.selfKillPenaltyWeight || 2),
                noGlobalDedupe: !!cfg.noGlobalDedupe,
                currentDensity: cfg.randomItemDensity,
                currentItemRespawnMs: cfg.itemRespawnMs,
                currentItemSafeRadius: cfg.itemSafeRadius,
                currentOpponentThinkMs: 95,
                lastTerminal: null,
                terminalEvents: 0,
                mapRows: 13,
                mapCols: 15,
                currentScenario: "open_random",
                scenarioHist: {},
                originalNonRigidCells: null,
                currentEpisodeMeta: null,
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
                if (pool.length === 1) {
                    return pool[0];
                }
                const idx = Math.floor(rng() * pool.length);
                return pool[idx] || "heuristic_v1";
            }

            function pickAgentMode(matchIndex) {
                const pool = window.__combatCollect.agentPool;
                if (!pool || pool.length === 0) {
                    return "heuristic_v2";
                }
                if (pool.length === 1) {
                    return pool[0];
                }
                const idx = Math.floor(rng() * pool.length);
                return pool[idx] || "heuristic_v2";
            }

            function pickScenario() {
                const pool = window.__combatCollect.scenarioBuckets || ["open_random"];
                const idx = Math.max(0, (Number(window.__combatCollect.matchIndex || 1) - 1) % Math.max(1, pool.length));
                const scenario = pool[idx] || "open_random";
                window.__combatCollect.currentScenario = scenario;
                window.__combatCollect.scenarioHist[scenario] = (window.__combatCollect.scenarioHist[scenario] || 0) + 1;
                return scenario;
            }

            function resetMirrorCollector() {
                window.__combatMirrorCollect = {
                    rowsReady: [],
                    stagingRows: [],
                    lastOpenSample: null,
                    sampleIdSeed: 0,
                    preDeathWindowMs: 1500
                };
            }

            function mirrorNormalizeOutcome(outcomeTag, preDeath, done) {
                const tag = String(outcomeTag || "").toLowerCase();
                if (tag === "win" || tag === "loss" || tag === "draw" || tag === "self_kill") {
                    return tag;
                }
                if (!done) return "ongoing";
                return preDeath ? "self_kill" : "draw";
            }

            function mirrorFlushStaging(forceAll) {
                const store = window.__combatMirrorCollect;
                if (!store) return;
                const now = Date.now();
                const keep = [];
                const limitTs = now - Number(store.preDeathWindowMs || 1500);
                for (let i = 0; i < store.stagingRows.length; i++) {
                    const row = store.stagingRows[i];
                    if (forceAll || row.done || row.ts <= limitTs) {
                        store.rowsReady.push(row);
                    } else {
                        keep.push(row);
                    }
                }
                store.stagingRows = keep;
            }

            function mirrorInferRiskLabel(sample, done) {
                if (done || (sample && sample.pre_death)) return 1;
                if (window.BNBMLDatasetCollector && typeof window.BNBMLDatasetCollector.InferRiskLabel === "function") {
                    return window.BNBMLDatasetCollector.InferRiskLabel(sample, !!done);
                }
                return 0;
            }

            function mirrorFinalizeOpenSample(nextState, reward, done) {
                const store = window.__combatMirrorCollect;
                if (!store || !store.lastOpenSample) return;
                const sample = store.lastOpenSample;
                sample.next_state = nextState || sample.state || null;
                sample.done = !!done;
                sample.reward = typeof reward === "number" ? reward : (sample.done ? -1.5 : 0.03);
                sample.risk_label = mirrorInferRiskLabel(sample, sample.done);
                sample.outcome_tag = mirrorNormalizeOutcome(sample.outcome_tag, sample.pre_death, sample.done);
                store.stagingRows.push(sample);
                store.lastOpenSample = null;
            }

            function mirrorRecordRoleFrame(role, actionOverride, policyTag, episodeId) {
                const store = window.__combatMirrorCollect;
                if (!window.__combatCollect.mirrorSampling || !store || !role || role.IsDeath || role.IsInPaopao || typeof role.CurrentMapID !== "function") {
                    return;
                }
                if (!window.BNBMLFeatureEncoder || !window.BNBMLDatasetCollector) {
                    return;
                }
                const currentMap = role.CurrentMapID();
                const snapshot = typeof window.BuildThreatSnapshot === "function" ? window.BuildThreatSnapshot() : null;
                const fakeMonster = { Role: role };
                const encoded = window.BNBMLFeatureEncoder.Encode(role, currentMap, snapshot);
                const actionMask = window.BNBMLDatasetCollector.BuildActionMask(fakeMonster, currentMap, snapshot, 6);
                let action = typeof actionOverride === "number" ? actionOverride : 0;
                action = typeof window.NormalizeActionId === "function" ? window.NormalizeActionId(action) : Math.max(0, Math.min(5, parseInt(action, 10) || 0));
                if (action < 0 || action >= actionMask.length || Number(actionMask[action] || 0) <= 0) {
                    action = 0;
                }
                if (store.lastOpenSample) {
                    mirrorFinalizeOpenSample(encoded, null, false);
                }
                const sample = {
                    id: "M" + (++store.sampleIdSeed),
                    ts: Date.now(),
                    state: {
                        state_map: encoded.state_map,
                        state_vector: encoded.state_vector
                    },
                    action,
                    action_mask: actionMask,
                    reward: 0,
                    done: false,
                    next_state: null,
                    pre_death: false,
                    risk_label: 0,
                    policy_tag: policyTag || "opponent_expert",
                    episode_id: episodeId || ("m" + window.__combatCollect.matchIndex + "_opponent"),
                    agent_id: "role_" + role.RoleNumber,
                    opponent_id: "role_unknown",
                    outcome_tag: "ongoing",
                    meta: window.BNBMLDatasetCollector.BuildSampleMeta(fakeMonster, currentMap, snapshot, null)
                };
                sample.meta.action_source = policyTag || "opponent_expert";
                sample.meta.perspective = "opponent";
                sample.meta.mirror_sample = 1;
                sample.meta.scenarioName = window.__combatCollect.currentScenario || "";
                store.lastOpenSample = sample;
                mirrorFlushStaging(false);
            }

            function mirrorFinalizeEpisode(outcomeTag, opts) {
                const store = window.__combatMirrorCollect;
                if (!store) return false;
                const finalOpts = opts || {};
                const done = finalOpts.done !== false;
                const preDeath = !!finalOpts.preDeath;
                const reward = typeof finalOpts.reward === "number" ? finalOpts.reward : null;
                const normalized = mirrorNormalizeOutcome(outcomeTag, preDeath, done);
                if (store.lastOpenSample) {
                    store.lastOpenSample.pre_death = preDeath || !!store.lastOpenSample.pre_death;
                    store.lastOpenSample.outcome_tag = normalized;
                    mirrorFinalizeOpenSample(store.lastOpenSample.state || null, reward, done);
                }
                const cutoff = Date.now() - Number(store.preDeathWindowMs || 1500);
                for (let i = store.stagingRows.length - 1; i >= 0; i--) {
                    const row = store.stagingRows[i];
                    if (row.ts < cutoff) break;
                    row.done = done;
                    row.pre_death = preDeath || !!row.pre_death;
                    row.outcome_tag = normalized;
                    if (typeof reward === "number") row.reward = reward;
                    row.risk_label = mirrorInferRiskLabel(row, done);
                }
                mirrorFlushStaging(true);
                return true;
            }

            function mirrorDrain(maxRows, forceAll) {
                const store = window.__combatMirrorCollect;
                if (!store) return [];
                if (forceAll && store.lastOpenSample) {
                    mirrorFinalizeOpenSample(store.lastOpenSample.state || null, null, false);
                }
                mirrorFlushStaging(!!forceAll);
                return store.rowsReady.splice(0, Math.max(1, parseInt(maxRows, 10) || 2048));
            }

            resetMirrorCollector();
            window.__combatMirrorRecordRoleFrame = mirrorRecordRoleFrame;
            window.__combatMirrorFinalizeEpisode = mirrorFinalizeEpisode;
            window.__combatMirrorDrain = mirrorDrain;

            function emitMicroScenarioSamples(scenario) {
                if (!window.__combatCollect.microScenario || !window.__combatCollect.microSamplesPerEpisode) {
                    return 0;
                }
                if (!window.BNBMLFeatureEncoder || !window.BNBMLDatasetCollector) {
                    return 0;
                }
                const roles = findPlayerAndAiRoles();
                const roleItems = [
                    { role: roles.ai, perspective: "agent", policy: "micro_agent_" + String(window.__combatCollect.currentAgentExpert || "heuristic_v2"), episode: "m" + window.__combatCollect.matchIndex + "_agent" },
                    { role: roles.player, perspective: "opponent", policy: "micro_opponent_" + String(window.__combatCollect.currentOpponent || "heuristic_v2"), episode: "m" + window.__combatCollect.matchIndex + "_opponent" }
                ].filter(function(item) {
                    return item.role && !item.role.IsDeath && typeof item.role.CurrentMapID === "function";
                });
                if (!roleItems.length) {
                    return 0;
                }
                window.__combatMicroRows = window.__combatMicroRows || [];
                const perEpisode = Math.max(0, Number(window.__combatCollect.microSamplesPerEpisode || 0));
                const dirs = [[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0], [0, -2], [0, 2], [-2, 0], [2, 0]];
                let emitted = 0;
                for (let i = 0; i < perEpisode; i++) {
                    const item = roleItems[i % roleItems.length];
                    const role = item.role;
                    const original = role.CurrentMapID();
                    if (!original) continue;
                    const d = dirs[Math.floor(rng() * dirs.length)] || dirs[0];
                    const tx = Math.max(0, Math.min(14, original.X + d[0]));
                    const ty = Math.max(0, Math.min(12, original.Y + d[1]));
                    if (!isWalkableEmpty(tx, ty) || isOccupiedByRole(tx, ty)) {
                        continue;
                    }
                    if (typeof role.SetToMap === "function") {
                        role.SetToMap(tx, ty);
                    }
                    const currentMap = role.CurrentMapID();
                    const snapshot = typeof window.BuildThreatSnapshot === "function" ? window.BuildThreatSnapshot() : null;
                    const fakeMonster = { Role: role };
                    const encoded = window.BNBMLFeatureEncoder.Encode(role, currentMap, snapshot);
                    const actionMask = window.BNBMLDatasetCollector.BuildActionMask(fakeMonster, currentMap, snapshot, 6);
                    const legal = [];
                    for (let a = 0; a < actionMask.length; a++) {
                        if (Number(actionMask[a]) === 1) legal.push(a);
                    }
                    let action = 0;
                    if (Number(actionMask[5]) === 1 && rng() < (scenario === "open_random" ? 0.62 : 0.88)) {
                        action = 5;
                    } else {
                        const moving = legal.filter(function(a) { return a >= 1 && a <= 4; });
                        action = moving.length ? moving[Math.floor(rng() * moving.length)] : (legal[0] || 0);
                    }
                    const meta = window.BNBMLDatasetCollector.BuildSampleMeta(fakeMonster, currentMap, snapshot, null);
                    meta.action_source = "micro_scenario";
                    meta.micro_scenario_sample = 1;
                    meta.perspective = item.perspective;
                    meta.scenarioName = scenario;
                    meta.myBombThreatScore = Math.max(Number(meta.myBombThreatScore || 0), action === 5 ? 0.35 : 0.12);
                    meta.closeRangeDuelScore = Math.max(Number(meta.closeRangeDuelScore || 0), Number(window.__combatCollect.currentEpisodeMeta && window.__combatCollect.currentEpisodeMeta.closeRangeDuelScore || 0));
                    const row = {
                        id: "G" + window.__combatCollect.matchIndex + "_" + emitted + "_" + Math.floor(rng() * 1e9),
                        ts: Date.now(),
                        state: {
                            state_map: encoded.state_map,
                            state_vector: encoded.state_vector
                        },
                        action,
                        action_mask: actionMask,
                        reward: action === 5 ? 0.16 : 0.05,
                        done: false,
                        next_state: {
                            state_map: encoded.state_map,
                            state_vector: encoded.state_vector
                        },
                        pre_death: false,
                        risk_label: 0,
                        policy_tag: item.policy,
                        episode_id: item.episode,
                        agent_id: "role_" + role.RoleNumber,
                        opponent_id: "role_micro",
                        outcome_tag: "ongoing",
                        meta,
                        sample_bucket: action === 5 ? "drop_bomb_safe" : "ongoing"
                    };
                    window.__combatMicroRows.push(row);
                    emitted += 1;
                    if (typeof role.SetToMap === "function") {
                        role.SetToMap(original.X, original.Y);
                    }
                }
                return emitted;
            }

            window.__combatMicroDrain = function(maxRows) {
                window.__combatMicroRows = window.__combatMicroRows || [];
                return window.__combatMicroRows.splice(0, Math.max(1, parseInt(maxRows, 10) || 2048));
            };

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
                    for (let x = 0; x < window.townBarrierMap[y].length; x++) {
                        if (isWalkableEmpty(x, y)) {
                            out.push({ X: x, Y: y });
                        }
                    }
                }
                return out;
            }

            function isRigidBarrierNo(no) {
                return typeof window.IsRigidBarrierNo === "function"
                    ? !!window.IsRigidBarrierNo(Number(no || 0))
                    : (Number(no || 0) > 0 && Number(no || 0) < 100 && Number(no || 0) !== 3 && Number(no || 0) !== 8);
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
                if (window.__combatCollect.originalNonRigidCells || !window.townBarrierMap) {
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
                window.__combatCollect.originalNonRigidCells = cells;
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
                if (!window.__combatCollect.clearNonRigid || !window.townBarrierMap) {
                    return 0;
                }
                const cells = (window.__combatCollect.originalNonRigidCells || []).slice();
                for (let i = cells.length - 1; i > 0; i--) {
                    const j = Math.floor(rng() * (i + 1));
                    const tmp = cells[i];
                    cells[i] = cells[j];
                    cells[j] = tmp;
                }
                const keepMin = Math.max(0, Math.floor(Number(window.__combatCollect.softObstacleKeepMin || 0)));
                const keepMax = Math.max(keepMin, Math.floor(Number(window.__combatCollect.softObstacleKeepMax || 20)));
                const keepCount = window.__combatCollect.specialBombEscape
                    ? 0
                    : Math.min(cells.length, keepMin + Math.floor(rng() * (keepMax - keepMin + 1)));
                const keep = {};
                for (let i = 0; i < keepCount; i++) {
                    keep[`${cells[i].X}_${cells[i].Y}`] = true;
                }
                let cleared = 0;
                for (let i = 0; i < cells.length; i++) {
                    const c = cells[i];
                    disposeBarrierAt(c.X, c.Y);
                    if (keep[`${c.X}_${c.Y}`]) {
                        window.townBarrierMap[c.Y][c.X] = c.No;
                        if (window.Barrier && typeof window.Barrier.Create === "function") {
                            window.Barrier.Create(c.X, c.Y, c.No);
                        }
                    } else {
                        const itemProb = window.__combatCollect.specialBombEscape
                            ? 0
                            : Math.max(0, Math.min(0.5, Number(window.__combatCollect.currentDensity || 0)));
                        if (itemProb > 0 && rng() < itemProb) {
                            const giftNo = typeof window.CreateRandomGift === "function"
                                ? window.CreateRandomGift()
                                : ([101, 102, 103][Math.floor(rng() * 3)]);
                            window.townBarrierMap[c.Y][c.X] = giftNo;
                            if (window.Barrier && typeof window.Barrier.Create === "function") {
                                window.Barrier.Create(c.X, c.Y, giftNo);
                            }
                        } else {
                            window.townBarrierMap[c.Y][c.X] = 0;
                        }
                        cleared += 1;
                    }
                }
                window.__combatCollect.currentEpisodeMeta.softObstaclesKept = keepCount;
                window.__combatCollect.currentEpisodeMeta.softObstaclesCleared = cleared;
                return cleared;
            }

            function pickSpawnPairByShortestPath(scenario) {
                const cells = collectWalkableCells();
                if (!cells.length) {
                    return null;
                }
                const spawnBucketTarget = Math.max(0, (Number(window.__combatCollect.matchIndex || 1) - 1) % 3);
                const bucketRanges = [[1, 3], [4, 6], [7, 10]];
                const targetBucket = bucketRanges[spawnBucketTarget] || bucketRanges[0];
                const tries = Math.max(80, cells.length * 2);
                let best = null;
                let bestScore = -1e9;
                for (let i = 0; i < tries; i++) {
                    const aiCell = cells[Math.floor(rng() * cells.length)];
                    const playerCell = cells[Math.floor(rng() * cells.length)];
                    if (!aiCell || !playerCell) continue;
                    if (aiCell.X === playerCell.X && aiCell.Y === playerCell.Y) continue;
                    const pathDist = bfsDistance(aiCell, playerCell);
                    if (pathDist >= 999) continue;
                    const inRange = pathDist >= window.__combatCollect.spawnShortestPathMin
                        && pathDist <= window.__combatCollect.spawnShortestPathMax;
                    const inTargetBucket = pathDist >= Math.max(targetBucket[0], window.__combatCollect.spawnShortestPathMin)
                        && pathDist <= Math.min(targetBucket[1], window.__combatCollect.spawnShortestPathMax);
                    let score = inTargetBucket
                        ? 120
                        : (inRange ? 70 : -Math.abs(pathDist - window.__combatCollect.spawnShortestPathMax));
                    if (scenario === "deadend_chase" || scenario === "enemy_choke" || scenario === "escape_after_bomb") {
                        score += Math.max(0, 8 - pathDist) * 6;
                    } else {
                        score += pathDist * 0.5;
                    }
                    score += rng();
                    if (score > bestScore) {
                        bestScore = score;
                        best = { aiCell, playerCell, pathDist };
                    }
                }
                return best;
            }

            function pickWalkableCell(anchors, minDist, preferDeadend) {
                const cells = collectWalkableCells();
                let best = null;
                let bestScore = -1e9;
                for (let i = 0; i < cells.length; i++) {
                    const c = cells[i];
                    if (isOccupiedByRole(c.X, c.Y) || isNearAnchors(c.X, c.Y, anchors || [], minDist || 0)) {
                        continue;
                    }
                    let neighbors = 0;
                    const dirs = [[0, -1], [0, 1], [-1, 0], [1, 0]];
                    for (let d = 0; d < dirs.length; d++) {
                        if (isWalkableEmpty(c.X + dirs[d][0], c.Y + dirs[d][1])) {
                            neighbors += 1;
                        }
                    }
                    let score = rng();
                    if (preferDeadend) {
                        score += Math.max(0, 3 - neighbors) * 2;
                    } else {
                        score += neighbors * 0.2;
                    }
                    if (score > bestScore) {
                        bestScore = score;
                        best = c;
                    }
                }
                return best || cells[Math.floor(rng() * Math.max(1, cells.length))] || null;
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

            function reinjectSoftObstacles() {
                const ratio = Math.max(0, Math.min(0.35, window.__combatCollect.softObstacleReinjectRatio || 0));
                if (ratio <= 0 || !window.townBarrierMap || !Array.isArray(window.townBarrierMap)) {
                    return 0;
                }
                const rows = window.townBarrierMap.length;
                const cols = Array.isArray(window.townBarrierMap[0]) ? window.townBarrierMap[0].length : 0;
                const anchors = collectSpawnAnchors();
                const target = Math.floor(rows * cols * ratio);
                let added = 0;
                let attempts = 0;
                while (added < target && attempts < rows * cols * 8) {
                    attempts += 1;
                    const x = Math.floor(rng() * cols);
                    const y = Math.floor(rng() * rows);
                    if (!isWalkableEmpty(x, y) || isOccupiedByRole(x, y) || isNearAnchors(x, y, anchors, 2)) {
                        continue;
                    }
                    window.townBarrierMap[y][x] = 3;
                    if (window.Barrier && typeof window.Barrier.Create === "function") {
                        window.Barrier.Create(x, y, 3);
                    }
                    added += 1;
                }
                return added;
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
                const densityTarget = Math.floor(rows * cols * Math.max(0, Math.min(0.4, window.__combatCollect.currentDensity || 0)));
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
                    if (isNearAnchors(x, y, anchors, window.__combatCollect.currentItemSafeRadius || 0)) {
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
                if ((window.__combatCollect.currentItemRespawnMs || 0) <= 0) {
                    return;
                }
                window.__combatCollect.itemRespawnTicker = setInterval(function() {
                    if (!window.gameRunning) {
                        return;
                    }
                    trySpawnRandomItems();
                }, window.__combatCollect.currentItemRespawnMs);
            }

            function rollMapItemParams() {
                const cfg = window.__combatCollect;
                if (cfg.specialBombEscape) {
                    cfg.currentDensity = Math.max(0, Math.min(0.4, Number(cfg.randomItemDensity || 0.12)));
                    cfg.currentItemRespawnMs = Math.max(200, Number(cfg.specialItemRespawnMs || 2000));
                    cfg.currentItemSafeRadius = Math.max(0, Math.min(8, Number(cfg.itemSafeRadius || 2)));
                    cfg.currentOpponentThinkMs = Math.max(20, Number(cfg.thinkMs || 95));
                    return;
                }
                const densityNoise = (rng() * 2 - 1) * Math.max(0, Number(cfg.randomItemDensityJitter || 0));
                const density = Math.max(0, Math.min(0.4, Number(cfg.randomItemDensity || 0) + densityNoise));
                const respawnNoise = (rng() * 2 - 1) * Math.max(0, Number(cfg.itemRespawnJitterRatio || 0));
                const respawnMs = Math.max(220, Math.round(Number(cfg.itemRespawnMs || 0) * (1 + respawnNoise)));
                const safeNoise = Math.round((rng() * 2 - 1) * Math.max(0, Number(cfg.itemSafeRadiusJitter || 0)));
                const safeRadius = Math.max(0, Math.min(8, Number(cfg.itemSafeRadius || 0) + safeNoise));
                const thinkNoise = (rng() * 2 - 1) * Math.max(0, Number(cfg.opponentThinkJitterRatio || 0));
                const baseThinkMs = Math.max(20, Number(cfg.thinkMs || 95));
                const thinkMs = Math.max(20, Math.round(baseThinkMs * (1 + thinkNoise)));

                cfg.currentDensity = density;
                cfg.currentItemRespawnMs = respawnMs;
                cfg.currentItemSafeRadius = safeRadius;
                cfg.currentOpponentThinkMs = thinkMs;
            }

            function tuneRole(role, profile) {
                if (!role) return;
                const bubble = profile === "power" ? 2 + Math.floor(rng() * 3) : 1 + Math.floor(rng() * 2);
                const strong = profile === "power" ? 2 + Math.floor(rng() * 5) : 1 + Math.floor(rng() * 3);
                role.CanPaopaoLength = Math.max(1, bubble);
                role.PaopaoStrong = Math.max(1, strong);
                role.PaopaoCount = Math.max(0, Math.min(role.CanPaopaoLength - 1, Math.floor(rng() * Math.max(1, role.CanPaopaoLength))));
                role.LastBombAt = 0;
                if (typeof role.SetMoveSpeedPxPerSec === "function") {
                    role.SetMoveSpeedPxPerSec(105 + Math.floor(rng() * 75));
                }
            }

            function placeRolesForScenario(scenario) {
                const roles = findPlayerAndAiRoles();
                const player = roles.player;
                const ai = roles.ai;
                if (!player || !ai || typeof player.SetToMap !== "function" || typeof ai.SetToMap !== "function") {
                    return;
                }
                const pair = pickSpawnPairByShortestPath(scenario);
                if (!pair || !pair.aiCell || !pair.playerCell) return;
                ai.SetToMap(pair.aiCell.X, pair.aiCell.Y);
                player.SetToMap(pair.playerCell.X, pair.playerCell.Y);
                tuneRole(ai, scenario === "enemy_choke" || scenario === "escape_after_bomb" ? "power" : "normal");
                tuneRole(player, scenario === "item_race" ? "power" : "normal");
                window.__combatCollect.currentEpisodeMeta = {
                    spawnShortestPathDist: pair.pathDist,
                    spawnShortestPathDistNorm: Math.max(0, Math.min(1, pair.pathDist / 28)),
                    minEnemyDist: pair.pathDist,
                    property_bucket_score: 0,
                    lastAiThreatTs: 0,
                    myBombThreatScore: 0,
                    closeRangeDuelScore: Math.max(0, Math.min(1, 1 - pair.pathDist / 10)),
                    winningBombSourceRecent: 0,
                    scenarioName: scenario,
                    discardEpisode: false,
                };
                const aiPowerScore = (Number(ai.CanPaopaoLength || 0) / 6) * 0.4
                    + (Number(ai.PaopaoStrong || 0) / 8) * 0.35
                    + (Number(ai.MoveStep || 0) / 10) * 0.25;
                window.__combatCollect.currentEpisodeMeta.property_bucket_score = Math.max(0, Math.min(1, aiPowerScore));
            }

            function placeScenarioItems(scenario) {
                if (scenario !== "item_race") {
                    return;
                }
                const roles = findPlayerAndAiRoles();
                const pMap = getRoleMap(roles.player);
                const aiMap = getRoleMap(roles.ai);
                if (!pMap || !aiMap || !window.townBarrierMap) {
                    return;
                }
                const mid = {
                    X: Math.max(0, Math.min(14, Math.round((pMap.X + aiMap.X) / 2))),
                    Y: Math.max(0, Math.min(12, Math.round((pMap.Y + aiMap.Y) / 2))),
                };
                const candidates = [
                    mid,
                    { X: mid.X + 1, Y: mid.Y },
                    { X: mid.X - 1, Y: mid.Y },
                    { X: mid.X, Y: mid.Y + 1 },
                    { X: mid.X, Y: mid.Y - 1 },
                ];
                for (let i = 0; i < candidates.length; i++) {
                    const c = candidates[i];
                    if (!isWalkableEmpty(c.X, c.Y)) continue;
                    const giftNo = [101, 102, 103][Math.floor(rng() * 3)];
                    window.townBarrierMap[c.Y][c.X] = giftNo;
                    if (window.Barrier && typeof window.Barrier.Create === "function") {
                        window.Barrier.Create(c.X, c.Y, giftNo);
                    }
                    break;
                }
            }

            function placeSoftObstacleIfPossible(x, y) {
                if (!window.townBarrierMap || !window.townBarrierMap[y] || typeof window.townBarrierMap[y][x] === "undefined") {
                    return false;
                }
                if (!isWalkableEmpty(x, y) || isOccupiedByRole(x, y)) {
                    return false;
                }
                window.townBarrierMap[y][x] = 3;
                if (window.Barrier && typeof window.Barrier.Create === "function") {
                    window.Barrier.Create(x, y, 3);
                }
                return true;
            }

            function applyMicroScenarioGeometry(scenario) {
                if (!window.__combatCollect.microScenario) {
                    return;
                }
                const roles = findPlayerAndAiRoles();
                const player = roles.player;
                const ai = roles.ai;
                if (!player || !ai || !window.townBarrierMap) {
                    return;
                }
                const pMap = getRoleMap(player);
                const aiMap = getRoleMap(ai);
                if (!pMap || !aiMap) {
                    return;
                }
                const dirs = [[0, -1], [0, 1], [-1, 0], [1, 0]];
                if (scenario === "enemy_choke" || scenario === "deadend_chase") {
                    const keepDx = Math.sign(aiMap.X - pMap.X);
                    const keepDy = keepDx === 0 ? Math.sign(aiMap.Y - pMap.Y) : 0;
                    for (let i = 0; i < dirs.length; i++) {
                        const dx = dirs[i][0];
                        const dy = dirs[i][1];
                        if (dx === keepDx && dy === keepDy) {
                            continue;
                        }
                        if (rng() < 0.72) {
                            placeSoftObstacleIfPossible(pMap.X + dx, pMap.Y + dy);
                        }
                    }
                }
                if (scenario === "escape_after_bomb") {
                    const keepDx = Math.sign(pMap.X - aiMap.X);
                    const keepDy = keepDx === 0 ? Math.sign(pMap.Y - aiMap.Y) : 0;
                    for (let i = 0; i < dirs.length; i++) {
                        const dx = dirs[i][0];
                        const dy = dirs[i][1];
                        if (dx === keepDx && dy === keepDy) {
                            continue;
                        }
                        if (rng() < 0.48) {
                            placeSoftObstacleIfPossible(aiMap.X + dx, aiMap.Y + dy);
                        }
                    }
                }
            }

            function injectScenarioPressure(scenario) {
                const roles = findPlayerAndAiRoles();
                const player = roles.player;
                const ai = roles.ai;
                if (!player || !ai || typeof window.RoleKeyEvent !== "function") {
                    return;
                }
                function tryBomb(role, delayMs) {
                    setTimeout(function() {
                        if (role && !role.IsDeath && !role.IsInPaopao && role.CanPaopaoLength > role.PaopaoCount) {
                            window.RoleKeyEvent(32, role);
                        }
                    }, Math.max(0, delayMs || 0));
                }
                if (scenario === "escape_after_bomb") {
                    tryBomb(ai, 0);
                    tryBomb(ai, 300);
                    tryBomb(ai, 680);
                    tryBomb(player, 520);
                }
                if (scenario === "item_race") {
                    tryBomb(ai, 240);
                    tryBomb(ai, 760);
                    tryBomb(player, 520);
                }
                if (scenario === "open_random") {
                    tryBomb(ai, 680);
                }
                if (scenario === "enemy_choke" || scenario === "deadend_chase") {
                    tryBomb(ai, 0);
                    tryBomb(ai, 260);
                    tryBomb(ai, 720);
                    tryBomb(player, 0);
                    tryBomb(player, 420);
                }
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
                    if (mode === "stationary_dummy") {
                        if (typeof window.__combatMirrorRecordRoleFrame === "function") {
                            window.__combatMirrorRecordRoleFrame(
                                player,
                                0,
                                "opponent_stationary_dummy",
                                "m" + window.__combatCollect.matchIndex + "_opponent"
                            );
                        }
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

                    let recordedAction = 0;
                    if (best && typeof window.RoleKeyEvent === "function" && typeof window.RoleKeyEventEnd === "function") {
                        recordedAction = best.key === 38 ? 1 : (best.key === 40 ? 2 : (best.key === 37 ? 3 : (best.key === 39 ? 4 : 0)));
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
                            recordedAction = 5;
                        }
                    }
                    if (typeof window.__combatMirrorRecordRoleFrame === "function") {
                        window.__combatMirrorRecordRoleFrame(
                            player,
                            recordedAction,
                            "opponent_" + mode,
                            "m" + window.__combatCollect.matchIndex + "_opponent"
                        );
                    }
                }, Math.max(20, window.__combatCollect.currentOpponentThinkMs || Number(window.__combatCollect.thinkMs || 95)));
            }

            function refreshEpisodeMeta() {
                const roles = findPlayerAndAiRoles();
                const player = roles.player;
                const ai = roles.ai;
                const meta = window.__combatCollect.currentEpisodeMeta || {};
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
                const roundSec = Math.max(10, Number(window.__combatCollect.specialRoundSec || window.__combatCollect.matchDurationSec || 60));
                meta.roundKillCredit = clamp01(aiKills / Math.max(1, roundSec / 8));
                meta.roundSelfKillPenalty = clamp01(aiSelfKills / Math.max(1, roundSec / 10));
                meta.roundNetKdCredit = clamp01((aiKills - aiSelfKills + 6) / 12);
                window.__combatCollect.currentEpisodeMeta = meta;
                return meta;
            }

            function classifyOutcome(reason) {
                const roles = findPlayerAndAiRoles();
                const player = roles.player;
                const ai = roles.ai;
                const playerKills = player ? Number(player.kills || 0) : 0;
                const aiKills = ai ? Number(ai.kills || 0) : 0;
                const aiDeaths = ai ? Number(ai.deaths || 0) : 0;
                const aiSelfKills = Math.max(0, aiDeaths - playerKills);
                const meta = refreshEpisodeMeta();
                if (window.__combatCollect.specialBombEscape) {
                    const killDiff = aiKills - playerKills;
                    const killCredit = clamp01(aiKills / Math.max(1, Number(window.__combatCollect.specialRoundSec || 60) / 8));
                    const selfKillPenalty = clamp01(aiSelfKills / Math.max(1, Number(window.__combatCollect.specialRoundSec || 60) / 10));
                    const netKdCredit = clamp01((killDiff + 6) / 12);
                    let outcomeTag = "draw";
                    let reward = 0;
                    let mirrorOutcomeTag = "draw";
                    let mirrorReward = 0;
                    if (killDiff > 0) {
                        outcomeTag = "win";
                        mirrorOutcomeTag = "loss";
                    } else if (killDiff < 0) {
                        outcomeTag = "loss";
                        mirrorOutcomeTag = "win";
                    }
                    reward = Math.max(-2.5, Math.min(2.5, killCredit * 1.2 - selfKillPenalty * 1.8 + killDiff * 0.08));
                    mirrorReward = -reward;
                    meta.roundKillCredit = killCredit;
                    meta.roundSelfKillPenalty = selfKillPenalty;
                    meta.roundNetKdCredit = netKdCredit;
                    return {
                        outcome_tag: outcomeTag,
                        terminal_reason: "round_end",
                        discard_episode: false,
                        episode_id: "m" + window.__combatCollect.matchIndex + "_agent",
                        perspective: "agent",
                        mirror_episode_id: "m" + window.__combatCollect.matchIndex + "_opponent",
                        mirror_outcome_tag: mirrorOutcomeTag,
                        mirror_terminal_reason: "round_end",
                        mirror_discard_episode: false,
                        mirror_reward: mirrorReward,
                        mirror_pre_death: outcomeTag === "win",
                        ai_kills: aiKills,
                        player_kills: playerKills,
                        ai_deaths: aiDeaths,
                        ai_self_kills: aiSelfKills,
                        reward,
                        spawn_shortest_path_dist: Number(meta.spawnShortestPathDist || null),
                        property_bucket_score: Number(meta.property_bucket_score || 0),
                        my_bomb_threat_score: Number(meta.myBombThreatScore || 0),
                        close_range_duel_score: Number(meta.closeRangeDuelScore || 0),
                        winning_bomb_source_recent: Number(meta.winningBombSourceRecent || 0),
                        danger_cells_created_score: Number(meta.dangerCellsCreatedScore || 0),
                        round_kill_credit: Number(meta.roundKillCredit || 0),
                        round_self_kill_penalty: Number(meta.roundSelfKillPenalty || 0),
                        round_net_kd_credit: Number(meta.roundNetKdCredit || 0),
                        ts: Date.now(),
                    };
                }
                let outcomeTag = "draw";
                let terminalReason = String(reason || "draw");
                let discardEpisode = false;
                let reward = -0.2;
                let mirrorOutcomeTag = "draw";
                let mirrorTerminalReason = "stall_abort";
                let mirrorDiscardEpisode = true;
                let mirrorReward = 0;
                let mirrorPreDeath = false;
                if (terminalReason === "caught_self") {
                    outcomeTag = "self_kill";
                    reward = -1.5;
                    mirrorOutcomeTag = "draw";
                    mirrorTerminalReason = "enemy_self_kill_discard";
                    mirrorDiscardEpisode = true;
                } else if (terminalReason === "caught_enemy") {
                    const aiThreatRecent = Number(meta.lastAiThreatTs || 0) > 0 && (Date.now() - Number(meta.lastAiThreatTs || 0)) <= 2600;
                    if (window.__combatCollect.ignoreEnemySelfKill && !aiThreatRecent && Number(meta.myBombThreatScore || 0) < 0.20) {
                        terminalReason = "enemy_self_kill_discard";
                        discardEpisode = true;
                        outcomeTag = "draw";
                        reward = 0;
                        mirrorOutcomeTag = "self_kill";
                        mirrorTerminalReason = "caught_self";
                        mirrorDiscardEpisode = false;
                        mirrorReward = -1.5;
                        mirrorPreDeath = true;
                    } else {
                        outcomeTag = "win";
                        reward = 1.0
                            + Math.max(0, Math.min(0.45, Number(meta.closeRangeDuelScore || 0) * 0.45))
                            + Math.max(0, Math.min(0.35, Number(meta.myBombThreatScore || 0) * 0.35));
                        meta.winningBombSourceRecent = aiThreatRecent ? 1 : 0;
                        mirrorOutcomeTag = "loss";
                        mirrorTerminalReason = "caught_self";
                        mirrorDiscardEpisode = false;
                        mirrorReward = -1.0;
                        mirrorPreDeath = true;
                    }
                } else if (terminalReason === "stall_abort") {
                    discardEpisode = true;
                    outcomeTag = "draw";
                    reward = 0;
                    mirrorOutcomeTag = "draw";
                    mirrorTerminalReason = "stall_abort";
                    mirrorDiscardEpisode = true;
                } else if (aiSelfKills > 0) {
                    outcomeTag = "self_kill";
                    reward = -1.5;
                    mirrorOutcomeTag = "draw";
                    mirrorTerminalReason = "enemy_self_kill_discard";
                    mirrorDiscardEpisode = true;
                } else if (aiKills > playerKills) {
                    outcomeTag = "win";
                    reward = 1.0;
                    mirrorOutcomeTag = "loss";
                    mirrorTerminalReason = "caught_self";
                    mirrorDiscardEpisode = false;
                    mirrorReward = -1.0;
                    mirrorPreDeath = true;
                } else if (aiKills < playerKills) {
                    outcomeTag = "loss";
                    reward = -1.0;
                    mirrorOutcomeTag = "win";
                    mirrorTerminalReason = "caught_enemy";
                    mirrorDiscardEpisode = false;
                    mirrorReward = 1.0;
                }

                return {
                    outcome_tag: outcomeTag,
                    terminal_reason: terminalReason,
                    discard_episode: discardEpisode,
                    episode_id: "m" + window.__combatCollect.matchIndex + "_agent",
                    perspective: "agent",
                    mirror_episode_id: "m" + window.__combatCollect.matchIndex + "_opponent",
                    mirror_outcome_tag: mirrorOutcomeTag,
                    mirror_terminal_reason: mirrorTerminalReason,
                    mirror_discard_episode: mirrorDiscardEpisode,
                    mirror_reward: mirrorReward,
                    mirror_pre_death: mirrorPreDeath,
                    ai_kills: aiKills,
                    player_kills: playerKills,
                    ai_deaths: aiDeaths,
                    ai_self_kills: aiSelfKills,
                    reward,
                    spawn_shortest_path_dist: Number(meta.spawnShortestPathDist || null),
                    property_bucket_score: Number(meta.property_bucket_score || 0),
                    my_bomb_threat_score: Number(meta.myBombThreatScore || 0),
                    close_range_duel_score: Number(meta.closeRangeDuelScore || 0),
                    winning_bomb_source_recent: Number(meta.winningBombSourceRecent || 0),
                    ts: Date.now(),
                };
            }

            function finalizeCurrentMatch(reason) {
                if (window.__combatCollect.matchFinalized) {
                    return null;
                }
                const terminal = classifyOutcome(reason);
                let collectorFinalized = false;
                if (typeof window.BNBMLCollectorFinalizeEpisode === "function") {
                    collectorFinalized = !!window.BNBMLCollectorFinalizeEpisode(
                        terminal.outcome_tag,
                        {
                            done: true,
                            preDeath: terminal.outcome_tag === "self_kill" || terminal.terminal_reason === "caught_self",
                            reward: terminal.reward,
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
                                preDeath: terminal.outcome_tag === "self_kill" || terminal.terminal_reason === "caught_self",
                                reward: terminal.reward,
                                forceFlush: true,
                            }
                        );
                    }
                }
                terminal.collector_finalized = collectorFinalized;
                if (window.__combatCollect.currentEpisodeMeta) {
                    window.__combatCollect.currentEpisodeMeta.discardEpisode = !!terminal.discard_episode;
                }
                window.__combatCollect.matchFinalized = true;
                window.__combatCollect.lastTerminal = terminal;
                window.__combatCollect.terminalEvents += 1;
                return terminal;
            }

            window.__combatCollectFindRoles = findPlayerAndAiRoles;
            window.__combatCollectCountItems = countItems;
            window.__combatCollectRefreshEpisodeMeta = refreshEpisodeMeta;

            function applySpecialBombEscapePatch() {
                if (!window.__combatCollect.specialBombEscape || !window.Role || !window.Role.prototype) {
                    return;
                }
                if (window.__combatCollect.__specialBombPatched) {
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
                window.__combatCollect.__specialBombPatched = true;
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
                window.roundDurationSeconds = window.__combatCollect.suddenDeath ? 300 : window.__combatCollect.matchDurationSec;
                if (typeof window.StartSinglePlayerGame === "function") {
                    window.StartSinglePlayerGame(1);
                }
                if (window.__combatCollect.specialBombEscape) {
                    window.respawnDelayMs = Math.max(0, Number(window.__combatCollect.specialRespawnDelayMs || 0));
                    window.respawnInvincibleMs = Math.max(0, Number(window.__combatCollect.specialRespawnInvincibleMs || 300));
                    applySpecialBombEscapePatch();
                } else if (window.__combatCollect.disableRevive) {
                    window.respawnDelayMs = 999999;
                    window.respawnInvincibleMs = 0;
                }

                window.__combatCollect.matchIndex += 1;
                window.__combatCollect.matchFinalized = false;
                resetMirrorCollector();
                window.__combatCollect.lastStartAt = Date.now();
                window.__combatCollect.lastProgressAt = Date.now();
                window.__combatCollect.lastProgressSignature = "";
                window.__combatCollect.currentEpisodeMeta = {
                    spawnShortestPathDist: null,
                    spawnShortestPathDistNorm: 1,
                    minEnemyDist: 999,
                    property_bucket_score: 0,
                    lastAiThreatTs: 0,
                    myBombThreatScore: 0,
                    closeRangeDuelScore: 0,
                    winningBombSourceRecent: 0,
                    discardEpisode: false,
                };
                rollMapItemParams();
                const scenario = pickScenario();
                window.__combatCollect.currentScenario = scenario;
                window.__combatCollect.currentAgentExpert = pickAgentMode(window.__combatCollect.matchIndex);
                startPlayerController(pickOpponentMode(window.__combatCollect.matchIndex));

                setTimeout(function() {
                    applyPartialClearMap();
                    reinjectSoftObstacles();
                    if (window.__combatCollect.spawnRandomize) {
                        placeRolesForScenario(scenario);
                    }
                    applyMicroScenarioGeometry(scenario);
                    placeScenarioItems(scenario);
                    trySpawnRandomItems();
                    emitMicroScenarioSamples(scenario);
                    injectScenarioPressure(scenario);
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
            agentPool,
            agentExpertDuel,
            mapId,
            thinkMs,
            matchDurationSec: runtimeMatchDurationSec,
            clearNonRigid: runtimeClearNonRigid,
            randomItemDensity,
            randomItemDensityJitter: runtimeRandomItemDensityJitter,
            itemRespawnMs: runtimeItemRespawnMs,
            itemRespawnJitterRatio: runtimeItemRespawnJitterRatio,
            itemSafeRadius,
            itemSafeRadiusJitter,
            itemMax,
            opponentThinkJitterRatio,
            balanced,
            scenarioBuckets,
            softObstacleReinjectRatio,
            spawnRandomize,
            terminalTailMs,
            suddenDeath: runtimeSuddenDeath,
            disableRevive: runtimeDisableRevive,
            ignoreEnemySelfKill: runtimeIgnoreEnemySelfKill,
            stallNoProgressMs,
            maxEpisodeMs,
            mirrorSampling,
            earlyCommitHighValue,
            microScenario,
            microSamplesPerEpisode,
            partialClearMinRatio,
            partialClearMaxRatio,
            softObstacleKeepMin: runtimeSoftObstacleKeepMin,
            softObstacleKeepMax: runtimeSoftObstacleKeepMax,
            spawnShortestPathMin,
            spawnShortestPathMax,
            specialBombEscape,
            specialRoundSec,
            specialItemRespawnMs,
            specialRespawnDelayMs,
            specialRespawnInvincibleMs,
            dangerScoreWeight,
            killScoreWeight,
            selfKillPenaltyWeight,
            noGlobalDedupe,
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
                if (typeof window.__combatMicroDrain === "function") {
                    const microRows = window.__combatMicroDrain(maxRows) || [];
                    if (Array.isArray(microRows) && microRows.length > 0) {
                        for (let i = 0; i < microRows.length; i++) {
                            rows.push(microRows[i]);
                        }
                    }
                }
                const runtime = window.BNBTrainingRuntimeState || null;
                const collector = window.BNBMLDatasetCollectorState || null;
                const collectState = window.__combatCollect || null;
                const running = !!window.gameRunning;
                const activeEpisodeId = collectState
                    ? ("m" + Math.max(1, Number(collectState.matchIndex || 1)) + "_a1")
                    : "runtime";
                const roles = typeof window.__combatCollectFindRoles === "function"
                    ? window.__combatCollectFindRoles()
                    : { player: null, ai: null };
                const playerTrapped = !!(roles.player && !roles.player.IsDeath && roles.player.IsInPaopao);
                const aiTrapped = !!(roles.ai && !roles.ai.IsDeath && roles.ai.IsInPaopao);
                const timedOut = !!(running
                    && collectState
                    && collectState.lastStartAt
                    && !collectState.suddenDeath
                    && collectState.matchDurationSec
                    && Date.now() - collectState.lastStartAt >= collectState.matchDurationSec * 1000);
                let stalled = false;
                const episodeExceeded = !!(running
                    && collectState
                    && collectState.suddenDeath
                    && collectState.maxEpisodeMs
                    && collectState.lastStartAt
                    && Date.now() - collectState.lastStartAt >= collectState.maxEpisodeMs);
                if (running && collectState && collectState.suddenDeath && collectState.currentEpisodeMeta) {
                    const meta = typeof window.__combatCollectRefreshEpisodeMeta === "function"
                        ? window.__combatCollectRefreshEpisodeMeta()
                        : (collectState.currentEpisodeMeta || {});
                    const currentSig = JSON.stringify({
                        player: roles.player && typeof roles.player.CurrentMapID === "function" ? roles.player.CurrentMapID() : null,
                        ai: roles.ai && typeof roles.ai.CurrentMapID === "function" ? roles.ai.CurrentMapID() : null,
                        bombs: typeof window.CountActiveBombs === "function" ? window.CountActiveBombs() : 0,
                        items: typeof window.__combatCollectCountItems === "function" ? window.__combatCollectCountItems() : 0,
                        minEnemyDist: meta && typeof meta.minEnemyDist === "number" ? meta.minEnemyDist : null,
                        threat: meta && typeof meta.myBombThreatScore === "number" ? meta.myBombThreatScore : 0
                    });
                    if (collectState.lastProgressSignature !== currentSig) {
                        collectState.lastProgressSignature = currentSig;
                        collectState.lastProgressAt = Date.now();
                    }
                    stalled = !!(collectState.lastProgressAt && (Date.now() - collectState.lastProgressAt >= collectState.stallNoProgressMs));
                }
                let terminal = null;

                if (running && collectState && collectState.suddenDeath && (playerTrapped || aiTrapped || stalled || episodeExceeded)) {
                    window.gameRunning = false;
                    terminal = typeof window.__combatCollectFinalizeCurrentMatch === "function"
                        ? window.__combatCollectFinalizeCurrentMatch(
                            playerTrapped ? "caught_enemy" : (aiTrapped ? "caught_self" : "stall_abort")
                        )
                        : null;
                }

                if (!running || timedOut || stalled || episodeExceeded || playerTrapped || aiTrapped) {
                    if (typeof window.__combatCollectFinalizeCurrentMatch === "function") {
                        terminal = terminal || window.__combatCollectFinalizeCurrentMatch();
                    }
                    if (typeof window.BNBMLCollectorDrainAll === "function") {
                        const terminalRows = window.BNBMLCollectorDrainAll() || [];
                        if (Array.isArray(terminalRows) && terminalRows.length > 0) {
                            for (let i = 0; i < terminalRows.length; i++) {
                                rows.push(terminalRows[i]);
                            }
                        }
                    }
                    if (terminal && typeof window.__combatMirrorFinalizeEpisode === "function") {
                        window.__combatMirrorFinalizeEpisode(
                            terminal.mirror_outcome_tag || "draw",
                            {
                                done: true,
                                preDeath: !!terminal.mirror_pre_death,
                                reward: typeof terminal.mirror_reward === "number" ? terminal.mirror_reward : 0,
                            }
                        );
                    }
                    if (typeof window.__combatMirrorDrain === "function") {
                        const mirrorRows = window.__combatMirrorDrain(maxRows, !!terminal) || [];
                        if (Array.isArray(mirrorRows) && mirrorRows.length > 0) {
                            for (let i = 0; i < mirrorRows.length; i++) {
                                rows.push(mirrorRows[i]);
                            }
                        }
                    }
                    if (terminal && collectState && collectState.terminalTailMs > 0) {
                        const tailCutoff = Number(terminal.ts || Date.now()) - Number(collectState.terminalTailMs || 0);
                        const terminalReward = terminal.outcome_tag === "win"
                            ? 1.5
                            : (terminal.outcome_tag === "self_kill" ? -2.0 : (terminal.outcome_tag === "loss" ? -1.2 : -0.2));
                        for (let i = 0; i < rows.length; i++) {
                            const row = rows[i];
                            if (!row || Number(row.ts || 0) < tailCutoff) {
                                continue;
                            }
                            row.done = true;
                            row.outcome_tag = terminal.outcome_tag;
                            row.reward = terminalReward;
                            row.pre_death = terminal.outcome_tag === "self_kill" || !!row.pre_death;
                            row.risk_label = row.pre_death || row.done ? 1 : Number(row.risk_label || 0);
                            row.meta = row.meta || {};
                            row.meta.terminal_tail = 1;
                            row.meta.myBombThreatScore = Number(terminal.my_bomb_threat_score || row.meta.myBombThreatScore || 0);
                            row.meta.closeRangeDuelScore = Number(terminal.close_range_duel_score || row.meta.closeRangeDuelScore || 0);
                            row.meta.winningBombSourceRecent = Number(terminal.winning_bomb_source_recent || row.meta.winningBombSourceRecent || 0);
                            row.meta.spawnShortestPathDist = Number(terminal.spawn_shortest_path_dist || row.meta.spawnShortestPathDist || 0);
                            row.meta.dangerCellsCreatedScore = Number(terminal.danger_cells_created_score || row.meta.dangerCellsCreatedScore || 0);
                            row.meta.roundKillCredit = Number(terminal.round_kill_credit || row.meta.roundKillCredit || 0);
                            row.meta.roundSelfKillPenalty = Number(terminal.round_self_kill_penalty || row.meta.roundSelfKillPenalty || 0);
                            row.meta.roundNetKdCredit = Number(terminal.round_net_kd_credit || row.meta.roundNetKdCredit || 0);
                            row.meta.scenarioName = (collectState && collectState.currentScenario) || row.meta.scenarioName || "";
                            if (row.meta.perspective === "opponent") {
                                row.outcome_tag = terminal.mirror_outcome_tag || row.outcome_tag;
                                row.reward = typeof terminal.mirror_reward === "number" ? terminal.mirror_reward : row.reward;
                                row.pre_death = !!terminal.mirror_pre_death || !!row.pre_death;
                                row.terminal_reason = terminal.mirror_terminal_reason || row.terminal_reason || row.outcome_tag;
                            } else {
                                row.terminal_reason = terminal.terminal_reason || row.terminal_reason || row.outcome_tag;
                            }
                        }
                    }
                    if (!terminal && typeof window.__combatMirrorDrain === "function") {
                        const mirrorRows = window.__combatMirrorDrain(maxRows, false) || [];
                        if (Array.isArray(mirrorRows) && mirrorRows.length > 0) {
                            for (let i = 0; i < mirrorRows.length; i++) {
                                rows.push(mirrorRows[i]);
                            }
                        }
                    }
                    if (typeof window.__combatCollectStartMatch === "function") {
                        window.__combatCollectStartMatch();
                    }
                }
                if (running && !terminal && typeof window.__combatMirrorDrain === "function") {
                    const mirrorRows = window.__combatMirrorDrain(maxRows, false) || [];
                    if (Array.isArray(mirrorRows) && mirrorRows.length > 0) {
                        for (let i = 0; i < mirrorRows.length; i++) {
                            rows.push(mirrorRows[i]);
                        }
                    }
                }

                return {
                    rows,
                    runtime,
                    collector,
                    running,
                    terminal,
                    episodeId: activeEpisodeId,
                    matchIndex: collectState ? collectState.matchIndex : 0,
                    aiRoleNumber: collectState ? collectState.aiRoleNumber : null,
                    opponentMode: collectState ? collectState.currentOpponent : "heuristic_v1",
                    agentMode: collectState ? collectState.currentAgentExpert : "heuristic_v2",
                    terminalEvents: collectState ? collectState.terminalEvents : 0,
                    episodeMeta: collectState ? collectState.currentEpisodeMeta : null,
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
                    raw.meta = raw.meta || {};
                    const rawRoleNumber = Number(raw.meta.roleNumber);
                    const isMirrorSample = raw.meta.perspective === "opponent" || Number(raw.meta.mirror_sample || 0) > 0;
                    const matchEpisodePrefix = "m" + Math.max(1, Number(batchResult.matchIndex || 1));
                    if (isMirrorSample) {
                        raw.episode_id = matchEpisodePrefix + "_opponent";
                        mirrorRowsSeen += 1;
                    } else if (Number.isFinite(rawRoleNumber) && Number.isFinite(Number(batchResult.aiRoleNumber)) && rawRoleNumber !== Number(batchResult.aiRoleNumber)) {
                        raw.episode_id = matchEpisodePrefix + "_opponent";
                        raw.meta.perspective = "opponent";
                        mirrorRowsSeen += 1;
                    } else {
                        raw.episode_id = matchEpisodePrefix + "_agent";
                        raw.meta.perspective = "agent";
                    }
                    if (Number(raw.meta.micro_scenario_sample || 0) > 0) {
                        microRowsSeen += 1;
                    }
                    if (!raw.meta.scenarioName && batchResult.episodeMeta && batchResult.episodeMeta.scenarioName) {
                        raw.meta.scenarioName = batchResult.episodeMeta.scenarioName;
                    }
                    if (batchResult.episodeMeta) {
                        raw.meta.dangerCellsCreatedScore = Number(batchResult.episodeMeta.dangerCellsCreatedScore || raw.meta.dangerCellsCreatedScore || 0);
                        raw.meta.roundKillCredit = Number(batchResult.episodeMeta.roundKillCredit || raw.meta.roundKillCredit || 0);
                        raw.meta.roundSelfKillPenalty = Number(batchResult.episodeMeta.roundSelfKillPenalty || raw.meta.roundSelfKillPenalty || 0);
                        raw.meta.roundNetKdCredit = Number(batchResult.episodeMeta.roundNetKdCredit || raw.meta.roundNetKdCredit || 0);
                    }
                    raw.meta.dangerScoreWeight = dangerScoreWeight;
                    raw.meta.killScoreWeight = killScoreWeight;
                    raw.meta.selfKillPenaltyWeight = selfKillPenaltyWeight;
                    if (batchResult.agentMode) {
                        raw.meta.agentExpertMode = batchResult.agentMode;
                    }
                    if (batchResult.opponentMode) {
                        raw.meta.opponentExpertMode = batchResult.opponentMode;
                    }
                    bufferEpisodeRow(raw);
                }
            }
            if (batchResult.terminal && batchResult.terminal.episode_id) {
                commitEpisodeRows(batchResult.terminal.episode_id, batchResult.terminal);
                if (batchResult.terminal.mirror_episode_id) {
                    commitEpisodeRows(batchResult.terminal.mirror_episode_id, {
                        outcome_tag: batchResult.terminal.mirror_outcome_tag,
                        terminal_reason: batchResult.terminal.mirror_terminal_reason,
                        discard_episode: !!batchResult.terminal.mirror_discard_episode,
                        reward: batchResult.terminal.mirror_reward,
                        pre_death: !!batchResult.terminal.mirror_pre_death,
                        perspective: "opponent",
                        spawn_shortest_path_dist: batchResult.terminal.spawn_shortest_path_dist,
                        property_bucket_score: batchResult.terminal.property_bucket_score,
                        my_bomb_threat_score: batchResult.terminal.my_bomb_threat_score,
                        close_range_duel_score: batchResult.terminal.close_range_duel_score,
                        winning_bomb_source_recent: batchResult.terminal.winning_bomb_source_recent,
                    });
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
                    `agent=${batchResult.agentMode || "heuristic_v2"}`,
                    `ai_role=${batchResult.aiRoleNumber == null ? "na" : batchResult.aiRoleNumber}`,
                    `valid_eps=${validEpisodeCount}`,
                    `burst_drop=${burstCapDropped}`,
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
            bufferEpisodeRow(raw);
        }
        for (const [episodeId, rows] of episodeBuffers.entries()) {
            dropped += rows.length;
            markDrop("unfinished_episode");
            episodeBuffers.delete(episodeId);
        }

        if (screenshotEnabled) {
            try {
                await page.screenshot({ path: screenshotPath, fullPage: true, timeout: 8000 });
            } catch (err) {
                console.log("[WARN] screenshot skipped:", String(err && err.message ? err.message : err));
            }
        }
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
        illegal_action_rows: Number(dropReasons.action_not_legal_by_mask || 0),
        repaired_illegal_action_rows: repairedIllegalActionRows,
        burst_cap_dropped_rows: burstCapDropped,
        early_committed_rows: earlyCommittedRows,
        streaming_written_rows: streamingWrittenRows,
        episode_buffer_written_rows: episodeBufferWrittenRows,
        terminal_credit_applied_rows: terminalCreditAppliedRows,
        stall_kept_rows: stallKeptRows,
        high_value_behavior_rows: highValueBehaviorRows,
        high_value_ratio: wrote > 0 ? highValueBehaviorRows / wrote : 0,
        mirror_rows_seen: mirrorRowsSeen,
        mirror_episodes_committed: mirrorEpisodesCommitted,
        micro_rows_seen: microRowsSeen,
        unique_transition_rows: uniqueTransitions,
        valid_episode_count: validEpisodeCount,
        discarded_episode_count: discardedEpisodeCount,
        drop_reasons: dropReasons,
        rows_per_sec: durationSec > 0 ? wrote / durationSec : 0,
        duration_sec: durationSec,
        action_hist_written: actionHist,
        done_hist_written: doneHist,
        pre_death_hist_written: preDeathHist,
        outcome_hist_written: outcomeHist,
        bucket_hist_written: bucketHist,
        terminal_reason_hist: terminalReasonHist,
        spawn_dist_hist: spawnDistHist,
        property_bucket_hist: propertyBucketHist,
        aux_label_hist_written: auxLabelHist,
        behavior_score_hist: behaviorScoreHist,
        behavior_score_mean: wrote > 0 ? behaviorScoreSum / wrote : 0,
        behavior_score_breakdown_hist: behaviorBreakdownHist,
        terminal_outcome_events: terminalOutcomeEvents,
        terminal_finalize_calls: terminalFinalizeCalls,
        terminal_collector_finalize_ok: terminalCollectorFinalizeOk,
        terminal_collector_finalize_miss: terminalCollectorFinalizeMiss,
        arena,
        action_space: actionSpace,
        map_id: mapId,
        opponent_pool: opponentPool,
        agent_pool: agentPool,
        agent_expert_duel: agentExpertDuel,
        clear_nonrigid: runtimeClearNonRigid,
        random_item_density: randomItemDensity,
        random_item_density_jitter: runtimeRandomItemDensityJitter,
        item_respawn_ms: runtimeItemRespawnMs,
        item_respawn_jitter_ratio: runtimeItemRespawnJitterRatio,
        item_safe_radius: itemSafeRadius,
        item_safe_radius_jitter: itemSafeRadiusJitter,
        item_max: itemMax,
        opponent_think_jitter_ratio: opponentThinkJitterRatio,
        balanced,
        scenario_buckets: scenarioBuckets.split(",").map((s) => s.trim()).filter(Boolean),
        soft_obstacle_reinject_ratio: softObstacleReinjectRatio,
        spawn_randomize: spawnRandomize,
        terminal_tail_ms: terminalTailMs,
        sudden_death: runtimeSuddenDeath,
        disable_revive: runtimeDisableRevive,
        ignore_enemy_self_kill: runtimeIgnoreEnemySelfKill,
        stall_no_progress_ms: stallNoProgressMs,
        max_episode_ms: maxEpisodeMs,
        mirror_sampling: mirrorSampling,
        early_commit_high_value: earlyCommitHighValue,
        behavior_scoring: behaviorScoring,
        behavior_score_threshold: behaviorScoreThreshold,
        high_value_behavior_threshold: highValueBehaviorThreshold,
        behavior_credit_window_ms: behaviorCreditWindowMs,
        micro_scenario: microScenario,
        micro_samples_per_episode: microSamplesPerEpisode,
        partial_clear_min_ratio: partialClearMinRatio,
        partial_clear_max_ratio: partialClearMaxRatio,
        soft_obstacle_keep_min: runtimeSoftObstacleKeepMin,
        soft_obstacle_keep_max: runtimeSoftObstacleKeepMax,
        spawn_shortest_path_min: spawnShortestPathMin,
        spawn_shortest_path_max: spawnShortestPathMax,
        special_bomb_escape: specialBombEscape,
        special_round_sec: specialRoundSec,
        special_item_respawn_ms: specialItemRespawnMs,
        respawn_delay_ms: specialRespawnDelayMs,
        respawn_invincible_ms: specialRespawnInvincibleMs,
        no_global_dedupe: noGlobalDedupe,
        danger_score_weight: dangerScoreWeight,
        kill_score_weight: killScoreWeight,
        self_kill_penalty_weight: selfKillPenaltyWeight,
        latest_training_runtime_state: lastRuntimeState,
        screenshot: screenshotEnabled ? screenshotPath : null,
        screenshot_enabled: screenshotEnabled
    };

    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
    console.log("[DONE]", JSON.stringify(report));
}

main().catch((err) => {
    console.error(err);
    process.exit(1);
});
