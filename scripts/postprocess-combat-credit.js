#!/usr/bin/env node
const fs = require("fs");
const path = require("path");
const readline = require("readline");

function getArg(name, fallback) {
    const prefix = "--" + name + "=";
    for (const arg of process.argv.slice(2)) {
        if (arg.startsWith(prefix)) {
            return arg.slice(prefix.length);
        }
    }
    return fallback;
}

function asFloat(v, fallback) {
    const n = Number(v);
    return Number.isFinite(n) ? n : fallback;
}

function asInt(v, fallback) {
    const n = parseInt(v, 10);
    return Number.isFinite(n) ? n : fallback;
}

function clamp(v, lo, hi) {
    const n = Number(v);
    if (!Number.isFinite(n)) return lo;
    return Math.max(lo, Math.min(hi, n));
}

function clamp01(v) {
    return clamp(v, 0, 1);
}

function ensureParent(p) {
    fs.mkdirSync(path.dirname(p), { recursive: true });
}

function compact(v) {
    const n = Number(v);
    if (!Number.isFinite(n)) return 0;
    return Math.round(n * 1e6) / 1e6;
}

function getVec(row, key) {
    if (!row || typeof row !== "object") return [];
    const state = key === "next"
        ? (row.next_state || {})
        : (row.state || {});
    const vec = Array.isArray(state.state_vector)
        ? state.state_vector
        : (key === "next" ? row.next_state_vector : row.state_vector);
    return Array.isArray(vec) ? vec : [];
}

function getSelfCap(vec) {
    if (!Array.isArray(vec)) return 0;
    if (vec.length > 24) return clamp01(vec[24]);
    if (vec.length > 10) return clamp01(vec[10]);
    return 0;
}

function getSelfPower(vec) {
    return Array.isArray(vec) && vec.length > 11 ? clamp01(vec[11]) : 0;
}

function getSelfSpeed(vec) {
    return Array.isArray(vec) && vec.length > 12 ? clamp01(vec[12]) : 0;
}

function deriveItemGain(row) {
    const aux = row && row.aux_labels && typeof row.aux_labels === "object" ? row.aux_labels : {};
    const meta = row && row.meta && typeof row.meta === "object" ? row.meta : {};
    if (Number.isFinite(Number(aux.item_gain))) {
        return clamp01(aux.item_gain);
    }
    if (Number.isFinite(Number(meta.itemGain))) {
        return clamp01(meta.itemGain);
    }
    const cur = getVec(row, "state");
    const nxt = getVec(row, "next");
    if (!cur.length || !nxt.length) {
        return 0;
    }
    const dCap = Math.max(0, getSelfCap(nxt) - getSelfCap(cur));
    const dPower = Math.max(0, getSelfPower(nxt) - getSelfPower(cur));
    const dSpeed = Math.max(0, getSelfSpeed(nxt) - getSelfSpeed(cur));
    return clamp01((dCap + dPower + dSpeed) * 2.5);
}

function getDangerScore(row) {
    const aux = row && row.aux_labels && typeof row.aux_labels === "object" ? row.aux_labels : {};
    const meta = row && row.meta && typeof row.meta === "object" ? row.meta : {};
    return clamp01(
        Number(aux.danger_cells_created_score)
        || Number(meta.dangerCellsCreatedScore)
        || 0
    );
}

function getKillScore(row) {
    const aux = row && row.aux_labels && typeof row.aux_labels === "object" ? row.aux_labels : {};
    const meta = row && row.meta && typeof row.meta === "object" ? row.meta : {};
    return clamp01(
        Number(aux.round_kill_credit)
        || Number(meta.roundKillCredit)
        || 0
    );
}

function getEscapeScore(row) {
    const aux = row && row.aux_labels && typeof row.aux_labels === "object" ? row.aux_labels : {};
    return clamp01(
        Number(aux.bomb_escape_success_label)
        || 0
    );
}

function getSelfKillPenalty(row) {
    const aux = row && row.aux_labels && typeof row.aux_labels === "object" ? row.aux_labels : {};
    const meta = row && row.meta && typeof row.meta === "object" ? row.meta : {};
    let penalty = clamp01(
        Number(aux.round_self_kill_penalty)
        || Number(meta.roundSelfKillPenalty)
        || Number(aux.bomb_self_trap_risk)
        || 0
    );
    const outcome = String(row && row.outcome_tag || "").toLowerCase();
    if (outcome === "self_kill" || !!(row && row.pre_death)) {
        penalty = Math.max(penalty, 0.5);
    }
    return clamp01(penalty);
}

async function eachJsonlRow(filePath, onRow) {
    const stream = fs.createReadStream(filePath);
    const rl = readline.createInterface({
        input: stream,
        crlfDelay: Infinity,
    });
    let lineNo = 0;
    for await (const line of rl) {
        lineNo += 1;
        if (!line || !line.trim()) {
            continue;
        }
        let row = null;
        try {
            row = JSON.parse(line);
        } catch (err) {
            throw new Error(`bad_json_line at ${lineNo}: ${err.message}`);
        }
        await onRow(row, lineNo);
    }
}

function writeLine(stream, line) {
    return new Promise((resolve, reject) => {
        const ok = stream.write(line + "\n", "utf8");
        if (ok) {
            resolve();
            return;
        }
        const onDrain = () => {
            stream.off("error", onError);
            resolve();
        };
        const onError = (err) => {
            stream.off("drain", onDrain);
            reject(err);
        };
        stream.once("drain", onDrain);
        stream.once("error", onError);
    });
}

function deriveEpisodeTs(row, fallback) {
    const ts = Number(row && row.ts);
    return Number.isFinite(ts) ? ts : fallback;
}

async function main() {
    const inputPath = path.resolve(getArg("input", getArg("dataset-path", "output/ml/datasets/combat_phase0_v4_suddendeath.jsonl")));
    const outputPath = path.resolve(getArg("output", getArg("out-dataset", inputPath.replace(/\.jsonl$/i, "_credit.jsonl"))));
    const reportPath = path.resolve(getArg("report-path", `output/ml/reports/combat_phase0_credit_${Date.now()}.json`));

    const gamma = clamp(asFloat(getArg("credit-gamma", "0.995"), 0.995), 0.8, 0.9999);
    const creditHorizonMs = Math.max(1000, asInt(getArg("credit-horizon-ms", "8000"), 8000));
    const rewardBlendAlpha = clamp(asFloat(getArg("reward-blend-alpha", "0.65"), 0.65), 0, 1);
    const rewardMin = asFloat(getArg("reward-clip-min", "-3"), -3);
    const rewardMax = asFloat(getArg("reward-clip-max", "3"), 3);

    const wDanger = asFloat(getArg("w-danger", "0.35"), 0.35);
    const wItem = asFloat(getArg("w-item", "0.25"), 0.25);
    const wKill = asFloat(getArg("w-kill", "0.85"), 0.85);
    const wEscape = asFloat(getArg("w-escape", "0.30"), 0.30);
    const wSelfPenalty = asFloat(getArg("w-self-penalty", "1.10"), 1.10);

    if (!fs.existsSync(inputPath)) {
        throw new Error(`input dataset missing: ${inputPath}`);
    }

    ensureParent(outputPath);
    ensureParent(reportPath);

    let creditNonZero = 0;
    let rewardDenseSum = 0;
    let rewardTrainSum = 0;
    let rewardRawSum = 0;
    let terminalRows = 0;
    let processedRows = 0;
    let episodeCount = 0;
    let splitByTsReset = 0;
    let lineIndex = 0;

    const segmentCounterByEpisode = new Map();
    const activeSegments = new Map();

    const outStream = fs.createWriteStream(outputPath, { flags: "w" });

    async function flushSegment(baseEpisodeId, segment) {
        if (!segment || !Array.isArray(segment.rows) || segment.rows.length === 0) {
            return;
        }
        const items = segment.rows;
        const dense = new Array(items.length).fill(0);
        const done = new Array(items.length).fill(false);
        const ts = new Array(items.length).fill(0);

        for (let p = 0; p < items.length; p++) {
            const row = items[p];
            const rewardRaw = Number(row && row.reward);
            const raw = Number.isFinite(rewardRaw) ? rewardRaw : 0;
            const danger = getDangerScore(row);
            const itemGain = deriveItemGain(row);
            const kill = getKillScore(row);
            const escape = getEscapeScore(row);
            const selfPenalty = getSelfKillPenalty(row);
            dense[p] = raw
                + wDanger * danger
                + wItem * itemGain
                + wKill * kill
                + wEscape * escape
                - wSelfPenalty * selfPenalty;
            done[p] = !!(row && row.done);
            ts[p] = deriveEpisodeTs(row, segment.ts[p]);
        }

        for (let p = 0; p < items.length; p++) {
            let acc = 0;
            let g = 1;
            for (let q = p; q < items.length; q++) {
                if (ts[q] - ts[p] > creditHorizonMs) {
                    break;
                }
                acc += g * dense[q];
                if (done[q]) {
                    break;
                }
                g *= gamma;
            }

            const row = items[p] || {};
            const raw = Number.isFinite(Number(row.reward)) ? Number(row.reward) : 0;
            const denseReward = dense[p];
            const rewardTrain = clamp(
                (1 - rewardBlendAlpha) * denseReward + rewardBlendAlpha * acc,
                rewardMin,
                rewardMax
            );

            row.meta = row.meta && typeof row.meta === "object" ? row.meta : {};
            row.meta.reward_raw = compact(raw);
            row.meta.reward_dense = compact(denseReward);
            row.meta.return_discounted = compact(acc);
            row.meta.credit_horizon_ms = creditHorizonMs;
            row.meta.sequence_index = p;
            row.meta.credit_gamma = gamma;
            row.meta.reward_blend_alpha = rewardBlendAlpha;
            row.meta.credit_episode_key = `${baseEpisodeId}#${segment.index}`;
            row.reward_train = compact(rewardTrain);
            row.reward = row.reward_train;

            await writeLine(outStream, JSON.stringify(row));
            processedRows += 1;
            rewardRawSum += raw;
            rewardDenseSum += denseReward;
            rewardTrainSum += rewardTrain;
            if (Math.abs(acc - denseReward) > 1e-6) {
                creditNonZero += 1;
            }
            if (row && row.done) {
                terminalRows += 1;
            }
        }
        episodeCount += 1;
    }

    await eachJsonlRow(inputPath, async (row) => {
        lineIndex += 1;
        const baseEpisodeId = String(row && row.episode_id ? row.episode_id : "runtime");
        const ts = deriveEpisodeTs(row, lineIndex);
        let segment = activeSegments.get(baseEpisodeId);
        if (!segment) {
            const idx = segmentCounterByEpisode.get(baseEpisodeId) || 0;
            segment = { index: idx, rows: [], ts: [], lastTs: ts };
            activeSegments.set(baseEpisodeId, segment);
        } else if (segment.rows.length > 0) {
            const previousDone = !!(segment.rows[segment.rows.length - 1] && segment.rows[segment.rows.length - 1].done);
            const tsReset = ts < segment.lastTs;
            if (previousDone || tsReset) {
                await flushSegment(baseEpisodeId, segment);
                const nextIdx = (segmentCounterByEpisode.get(baseEpisodeId) || segment.index) + 1;
                segmentCounterByEpisode.set(baseEpisodeId, nextIdx);
                if (tsReset) {
                    splitByTsReset += 1;
                }
                segment = { index: nextIdx, rows: [], ts: [], lastTs: ts };
                activeSegments.set(baseEpisodeId, segment);
            }
        }

        segment.rows.push(row);
        segment.ts.push(ts);
        segment.lastTs = ts;

        if (row && row.done) {
            await flushSegment(baseEpisodeId, segment);
            const nextIdx = (segmentCounterByEpisode.get(baseEpisodeId) || segment.index) + 1;
            segmentCounterByEpisode.set(baseEpisodeId, nextIdx);
            activeSegments.delete(baseEpisodeId);
        }
    });

    for (const [baseEpisodeId, segment] of activeSegments.entries()) {
        await flushSegment(baseEpisodeId, segment);
    }

    await new Promise((resolve, reject) => {
        outStream.end((err) => {
            if (err) reject(err);
            else resolve();
        });
    });

    if (!processedRows) {
        throw new Error("input dataset is empty");
    }

    const report = {
        ts: Date.now(),
        input_path: inputPath,
        output_path: outputPath,
        rows: processedRows,
        episodes: episodeCount,
        split_by_ts_reset: splitByTsReset,
        terminal_rows: terminalRows,
        credit_nonzero_rows: creditNonZero,
        credit_nonzero_ratio: processedRows > 0 ? creditNonZero / processedRows : 0,
        reward_raw_mean: processedRows > 0 ? rewardRawSum / processedRows : 0,
        reward_dense_mean: processedRows > 0 ? rewardDenseSum / processedRows : 0,
        reward_train_mean: processedRows > 0 ? rewardTrainSum / processedRows : 0,
        config: {
            credit_gamma: gamma,
            credit_horizon_ms: creditHorizonMs,
            reward_blend_alpha: rewardBlendAlpha,
            reward_clip_min: rewardMin,
            reward_clip_max: rewardMax,
            w_danger: wDanger,
            w_item: wItem,
            w_kill: wKill,
            w_escape: wEscape,
            w_self_penalty: wSelfPenalty,
        },
    };
    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));

    console.log("[DONE]", JSON.stringify({
        output_path: outputPath,
        report_path: reportPath,
        rows: processedRows,
        credit_nonzero_ratio: report.credit_nonzero_ratio,
    }));
}

main().catch((err) => {
    console.error(err && err.stack ? err.stack : String(err));
    process.exit(1);
});
