#!/usr/bin/env node
const fs = require("fs");
const path = require("path");
const { execSync } = require("child_process");

const ROOT = path.resolve(__dirname, "..");

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

function run(cmd) {
    console.log("[RUN]", cmd);
    execSync(cmd, { cwd: ROOT, stdio: "inherit" });
}

function ensureServer(baseUrl) {
    try {
        execSync(`curl -fsS "${baseUrl}" >/dev/null`, { cwd: ROOT, stdio: "ignore" });
    } catch (_) {
        throw new Error(`game server is not reachable at ${baseUrl}. run npm start first.`);
    }
}

function readJsonMaybe(p) {
    if (!fs.existsSync(p)) return null;
    return JSON.parse(fs.readFileSync(p, "utf8"));
}

function summarizeEvalReport(report) {
    const s = report && report.summary ? report.summary : {};
    const elo = s && s.elo_result ? s.elo_result : {};
    return {
        win_rate: Number(s.win_rate || 0),
        self_kill_rate: Number(s.self_kill_rate || 0),
        draw_rate: Number(s.draw_rate || 0),
        item_control: Number(s.item_control || 0),
        sequence_path_hit_rate: Number(s.sequence_path_hit_rate || 0),
        elo_delta: Number(elo.ai_delta || 0),
    };
}

function delta(cur, prev) {
    const out = {};
    for (const k of Object.keys(cur)) {
        out[k + "_delta"] = Number(cur[k] || 0) - Number(prev[k] || 0);
    }
    return out;
}

function getNested(obj, keys, fallback) {
    let cur = obj;
    for (const k of keys) {
        if (!cur || typeof cur !== "object" || !(k in cur)) {
            return fallback;
        }
        cur = cur[k];
    }
    return cur;
}

function main() {
    const ts = Date.now();
    const baseUrl = getArg("base-url", "http://127.0.0.1:4000/");
    const mapId = getArg("map", "windmill-heart");
    const workers = asPositiveInt(getArg("workers", "10"), 10);
    const targetFrames = asPositiveInt(getArg("target-frames", "200000"), 200000);
    const maxWallSec = asPositiveInt(getArg("collect-max-wall-sec", "7200"), 7200);
    const runs = asPositiveInt(getArg("runs", "50"), 50);
    const parallel = asPositiveInt(getArg("parallel", "8"), 8);
    const liveView = getArg("live-view", "1");
    const baselineEvalEnabled = getArg("baseline-eval", "1") !== "0";
    const collectMaxAttempts = Math.max(1, asPositiveInt(getArg("collect-max-attempts", "4"), 4));
    const collectRetryDelayMs = Math.max(0, asPositiveInt(getArg("collect-retry-delay-ms", "5000"), 5000));
    const collectMinFinalRatio = Math.max(0.90, Math.min(1.0, asFloat(getArg("collect-min-final-ratio", "0.98"), 0.98)));
    const collectMatchDurationSec = Math.max(12, asPositiveInt(getArg("collect-match-duration-sec", "18"), 18));
    const collectPollMs = Math.max(120, asPositiveInt(getArg("collect-poll-ms", "250"), 250));
    const collectBatchSize = Math.max(512, asPositiveInt(getArg("collect-batch-size", "4096"), 4096));
    const collectThinkIntervalMs = Math.max(1, asPositiveInt(getArg("collect-think-interval-ms", "4"), 4));
    const collectBurstCapOngoing = Math.max(6, asPositiveInt(getArg("collect-burst-cap-ongoing", "20"), 20));
    const collectBalanced = getArg("collect-balanced", "1") !== "0";
    const collectOfflineMicro = getArg("collect-offline-micro", "0") !== "0";
    const collectBehaviorScoring = getArg("collect-behavior-scoring", collectBalanced ? "1" : "0") !== "0";
    const collectEarlyCommitHighValue = getArg("collect-early-commit-high-value", collectBalanced ? "1" : "0") !== "0";
    const collectMicroScenario = getArg("collect-micro-scenario", collectBalanced ? "1" : "0") !== "0";
    const collectMicroSamplesPerEpisode = Math.max(0, asInt(getArg("collect-micro-samples-per-episode", collectBalanced ? "32" : "0"), collectBalanced ? 32 : 0));
    const collectMirrorSampling = getArg("collect-mirror-sampling", collectBalanced ? "1" : "0") !== "0";
    const collectSpawnRandomize = getArg("collect-spawn-randomize", collectBalanced ? "1" : "0") !== "0";

    const sequenceLen = asPositiveInt(getArg("sequence-len", "8"), 8);
    const sequenceStride = asPositiveInt(getArg("sequence-stride", "1"), 1);
    const creditGamma = asFloat(getArg("credit-gamma", "0.995"), 0.995);
    const creditHorizonMs = asPositiveInt(getArg("credit-horizon-ms", "8000"), 8000);
    const rewardBlendAlpha = asFloat(getArg("reward-blend-alpha", "0.65"), 0.65);

    const rawDataset = path.resolve(getArg("raw-dataset", `output/ml/datasets/combat_phase0_seq_raw_${ts}.jsonl`));
    const creditDataset = path.resolve(getArg("credit-dataset", `output/ml/datasets/combat_phase0_seq_credit_${ts}.jsonl`));
    const collectReport = path.resolve(getArg("collect-report", `output/ml/reports/combat_phase0_seq_collect_${ts}.json`));
    const creditReport = path.resolve(getArg("credit-report", `output/ml/reports/combat_phase0_seq_credit_${ts}.json`));
    const trainReport = path.resolve(getArg("train-report", `output/ml/reports/combat_phase0_seq_train_${ts}.json`));
    const evalReport = path.resolve(getArg("eval-report", `output/ml/reports/combat_phase0_seq_eval_${ts}.json`));
    const compareReport = path.resolve(getArg("compare-report", `output/ml/reports/combat_phase0_seq_compare_${ts}.json`));
    const baselineEvalReport = path.resolve(getArg("baseline-eval-report", `output/ml/reports/combat_phase0_seq_baseline_eval_${ts}.json`));

    const initPt = path.resolve(getArg("init-pt", "output/ml/models/dodge_iql_v1.pt"));
    const outPt = path.resolve(getArg("out-pt", "output/ml/models/combat_phase0_iql_phase0_seq.pt"));
    const outOnnx = path.resolve(getArg("out-onnx", "output/ml/models/combat_phase0_iql_phase0_seq.onnx"));
    const modelUrl = getArg("model-url", "/output/ml/models/combat_phase0_iql_phase0_seq.onnx");
    const baselineModelUrl = getArg("baseline-model-url", "/output/ml/models/combat_phase0_iql_v4_suddendeath.onnx");

    [
        rawDataset,
        creditDataset,
        collectReport,
        creditReport,
        trainReport,
        evalReport,
        compareReport,
        baselineEvalReport,
        outPt,
        outOnnx,
    ].forEach(ensureParent);

    ensureServer(baseUrl);

    const collectCmd =
        "node scripts/collect-combat-dataset-parallel.js "
        + `--workers=${workers} --target-frames=${targetFrames} --max-wall-sec=${maxWallSec} `
        + `--arena=1v1 --map=${mapId} --action-space=discrete6 --fresh=1 `
        + `--dataset-path=${rawDataset} --report-path=${collectReport} `
        + `--min-final-ratio=${collectMinFinalRatio} `
        + `--match-duration-sec=${collectMatchDurationSec} `
        + `--poll-ms=${collectPollMs} --batch-size=${collectBatchSize} --think-interval-ms=${collectThinkIntervalMs} `
        + `--burst-cap-ongoing=${collectBurstCapOngoing} `
        + `--balanced=${collectBalanced ? 1 : 0} --offline-micro=${collectOfflineMicro ? 1 : 0} `
        + `--behavior-scoring=${collectBehaviorScoring ? 1 : 0} --early-commit-high-value=${collectEarlyCommitHighValue ? 1 : 0} `
        + `--micro-scenario=${collectMicroScenario ? 1 : 0} --micro-samples-per-episode=${collectMicroSamplesPerEpisode} `
        + `--mirror-sampling=${collectMirrorSampling ? 1 : 0} --spawn-randomize=${collectSpawnRandomize ? 1 : 0} `
        + "--clear-nonrigid=1 --sudden-death=1 --disable-revive=1 --ignore-enemy-self-kill=1 "
        + "--stall-no-progress-ms=12000 --partial-clear-min-ratio=0.35 --partial-clear-max-ratio=0.75 "
        + "--spawn-shortest-path-min=1 --spawn-shortest-path-max=10 "
        + "--opponent-pool=heuristic_v2,aggressive_trapper --agent-pool=heuristic_v2,aggressive_trapper --agent-expert-duel=1";

    let collectOk = false;
    for (let attempt = 1; attempt <= collectMaxAttempts; attempt++) {
        console.log(`[INFO] collect attempt ${attempt}/${collectMaxAttempts}`);
        try {
            run(collectCmd);
        } catch (err) {
            if (attempt >= collectMaxAttempts) {
                throw err;
            }
            console.warn("[WARN] collect command failed, retry:", String(err && err.message ? err.message : err));
            if (collectRetryDelayMs > 0) {
                execSync(`sleep ${Math.max(0, Math.round(collectRetryDelayMs / 1000))}`, { cwd: ROOT, stdio: "inherit" });
            }
            continue;
        }
        const collect = readJsonMaybe(collectReport) || {};
        const rows = Number(getNested(collect, ["merge", "rows_written"], 0)) || 0;
        const minRequiredRows = Math.max(1, Math.floor(targetFrames * collectMinFinalRatio));
        const collectionComplete = getNested(collect, ["collection_complete"], rows >= minRequiredRows) === true;
        if (collectionComplete && rows >= minRequiredRows) {
            collectOk = true;
            break;
        }
        console.warn("[WARN] collect report insufficient rows:", JSON.stringify({
            rows_written: rows,
            min_required_rows: minRequiredRows,
            collection_complete: collectionComplete,
            worker_failure_count: Number(getNested(collect, ["worker_failure_count"], 0)) || 0,
            attempt,
        }));
        if (attempt < collectMaxAttempts && collectRetryDelayMs > 0) {
            execSync(`sleep ${Math.max(0, Math.round(collectRetryDelayMs / 1000))}`, { cwd: ROOT, stdio: "inherit" });
        }
    }
    if (!collectOk) {
        throw new Error("collect_failed_after_retries");
    }

    run(
        "node scripts/postprocess-combat-credit.js "
        + `--input=${rawDataset} --output=${creditDataset} --report-path=${creditReport} `
        + `--credit-gamma=${creditGamma} --credit-horizon-ms=${creditHorizonMs} --reward-blend-alpha=${rewardBlendAlpha}`
    );

    run(
        "python3 ml/train_iql_combat.py "
        + `--dataset ${creditDataset} --init-pt ${initPt} `
        + `--sequence-len ${sequenceLen} --sequence-stride ${sequenceStride} `
        + `--credit-gamma ${creditGamma} --credit-horizon-ms ${creditHorizonMs} --reward-blend-alpha ${rewardBlendAlpha} `
        + `--out-pt ${outPt} --out-onnx ${outOnnx} --out-metrics ${trainReport}`
    );

    if (baselineEvalEnabled && baselineModelUrl && String(baselineModelUrl).trim()) {
        try {
            run(
                "node scripts/eval-combat-1v1.js "
                + `--model-url=${baselineModelUrl} --opponent=heuristic_v2 --runs=${runs} --parallel=${parallel} --map=${mapId} `
                + "--match-duration-sec=45 --clear-nonrigid=1 --sudden-death=1 --disable-revive=1 --ignore-enemy-self-kill=1 "
                + "--stall-no-progress-ms=12000 --partial-clear-min-ratio=0.35 --partial-clear-max-ratio=0.75 "
                + "--spawn-shortest-path-min=1 --spawn-shortest-path-max=10 --random-item-density=0.12 "
                + `--live-view=${liveView} --report-path=${baselineEvalReport}`
            );
        } catch (err) {
            console.warn("[WARN] baseline eval failed, continue:", String(err && err.message ? err.message : err));
        }
    }

    run(
        "node scripts/eval-combat-1v1.js "
        + `--model-url=${modelUrl} --opponent=heuristic_v2 --runs=${runs} --parallel=${parallel} --map=${mapId} `
        + "--match-duration-sec=45 --clear-nonrigid=1 --sudden-death=1 --disable-revive=1 --ignore-enemy-self-kill=1 "
        + "--stall-no-progress-ms=12000 --partial-clear-min-ratio=0.35 --partial-clear-max-ratio=0.75 "
        + "--spawn-shortest-path-min=1 --spawn-shortest-path-max=10 --random-item-density=0.12 "
        + `--live-view=${liveView} --report-path=${evalReport}`
    );

    const credit = readJsonMaybe(creditReport) || {};
    const trainedEval = summarizeEvalReport(readJsonMaybe(evalReport));
    const baselineEval = fs.existsSync(baselineEvalReport)
        ? summarizeEvalReport(readJsonMaybe(baselineEvalReport))
        : null;

    const compare = {
        ts: Date.now(),
        map_id: mapId,
        protocol: {
            workers,
            target_frames: targetFrames,
            collect_max_wall_sec: maxWallSec,
            runs,
            parallel,
            sequence_len: sequenceLen,
            sequence_stride: sequenceStride,
            credit_gamma: creditGamma,
            credit_horizon_ms: creditHorizonMs,
            reward_blend_alpha: rewardBlendAlpha,
            eval_opponent: "heuristic_v2",
            eval_live_view: liveView !== "0",
        },
        datasets: {
            raw_dataset: rawDataset,
            credit_dataset: creditDataset,
            collect_report: collectReport,
            credit_report: creditReport,
            credit_nonzero_ratio: Number(credit.credit_nonzero_ratio || 0),
        },
        models: {
            init_pt: initPt,
            out_pt: outPt,
            out_onnx: outOnnx,
            model_url: modelUrl,
            baseline_model_url: baselineModelUrl,
        },
        reports: {
            train_report: trainReport,
            eval_report: evalReport,
            baseline_eval_report: fs.existsSync(baselineEvalReport) ? baselineEvalReport : null,
            compare_report: compareReport,
        },
        baseline_eval: baselineEval,
        trained_eval: trainedEval,
        delta_vs_baseline: baselineEval ? delta(trainedEval, baselineEval) : null,
        gates: {
            credit_nonzero_ratio_gt_0_30: Number(credit.credit_nonzero_ratio || 0) > 0.30,
            sequence_path_hit_rate_gt_0_99: Number(trainedEval.sequence_path_hit_rate || 0) > 0.99,
        },
    };

    fs.writeFileSync(compareReport, JSON.stringify(compare, null, 2));
    console.log("[DONE]", JSON.stringify({
        compare_report: compareReport,
        credit_nonzero_ratio: compare.datasets.credit_nonzero_ratio,
        sequence_path_hit_rate: trainedEval.sequence_path_hit_rate,
    }));
}

main();
