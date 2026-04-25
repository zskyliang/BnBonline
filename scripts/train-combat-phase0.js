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

function run(cmd) {
    console.log("[RUN]", cmd);
    execSync(cmd, { cwd: ROOT, stdio: "inherit" });
}

function runQuiet(cmd) {
    return execSync(cmd, { cwd: ROOT, stdio: ["ignore", "pipe", "pipe"] }).toString("utf8");
}

function ensureServer(baseUrl) {
    try {
        execSync(`curl -fsS "${baseUrl}" >/dev/null`, { cwd: ROOT, stdio: "ignore" });
    } catch (_) {
        throw new Error(`game server is not reachable at ${baseUrl}. please run npm start first.`);
    }
}

function ensureParent(p) {
    fs.mkdirSync(path.dirname(p), { recursive: true });
}

function readJsonMaybe(p) {
    if (!fs.existsSync(p)) return null;
    return JSON.parse(fs.readFileSync(p, "utf8"));
}

function countLines(filePath) {
    if (!fs.existsSync(filePath)) return 0;
    const out = runQuiet(`wc -l "${filePath}"`);
    const n = parseInt(String(out).trim().split(/\s+/)[0], 10);
    return Number.isFinite(n) ? n : 0;
}

function sampleLines(filePath, targetCount, seed) {
    if (!fs.existsSync(filePath) || targetCount <= 0) return [];
    const lines = fs.readFileSync(filePath, "utf8").split(/\n/).filter(Boolean);
    if (lines.length <= targetCount) return lines;
    let s = (seed >>> 0) || 1;
    function rnd() {
        s = (s * 1664525 + 1013904223) >>> 0;
        return s / 0x100000000;
    }
    for (let i = lines.length - 1; i > 0; i--) {
        const j = Math.floor(rnd() * (i + 1));
        const t = lines[i];
        lines[i] = lines[j];
        lines[j] = t;
    }
    return lines.slice(0, targetCount);
}

function mixDatasets(specialPath, legacyPath, legacyRows, outPath, seed) {
    if (!fs.existsSync(specialPath)) {
        throw new Error(`special dataset not found: ${specialPath}`);
    }
    const specialLines = fs.readFileSync(specialPath, "utf8").split(/\n/).filter(Boolean);
    const legacyLines = sampleLines(legacyPath, legacyRows, seed ^ 0x9e3779b9);
    const merged = specialLines.concat(legacyLines);
    let s = (seed >>> 0) || 1;
    function rnd() {
        s = (s * 1103515245 + 12345) >>> 0;
        return s / 0x100000000;
    }
    for (let i = merged.length - 1; i > 0; i--) {
        const j = Math.floor(rnd() * (i + 1));
        const t = merged[i];
        merged[i] = merged[j];
        merged[j] = t;
    }
    ensureParent(outPath);
    fs.writeFileSync(outPath, merged.join("\n") + "\n");
    return {
        special_rows: specialLines.length,
        legacy_rows: legacyLines.length,
        mixed_rows: merged.length,
    };
}

function evalMetricPack(report) {
    const s = (report && report.summary) || {};
    const elo = (s && s.elo_result) || {};
    return {
        win_rate: Number(s.win_rate || 0),
        self_kill_rate: Number(s.self_kill_rate || 0),
        draw_rate: Number(s.draw_rate || 0),
        item_control: Number(s.item_control || 0),
        elo_delta: Number(elo.ai_delta || 0),
        kills_per_min: Number(s.kills_per_min || 0),
        self_kills_per_min: Number(s.self_kills_per_min || 0),
        net_kd: Number(s.net_kd || 0),
        danger_cells_per_min: Number(s.danger_cells_per_min || 0),
    };
}

function delta(cur, prev) {
    const out = {};
    for (const k of Object.keys(cur)) {
        out[k + "_delta"] = Number(cur[k] || 0) - Number(prev[k] || 0);
    }
    return out;
}

function collectQualityFromParallelReport(report) {
    const m = report && report.merge ? report.merge : {};
    const rows = Number(m.rows_written || 0);
    const action5 = Number((m.action_hist_written || {})["5"] || 0);
    const selfKill = Number((m.outcome_hist_written || {}).self_kill || 0);
    const illegalActionRows = Number(report && report.worker_reports
        ? report.worker_reports
            .map((x) => (x.report && Number(x.report.illegal_action_rows || 0)) || 0)
            .reduce((a, b) => a + b, 0)
        : 0);
    return {
        rows_written: rows,
        rows_per_sec: Number(report && report.rows_per_sec || 0),
        action5_ratio: rows > 0 ? action5 / rows : 0,
        self_kill_rows: selfKill,
        danger_nonzero_rows: Number(((m.aux_label_hist_written || {}).danger_cells_created_score_nonzero) || 0),
        self_kill_penalty_rows: Number(((m.aux_label_hist_written || {}).round_self_kill_penalty_nonzero) || 0),
        illegal_action_rows: illegalActionRows,
    };
}

function verifyGate(q, targetRows) {
    return q.rows_written >= targetRows
        && q.rows_per_sec >= 20
        && q.action5_ratio >= 0.08
        && (q.self_kill_rows > 0 || q.self_kill_penalty_rows > 0)
        && q.danger_nonzero_rows > 0
        && q.illegal_action_rows === 0;
}

function buildCollectCmd(opts) {
    return "node scripts/collect-combat-dataset-parallel.js "
        + `--workers=${opts.workers} --target-frames=${opts.targetFrames} --max-wall-sec=${opts.maxWallSec} `
        + `--arena=1v1 --map=${opts.mapId} --action-space=discrete6 --fresh=1 `
        + `--dataset-path=${opts.datasetPath} --report-path=${opts.reportPath} `
        + `--clear-nonrigid=1 --random-item-density=${opts.randomItemDensity} --random-item-density-jitter=0 `
        + `--item-respawn-ms=2000 --item-respawn-jitter-ratio=0 --item-safe-radius=2 --item-safe-radius-jitter=0 --item-max=${opts.itemMax} `
        + `--opponent-pool=stationary_dummy --agent-pool=heuristic_v2,aggressive_trapper --agent-expert-duel=1 `
        + `--balanced=1 --spawn-randomize=1 --scenario-buckets=open_random,escape_after_bomb,enemy_choke,item_race,deadend_chase `
        + `--soft-obstacle-reinject-ratio=0 --partial-clear-min-ratio=1 --partial-clear-max-ratio=1 `
        + `--soft-obstacle-keep-min=0 --soft-obstacle-keep-max=0 --spawn-shortest-path-min=1 --spawn-shortest-path-max=10 `
        + `--special-bomb-escape=1 --special-round-sec=60 --special-item-respawn-ms=2000 `
        + `--respawn-delay-ms=0 --respawn-invincible-ms=300 --no-global-dedupe=1 `
        + `--danger-score-weight=${opts.dangerScoreWeight} --kill-score-weight=${opts.killScoreWeight} --self-kill-penalty-weight=${opts.selfKillPenaltyWeight} `
        + `--sudden-death=0 --disable-revive=0 --ignore-enemy-self-kill=0 --match-duration-sec=60 `
        + `--burst-cap-ongoing=6 --behavior-scoring=1 --behavior-score-threshold=0.12 --high-value-behavior-threshold=0.35 --behavior-credit-window-ms=6000 `
        + `--offline-micro=0 --micro-scenario=0 --screenshot=0`;
}

function buildEvalCmd(opts) {
    return "node scripts/eval-combat-1v1.js "
        + `--eval-profile=${opts.profile} --model-url=${opts.modelUrl} --runs=${opts.runs} --parallel=${opts.parallel} `
        + `--map=${opts.mapId} --report-path=${opts.reportPath} --live-view=${opts.liveView} `
        + `--clear-nonrigid=1 --random-item-density=0.12 --random-item-density-jitter=0 --item-respawn-ms=2000 `
        + `--item-respawn-jitter-ratio=0 --item-safe-radius=2 --item-safe-radius-jitter=0 --item-max=24 `
        + `--special-bomb-escape=${opts.profile === "special_respawn_duel" ? 1 : 0} --special-round-sec=60 `
        + `--special-item-respawn-ms=2000 --respawn-delay-ms=0 --respawn-invincible-ms=300 `
        + `--opponent=${opts.profile === "special_respawn_duel" ? "stationary_dummy" : "heuristic_v2"} `
        + `--match-duration-sec=${opts.profile === "special_respawn_duel" ? 60 : 45}`;
}

function trainMetricsPack(report) {
    return {
        val_policy_f1: Number(report && report.val_policy_f1 || 0),
        action5_recall: Number(report && report.action5_recall || 0),
        illegal_action_pred_rate: Number(report && report.illegal_action_pred_rate || 0),
    };
}

function main() {
    const ts = Date.now();
    const baseUrl = getArg("base-url", "http://127.0.0.1:4000/");
    const mapId = getArg("map", "windmill-heart");
    const runs = asPositiveInt(getArg("runs", "50"), 50);
    const parallel = asPositiveInt(getArg("parallel", "8"), 8);
    const workers = asPositiveInt(getArg("workers", "10"), 10);
    const verifyFrames = asPositiveInt(getArg("verify-frames", "10000"), 10000);
    const specialFrames = asPositiveInt(getArg("special-frames", "70000"), 70000);
    const legacyMixRows = asPositiveInt(getArg("legacy-mix-rows", "30000"), 30000);
    const collectVerifyWallSec = asPositiveInt(getArg("verify-max-wall-sec", "1800"), 1800);
    const collectWallSec = asPositiveInt(getArg("collect-max-wall-sec", "7200"), 7200);
    const epochs = asPositiveInt(getArg("epochs", "60"), 60);
    const batchSize = asPositiveInt(getArg("batch-size", "512"), 512);
    const freezeConvEpochs = asInt(getArg("freeze-conv-epochs", "50"), 50);
    const dangerScoreWeight = Number(getArg("danger-score-weight", "1.0"));
    const killScoreWeight = Number(getArg("kill-score-weight", "1.0"));
    const selfKillPenaltyWeight = Number(getArg("self-kill-penalty-weight", "2.0"));
    const evalLiveView = getArg("eval-live-view", "0");

    const preModelUrl = getArg("pre-model-url", "/output/ml/models/combat_phase0_iql_v1.onnx");
    const initPt = path.resolve(getArg("init-pt", "output/ml/models/dodge_iql_v1.pt"));
    const outPt = path.resolve(getArg("out-pt", "output/ml/models/combat_phase0_iql_special_v1.pt"));
    const outOnnx = path.resolve(getArg("out-onnx", "output/ml/models/combat_phase0_iql_special_v1.onnx"));
    const modelUrl = getArg("model-url", "/output/ml/models/combat_phase0_iql_special_v1.onnx");

    const verifyDataset = path.resolve(getArg("verify-dataset", `output/ml/datasets/combat_phase0_special_verify_${ts}.jsonl`));
    const specialDataset = path.resolve(getArg("special-dataset", `output/ml/datasets/combat_phase0_special_70k_${ts}.jsonl`));
    const mixedDataset = path.resolve(getArg("mixed-dataset", `output/ml/datasets/combat_phase0_special_mix_100k_${ts}.jsonl`));
    const legacyDataset = path.resolve(getArg("legacy-dataset", "output/ml/datasets/combat_phase0_v2_features24.jsonl"));

    const preEvalStandard = path.resolve(getArg("pre-eval-standard", `output/ml/reports/combat_phase0_pre_eval_standard_${ts}.json`));
    const preEvalSpecial = path.resolve(getArg("pre-eval-special", `output/ml/reports/combat_phase0_pre_eval_special_${ts}.json`));
    const verifyCollectReport = path.resolve(getArg("verify-collect-report", `output/ml/reports/combat_phase0_special_collect_verify_${ts}.json`));
    const collectReport = path.resolve(getArg("collect-report", `output/ml/reports/combat_phase0_special_collect_70k_${ts}.json`));
    const trainReport = path.resolve(getArg("train-report", `output/ml/reports/combat_phase0_special_train_${ts}.json`));
    const postEvalStandard = path.resolve(getArg("post-eval-standard", `output/ml/reports/combat_phase0_post_eval_standard_${ts}.json`));
    const postEvalSpecial = path.resolve(getArg("post-eval-special", `output/ml/reports/combat_phase0_post_eval_special_${ts}.json`));
    const compareReport = path.resolve(getArg("compare-report", `output/ml/reports/combat_phase0_compare_special_${ts}.json`));

    [
        verifyDataset,
        specialDataset,
        mixedDataset,
        preEvalStandard,
        preEvalSpecial,
        verifyCollectReport,
        collectReport,
        trainReport,
        postEvalStandard,
        postEvalSpecial,
        compareReport,
        outPt,
        outOnnx,
    ].forEach(ensureParent);

    ensureServer(baseUrl);

    run(buildEvalCmd({ profile: "standard", modelUrl: preModelUrl, runs, parallel, mapId, reportPath: preEvalStandard, liveView: evalLiveView }));
    run(buildEvalCmd({ profile: "special_respawn_duel", modelUrl: preModelUrl, runs, parallel, mapId, reportPath: preEvalSpecial, liveView: evalLiveView }));

    run(buildCollectCmd({
        workers,
        targetFrames: verifyFrames,
        maxWallSec: collectVerifyWallSec,
        mapId,
        datasetPath: verifyDataset,
        reportPath: verifyCollectReport,
        randomItemDensity: 0.12,
        itemMax: 24,
        dangerScoreWeight,
        killScoreWeight,
        selfKillPenaltyWeight,
    }));

    const verifyReport = readJsonMaybe(verifyCollectReport);
    const verifyQuality = collectQualityFromParallelReport(verifyReport);
    if (!verifyGate(verifyQuality, verifyFrames)) {
        throw new Error(`verify gate failed: ${JSON.stringify(verifyQuality)}`);
    }

    run(buildCollectCmd({
        workers,
        targetFrames: specialFrames,
        maxWallSec: collectWallSec,
        mapId,
        datasetPath: specialDataset,
        reportPath: collectReport,
        randomItemDensity: 0.12,
        itemMax: 24,
        dangerScoreWeight,
        killScoreWeight,
        selfKillPenaltyWeight,
    }));

    const mixStats = mixDatasets(specialDataset, legacyDataset, legacyMixRows, mixedDataset, ts >>> 0);
    const mixedRows = countLines(mixedDataset);
    if (mixedRows <= 0) {
        throw new Error("mixed dataset is empty");
    }

    const progressLog = trainReport + ".progress.log";
    run(
        "python3 ml/train_iql_combat.py "
        + `--dataset ${mixedDataset} --epochs ${epochs} --batch-size ${batchSize} `
        + `--freeze-conv-epochs ${freezeConvEpochs} --init-pt ${initPt} `
        + `--out-pt ${outPt} --out-onnx ${outOnnx} --out-metrics ${trainReport} --progress-log ${progressLog}`
    );

    run(buildEvalCmd({ profile: "standard", modelUrl, runs, parallel, mapId, reportPath: postEvalStandard, liveView: evalLiveView }));
    run(buildEvalCmd({ profile: "special_respawn_duel", modelUrl, runs, parallel, mapId, reportPath: postEvalSpecial, liveView: evalLiveView }));

    const preStd = evalMetricPack(readJsonMaybe(preEvalStandard));
    const preSpecial = evalMetricPack(readJsonMaybe(preEvalSpecial));
    const postStd = evalMetricPack(readJsonMaybe(postEvalStandard));
    const postSpecial = evalMetricPack(readJsonMaybe(postEvalSpecial));
    const trainMetrics = trainMetricsPack(readJsonMaybe(trainReport));

    const compare = {
        ts: Date.now(),
        map_id: mapId,
        pipeline: {
            runs,
            parallel,
            workers,
            verify_frames: verifyFrames,
            special_frames: specialFrames,
            legacy_mix_rows: legacyMixRows,
            epochs,
            batch_size: batchSize,
            freeze_conv_epochs: freezeConvEpochs,
        },
        datasets: {
            verify_dataset: verifyDataset,
            special_dataset: specialDataset,
            legacy_dataset: legacyDataset,
            mixed_dataset: mixedDataset,
            mix_stats: mixStats,
            mixed_rows: mixedRows,
            verify_quality: verifyQuality,
        },
        reports: {
            pre_eval_standard: preEvalStandard,
            pre_eval_special: preEvalSpecial,
            verify_collect_report: verifyCollectReport,
            collect_report: collectReport,
            train_report: trainReport,
            post_eval_standard: postEvalStandard,
            post_eval_special: postEvalSpecial,
            compare_report: compareReport,
            progress_log: progressLog,
        },
        train_metrics: trainMetrics,
        pre: {
            standard: preStd,
            special_respawn_duel: preSpecial,
        },
        post: {
            standard: postStd,
            special_respawn_duel: postSpecial,
        },
        delta: {
            standard: delta(postStd, preStd),
            special_respawn_duel: delta(postSpecial, preSpecial),
        },
        phase1_gate: {
            based_on_standard_eval: {
                win_rate_threshold: 0.35,
                self_kill_rate_threshold: 0.25,
                pass: postStd.win_rate >= 0.35 && postStd.self_kill_rate <= 0.25,
            },
        },
    };

    fs.writeFileSync(compareReport, JSON.stringify(compare, null, 2));
    console.log("[DONE]", JSON.stringify({
        compare_report: compareReport,
        mixed_dataset: mixedDataset,
        out_onnx: outOnnx,
        post_eval_standard: postEvalStandard,
        post_eval_special: postEvalSpecial,
    }));
}

main();
