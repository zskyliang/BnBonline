const fs = require("fs");
const path = require("path");
const { execSync } = require("child_process");

const ROOT = path.resolve(__dirname, "..");
const DATASET_PATH = path.join(ROOT, "output/ml/datasets/dodge_bc_v1.jsonl");
const COLLECT_REPORT = path.join(ROOT, "output/ml/reports/dodge_bc_v1_collect_stats.json");
const TRAIN_REPORT = path.join(ROOT, "output/ml/reports/bc_v1_metrics.json");
const EVAL_REPORT = path.join(ROOT, "output/ml/reports/dodge_ml_eval_report.json");
const ROUND_REPORT = path.join(ROOT, "output/ml/reports/bc_v1_1_rounds.json");

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
    return Number.isFinite(n) && n > 0 ? n : fallback;
}

function asFloat(v, fallback) {
    const n = parseFloat(v);
    return Number.isFinite(n) ? n : fallback;
}

function countLines(filePath) {
    if (!fs.existsSync(filePath)) {
        return 0;
    }
    try {
        const raw = execSync(`wc -l < "${filePath}"`, { cwd: ROOT, encoding: "utf-8" });
        return asInt((raw || "").trim(), 0);
    } catch (err) {
        return 0;
    }
}

function readJson(filePath) {
    if (!fs.existsSync(filePath)) {
        return null;
    }
    return JSON.parse(fs.readFileSync(filePath, "utf-8"));
}

function run(cmd) {
    console.log("[RUN]", cmd);
    execSync(cmd, { cwd: ROOT, stdio: "inherit" });
}

function ensureServer() {
    const baseUrl = getArg("base-url", "http://127.0.0.1:4000/");
    const curlCmd = `curl -fsS "${baseUrl}" >/dev/null`;
    try {
        execSync(curlCmd, { cwd: ROOT, stdio: "ignore" });
    } catch (err) {
        throw new Error(
            "game server is not reachable at " +
            baseUrl +
            ". Please run `npm start` in another terminal first."
        );
    }
}

function main() {
    const targets = getArg("targets", "600000,900000,1200000")
        .split(",")
        .map((s) => asInt(s.trim(), 0))
        .filter((n) => n > 0);
    const survivalTarget = asFloat(getArg("survival-target", "0.95"), 0.95);
    const evalRuns = asInt(getArg("eval-runs", "3"), 3);
    const evalDurationMs = asInt(getArg("eval-duration-ms", "60000"), 60000);
    const resume = getArg("resume", "0") === "1";

    if (targets.length === 0) {
        throw new Error("no valid targets");
    }

    ensureServer();
    fs.mkdirSync(path.dirname(ROUND_REPORT), { recursive: true });

    let roundSummaries = [];
    let currentRows = resume ? countLines(DATASET_PATH) : 0;
    if (!resume && fs.existsSync(DATASET_PATH)) {
        fs.unlinkSync(DATASET_PATH);
    }

    for (let i = 0; i < targets.length; i++) {
        const target = targets[i];
        const need = Math.max(0, target - currentRows);
        if (need > 0) {
            const freshFlag = currentRows === 0 ? "1" : "0";
            run(
                `node scripts/collect-dodge-dataset.js --target-frames=${need} --fresh=${freshFlag}`
            );
            currentRows = countLines(DATASET_PATH);
        }

        run("python3 ml/train_bc.py --dataset output/ml/datasets/dodge_bc_v1.jsonl");
        run(
            `node scripts/eval-ml-dodge.js --duration-ms=${evalDurationMs} --runs=${evalRuns} --model-url=/output/ml/models/dodge_bc_v1.onnx`
        );

        const collectReport = readJson(COLLECT_REPORT) || {};
        const trainReport = readJson(TRAIN_REPORT) || {};
        const evalReport = readJson(EVAL_REPORT) || {};
        const modelMean =
            evalReport &&
            evalReport.model &&
            typeof evalReport.model.survival_rate_mean === "number"
                ? evalReport.model.survival_rate_mean
                : null;

        const summary = {
            round: i + 1,
            target_rows: target,
            dataset_rows: currentRows,
            collect: {
                spawned_bubbles_effective: collectReport.spawned_bubbles_effective || collectReport.spawned_bubbles || 0,
                spawned_bubbles_ignored_trapped: collectReport.spawned_bubbles_ignored_trapped || 0,
                bombed_count: collectReport.bombed_count || 0,
                survival_rate: collectReport.survival_rate
            },
            train: {
                dataset_size_used: trainReport.dataset_size_used || 0,
                val_acc: trainReport.val_acc || 0,
                val_macro_f1: trainReport.val_macro_f1 || 0
            },
            eval: {
                runs: evalReport.runs || 0,
                model_survival_rate_mean: modelMean,
                model_fallback_rate_mean:
                    evalReport &&
                    evalReport.model &&
                    typeof evalReport.model.fallback_rate_mean === "number"
                        ? evalReport.model.fallback_rate_mean
                        : 0,
                baseline_survival_rate_mean:
                    evalReport &&
                    evalReport.baseline &&
                    typeof evalReport.baseline.survival_rate_mean === "number"
                        ? evalReport.baseline.survival_rate_mean
                        : 0
            }
        };
        roundSummaries.push(summary);
        fs.writeFileSync(
            ROUND_REPORT,
            JSON.stringify(
                {
                    ts: Date.now(),
                    survival_target: survivalTarget,
                    targets,
                    rounds: roundSummaries
                },
                null,
                2
            )
        );

        if (typeof modelMean === "number" && modelMean >= survivalTarget) {
            console.log(
                "[STOP] survival target reached at round " +
                    (i + 1) +
                    ": " +
                    modelMean.toFixed(4) +
                    " >= " +
                    survivalTarget
            );
            return;
        }
    }

    console.log("[DONE] reached max rounds without hitting target.");
}

main();
