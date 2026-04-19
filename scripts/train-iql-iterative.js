const fs = require("fs");
const path = require("path");
const { execSync } = require("child_process");

const ROOT = path.resolve(__dirname, "..");
const DATASET_PATH_DEFAULT = path.join(ROOT, "output/ml/datasets/dodge_iql_v1_mixed_200k.jsonl");
const COLLECT_REPORT_DEFAULT = path.join(ROOT, "output/ml/reports/dodge_iql_v1_collect_stats.json");
const TRAIN_REPORT_DEFAULT = path.join(ROOT, "output/ml/reports/iql_v1_metrics.json");
const EVAL_REPORT = path.join(ROOT, "output/ml/reports/dodge_ml_eval_report.json");
const ROUND_REPORT_DEFAULT = path.join(ROOT, "output/ml/reports/iql_v1_rounds.json");

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
    const targets = getArg("targets", "200000,400000,600000,900000")
        .split(",")
        .map((s) => asInt(s.trim(), 0))
        .filter((n) => n > 0);
    const survivalTarget = asFloat(getArg("survival-target", "0.98"), 0.98);
    const evalRuns = asInt(getArg("eval-runs", "5"), 5);
    const evalDurationMs = asInt(getArg("eval-duration-ms", "60000"), 60000);
    const resume = getArg("resume", "0") === "1";

    const datasetPath = path.resolve(getArg("dataset-path", DATASET_PATH_DEFAULT));
    const collectReportPath = path.resolve(getArg("collect-report", COLLECT_REPORT_DEFAULT));
    const trainReportPath = path.resolve(getArg("train-report", TRAIN_REPORT_DEFAULT));
    const roundReportPath = path.resolve(getArg("round-report", ROUND_REPORT_DEFAULT));

    const mixExpertRatio = asFloat(getArg("mix-expert-ratio", "0.6"), 0.6);
    const mixRandomRatio = asFloat(getArg("mix-random-ratio", "0.2"), 0.2);
    const mixEpsilonRatio = asFloat(getArg("mix-epsilon-ratio", "0.2"), 0.2);
    const quotaEnabled = getArg("quota-enabled", "0") !== "0";
    const policyBalanceEnabled = getArg("policy-balance-enabled", "1") !== "0";

    if (targets.length === 0) {
        throw new Error("no valid targets");
    }

    ensureServer();
    fs.mkdirSync(path.dirname(roundReportPath), { recursive: true });

    const roundSummaries = [];
    let currentRows = resume ? countLines(datasetPath) : 0;
    if (!resume && fs.existsSync(datasetPath)) {
        fs.unlinkSync(datasetPath);
    }

    for (let i = 0; i < targets.length; i++) {
        const target = targets[i];
        const need = Math.max(0, target - currentRows);

        if (need > 0) {
            const freshFlag = currentRows === 0 ? "1" : "0";
            run(
                "node scripts/collect-dodge-dataset.js " +
                `--target-frames=${need} --fresh=${freshFlag} ` +
                `--dataset-path=${datasetPath} --report-path=${collectReportPath} ` +
                "--screenshot-path=output/ml/reports/dodge_iql_collect_latest.png " +
                "--ml-runtime=0 --policy-mode=pure --iql-mix=1 " +
                `--mix-expert-ratio=${mixExpertRatio} --mix-random-ratio=${mixRandomRatio} --mix-epsilon-ratio=${mixEpsilonRatio} ` +
                `--quota-enabled=${quotaEnabled ? "1" : "0"} --policy-balance-enabled=${policyBalanceEnabled ? "1" : "0"}`
            );
            currentRows = countLines(datasetPath);
        }

        run(
            "python3 ml/train_iql.py " +
            `--dataset ${datasetPath} ` +
            "--epochs=12 --batch-size=512 --sampler=balanced " +
            "--out-pt output/ml/models/dodge_iql_v1.pt " +
            "--out-onnx output/ml/models/dodge_iql_v1.onnx " +
            `--out-metrics ${trainReportPath}`
        );

        run(
            "node scripts/eval-ml-dodge.js " +
            `--duration-ms=${evalDurationMs} --runs=${evalRuns} ` +
            "--policy-mode=pure --model-url=/output/ml/models/dodge_iql_v1.onnx"
        );

        const collectReport = readJson(collectReportPath) || {};
        const trainReport = readJson(trainReportPath) || {};
        const evalReport = readJson(EVAL_REPORT) || {};
        const modelMean =
            evalReport &&
            evalReport.model &&
            typeof evalReport.model.survival_rate_mean === "number"
                ? evalReport.model.survival_rate_mean
                : null;
        const ruleCallsMean =
            evalReport && evalReport.model && typeof evalReport.model.rule_calls_mean === "number"
                ? evalReport.model.rule_calls_mean
                : null;
        const pureViolationMean =
            evalReport && evalReport.model && typeof evalReport.model.pure_violation_count_mean === "number"
                ? evalReport.model.pure_violation_count_mean
                : null;

        const summary = {
            round: i + 1,
            target_rows: target,
            dataset_rows: currentRows,
            collect: {
                spawned_bubbles_effective: collectReport.spawned_bubbles_effective || collectReport.spawned_bubbles || 0,
                spawned_bubbles_ignored_trapped: collectReport.spawned_bubbles_ignored_trapped || 0,
                bombed_count: collectReport.bombed_count || 0,
                survival_rate: collectReport.survival_rate,
                policy_tag_hist_written: collectReport.policy_tag_hist_written || {},
                action_hist_written: collectReport.action_hist_written || {}
            },
            train: {
                dataset_size: trainReport.dataset_size || 0,
                val_policy_acc: trainReport.val_policy_acc || 0,
                val_policy_f1: trainReport.val_policy_f1 || 0,
                val_risk_acc: trainReport.val_risk_acc || 0
            },
            eval: {
                runs: evalReport.runs || 0,
                model_survival_rate_mean: modelMean,
                baseline_survival_rate_mean:
                    evalReport && evalReport.baseline && typeof evalReport.baseline.survival_rate_mean === "number"
                        ? evalReport.baseline.survival_rate_mean
                        : 0,
                model_rule_calls_mean: ruleCallsMean,
                model_pure_violation_count_mean: pureViolationMean,
                model_fallback_rate_mean:
                    evalReport && evalReport.model && typeof evalReport.model.fallback_rate_mean === "number"
                        ? evalReport.model.fallback_rate_mean
                        : 0
            }
        };

        roundSummaries.push(summary);
        fs.writeFileSync(
            roundReportPath,
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

        if (
            typeof modelMean === "number" &&
            modelMean >= survivalTarget &&
            (ruleCallsMean == null || ruleCallsMean === 0) &&
            (pureViolationMean == null || pureViolationMean === 0)
        ) {
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
