const fs = require("fs");
const path = require("path");

const skillPlaywrightPath = path.join(
    process.env.HOME || "",
    ".codex/skills/develop-web-game/node_modules/playwright"
);
const { chromium } = require(skillPlaywrightPath);

const OUT_DIR = path.resolve(__dirname, "../output/ml/reports");
const REPORT_PATH = path.join(OUT_DIR, "dodge_ml_eval_report.json");

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

function asFloatOrNull(v) {
    if (v === "" || v === null || v === undefined) {
        return null;
    }
    const n = parseFloat(v);
    return Number.isFinite(n) ? n : null;
}

async function runScenario(mlEnabled, durationMs, modelUrl, runIndex, cfg) {
    const flag = mlEnabled ? "1" : "0";
    const mlConf = cfg.mlConf;
    const mlMoveConf = cfg.mlMoveConf;
    const mlMargin = cfg.mlMargin;
    const mlForceMoveEta = cfg.mlForceMoveEta;
    const mlWaitBlockEta = cfg.mlWaitBlockEta;
    const mlMoveThreatMs = cfg.mlMoveThreatMs;
    const policyMode = cfg.policyMode || "pure";
    const url =
        "http://127.0.0.1:4000/?train=1&autostart=0&ml=" +
        flag +
        "&ml_collect=0&ml_freeze=1" +
        "&ml_policy_mode=" + encodeURIComponent(policyMode) +
        (typeof mlConf === "number" ? "&ml_conf=" + encodeURIComponent(String(mlConf)) : "") +
        (typeof mlMoveConf === "number" ? "&ml_move_conf=" + encodeURIComponent(String(mlMoveConf)) : "") +
        (typeof mlMargin === "number" ? "&ml_margin=" + encodeURIComponent(String(mlMargin)) : "") +
        (typeof mlForceMoveEta === "number" ? "&ml_force_move_eta=" + encodeURIComponent(String(mlForceMoveEta)) : "") +
        (typeof mlWaitBlockEta === "number" ? "&ml_wait_block_eta=" + encodeURIComponent(String(mlWaitBlockEta)) : "") +
        (typeof mlMoveThreatMs === "number" ? "&ml_move_threat_ms=" + encodeURIComponent(String(mlMoveThreatMs)) : "") +
        (modelUrl ? "&ml_model=" + encodeURIComponent(modelUrl) : "");
    let browser;
    try {
        browser = await chromium.launch({
            headless: true,
            args: ["--use-gl=angle", "--use-angle=swiftshader"]
        });
    } catch (err) {
        const chromePath = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
        if (!fs.existsSync(chromePath)) {
            throw err;
        }
        browser = await chromium.launch({
            headless: true,
            executablePath: chromePath,
            args: ["--use-gl=angle", "--use-angle=swiftshader"]
        });
    }
    const page = await browser.newPage({ viewport: { width: 1460, height: 940 } });

    try {
        await page.goto(url, { waitUntil: "domcontentloaded" });
        await page.waitForTimeout(200);
        await page.evaluate((enabled) => {
            window.BNBMLFreezeExpertPolicy = true;
            if (typeof window.BNBMLRefreshConfig === "function") {
                window.BNBMLRefreshConfig();
            }
            if (enabled) {
                window.BNBPaopaoFuseMs = 1200;
            }
            if (typeof StartAIDodgeTraining === "function") {
                StartAIDodgeTraining();
            }
        }, mlEnabled);

        const startedAt = Date.now();
        while (Date.now() - startedAt < durationMs) {
            await page.waitForTimeout(1000);
        }
        const state = await page.evaluate(() => {
            return {
                runtime: window.BNBMLRuntimeState || null,
                trainer: window.BNBTrainingRuntimeState || null
            };
        });
        const screenshot = path.join(
            OUT_DIR,
            (mlEnabled ? "eval_ml_on_" : "eval_ml_off_") + String(runIndex || 1) + ".png"
        );
        await page.screenshot({ path: screenshot, fullPage: true });

        const runtime = state.runtime || {};
        const spawned = runtime.spawned_bubbles_effective || runtime.spawned_bubbles || 0;
        const ignored = runtime.spawned_bubbles_ignored_trapped || 0;
        const bombed = runtime.bombed_count || 0;
        const survival =
            typeof runtime.survival_rate === "number"
                ? runtime.survival_rate
                : (spawned > 0 ? (1 - bombed / spawned) : 1);
        return {
            mode: mlEnabled ? "ml_on" : "ml_off",
            run_index: runIndex || 1,
            duration_ms: durationMs,
            spawned_bubbles: spawned,
            spawned_bubbles_effective: spawned,
            spawned_bubbles_ignored_trapped: ignored,
            bombed_count: bombed,
            survival_rate: survival,
            fallback_rate: runtime.fallback_rate || 0,
            rule_calls: runtime.rule_calls || 0,
            pure_violation_count: runtime.pure_violation_count || 0,
            avg_latency_ms: runtime.avg_latency_ms || 0,
            screenshot,
            runtime_state: runtime,
            trainer_state: state.trainer || {}
        };
    } finally {
        await browser.close();
    }
}

function summarizeRuns(mode, runs) {
    const n = Math.max(1, runs.length);
    let survivalSum = 0;
    let fallbackSum = 0;
    let latencySum = 0;
    let spawnSum = 0;
    let ignoredSum = 0;
    let bombedSum = 0;
    let ruleCallsSum = 0;
    let pureViolationSum = 0;
    for (const run of runs) {
        survivalSum += run.survival_rate || 0;
        fallbackSum += run.fallback_rate || 0;
        latencySum += run.avg_latency_ms || 0;
        spawnSum += run.spawned_bubbles_effective || run.spawned_bubbles || 0;
        ignoredSum += run.spawned_bubbles_ignored_trapped || 0;
        bombedSum += run.bombed_count || 0;
        ruleCallsSum += run.rule_calls || 0;
        pureViolationSum += run.pure_violation_count || 0;
    }
    return {
        mode,
        runs: runs.length,
        survival_rate_mean: survivalSum / n,
        fallback_rate_mean: fallbackSum / n,
        rule_calls_mean: ruleCallsSum / n,
        pure_violation_count_mean: pureViolationSum / n,
        avg_latency_ms_mean: latencySum / n,
        spawned_bubbles_effective_mean: spawnSum / n,
        spawned_bubbles_ignored_trapped_mean: ignoredSum / n,
        bombed_count_mean: bombedSum / n
    };
}

async function main() {
    const durationMs = asInt(getArg("duration-ms", "60000"), 60000);
    const runs = asInt(getArg("runs", "3"), 3);
    const modelUrl = getArg("model-url", "/output/ml/models/dodge_iql_v1.onnx");
    const mlConfRaw = getArg("ml-conf", "");
    const cfg = {
        mlConf: asFloatOrNull(mlConfRaw),
        mlMoveConf: asFloatOrNull(getArg("ml-move-conf", "")),
        mlMargin: asFloatOrNull(getArg("ml-margin", "")),
        mlForceMoveEta: asFloatOrNull(getArg("ml-force-move-eta", "")),
        mlWaitBlockEta: asFloatOrNull(getArg("ml-wait-block-eta", "")),
        mlMoveThreatMs: asFloatOrNull(getArg("ml-move-threat-ms", "")),
        policyMode: getArg("policy-mode", "pure")
    };
    fs.mkdirSync(OUT_DIR, { recursive: true });

    const baselineRuns = [];
    const modelRuns = [];
    for (let i = 0; i < runs; i++) {
        baselineRuns.push(await runScenario(false, durationMs, modelUrl, i + 1, cfg));
    }
    for (let i = 0; i < runs; i++) {
        modelRuns.push(await runScenario(true, durationMs, modelUrl, i + 1, cfg));
    }
    const baseline = summarizeRuns("ml_off", baselineRuns);
    const model = summarizeRuns("ml_on", modelRuns);

    const report = {
        ts: Date.now(),
        duration_ms: durationMs,
        runs,
        ml_cfg: cfg,
        baseline,
        model,
        baseline_runs: baselineRuns,
        model_runs: modelRuns,
        delta_survival_rate_mean: model.survival_rate_mean - baseline.survival_rate_mean
    };
    fs.writeFileSync(REPORT_PATH, JSON.stringify(report, null, 2));
    console.log("[EVAL]", JSON.stringify(report));
}

main().catch((err) => {
    console.error(err);
    process.exit(1);
});
