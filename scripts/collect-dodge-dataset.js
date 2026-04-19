const fs = require("fs");
const path = require("path");

const skillPlaywrightPath = path.join(
    process.env.HOME || "",
    ".codex/skills/develop-web-game/node_modules/playwright"
);
const { chromium } = require(skillPlaywrightPath);

const OUT_ROOT = path.resolve(__dirname, "../output/ml");
const DATASET_PATH_DEFAULT = path.join(OUT_ROOT, "datasets", "dodge_iql_v1_mixed_200k.jsonl");
const REPORT_PATH_DEFAULT = path.join(OUT_ROOT, "reports", "dodge_iql_v1_collect_stats.json");
const SCREENSHOT_PATH_DEFAULT = path.join(OUT_ROOT, "reports", "dodge_iql_v1_collect_final.png");
const URL_BASE = "http://127.0.0.1:4000/?train=1&autostart=0&ml_collect=1&ml_freeze=1&ml_wait_keep=1";

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

function normalizeAction(v) {
    const a = parseInt(v, 10);
    return Number.isFinite(a) && a >= 0 && a <= 4 ? a : 0;
}

function normalizePolicyTag(v) {
    if (v === "random" || v === "epsilon" || v === "expert") {
        return v;
    }
    return "expert";
}

async function ensureServerReady(url) {
    const maxRetry = 30;
    for (let i = 0; i < maxRetry; i++) {
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

async function main() {
    const targetFrames = asInt(getArg("target-frames", "200000"), 200000);
    const batchSize = asInt(getArg("batch-size", "2048"), 2048);
    const pollMs = asInt(getArg("poll-ms", "500"), 500);
    const fresh = getArg("fresh", "1") !== "0";
    const mlRuntimeEnabled = getArg("ml-runtime", "0") === "1";
    const quotaEnabled = getArg("quota-enabled", "1") !== "0";
    const waitMaxRatioBase = Math.min(0.95, Math.max(0.1, asFloat(getArg("wait-max-ratio", "0.55"), 0.55)));
    const minMoveRatioBase = Math.min(0.25, Math.max(0, asFloat(getArg("min-move-ratio", "0.08"), 0.08)));
    const quotaStallMs = asInt(getArg("quota-stall-ms", "15000"), 15000);
    const quotaRelaxStep = Math.min(0.3, Math.max(0.01, asFloat(getArg("quota-relax-step", "0.08"), 0.08)));
    const quotaWarmupRows = asInt(getArg("quota-warmup-rows", "800"), 800);
    const thinkIntervalMs = asInt(getArg("think-interval-ms", "8"), 8);
    const policyMode = getArg("policy-mode", "pure");
    const iqlMixEnabled = getArg("iql-mix", "1") !== "0";
    let mixExpertRatio = asFloat(getArg("mix-expert-ratio", "0.6"), 0.6);
    let mixRandomRatio = asFloat(getArg("mix-random-ratio", "0.2"), 0.2);
    let mixEpsilonRatio = asFloat(getArg("mix-epsilon-ratio", "0.2"), 0.2);
    const policyBalanceEnabled = getArg("policy-balance-enabled", "1") !== "0";
    const policyBalanceSlack = Math.max(0, Math.min(0.3, asFloat(getArg("policy-balance-slack", "0.06"), 0.06)));
    const mixSum = Math.max(1e-6, Math.max(0, mixExpertRatio) + Math.max(0, mixRandomRatio) + Math.max(0, mixEpsilonRatio));
    mixExpertRatio = Math.max(0, mixExpertRatio) / mixSum;
    mixRandomRatio = Math.max(0, mixRandomRatio) / mixSum;
    mixEpsilonRatio = Math.max(0, mixEpsilonRatio) / mixSum;
    const datasetPath = path.resolve(getArg("dataset-path", DATASET_PATH_DEFAULT));
    const reportPath = path.resolve(getArg("report-path", REPORT_PATH_DEFAULT));
    const screenshotPath = path.resolve(getArg("screenshot-path", SCREENSHOT_PATH_DEFAULT));
    const url = URL_BASE
        + "&ml=" + (mlRuntimeEnabled ? "1" : "0")
        + "&ml_policy_mode=" + encodeURIComponent(policyMode)
        + "&ml_iql_mix=" + (iqlMixEnabled ? "1" : "0")
        + "&ml_mix_expert=" + encodeURIComponent(String(mixExpertRatio))
        + "&ml_mix_random=" + encodeURIComponent(String(mixRandomRatio))
        + "&ml_mix_epsilon=" + encodeURIComponent(String(mixEpsilonRatio));

    fs.mkdirSync(path.dirname(datasetPath), { recursive: true });
    fs.mkdirSync(path.dirname(reportPath), { recursive: true });

    if (fresh && fs.existsSync(datasetPath)) {
        fs.unlinkSync(datasetPath);
    }

    await ensureServerReady("http://127.0.0.1:4000/");

    const stream = fs.createWriteStream(datasetPath, { flags: "a" });
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

    page.on("pageerror", (err) => {
        console.log("[PAGEERROR]", String(err));
    });
    page.on("console", (msg) => {
        if (msg.type() === "error") {
            console.log("[CONSOLE.ERROR]", msg.text());
        }
    });

    const startedAt = Date.now();
    let wrote = 0;
    let droppedByQuota = 0;
    let droppedByPolicyBalance = 0;
    let quotaScale = 1;
    let waitMaxRatio = waitMaxRatioBase;
    let lastAcceptAt = Date.now();
    const acceptedActionHist = { "0": 0, "1": 0, "2": 0, "3": 0, "4": 0 };
    const acceptedPolicyTagHist = { expert: 0, random: 0, epsilon: 0 };
    let lastLogAt = 0;
    let lastState = null;

    function minMoveCountEach() {
        return Math.floor(targetFrames * minMoveRatioBase * quotaScale);
    }

    function moveDeficitAfter(action) {
        let deficit = 0;
        const minEach = minMoveCountEach();
        for (let a = 1; a <= 4; a++) {
            const have = acceptedActionHist[String(a)] + (action === a ? 1 : 0);
            deficit += Math.max(0, minEach - have);
        }
        return deficit;
    }

    function shouldAcceptRow(row) {
        const action = normalizeAction(row && row.action);
        const policyTag = normalizePolicyTag(row && row.policy_tag);
        const nextTotal = wrote + 1;
        if (!quotaEnabled) {
            if (!policyBalanceEnabled || !iqlMixEnabled) {
                return true;
            }
            return shouldAcceptByPolicyMix(policyTag, nextTotal);
        }
        if (wrote < quotaWarmupRows) {
            return true;
        }
        if (action !== 0) {
            return true;
        }
        const nextWaitRatio = (acceptedActionHist["0"] + 1) / Math.max(1, nextTotal);
        if (nextWaitRatio > waitMaxRatio) {
            return false;
        }
        const remaining = targetFrames - nextTotal;
        const moveNeed = moveDeficitAfter(0);
        if (remaining < moveNeed) {
            return false;
        }
        if (policyBalanceEnabled && iqlMixEnabled && !shouldAcceptByPolicyMix(policyTag, nextTotal)) {
            return false;
        }
        return true;
    }

    function shouldAcceptByPolicyMix(tag, nextTotal) {
        const targetRatio =
            tag === "random" ? mixRandomRatio :
            tag === "epsilon" ? mixEpsilonRatio :
            mixExpertRatio;
        const projected = (acceptedPolicyTagHist[tag] || 0) + 1;
        const target = nextTotal * targetRatio;
        const slackAbs = Math.max(30, nextTotal * policyBalanceSlack);
        if (projected <= target + slackAbs) {
            return true;
        }
        return false;
    }

    try {
        await page.goto(url, { waitUntil: "domcontentloaded" });
        await page.waitForTimeout(200);
        await page.evaluate((thinkMs) => {
            window.BNBMLFreezeExpertPolicy = true;
            window.BNBMLCollectWaitKeepProb = 1;
            window.BNBPaopaoFuseMs = 1200;
            if (typeof window.BNBMLRefreshConfig === "function") {
                window.BNBMLRefreshConfig();
            }
            if (window.AIDodgeTrainer && window.AIDodgeTrainer.Config) {
                window.AIDodgeTrainer.Config.trainingThinkIntervalMs = thinkMs;
            }
            if (typeof StartAIDodgeTraining === "function") {
                StartAIDodgeTraining();
            }
        }, thinkIntervalMs);

        while (wrote < targetFrames) {
            const batch = await page.evaluate((maxRows) => {
                if (typeof window.BNBMLCollectorDrain === "function") {
                    return window.BNBMLCollectorDrain(maxRows) || [];
                }
                return [];
            }, batchSize);

            if (batch.length > 0) {
                for (const row of batch) {
                    const action = normalizeAction(row && row.action);
                    const policyTag = normalizePolicyTag(row && row.policy_tag);
                    if (!shouldAcceptRow(row)) {
                        if (policyBalanceEnabled && iqlMixEnabled && !shouldAcceptByPolicyMix(policyTag, wrote + 1)) {
                            droppedByPolicyBalance += 1;
                        }
                        else {
                            droppedByQuota += 1;
                        }
                        continue;
                    }
                    stream.write(JSON.stringify(row) + "\n");
                    wrote += 1;
                    acceptedActionHist[String(action)] = (acceptedActionHist[String(action)] || 0) + 1;
                    acceptedPolicyTagHist[policyTag] = (acceptedPolicyTagHist[policyTag] || 0) + 1;
                    lastAcceptAt = Date.now();
                    if (wrote >= targetFrames) {
                        break;
                    }
                }
            }

            lastState = await page.evaluate(() => {
                return {
                    collector: window.BNBMLDatasetCollectorState || null,
                    runtime: window.BNBMLRuntimeState || null,
                    trainer: window.BNBTrainingRuntimeState || null
                };
            });

            const now = Date.now();
            if (now - lastLogAt > 2000) {
                lastLogAt = now;
                const c = (lastState && lastState.collector) || {};
                const t = (lastState && lastState.trainer) || {};
                console.log(
                    "[COLLECT]",
                    "rows=" + wrote + "/" + targetFrames,
                    "ready=" + (c.rows_ready || 0),
                    "spawned_effective=" + (c.spawned_bubbles_effective || c.spawned_bubbles || 0),
                    "spawned_ignored=" + (c.spawned_bubbles_ignored_trapped || 0),
                    "bombed=" + (c.bombed_count || 0),
                    "survival=" + (typeof c.survival_rate === "number" ? c.survival_rate.toFixed(4) : "na"),
                    "quota_scale=" + quotaScale.toFixed(2),
                    "wait_max=" + waitMaxRatio.toFixed(2),
                    "drop_quota=" + droppedByQuota,
                    "drop_policy=" + droppedByPolicyBalance,
                    "match=" + (t.matchIndex || 0),
                    "attempt=" + (t.matchAttempt || 0)
                );
            }

            if (
                quotaEnabled &&
                wrote < targetFrames &&
                Date.now() - lastAcceptAt > quotaStallMs
            ) {
                quotaScale = Math.max(0.55, quotaScale - quotaRelaxStep);
                waitMaxRatio = Math.min(0.92, waitMaxRatio + quotaRelaxStep * 0.35);
                lastAcceptAt = Date.now();
                console.log(
                    "[QUOTA_RELAX]",
                    "quota_scale=" + quotaScale.toFixed(2),
                    "wait_max=" + waitMaxRatio.toFixed(2)
                );
            }

            await page.waitForTimeout(pollMs);
        }

        const tail = await page.evaluate(() => {
            if (typeof window.BNBMLCollectorDrainAll === "function") {
                return window.BNBMLCollectorDrainAll() || [];
            }
            return [];
        });
        for (const row of tail) {
            if (wrote >= targetFrames) {
                break;
            }
            const action = normalizeAction(row && row.action);
            const policyTag = normalizePolicyTag(row && row.policy_tag);
            if (!shouldAcceptRow(row)) {
                if (policyBalanceEnabled && iqlMixEnabled && !shouldAcceptByPolicyMix(policyTag, wrote + 1)) {
                    droppedByPolicyBalance += 1;
                }
                else {
                    droppedByQuota += 1;
                }
                continue;
            }
            stream.write(JSON.stringify(row) + "\n");
            wrote += 1;
            acceptedActionHist[String(action)] = (acceptedActionHist[String(action)] || 0) + 1;
            acceptedPolicyTagHist[policyTag] = (acceptedPolicyTagHist[policyTag] || 0) + 1;
        }

        await page.screenshot({ path: screenshotPath, fullPage: true });
    } finally {
        stream.end();
        await browser.close();
    }

    const durationSec = (Date.now() - startedAt) / 1000;
    const collector = (lastState && lastState.collector) || {};
    const runtime = (lastState && lastState.runtime) || {};
    const report = {
        ts: Date.now(),
        dataset_path: datasetPath,
        target_frames: targetFrames,
        rows_written: wrote,
        duration_sec: durationSec,
        rows_per_sec: durationSec > 0 ? wrote / durationSec : 0,
        spawned_bubbles: collector.spawned_bubbles_effective || collector.spawned_bubbles || 0,
        spawned_bubbles_effective: collector.spawned_bubbles_effective || collector.spawned_bubbles || 0,
        spawned_bubbles_ignored_trapped: collector.spawned_bubbles_ignored_trapped || 0,
        bombed_count: collector.bombed_count || 0,
        survival_rate: typeof collector.survival_rate === "number" ? collector.survival_rate : 1,
        action_hist: collector.action_hist || {},
        action_hist_written: acceptedActionHist,
        dropped_by_quota: droppedByQuota,
        dropped_by_policy_balance: droppedByPolicyBalance,
        quota_enabled: quotaEnabled,
        quota_scale_final: quotaScale,
        wait_max_ratio_final: waitMaxRatio,
        wait_max_ratio_base: waitMaxRatioBase,
        min_move_ratio_base: minMoveRatioBase,
        pre_death_window_ms: collector.pre_death_window_ms || null,
        fallback_rate: typeof runtime.fallback_rate === "number" ? runtime.fallback_rate : 0,
        rule_calls: runtime.rule_calls || 0,
        pure_violation_count: runtime.pure_violation_count || 0,
        policy_tag_hist: collector.policy_tag_hist || {},
        policy_tag_hist_written: acceptedPolicyTagHist,
        risk_label_hist: collector.risk_label_hist || {},
        policy_mode: policyMode,
        iql_mix_enabled: iqlMixEnabled,
        policy_balance_enabled: policyBalanceEnabled,
        policy_balance_slack: policyBalanceSlack,
        mix_expert_ratio: mixExpertRatio,
        mix_random_ratio: mixRandomRatio,
        mix_epsilon_ratio: mixEpsilonRatio,
        screenshot: screenshotPath
    };

    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
    console.log("[DONE]", JSON.stringify(report));
}

main().catch((err) => {
    console.error(err);
    process.exit(1);
});
