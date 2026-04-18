const fs = require("fs");
const path = require("path");

const skillPlaywrightPath = path.join(
    process.env.HOME || "",
    ".codex/skills/develop-web-game/node_modules/playwright"
);
const { chromium } = require(skillPlaywrightPath);

const URL = "http://127.0.0.1:4000/?train=1&autostart=0";
const OUT_DIR = path.resolve(__dirname, "../output/web-game");
const STATUS_FILE = path.join(OUT_DIR, "training-iter5-status.json");
const LOCK_FILE = path.join(OUT_DIR, "training-runtime.lock.json");
const RUNTIME_FILE = path.join(OUT_DIR, "training-runtime-state.json");
const LIVE_FRAME_FILE = path.join(OUT_DIR, "training-iter5-live.png");
const COMPLETE_FRAME_FILE = path.join(OUT_DIR, "training-iter5-complete.png");
const TIMEOUT_FRAME_FILE = path.join(OUT_DIR, "training-iter5-timeout.png");
const MAX_WAIT_MS = Number.POSITIVE_INFINITY;
const LOCK_STALE_MS = 15000;
const STATE_POLL_MS = 1000;
const FRAME_CAPTURE_MS = 33;
const LOOP_SLEEP_MS = 20;
const FORCE_FRESH = process.argv.indexOf("--fresh") >= 0;

const SESSION_ID = "train-" + Date.now() + "-" + Math.floor(Math.random() * 100000);
let INTERRUPTED = false;

process.on("SIGINT", function() {
    INTERRUPTED = true;
});
process.on("SIGTERM", function() {
    INTERRUPTED = true;
});

function safeReadJSON(filepath) {
    try {
        if (!fs.existsSync(filepath)) return null;
        return JSON.parse(fs.readFileSync(filepath, "utf8"));
    } catch (err) {
        return null;
    }
}

function writeJSON(filepath, data) {
    fs.writeFileSync(filepath, JSON.stringify(data, null, 2));
}

function updateLock(active, extra) {
    const base = {
        active: !!active,
        sessionId: SESSION_ID,
        pid: process.pid,
        heartbeat: Date.now()
    };
    writeJSON(LOCK_FILE, Object.assign(base, extra || {}));
}

function isProcessAlive(pid) {
    if (typeof pid !== "number" || pid <= 0) {
        return false;
    }
    try {
        process.kill(pid, 0);
        return true;
    } catch (err) {
        return false;
    }
}

function buildResumeBootstrap() {
    if (FORCE_FRESH) {
        return null;
    }
    const runtime = safeReadJSON(RUNTIME_FILE);
    const status = safeReadJSON(STATUS_FILE);
    const state = runtime && runtime.state
        ? runtime.state
        : (status && status.state ? status.state : null);
    if (!state) {
        return null;
    }
    if (typeof state.baselineScore !== "number" && typeof state.targetScore !== "number") {
        return null;
    }
    return {
        matchIndex: typeof state.matchIndex === "number" ? state.matchIndex : 1,
        matchAttempt: typeof state.matchAttempt === "number" ? state.matchAttempt : 1,
        completedMatches: typeof state.completedMatches === "number" ? state.completedMatches : 0,
        totalAttempts: typeof state.totalAttempts === "number" ? state.totalAttempts : 0,
        totalScore: typeof state.totalScore === "number" ? state.totalScore : 0,
        baselineScore: typeof state.baselineScore === "number" ? state.baselineScore : null,
        targetScore: typeof state.targetScore === "number" ? state.targetScore : null
    };
}

async function readState(page) {
    return page.evaluate(function() {
        var trainer = window.AIDodgeTrainer;
        var state;
        var latestResult = window.BNBLatestTrainingResult || null;
        var runtime = window.BNBTrainingRuntimeState || null;
        if (!trainer) {
            return null;
        }
        state = trainer.State || {};
        return {
            running: !!trainer.IsRunning,
            matchIndex: state.matchIndex || 0,
            matchAttempt: state.matchAttempt || 0,
            roundIndex: state.roundIndex || 0,
            matchScore: state.matchScore || 0,
            matchDeaths: state.matchDeaths || 0,
            matchSurviveRounds: state.matchSurviveRounds || 0,
            completedMatches: state.completedMatches || 0,
            totalAttempts: state.totalAttempts || 0,
            totalScore: state.totalScore || 0,
            baselineScore: typeof state.baselineScore === "number" ? state.baselineScore : null,
            targetScore: typeof state.currentMatchTargetScore === "number" ? state.currentMatchTargetScore : null,
            latestLesson: state.latestLesson || "",
            latestDeathSummary: state.latestDeathSummary || "",
            latestReview: state.latestReview || null,
            roundReviewsCount: Array.isArray(state.roundReviews) ? state.roundReviews.length : 0,
            logTail: (state.logs || []).slice(-8),
            trainingCompleted: !!window.BNBTrainingCompleted,
            stopReason: state.stopReason || (runtime && runtime.stopReason) || (latestResult && latestResult.stopReason) || "",
            latestResult: latestResult
        };
    });
}

async function capture(page, filepath, options) {
    var tempPath = filepath + ".tmp";
    await page.screenshot(Object.assign({ path: tempPath, fullPage: false, type: "png" }, options || {}));
    fs.renameSync(tempPath, filepath);
}

function persistRuntime(state, extra) {
    const payload = Object.assign({
        ts: Date.now(),
        sessionId: SESSION_ID,
        pid: process.pid,
        liveFramePath: LIVE_FRAME_FILE
    }, state || {});
    payload.state = state;
    writeJSON(RUNTIME_FILE, Object.assign(payload, extra || {}));
}

function persistStatus(state, extra) {
    writeJSON(STATUS_FILE, Object.assign({ ts: Date.now(), state: state }, extra || {}));
}

async function bootstrapTraining(page, bootstrap) {
    await page.evaluate(function(payload) {
        window.BNBTrainingBootstrap = payload || null;
        if (typeof StartAIDodgeTraining === "function") {
            StartAIDodgeTraining();
        }
    }, bootstrap || null);
}

async function main() {
    var browser;
    var page;
    var launchOptions = {
        headless: true,
        args: ["--use-gl=angle", "--use-angle=swiftshader"]
    };
    var lastRound = -1;
    var lastMatch = -1;
    var lastAttempt = -1;
    var lastCompleted = -1;
    var lastLogTail = "";
    var startedAt = Date.now();
    var resume = buildResumeBootstrap();
    var done = false;
    var stopReason = "";
    var doneState = null;
    var lastStateAt = 0;
    var lastFrameAt = 0;
    var existingLock;
    var existingAge;
    var existingAlive;

    fs.mkdirSync(OUT_DIR, { recursive: true });

    existingLock = safeReadJSON(LOCK_FILE);
    if (existingLock && existingLock.active) {
        existingAge = Date.now() - (existingLock.heartbeat || 0);
        existingAlive = isProcessAlive(existingLock.pid);
        if (existingAlive && existingAge >= 0 && existingAge < LOCK_STALE_MS) {
            console.log("[LOCK] 已存在活跃训练会话，session="
                + (existingLock.sessionId || "unknown")
                + " pid=" + existingLock.pid
                + " heartbeatAgeMs=" + existingAge
                + "，本次不重复启动。");
            return;
        }
    }

    updateLock(true, { startedAt: startedAt, reason: "running" });

    try {
        browser = await chromium.launch(launchOptions);
    }
    catch (err) {
        var chromePath = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
        if (fs.existsSync(chromePath)) {
            browser = await chromium.launch({
                headless: true,
                executablePath: chromePath,
                args: ["--use-gl=angle", "--use-angle=swiftshader"]
            });
        }
        else {
            throw err;
        }
    }
    page = await browser.newPage({ viewport: { width: 1460, height: 940 } });
    page.on("pageerror", function(err) {
        console.log("[PAGEERROR]", String(err));
    });
    page.on("console", function(msg) {
        if (msg.type() === "error") {
            console.log("[CONSOLE.ERROR]", msg.text());
        }
    });

    await page.goto(URL, { waitUntil: "domcontentloaded" });
    await page.waitForTimeout(300);
    await bootstrapTraining(page, resume);
    await page.waitForTimeout(1400);
    await capture(page, LIVE_FRAME_FILE);

    if (resume) {
        console.log("[RESUME] 恢复基线=" + (resume.baselineScore == null ? "na" : resume.baselineScore)
            + " 目标=" + (resume.targetScore == null ? "na" : resume.targetScore)
            + " 轮次=" + resume.matchIndex + " 尝试=" + resume.matchAttempt);
    }

    while (Date.now() - startedAt < MAX_WAIT_MS) {
        var loopNow = Date.now();
        var s;
        if (loopNow - lastFrameAt >= FRAME_CAPTURE_MS) {
            await capture(page, LIVE_FRAME_FILE);
            lastFrameAt = Date.now();
        }
        if (loopNow - lastStateAt < STATE_POLL_MS) {
            await page.waitForTimeout(LOOP_SLEEP_MS);
            continue;
        }
        s = await readState(page);
        lastStateAt = Date.now();
        if (!s) {
            await page.waitForTimeout(LOOP_SLEEP_MS);
            continue;
        }

        updateLock(true, { startedAt: startedAt, reason: "running" });

        if (s.matchIndex !== lastMatch || s.matchAttempt !== lastAttempt || s.roundIndex !== lastRound) {
            console.log(
                "[PROGRESS] 轮" + s.matchIndex + " 尝试" + s.matchAttempt
                + " 事件" + s.roundIndex + " | 分" + s.matchScore
                + " 死" + s.matchDeaths
                + " | 基线 " + (s.baselineScore == null ? "na" : s.baselineScore)
                + " 目标 " + (s.targetScore == null ? "na" : s.targetScore)
                + " | 达标轮 " + s.completedMatches
            );
            lastMatch = s.matchIndex;
            lastAttempt = s.matchAttempt;
            lastRound = s.roundIndex;
        }

        if (s.completedMatches !== lastCompleted) {
            lastCompleted = s.completedMatches;
            await capture(
                page,
                path.join(OUT_DIR, "train-iter5-pass-" + String(lastCompleted + 100).slice(1) + ".png"),
                { fullPage: true }
            );
            console.log("[SNAP] 已保存达标轮截图 pass=" + lastCompleted);
        }

        var currentTail = (s.logTail || []).join("\n");
        if (currentTail && currentTail !== lastLogTail) {
            lastLogTail = currentTail;
            console.log("[LOGTAIL] " + currentTail.replace(/\n/g, " || "));
        }

        persistStatus(s, { done: false, sessionId: SESSION_ID });
        persistRuntime(s, { done: false });

        stopReason = s.stopReason || (s.latestResult && s.latestResult.stopReason) || "";
        if (stopReason || s.trainingCompleted) {
            done = true;
            doneState = s;
            await capture(page, COMPLETE_FRAME_FILE, { fullPage: true });
            persistStatus(s, { done: true, sessionId: SESSION_ID, stopReason: stopReason || "completed_flag" });
            persistRuntime(s, { done: true, stopReason: stopReason || "completed_flag" });
            if (stopReason === "gap_not_met_requires_review") {
                console.log("[STOP-REVIEW] 当前轮未达成基线+50，训练已停止；请先复盘日志并调整代码/参数后再继续。");
            }
            console.log("[DONE] 训练完成 " + JSON.stringify(s.latestResult || s));
            break;
        }

        await page.waitForTimeout(LOOP_SLEEP_MS);
    }

    if (!done && Date.now() - startedAt >= MAX_WAIT_MS) {
        var timeoutState = await readState(page);
        await capture(page, TIMEOUT_FRAME_FILE, { fullPage: true });
        persistStatus(timeoutState, { timeout: true, done: false, sessionId: SESSION_ID });
        persistRuntime(timeoutState, { timeout: true, done: false });
        console.log("[TIMEOUT] 达到最大等待时间，已保存当前状态。");
    }

    if (browser) {
        await browser.close();
    }

    updateLock(false, {
        startedAt: startedAt,
        endedAt: Date.now(),
        reason: done ? (stopReason || "completed_flag") : "stopped_or_timeout",
        finalState: doneState ? {
            matchIndex: doneState.matchIndex,
            matchAttempt: doneState.matchAttempt,
            matchScore: doneState.matchScore,
            baselineScore: doneState.baselineScore,
            targetScore: doneState.targetScore
        } : null
    });
}

main().catch(function(err) {
    var interrupted = INTERRUPTED || /Target page, context or browser has been closed/.test(String(err && err.message ? err.message : err));
    try {
        updateLock(false, {
            endedAt: Date.now(),
            reason: interrupted ? "interrupted" : "script_error",
            error: interrupted ? "" : String(err && err.stack ? err.stack : err)
        });
    } catch (e) {}
    if (!interrupted) {
        console.error(err);
        process.exit(1);
        return;
    }
    console.log("[STOP] 训练脚本已中断，锁已释放。");
    process.exit(0);
});
