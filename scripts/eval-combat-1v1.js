const fs = require("fs");
const path = require("path");

const skillPlaywrightPath = path.join(
    process.env.HOME || "",
    ".codex/skills/develop-web-game/node_modules/playwright"
);
const { chromium } = require(skillPlaywrightPath);

const OUT_DIR = path.resolve(__dirname, "../output/ml/reports");
const REPORT_PATH_DEFAULT = path.join(OUT_DIR, `combat_phase0_eval_${Date.now()}.json`);

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

function asFloatOrNull(v) {
    if (v === "" || v === null || v === undefined) {
        return null;
    }
    const n = parseFloat(v);
    return Number.isFinite(n) ? n : null;
}

function normalizeOpponentMode(raw) {
    const value = String(raw || "heuristic_v1").trim().toLowerCase();
    if (value === "heuristic_v2" || value === "scripted_heuristic_v2") {
        return "heuristic_v2";
    }
    return "heuristic_v1";
}

async function launchBrowser() {
    try {
        return await chromium.launch({
            headless: true,
            args: ["--use-gl=angle", "--use-angle=swiftshader"]
        });
    } catch (err) {
        const chromePath = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
        if (!fs.existsSync(chromePath)) {
            throw err;
        }
        return chromium.launch({
            headless: true,
            executablePath: chromePath,
            args: ["--use-gl=angle", "--use-angle=swiftshader"]
        });
    }
}

function computePowerScore(role) {
    if (!role) {
        return 0;
    }
    const canPaopaoLength = Number(role.CanPaopaoLength || 0);
    const moveStep = Number(role.MoveStep || 0);
    const paopaoStrong = Number(role.PaopaoStrong || 0);
    return canPaopaoLength + moveStep + paopaoStrong;
}

async function runMatch(runIndex, cfg) {
    const mlFlag = cfg.mlEnabled ? "1" : "0";
    const url =
        "http://127.0.0.1:4000/?autostart=0"
        + "&ml=" + mlFlag
        + "&ml_collect=0"
        + "&ml_freeze=1"
        + "&ml_policy_mode=" + encodeURIComponent(cfg.policyMode)
        + (cfg.modelUrl ? "&ml_model=" + encodeURIComponent(cfg.modelUrl) : "")
        + (typeof cfg.mlConf === "number" ? "&ml_conf=" + encodeURIComponent(String(cfg.mlConf)) : "")
        + (typeof cfg.mlMoveConf === "number" ? "&ml_move_conf=" + encodeURIComponent(String(cfg.mlMoveConf)) : "")
        + (typeof cfg.mlMargin === "number" ? "&ml_margin=" + encodeURIComponent(String(cfg.mlMargin)) : "")
        + (typeof cfg.mlForceMoveEta === "number" ? "&ml_force_move_eta=" + encodeURIComponent(String(cfg.mlForceMoveEta)) : "")
        + (typeof cfg.mlWaitBlockEta === "number" ? "&ml_wait_block_eta=" + encodeURIComponent(String(cfg.mlWaitBlockEta)) : "")
        + (typeof cfg.mlMoveThreatMs === "number" ? "&ml_move_threat_ms=" + encodeURIComponent(String(cfg.mlMoveThreatMs)) : "");

    let browser = null;
    let page = null;
    let timedOut = false;
    const hardTimeoutMs = Math.max(15000, (cfg.matchDurationSec + 20) * 1000);
    const hardTimer = setTimeout(() => {
        timedOut = true;
        if (browser) {
            browser.close().catch(() => {});
        }
    }, hardTimeoutMs);

    try {
        browser = await Promise.race([
            launchBrowser(),
            new Promise((_, reject) => {
                setTimeout(() => reject(new Error("browser_launch_timeout")), 12000);
            }),
        ]);
        page = await browser.newPage({ viewport: { width: 1460, height: 940 } });
        await page.goto(url, { waitUntil: "domcontentloaded", timeout: 15000 });
        await page.waitForTimeout(200);

        const seed = (cfg.seedBase >>> 0) + runIndex;
        await page.evaluate((runtimeCfg) => {
            function mulberry32(a) {
                let t = a >>> 0;
                return function() {
                    t += 0x6D2B79F5;
                    let z = Math.imul(t ^ (t >>> 15), 1 | t);
                    z ^= z + Math.imul(z ^ (z >>> 7), 61 | z);
                    return ((z ^ (z >>> 14)) >>> 0) / 4294967296;
                };
            }

            const rng = mulberry32(runtimeCfg.seed >>> 0);
            window.Math.random = function() {
                return rng();
            };

            window.alert = function() {};
            window.__combatEvalController = null;

            if (typeof window.BNBMLRefreshConfig === "function") {
                window.BNBMLRefreshConfig();
            }
            if (typeof window.ApplySelectedGameMap === "function" && runtimeCfg.mapId) {
                window.ApplySelectedGameMap(runtimeCfg.mapId, false);
            }
            if (typeof window.SetAIEnemyCount === "function") {
                window.SetAIEnemyCount(1, false);
            }

            window.roundDurationSeconds = runtimeCfg.matchDurationSec;
            if (typeof window.StartSinglePlayerGame === "function") {
                window.StartSinglePlayerGame(1);
            }

            window.__combatEvalController = setInterval(function() {
                if (!window.singlePlayerState || !window.gameRunning) {
                    return;
                }
                const player = window.singlePlayerState.Player;
                if (!player || player.IsDeath || player.IsInPaopao || typeof player.CurrentMapID !== "function") {
                    return;
                }

                const currentMap = player.CurrentMapID();
                if (!currentMap) {
                    return;
                }

                const snapshot = (typeof window.BuildThreatSnapshot === "function")
                    ? window.BuildThreatSnapshot()
                    : null;

                let aiMap = null;
                const fighters = window.singlePlayerState && Array.isArray(window.singlePlayerState.Fighters)
                    ? window.singlePlayerState.Fighters
                    : [];
                for (let i = 0; i < fighters.length; i++) {
                    if (fighters[i] && typeof fighters[i].id === "string" && fighters[i].id.indexOf("ai_") === 0) {
                        const aiRole = fighters[i].role;
                        if (aiRole && !aiRole.IsDeath && typeof aiRole.CurrentMapID === "function") {
                            aiMap = aiRole.CurrentMapID();
                            break;
                        }
                    }
                }

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
                    const nx = currentMap.X + d.dx;
                    const ny = currentMap.Y + d.dy;
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
                    score += safeN * 11;
                    score += isThreat ? -920 : 30;
                    score += typeof eta === "number" ? Math.min(eta, 1200) / 110 : 7;
                    if (runtimeCfg.opponentMode === "heuristic_v2") {
                        score += Math.max(0, 6 - distToAi) * 7.5;
                    } else {
                        score += Math.max(0, distToAi - 2) * 2.0;
                    }
                    score += (rng() - 0.5) * 2.0;

                    if (score > bestScore) {
                        bestScore = score;
                        best = d;
                    }
                }

                if (best && typeof window.RoleKeyEvent === "function" && typeof window.RoleKeyEventEnd === "function") {
                    window.RoleKeyEvent(best.key, player);
                    setTimeout(function() {
                        window.RoleKeyEventEnd(best.key, player);
                    }, 80);
                }

                if (aiMap && player.CanPaopaoLength > player.PaopaoCount && typeof window.RoleKeyEvent === "function") {
                    const dist = Math.abs(aiMap.X - currentMap.X) + Math.abs(aiMap.Y - currentMap.Y);
                    const bombProb = runtimeCfg.opponentMode === "heuristic_v2" ? 0.23 : 0.11;
                    if (dist <= 2 && rng() < bombProb) {
                        window.RoleKeyEvent(32, player);
                    }
                }
            }, Math.max(80, runtimeCfg.opponentThinkMs || 120));
        }, {
            seed,
            mapId: cfg.mapId,
            matchDurationSec: cfg.matchDurationSec,
            opponentMode: cfg.opponentMode,
            opponentThinkMs: cfg.opponentThinkMs
        });

        const timeoutMs = (cfg.matchDurationSec + 8) * 1000;
        const startedAt = Date.now();
        let loopTimedOut = false;
        while (Date.now() - startedAt < timeoutMs) {
            if (timedOut) {
                throw new Error("match_hard_timeout");
            }
            const done = await page.evaluate(() => !window.gameRunning);
            if (done) {
                break;
            }
            await page.waitForTimeout(250);
        }
        if (Date.now() - startedAt >= timeoutMs) {
            loopTimedOut = true;
        }

        const match = await page.evaluate((payload) => {
            const runNo = payload && typeof payload.runNo === "number" ? payload.runNo : 0;
            const timedOutByLoop = !!(payload && payload.timedOutByLoop);
            if (window.__combatEvalController) {
                clearInterval(window.__combatEvalController);
                window.__combatEvalController = null;
            }

            const state = window.singlePlayerState || null;
            const fighters = state && Array.isArray(state.Fighters) ? state.Fighters : [];
            let player = null;
            let ai = null;
            for (let i = 0; i < fighters.length; i++) {
                if (!player && fighters[i] && fighters[i].id === "player") {
                    player = fighters[i];
                }
                if (!ai && fighters[i] && typeof fighters[i].id === "string" && fighters[i].id.indexOf("ai_") === 0) {
                    ai = fighters[i];
                }
            }

            const playerKills = player ? (player.kills || 0) : 0;
            const playerDeaths = player ? (player.deaths || 0) : 0;
            const aiKills = ai ? (ai.kills || 0) : 0;
            const aiDeaths = ai ? (ai.deaths || 0) : 0;
            const aiSelfKills = Math.max(0, aiDeaths - playerKills);

            let result = "draw";
            if (aiKills > playerKills) {
                result = "win";
            } else if (aiKills < playerKills) {
                result = "loss";
            }

            const playerPower = player && player.role
                ? (Number(player.role.CanPaopaoLength || 0) + Number(player.role.MoveStep || 0) + Number(player.role.PaopaoStrong || 0))
                : 0;
            const aiPower = ai && ai.role
                ? (Number(ai.role.CanPaopaoLength || 0) + Number(ai.role.MoveStep || 0) + Number(ai.role.PaopaoStrong || 0))
                : 0;
            const itemControl = aiPower / Math.max(1, aiPower + playerPower);

            return {
                run_index: runNo,
                result,
                player_kills: playerKills,
                player_deaths: playerDeaths,
                ai_kills: aiKills,
                ai_deaths: aiDeaths,
                ai_self_kills: aiSelfKills,
                ai_power: aiPower,
                player_power: playerPower,
                item_control: itemControl,
                remaining_seconds: state && typeof state.RemainingSeconds === "number" ? state.RemainingSeconds : null,
                game_running: !!window.gameRunning,
                loop_timeout: !!timedOutByLoop
            };
        }, { runNo: runIndex + 1, timedOutByLoop: loopTimedOut });

        if (cfg.saveScreenshots) {
            const screenshotPath = path.join(OUT_DIR, `combat_1v1_eval_match_${String(runIndex + 1).padStart(3, "0")}.png`);
            await page.screenshot({ path: screenshotPath, fullPage: true });
            match.screenshot = screenshotPath;
        }

        return match;
    } catch (err) {
        return {
            run_index: runIndex + 1,
            result: "error",
            player_kills: 0,
            player_deaths: 0,
            ai_kills: 0,
            ai_deaths: 0,
            ai_self_kills: 0,
            ai_power: 0,
            player_power: 0,
            item_control: 0.5,
            remaining_seconds: null,
            game_running: false,
            error: String(err && err.message ? err.message : err),
        };
    } finally {
        clearTimeout(hardTimer);
        if (browser) {
            await browser.close().catch(() => {});
        }
    }
}

function estimateEloDelta(score) {
    const s = Math.min(0.99, Math.max(0.01, score));
    const delta = 400 * Math.log10(s / (1 - s));
    return Math.max(-800, Math.min(800, delta));
}

function summarize(matches) {
    let win = 0;
    let loss = 0;
    let draw = 0;
    let error = 0;
    let aiKills = 0;
    let aiDeaths = 0;
    let aiSelfKills = 0;
    let itemControlSum = 0;

    for (const m of matches) {
        if (m.result === "win") win += 1;
        else if (m.result === "loss") loss += 1;
        else if (m.result === "draw") draw += 1;
        else error += 1;

        aiKills += m.ai_kills || 0;
        aiDeaths += m.ai_deaths || 0;
        aiSelfKills += m.ai_self_kills || 0;
        itemControlSum += Number(m.item_control || 0);
    }

    const total = Math.max(1, matches.length);
    const score = (win + 0.5 * draw) / total;
    const eloDelta = estimateEloDelta(score);

    return {
        matches: matches.length,
        win_count: win,
        loss_count: loss,
        draw_count: draw,
        error_count: error,
        win_rate: win / total,
        loss_rate: loss / total,
        draw_rate: draw / total,
        error_rate: error / total,
        ai_kills_total: aiKills,
        ai_deaths_total: aiDeaths,
        ai_self_kills_total: aiSelfKills,
        self_kill_rate: aiSelfKills / Math.max(1, aiDeaths),
        ai_kd: aiKills / Math.max(1, aiDeaths),
        item_control: itemControlSum / total,
        elo_result: {
            ai_baseline: 1000,
            opponent_baseline: 1000,
            ai_estimated: 1000 + eloDelta,
            opponent_estimated: 1000 - eloDelta,
            ai_delta: eloDelta,
            score
        }
    };
}

async function main() {
    const runs = asPositiveInt(getArg("runs", getArg("matches", "200")), 200);
    const matchDurationSec = asPositiveInt(getArg("match-duration-sec", "45"), 45);
    const parallel = asPositiveInt(getArg("parallel", "8"), 8);
    const seedBase = asInt(getArg("seed-base", "20260419"), 20260419);
    const modelUrl = getArg("model-url", "/output/ml/models/combat_phase0_iql_v1.onnx");
    const reportPath = path.resolve(getArg("report-path", REPORT_PATH_DEFAULT));
    const mapId = getArg("map", getArg("map-id", "windmill-heart"));
    const policyMode = getArg("policy-mode", "pure");
    const opponentMode = normalizeOpponentMode(getArg("opponent", getArg("opponent-mode", "heuristic_v2")));
    const opponentThinkMs = asPositiveInt(getArg("opponent-think-ms", "120"), 120);
    const saveScreenshots = getArg("save-screenshots", "0") === "1";

    const cfg = {
        mlEnabled: true,
        modelUrl,
        mapId,
        policyMode,
        matchDurationSec,
        seedBase,
        opponentMode,
        opponentThinkMs,
        saveScreenshots,
        mlConf: asFloatOrNull(getArg("ml-conf", "")),
        mlMoveConf: asFloatOrNull(getArg("ml-move-conf", "")),
        mlMargin: asFloatOrNull(getArg("ml-margin", "")),
        mlForceMoveEta: asFloatOrNull(getArg("ml-force-move-eta", "")),
        mlWaitBlockEta: asFloatOrNull(getArg("ml-wait-block-eta", "")),
        mlMoveThreatMs: asFloatOrNull(getArg("ml-move-threat-ms", ""))
    };

    fs.mkdirSync(path.dirname(reportPath), { recursive: true });
    fs.mkdirSync(OUT_DIR, { recursive: true });

    const startedAt = Date.now();
    const allMatches = [];
    let completed = 0;
    let nextIndex = 0;

    async function worker() {
        while (true) {
            const i = nextIndex;
            nextIndex += 1;
            if (i >= runs) {
                return;
            }
            const m = await runMatch(i, cfg);
            allMatches.push(m);
            completed += 1;
            if (completed % 10 === 0 || completed === runs) {
                const s = summarize(allMatches);
                console.log(
                    "[COMBAT-EVAL]",
                    `progress=${completed}/${runs}`,
                    `win_rate=${s.win_rate.toFixed(4)}`,
                    `self_kill_rate=${s.self_kill_rate.toFixed(4)}`,
                    `draw_rate=${s.draw_rate.toFixed(4)}`,
                    `item_control=${s.item_control.toFixed(4)}`,
                    `elo_delta=${s.elo_result.ai_delta.toFixed(2)}`
                );
            }
        }
    }

    const workerCount = Math.max(1, Math.min(parallel, runs));
    await Promise.all(new Array(workerCount).fill(0).map(() => worker()));

    const summary = summarize(allMatches);
    const report = {
        ts: Date.now(),
        duration_sec: (Date.now() - startedAt) / 1000,
        protocol: {
            mode: "combat_1v1",
            fixed_map_id: mapId,
            fixed_opponent_mode: opponentMode,
            fixed_seed_base: seedBase,
            runs,
            match_duration_sec: matchDurationSec,
            opponent_think_ms: opponentThinkMs
        },
        ml_cfg: {
            model_url: modelUrl,
            policy_mode: policyMode,
            ml_conf: cfg.mlConf,
            ml_move_conf: cfg.mlMoveConf,
            ml_margin: cfg.mlMargin,
            ml_force_move_eta: cfg.mlForceMoveEta,
            ml_wait_block_eta: cfg.mlWaitBlockEta,
            ml_move_threat_ms: cfg.mlMoveThreatMs
        },
        summary,
        matches: allMatches
    };

    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
    console.log("[COMBAT-EVAL-DONE]", JSON.stringify({ report_path: reportPath, summary }));
}

main().catch((err) => {
    console.error(err);
    process.exit(1);
});
