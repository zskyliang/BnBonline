const fs = require("fs");
const path = require("path");
const { execSync } = require("child_process");

const ROOT = path.resolve(__dirname, "..");

const DATASET_PATH_DEFAULT = path.join(ROOT, "output/ml/datasets/combat_phase0_iql_v1.jsonl");
const COLLECT_REPORT_DEFAULT = path.join(ROOT, "output/ml/reports/combat_phase0_collect_stats.json");
const TRAIN_REPORT_DEFAULT = path.join(ROOT, "output/ml/reports/combat_phase0_iql_metrics.json");
const EVAL_REPORT_DEFAULT = path.join(ROOT, "output/ml/reports/combat_1v1_eval_report.json");
const ROUND_REPORT_DEFAULT = path.join(ROOT, "output/ml/reports/combat_phase0_rounds.json");

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
    const n = parseFloat(v);
    return Number.isFinite(n) ? n : fallback;
}

function toPp(deltaRate) {
    if (typeof deltaRate !== "number" || !Number.isFinite(deltaRate)) {
        return null;
    }
    return deltaRate * 100;
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
    try {
        return JSON.parse(fs.readFileSync(filePath, "utf-8"));
    } catch (err) {
        return null;
    }
}

function run(cmd, dryRun) {
    console.log("[RUN]", cmd);
    if (dryRun) {
        return;
    }
    execSync(cmd, { cwd: ROOT, stdio: "inherit" });
}

function ensureServer(baseUrl) {
    try {
        execSync(`curl -fsS "${baseUrl}" >/dev/null`, { cwd: ROOT, stdio: "ignore" });
    } catch (err) {
        throw new Error(
            "game server is not reachable at " + baseUrl + ". Please run `npm start` first."
        );
    }
}

function computeDominantActionRatio(actionHist) {
    const hist = actionHist || {};
    const values = Object.keys(hist).map((k) => Number(hist[k]) || 0);
    const total = values.reduce((acc, v) => acc + v, 0);
    if (total <= 0) {
        return 1;
    }
    const maxValue = values.reduce((acc, v) => Math.max(acc, v), 0);
    return maxValue / total;
}

function checkTrainHealth(trainReport) {
    const history = Array.isArray(trainReport && trainReport.history) ? trainReport.history : [];
    let finiteOk = true;
    for (const row of history) {
        const keys = ["train_loss", "val_policy_acc", "val_policy_f1", "val_risk_acc"];
        for (const key of keys) {
            if (typeof row[key] === "number" && !Number.isFinite(row[key])) {
                finiteOk = false;
            }
        }
    }
    if (typeof trainReport?.val_policy_f1 === "number" && !Number.isFinite(trainReport.val_policy_f1)) {
        finiteOk = false;
    }
    return {
        no_nan: finiteOk,
        converged_like: typeof trainReport?.val_policy_f1 === "number" && trainReport.val_policy_f1 > 0.2
    };
}

function buildOptimizationBranchTemplate() {
    return {
        activated: true,
        rule: "single_variable_ablation_only",
        resume_condition: "increment_effective returns true on same fixed eval protocol",
        experiments: [
            {
                id: "feat_c8_enemy_position",
                type: "feature",
                description: "Add/verify enemy position channel C8 and retrain on same cumulative data."
            },
            {
                id: "feat_c9_item_heat",
                type: "feature",
                description: "Add/verify item heat channel C9 and retrain on same cumulative data."
            },
            {
                id: "feat_temporal_recent_bomb_enemy_heading",
                type: "feature",
                description: "Add temporal signal: recent bomb placement + enemy heading."
            },
            {
                id: "data_resample_self_bomb_and_chase_fail",
                type: "data",
                description: "Oversample failure slices: self-bomb, chase-fail, corridor draw."
            },
            {
                id: "data_action_rebalance_hard_samples",
                type: "data",
                description: "Rebalance action histogram and hard-sample mining on difficult states."
            }
        ]
    };
}

function main() {
    const baseUrl = getArg("base-url", "http://127.0.0.1:4000/");
    const dryRun = getArg("dry-run", "0") === "1";
    const resume = getArg("resume", "0") === "1";

    const stepFrames = asPositiveInt(getArg("step-frames", "200000"), 200000);
    const maxFrames = asPositiveInt(getArg("max-frames", "1200000"), 1200000);

    const evalMatches = asPositiveInt(getArg("eval-matches", "200"), 200);
    const evalMatchDurationSec = asPositiveInt(getArg("eval-match-duration-sec", "45"), 45);
    const evalSeedBase = asInt(getArg("eval-seed-base", "20260419"), 20260419);
    const evalMapId = getArg("eval-map-id", "windmill-heart");
    const evalOpponentMode = getArg("eval-opponent-mode", "scripted_heuristic_v1");

    const gateWinPp = asFloat(getArg("gate-win-pp", "2.0"), 2.0);
    const gateSelfKillPp = asFloat(getArg("gate-self-kill-pp", "1.0"), 1.0);
    const gateDrawPp = asFloat(getArg("gate-draw-pp", "3.0"), 3.0);

    const phaseTargetWinRate = asFloat(getArg("phase-target-win-rate", "0.35"), 0.35);
    const phaseTargetSelfKillRate = asFloat(getArg("phase-target-self-kill-rate", "0.25"), 0.25);

    const actionCollapseMaxRatio = asFloat(getArg("action-collapse-max-ratio", "0.92"), 0.92);
    const riskMinRatio = asFloat(getArg("risk-min-ratio", "0.01"), 0.01);
    const riskMaxRatio = asFloat(getArg("risk-max-ratio", "0.99"), 0.99);

    const datasetPath = path.resolve(getArg("dataset-path", DATASET_PATH_DEFAULT));
    const collectReportPath = path.resolve(getArg("collect-report", COLLECT_REPORT_DEFAULT));
    const trainReportPath = path.resolve(getArg("train-report", TRAIN_REPORT_DEFAULT));
    const evalReportPath = path.resolve(getArg("eval-report", EVAL_REPORT_DEFAULT));
    const roundReportPath = path.resolve(getArg("round-report", ROUND_REPORT_DEFAULT));

    const modelPtPath = getArg("model-pt", "output/ml/models/combat_phase0_iql_v1.pt");
    const modelOnnxPath = getArg("model-onnx", "output/ml/models/combat_phase0_iql_v1.onnx");
    const modelUrl = getArg("model-url", "/output/ml/models/combat_phase0_iql_v1.onnx");

    const mixExpertRatio = asFloat(getArg("mix-expert-ratio", "0.6"), 0.6);
    const mixRandomRatio = asFloat(getArg("mix-random-ratio", "0.2"), 0.2);
    const mixEpsilonRatio = asFloat(getArg("mix-epsilon-ratio", "0.2"), 0.2);

    if (maxFrames < stepFrames) {
        throw new Error("max-frames must be >= step-frames");
    }

    ensureServer(baseUrl);
    fs.mkdirSync(path.dirname(roundReportPath), { recursive: true });
    fs.mkdirSync(path.dirname(datasetPath), { recursive: true });

    if (!resume && fs.existsSync(datasetPath) && !dryRun) {
        fs.unlinkSync(datasetPath);
    }

    let currentRows = resume ? countLines(datasetPath) : 0;
    let prevRound = null;
    let consecutiveIneffective = 0;
    const rounds = [];
    let stopReason = "";
    let optimizationBranch = null;

    for (let target = stepFrames; target <= maxFrames; target += stepFrames) {
        const roundIndex = rounds.length + 1;
        const rowsBefore = currentRows;
        const need = Math.max(0, target - currentRows);

        if (need > 0) {
            const freshFlag = rowsBefore <= 0 ? "1" : "0";
            run(
                "node scripts/collect-dodge-dataset.js "
                + `--target-frames=${need} --fresh=${freshFlag} `
                + `--dataset-path=${datasetPath} --report-path=${collectReportPath} `
                + "--screenshot-path=output/ml/reports/combat_phase0_collect_latest.png "
                + "--ml-runtime=0 --policy-mode=pure --iql-mix=1 --quota-enabled=0 --policy-balance-enabled=1 "
                + `--mix-expert-ratio=${mixExpertRatio} --mix-random-ratio=${mixRandomRatio} --mix-epsilon-ratio=${mixEpsilonRatio}`,
                dryRun
            );
        }

        currentRows = dryRun ? target : countLines(datasetPath);
        const addedRows = currentRows - rowsBefore;
        const incrementExpected = need;
        const incrementVerified = addedRows === incrementExpected;

        run(
            "python3 ml/train_iql.py "
            + `--dataset ${datasetPath} `
            + "--epochs=12 --batch-size=512 --sampler=balanced "
            + `--out-pt ${modelPtPath} --out-onnx ${modelOnnxPath} --out-metrics ${trainReportPath}`,
            dryRun
        );

        run(
            "node scripts/eval-combat-1v1.js "
            + `--matches=${evalMatches} --match-duration-sec=${evalMatchDurationSec} `
            + `--seed-base=${evalSeedBase} --map-id=${evalMapId} --opponent-mode=${evalOpponentMode} `
            + `--model-url=${modelUrl} --report-path=${evalReportPath} --policy-mode=pure`,
            dryRun
        );

        const collectReport = dryRun ? {} : (readJson(collectReportPath) || {});
        const trainReport = dryRun ? {} : (readJson(trainReportPath) || {});
        const evalReport = dryRun ? {} : (readJson(evalReportPath) || {});
        const evalSummary = evalReport.summary || {};

        const winRate = typeof evalSummary.win_rate === "number" ? evalSummary.win_rate : null;
        const selfKillRate = typeof evalSummary.self_kill_rate === "number" ? evalSummary.self_kill_rate : null;
        const drawRate = typeof evalSummary.draw_rate === "number" ? evalSummary.draw_rate : null;

        const deltaWinPp = prevRound && typeof winRate === "number" && typeof prevRound.win_rate === "number"
            ? toPp(winRate - prevRound.win_rate)
            : null;
        const deltaSelfKillPp = prevRound && typeof selfKillRate === "number" && typeof prevRound.self_kill_rate === "number"
            ? toPp(selfKillRate - prevRound.self_kill_rate)
            : null;
        const deltaDrawPp = prevRound && typeof drawRate === "number" && typeof prevRound.draw_rate === "number"
            ? toPp(drawRate - prevRound.draw_rate)
            : null;

        let incrementEffective = null;
        if (prevRound && deltaWinPp != null && deltaSelfKillPp != null && deltaDrawPp != null) {
            incrementEffective = (
                deltaWinPp >= gateWinPp
                && deltaSelfKillPp <= gateSelfKillPp
                && deltaDrawPp <= gateDrawPp
            );
        }

        if (prevRound && incrementEffective === false) {
            consecutiveIneffective += 1;
        } else if (incrementEffective === true) {
            consecutiveIneffective = 0;
        }

        const dominantActionRatio = computeDominantActionRatio(trainReport.action_hist);
        const riskLabelRatio = typeof trainReport.risk_label_ratio === "number" ? trainReport.risk_label_ratio : null;
        const dataDistributionOk = (
            dominantActionRatio <= actionCollapseMaxRatio
            && riskLabelRatio != null
            && riskLabelRatio >= riskMinRatio
            && riskLabelRatio <= riskMaxRatio
        );

        const trainHealth = checkTrainHealth(trainReport);
        const phaseTargetReached = (
            typeof winRate === "number"
            && typeof selfKillRate === "number"
            && winRate >= phaseTargetWinRate
            && selfKillRate <= phaseTargetSelfKillRate
        );

        const roundStopReasons = [];
        if (!incrementVerified) {
            roundStopReasons.push("data_increment_mismatch");
        }
        if (!dataDistributionOk) {
            roundStopReasons.push("data_distribution_collapse_risk");
        }
        if (!trainHealth.no_nan) {
            roundStopReasons.push("train_nan_or_inf_detected");
        }
        if (phaseTargetReached) {
            roundStopReasons.push("phase0_target_reached");
        }
        if (consecutiveIneffective >= 2) {
            roundStopReasons.push("enter_optimization_branch");
        }
        if (target >= maxFrames) {
            roundStopReasons.push("max_frames_reached");
        }

        const roundSummary = {
            round: roundIndex,
            target_rows: target,
            dataset_rows_before: rowsBefore,
            dataset_rows_after: currentRows,
            added_rows: addedRows,
            increment_expected_rows: incrementExpected,
            increment_verified: incrementVerified,

            win_rate: winRate,
            self_kill_rate: selfKillRate,
            draw_rate: drawRate,

            delta_win_rate: deltaWinPp,
            delta_self_kill_rate: deltaSelfKillPp,
            delta_draw_rate: deltaDrawPp,
            increment_effective: incrementEffective,

            collect: {
                spawned_bubbles_effective: collectReport.spawned_bubbles_effective || collectReport.spawned_bubbles || 0,
                spawned_bubbles_ignored_trapped: collectReport.spawned_bubbles_ignored_trapped || 0,
                bombed_count: collectReport.bombed_count || 0,
                survival_rate: collectReport.survival_rate,
                action_hist_written: collectReport.action_hist_written || {},
                policy_tag_hist_written: collectReport.policy_tag_hist_written || {}
            },
            train: {
                dataset_size: trainReport.dataset_size || 0,
                val_policy_acc: trainReport.val_policy_acc || 0,
                val_policy_f1: trainReport.val_policy_f1 || 0,
                val_risk_acc: trainReport.val_risk_acc || 0,
                risk_label_ratio: riskLabelRatio,
                dominant_action_ratio: dominantActionRatio,
                no_nan: trainHealth.no_nan,
                converged_like: trainHealth.converged_like
            },
            eval: {
                matches: evalSummary.matches || 0,
                win_count: evalSummary.win_count || 0,
                loss_count: evalSummary.loss_count || 0,
                draw_count: evalSummary.draw_count || 0,
                ai_kd: evalSummary.ai_kd || 0
            },
            stop_reason: roundStopReasons.length > 0 ? roundStopReasons.join(",") : ""
        };

        rounds.push(roundSummary);

        const report = {
            ts: Date.now(),
            phase: "phase0_incremental_gated",
            config: {
                dataset_path: datasetPath,
                collect_report_path: collectReportPath,
                train_report_path: trainReportPath,
                eval_report_path: evalReportPath,
                round_report_path: roundReportPath,
                step_frames: stepFrames,
                max_frames: maxFrames,
                phase_target: {
                    win_rate: phaseTargetWinRate,
                    self_kill_rate: phaseTargetSelfKillRate
                },
                gate_thresholds_pp: {
                    delta_win_rate_min: gateWinPp,
                    delta_self_kill_rate_max: gateSelfKillPp,
                    delta_draw_rate_max: gateDrawPp
                },
                fixed_eval_protocol: {
                    map_id: evalMapId,
                    opponent_mode: evalOpponentMode,
                    seed_base: evalSeedBase,
                    matches: evalMatches,
                    match_duration_sec: evalMatchDurationSec
                }
            },
            status: {
                consecutive_ineffective_rounds: consecutiveIneffective,
                latest_stop_reason: roundSummary.stop_reason || ""
            },
            rounds,
            optimization_branch: optimizationBranch
        };

        fs.writeFileSync(roundReportPath, JSON.stringify(report, null, 2));

        prevRound = {
            win_rate: winRate,
            self_kill_rate: selfKillRate,
            draw_rate: drawRate
        };

        if (phaseTargetReached) {
            stopReason = "phase0_target_reached";
            break;
        }
        if (consecutiveIneffective >= 2) {
            stopReason = "enter_optimization_branch";
            optimizationBranch = buildOptimizationBranchTemplate();
            break;
        }
        if (!incrementVerified || !trainHealth.no_nan) {
            stopReason = roundSummary.stop_reason || "fatal_validation_failure";
            break;
        }
    }

    if (!stopReason) {
        stopReason = "max_frames_reached";
    }

    const final = readJson(roundReportPath) || {};
    final.ts = Date.now();
    final.status = final.status || {};
    final.status.final_stop_reason = stopReason;
    if (optimizationBranch) {
        final.optimization_branch = optimizationBranch;
    }
    fs.writeFileSync(roundReportPath, JSON.stringify(final, null, 2));

    console.log("[PHASE0-DONE]", JSON.stringify({
        stop_reason: stopReason,
        rounds: rounds.length,
        report_path: roundReportPath
    }));
}

main();
