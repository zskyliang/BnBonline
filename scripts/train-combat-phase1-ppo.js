#!/usr/bin/env node
const fs = require("fs");
const path = require("path");
const { execSync, spawnSync } = require("child_process");

const ROOT = path.resolve(__dirname, "..");
const UV_CACHE_DIR = path.resolve(ROOT, "output", "uv-cache");

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

function ensureParent(filePath) {
    fs.mkdirSync(path.dirname(filePath), { recursive: true });
}

function run(cmd) {
    console.log("[RUN]", cmd);
    fs.mkdirSync(UV_CACHE_DIR, { recursive: true });
    execSync(cmd, {
        cwd: ROOT,
        stdio: "inherit",
        env: Object.assign({}, process.env, { UV_CACHE_DIR }),
    });
}

function runQuiet(cmd) {
    fs.mkdirSync(UV_CACHE_DIR, { recursive: true });
    return execSync(cmd, {
        cwd: ROOT,
        stdio: ["ignore", "pipe", "pipe"],
        env: Object.assign({}, process.env, { UV_CACHE_DIR }),
    }).toString("utf8").trim();
}

function readJsonMaybe(p) {
    if (!fs.existsSync(p)) return null;
    return JSON.parse(fs.readFileSync(p, "utf8"));
}

function mean(values) {
    if (!values || values.length <= 0) return 0;
    return values.reduce((a, b) => a + Number(b || 0), 0) / values.length;
}

function ensureServer(baseUrl) {
    const code = [
        "const http=require('http');",
        `const url=${JSON.stringify(baseUrl)};`,
        "const req=http.get(url,(res)=>{const ok=res.statusCode>=200&&res.statusCode<400;res.resume();process.exit(ok?0:2);});",
        "req.on('error',()=>process.exit(3));",
        "setTimeout(()=>process.exit(4),3000);",
    ].join("");
    const probe = require("child_process").spawnSync(process.execPath, ["-e", code], {
        cwd: ROOT,
        stdio: "ignore",
    });
    if (probe.status !== 0) {
        throw new Error(`game server is not reachable at ${baseUrl}. run npm start first.`);
    }
}

function ensureUv() {
    try {
        runQuiet("uv --version");
    } catch (_) {
        throw new Error("uv is required but not found in PATH.");
    }
}

function ensureCudaTorch(pyExe) {
    run(`uv pip install --python "${pyExe}" --index-url https://download.pytorch.org/whl/cu118 --force-reinstall torch==2.5.1+cu118`);
    const checkCode = [
        "import json, torch",
        "ok = bool(torch.cuda.is_available())",
        "name = torch.cuda.get_device_name(0) if ok else ''",
        "print(json.dumps({'cuda_available': ok, 'device_name': name}))",
        "raise SystemExit(0 if (ok and ('NVIDIA' in name.upper())) else 7)",
    ].join("; ");
    const probe = spawnSync(pyExe, ["-c", checkCode], {
        cwd: ROOT,
        encoding: "utf8",
    });
    const out = String(probe.stdout || "").trim();
    const err = String(probe.stderr || "").trim();
    let info = null;
    try {
        info = out ? JSON.parse(out) : null;
    } catch (_) {
        info = null;
    }
    if (probe.status !== 0) {
        throw new Error(
            "CUDA preflight failed. expected NVIDIA CUDA device for PPO. "
            + `stdout=${out || "<empty>"} stderr=${err || "<empty>"}`
        );
    }
    return info || { cuda_available: true, device_name: "NVIDIA" };
}

function tsTag() {
    const d = new Date();
    const yyyy = d.getFullYear();
    const mm = String(d.getMonth() + 1).padStart(2, "0");
    const dd = String(d.getDate()).padStart(2, "0");
    const hh = String(d.getHours()).padStart(2, "0");
    const mi = String(d.getMinutes()).padStart(2, "0");
    const ss = String(d.getSeconds()).padStart(2, "0");
    return `${yyyy}${mm}${dd}_${hh}${mi}${ss}`;
}

function main() {
    const budget = String(getArg("budget", "quick")).trim().toLowerCase();
    const budgetCfg = budget === "intensive"
        ? { timesteps: 600_000, targetFrames: 18_000, nEnvs: 10, nSteps: 512, batchSize: 1024, nEpochs: 3, workers: 1, evalRuns: 50, evalRounds: 3, evalParallel: 8, collectMaxWallSec: 28800, label: "intensive" }
        : (budget === "standard"
            ? { timesteps: 300_000, targetFrames: 12_000, nEnvs: 8, nSteps: 512, batchSize: 1024, nEpochs: 3, workers: 1, evalRuns: 50, evalRounds: 3, evalParallel: 8, collectMaxWallSec: 21600, label: "standard" }
            : (budget === "smoke"
                ? { timesteps: 20_000, targetFrames: 1_500, nEnvs: 4, nSteps: 256, batchSize: 256, nEpochs: 3, workers: 1, evalRuns: 10, evalRounds: 1, evalParallel: 4, collectMaxWallSec: 2400, label: "smoke" }
                : { timesteps: 120_000, targetFrames: 3_000, nEnvs: 6, nSteps: 256, batchSize: 384, nEpochs: 4, workers: 1, evalRuns: 50, evalRounds: 3, evalParallel: 8, collectMaxWallSec: 7200, label: "quick" }));

    const t = tsTag();
    const baseUrl = getArg("base-url", "http://127.0.0.1:4000/");
    const mapId = getArg("map", "windmill-heart");
    const fixedOpponent = getArg("opponent", "heuristic_v2");
    const workers = asPositiveInt(getArg("workers", String(budgetCfg.workers)), budgetCfg.workers);
    const parallelEval = asPositiveInt(getArg("eval-parallel", String(budgetCfg.evalParallel)), budgetCfg.evalParallel);
    const evalRuns = asPositiveInt(getArg("eval-runs", String(budgetCfg.evalRuns)), budgetCfg.evalRuns);
    const evalRounds = asPositiveInt(getArg("eval-rounds", String(budgetCfg.evalRounds)), budgetCfg.evalRounds);
    const seedBase = asInt(getArg("seed-base", "20260425"), 20260425);
    const totalTimesteps = asPositiveInt(getArg("total-timesteps", String(budgetCfg.timesteps)), budgetCfg.timesteps);
    const targetFrames = asPositiveInt(getArg("target-frames", String(budgetCfg.targetFrames)), budgetCfg.targetFrames);
    const nEnvs = asPositiveInt(getArg("n-envs", String(budgetCfg.nEnvs)), budgetCfg.nEnvs);
    const nSteps = asPositiveInt(getArg("n-steps", String(budgetCfg.nSteps)), budgetCfg.nSteps);
    const batchSize = asPositiveInt(getArg("batch-size", String(budgetCfg.batchSize)), budgetCfg.batchSize);
    const nEpochs = asPositiveInt(getArg("n-epochs", String(budgetCfg.nEpochs)), budgetCfg.nEpochs);
    const collectMaxWallSec = asPositiveInt(getArg("collect-max-wall-sec", String(budgetCfg.collectMaxWallSec)), budgetCfg.collectMaxWallSec);
    const ppoDevice = String(getArg("device", "cuda")).trim() || "cuda";
    const mirrorSampling = String(getArg("mirror-sampling", "1")).trim() === "0" ? "0" : "1";
    const collectCpuCapPercent = Math.max(5, Math.min(100, asFloat(getArg("collect-cpu-cap-percent", "70"), 70)));
    const collectCpuControlMs = asPositiveInt(getArg("collect-cpu-control-ms", "2500"), 2500);
    const collectCpuResumeHysteresis = Math.max(1, Math.min(30, asFloat(getArg("collect-cpu-resume-hysteresis", "8"), 8)));

    const initModelUrl = getArg(
        "init-onnx",
        "/output/ml/models/combat_phase0_iql_phase0_seq_full83539_quick8e_20260425.onnx"
    );

    const rolloutDataset = path.resolve(getArg("rollout-dataset", `output/ml/datasets/combat_phase1_rollout_${budgetCfg.label}_${t}.jsonl`));
    const collectReport = path.resolve(getArg("collect-report", `output/ml/reports/combat_phase1_collect_${budgetCfg.label}_${t}.json`));
    const trainReport = path.resolve(getArg("train-report", `output/ml/reports/combat_phase1_train_${budgetCfg.label}_${t}.json`));
    const finalReport = path.resolve(getArg("final-report", `output/ml/reports/combat_phase1_final_${budgetCfg.label}_${t}.json`));
    const outZip = path.resolve(getArg("out-zip", `output/ml/models/combat_phase1_ppo_${budgetCfg.label}_${t}.zip`));
    const outOnnx = path.resolve(getArg("out-onnx", `output/ml/models/combat_phase1_ppo_${budgetCfg.label}_${t}.onnx`));
    const outModelUrl = getArg("out-model-url", `/output/ml/models/${path.basename(outOnnx)}`);

    const evalReportA = path.resolve(getArg("eval-report-a", `output/ml/reports/combat_phase1_eval_a_${budgetCfg.label}_${t}.json`));
    const evalReportB = path.resolve(getArg("eval-report-b", `output/ml/reports/combat_phase1_eval_b_${budgetCfg.label}_${t}.json`));
    const evalReportC = path.resolve(getArg("eval-report-c", `output/ml/reports/combat_phase1_eval_c_${budgetCfg.label}_${t}.json`));
    const evalReportD = path.resolve(getArg("eval-report-d", `output/ml/reports/combat_phase1_eval_d_${budgetCfg.label}_${t}.json`));
    const evalPool = [evalReportA, evalReportB, evalReportC, evalReportD];
    const evalReports = evalPool.slice(0, Math.max(1, Math.min(evalRounds, evalPool.length)));

    [
        rolloutDataset,
        collectReport,
        trainReport,
        finalReport,
        outZip,
        outOnnx,
        ...evalReports,
    ].forEach(ensureParent);

    ensureServer(baseUrl);
    ensureUv();

    const venvDir = path.resolve(getArg("venv-dir", ".venv-phase1"));
    const pyExe = path.join(venvDir, "Scripts", "python.exe");

    if (!fs.existsSync(pyExe)) {
        run(`uv venv "${venvDir}" --python 3.11`);
    } else {
        console.log("[INFO] reuse existing venv:", venvDir);
    }
    run(`uv pip install --python "${pyExe}" -r ml/requirements.txt`);
    const cudaInfo = ensureCudaTorch(pyExe);
    console.log("[INFO] CUDA ready:", JSON.stringify(cudaInfo));

    run(
        "node scripts/collect-combat-dataset-parallel.js "
        + `--workers=${workers} --target-frames=${targetFrames} --max-wall-sec=${collectMaxWallSec} --min-final-ratio=0.20 `
        + `--arena=1v1 --map=${mapId} --action-space=discrete6 --fresh=1 `
        + `--dataset-path="${rolloutDataset}" --report-path="${collectReport}" `
        + "--clear-nonrigid=1 --sudden-death=1 --disable-revive=1 --ignore-enemy-self-kill=1 "
        + "--stall-no-progress-ms=12000 --partial-clear-min-ratio=0.35 --partial-clear-max-ratio=0.75 "
        + "--spawn-shortest-path-min=1 --spawn-shortest-path-max=10 "
        + `--opponent-pool=${fixedOpponent} --agent-pool=heuristic_v2 --agent-expert-duel=0 `
        + `--balanced=0 --mirror-sampling=${mirrorSampling} --early-commit-high-value=0 --behavior-scoring=0 `
        + `--cpu-cap-percent=${collectCpuCapPercent} --cpu-control-ms=${collectCpuControlMs} --cpu-resume-hysteresis=${collectCpuResumeHysteresis} `
        + `--ml-enabled=1 --ml-collect=1 --ml-freeze=1 --ml-policy-mode=pure --ml-iql-mix=0 --ml-model-url=${initModelUrl}`
    );

    run(
        `"${pyExe}" ml/train_phase1_ppo.py `
        + `--dataset "${rolloutDataset}" `
        + `--total-timesteps ${totalTimesteps} --n-envs ${nEnvs} --n-steps ${nSteps} --batch-size ${batchSize} --n-epochs ${nEpochs} --device ${ppoDevice} --seed ${seedBase} `
        + `--out-zip "${outZip}" --out-onnx "${outOnnx}" --out-report "${trainReport}"`
    );

    for (let i = 0; i < evalReports.length; i++) {
        const reportPath = evalReports[i];
        run(
            "node scripts/eval-combat-1v1.js "
            + `--model-url=${outModelUrl} --opponent=${fixedOpponent} --runs=${evalRuns} --parallel=${parallelEval} `
            + `--seed-base=${seedBase + i * 1000} --map=${mapId} --match-duration-sec=45 `
            + "--clear-nonrigid=1 --sudden-death=1 --disable-revive=1 --ignore-enemy-self-kill=1 "
            + "--stall-no-progress-ms=12000 --partial-clear-min-ratio=0.35 --partial-clear-max-ratio=0.75 "
            + "--spawn-shortest-path-min=1 --spawn-shortest-path-max=10 --random-item-density=0.12 "
            + `--live-view=0 --report-path="${reportPath}"`
        );
    }

    const evalSummaries = evalReports
        .map((p) => readJsonMaybe(p))
        .map((r) => (r && r.summary) ? r.summary : null)
        .filter(Boolean);

    const aggregated = {
        win_rate_mean: mean(evalSummaries.map((s) => s.win_rate)),
        self_kill_rate_mean: mean(evalSummaries.map((s) => s.self_kill_rate)),
        draw_rate_mean: mean(evalSummaries.map((s) => s.draw_rate)),
        item_control_mean: mean(evalSummaries.map((s) => s.item_control)),
        elo_delta_mean: mean(evalSummaries.map((s) => (s.elo_result ? s.elo_result.ai_delta : 0))),
    };
    const gates = {
        win_rate_gte_0_90: aggregated.win_rate_mean >= 0.90,
        self_kill_rate_lte_0_12: aggregated.self_kill_rate_mean <= 0.12,
        draw_rate_lte_0_25: aggregated.draw_rate_mean <= 0.25,
    };
    const passAll = gates.win_rate_gte_0_90 && gates.self_kill_rate_lte_0_12 && gates.draw_rate_lte_0_25;

    const report = {
        ts: Date.now(),
        phase: "phase1_ppo_fixed_opponent",
        budget: budgetCfg.label,
        fixed_opponent: fixedOpponent,
        map_id: mapId,
        init_model_url: initModelUrl,
        output_model_url: outModelUrl,
        artifacts: {
            rollout_dataset: rolloutDataset,
            collect_report: collectReport,
            train_report: trainReport,
            out_zip: outZip,
            out_onnx: outOnnx,
            eval_reports: evalReports,
        },
        training: {
            total_timesteps: totalTimesteps,
            n_envs: nEnvs,
            n_steps: nSteps,
            batch_size: batchSize,
            n_epochs: nEpochs,
            device: ppoDevice,
            target_frames: targetFrames,
            workers,
            collect_max_wall_sec: collectMaxWallSec,
            eval_rounds: evalReports.length,
            collect_cpu_cap_percent: collectCpuCapPercent,
            collect_cpu_control_ms: collectCpuControlMs,
            collect_cpu_resume_hysteresis: collectCpuResumeHysteresis,
            mirror_sampling: mirrorSampling !== "0",
            cuda_preflight: cudaInfo,
        },
        eval_summary: aggregated,
        eval_rounds: evalSummaries,
        gates,
        gate_pass: passAll,
    };

    fs.writeFileSync(finalReport, JSON.stringify(report, null, 2));
    console.log("[DONE]", JSON.stringify({ final_report: finalReport, gate_pass: passAll, output_model_url: outModelUrl }));
}

main();
