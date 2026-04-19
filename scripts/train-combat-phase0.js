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

function ensureServer(baseUrl) {
    try {
        execSync(`curl -fsS "${baseUrl}" >/dev/null`, { cwd: ROOT, stdio: "ignore" });
    } catch (err) {
        throw new Error(`game server is not reachable at ${baseUrl}. please run npm start first.`);
    }
}

function main() {
    const ts = Date.now();
    const baseUrl = getArg("base-url", "http://127.0.0.1:4000/");
    const targetFrames = asPositiveInt(getArg("target-frames", "200000"), 200000);
    const datasetPath = path.resolve(getArg("dataset", "output/ml/datasets/combat_phase0_v1.jsonl"));
    const collectReport = path.resolve(getArg("collect-report", `output/ml/reports/combat_phase0_collect_${ts}.json`));
    const trainReport = path.resolve(getArg("train-report", `output/ml/reports/combat_phase0_train_${ts}.json`));
    const evalReport = path.resolve(getArg("eval-report", `output/ml/reports/combat_phase0_eval_${ts}.json`));
    const modelPt = path.resolve(getArg("out-pt", "output/ml/models/combat_phase0_iql_v1.pt"));
    const modelOnnx = path.resolve(getArg("out-onnx", "output/ml/models/combat_phase0_iql_v1.onnx"));
    const modelUrl = getArg("model-url", "/output/ml/models/combat_phase0_iql_v1.onnx");

    const epochs = asPositiveInt(getArg("epochs", "60"), 60);
    const batchSize = asPositiveInt(getArg("batch-size", "512"), 512);
    const freezeConvEpochs = asInt(getArg("freeze-conv-epochs", "50"), 50);
    const initPt = path.resolve(getArg("init-pt", "output/ml/models/dodge_iql_v1.pt"));
    const runs = asPositiveInt(getArg("runs", "200"), 200);
    const opponent = getArg("opponent", "heuristic_v2");
    const mapId = getArg("map", "windmill-heart");

    fs.mkdirSync(path.dirname(collectReport), { recursive: true });
    fs.mkdirSync(path.dirname(datasetPath), { recursive: true });

    ensureServer(baseUrl);

    run(
        "node scripts/collect-combat-dataset.js "
        + `--target-frames=${targetFrames} --arena=1v1 --map=${mapId} --action-space=discrete6 --fresh=1 `
        + "--clear-nonrigid=1 --random-item-density=0.12 "
        + `--dataset-path=${datasetPath} --report-path=${collectReport}`
    );

    run(
        "python3 ml/train_iql_combat.py "
        + `--dataset ${datasetPath} --epochs ${epochs} --batch-size ${batchSize} `
        + `--freeze-conv-epochs ${freezeConvEpochs} --init-pt ${initPt} `
        + `--out-pt ${modelPt} --out-onnx ${modelOnnx} --out-metrics ${trainReport}`
    );

    run(
        "node scripts/eval-combat-1v1.js "
        + `--model-url=${modelUrl} --opponent=${opponent} --runs=${runs} --match-duration-sec=45 --map=${mapId} --report-path=${evalReport}`
    );

    console.log("[DONE]", JSON.stringify({
        dataset_path: datasetPath,
        collect_report: collectReport,
        train_report: trainReport,
        eval_report: evalReport,
        model_pt: modelPt,
        model_onnx: modelOnnx,
    }));
}

main();
