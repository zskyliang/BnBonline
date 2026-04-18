const { execSync } = require("child_process");
const path = require("path");

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

function run(cmd) {
    console.log("[RUN]", cmd);
    execSync(cmd, { cwd: ROOT, stdio: "inherit" });
}

function main() {
    const targetFrames = parseInt(getArg("target-frames", "200000"), 10) || 200000;
    const evalRuns = parseInt(getArg("eval-runs", "3"), 10) || 3;
    const evalDurationMs = parseInt(getArg("eval-duration-ms", "45000"), 10) || 45000;
    const waitMaxRatio = parseFloat(getArg("wait-max-ratio", "0.5"));
    const minMoveRatio = parseFloat(getArg("min-move-ratio", "0.1"));

    run(
        "node scripts/collect-dodge-dataset.js " +
        `--target-frames=${targetFrames} --fresh=1 --ml-runtime=1 ` +
        `--quota-enabled=1 --wait-max-ratio=${waitMaxRatio} --min-move-ratio=${minMoveRatio} ` +
        "--batch-size=4096 --poll-ms=300"
    );
    run(
        "python3 ml/train_bc.py " +
        "--dataset output/ml/datasets/dodge_bc_v1.jsonl " +
        "--epochs=8 --batch-size=512 --sampler=balanced --sampler-power=0.8 --max-wait-ratio-train=0.5 " +
        "--out-pt output/ml/models/best.pt --out-onnx output/ml/models/dodge_bc_v1.onnx --out-metrics output/ml/reports/bc_v1_metrics.json"
    );
    run(
        "node scripts/eval-ml-dodge.js " +
        `--duration-ms=${evalDurationMs} --runs=${evalRuns} --ml-conf=0.34 --model-url=/output/ml/models/dodge_bc_v1.onnx`
    );
}

main();
