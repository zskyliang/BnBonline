node scripts/eval-combat-1v1.js \
  --model-url=/output/ml/models/combat_phase0_iql_v1.onnx \
  --opponent=heuristic_v2 \
  --runs=5 \
  --parallel=1 \
  --match-duration-sec=45 \
  --live-view=1