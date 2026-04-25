#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$SCRIPT_DIR/.game-server.pid"
DEFAULT_BATTLE_URL="http://127.0.0.1:4000/?mode=battle&ml=1&ml_policy_mode=pure&ml_conf=0.26&ml_move_conf=0.34&ml_margin=0.03&ml_force_move_eta=460&ml_wait_block_eta=760&ml_move_threat_ms=300&ml_model=/output/ml/models/dodge_iql_v1.onnx"
BATTLE_URL="${BATTLE_URL:-$DEFAULT_BATTLE_URL}"

stop_server() {
  local pid="$1"
  local i

  if ! kill -0 "$pid" 2>/dev/null; then
    return 0
  fi

  kill "$pid" 2>/dev/null || true
  for i in {1..20}; do
    if ! kill -0 "$pid" 2>/dev/null; then
      return 0
    fi
    sleep 0.1
  done

  kill -9 "$pid" 2>/dev/null || true
}

stop_port_4000_servers() {
  local pids=""
  local pid=""
  pids="$(lsof -tiTCP:4000 -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -z "$pids" ]]; then
    return 0
  fi

  echo "Stopping process(es) listening on :4000 -> $pids"
  for pid in $pids; do
    stop_server "$pid"
  done
}

if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "${OLD_PID:-}" ]]; then
    echo "Stopping game server (PID: $OLD_PID)..."
    stop_server "$OLD_PID"
  fi
  rm -f "$PID_FILE"
fi

stop_port_4000_servers

"$SCRIPT_DIR/start-game.sh"

echo ""
echo "Battle URL:"
echo "$BATTLE_URL"
