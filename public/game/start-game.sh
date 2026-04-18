#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PID_FILE="$SCRIPT_DIR/.game-server.pid"
LOG_FILE="$SCRIPT_DIR/.game-server.log"

if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "${OLD_PID:-}" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "Game server is already running (PID: $OLD_PID)."
    echo "URL: http://127.0.0.1:4000"
    exit 0
  fi
  rm -f "$PID_FILE"
fi

cd "$ROOT_DIR"
nohup node app.js >> "$LOG_FILE" 2>&1 &
NEW_PID=$!
echo "$NEW_PID" > "$PID_FILE"

sleep 0.3
if kill -0 "$NEW_PID" 2>/dev/null; then
  echo "Game server started (PID: $NEW_PID)."
  echo "URL: http://127.0.0.1:4000"
  echo "Log: $LOG_FILE"
else
  echo "Failed to start game server. Check log: $LOG_FILE" >&2
  rm -f "$PID_FILE"
  exit 1
fi
