#!/usr/bin/env bash
set -Eeuo pipefail

APP_DIR="${APP_DIR:-/opt/final-face}"
SERVICE_NAME="${SERVICE_NAME:-final-face-listeners.service}"
LINES="${LINES:-250}"
SEARCH_TERM="${1:-${SEARCH_TERM:-}}"
TODAY="${TODAY:-$(TZ=Asia/Jakarta date +%F)}"
OUT_DIR="${OUT_DIR:-/tmp/final-face-diagnostics-$TODAY-$(date +%H%M%S)}"

mkdir -p "$OUT_DIR"

run_capture() {
  local name="$1"
  shift
  printf '\n=== %s ===\n' "$name" | tee -a "$OUT_DIR/summary.txt"
  { "$@" 2>&1 || true; } | tee "$OUT_DIR/$name.txt" | tail -n 80 | tee -a "$OUT_DIR/summary.txt"
}

cd "$APP_DIR"

run_capture service-status systemctl --no-pager --full status "$SERVICE_NAME"
run_capture listener-journal journalctl -u "$SERVICE_NAME" -n "$LINES" --no-pager
run_capture gate-journal bash -c 'journalctl -u "$1" -n "$2" --no-pager | grep -Ei "Gate|alwaysOpen|alwaysClose|manual-open|manual-closed|RemoteControl"' _ "$SERVICE_NAME" "$LINES"
run_capture git-status git status --short --branch
run_capture latest-commits git --no-pager log --oneline --decorate -n 8

if [[ -f "$APP_DIR/backend/data/attendance/$TODAY.json" ]]; then
  run_capture today-attendance tail -n 220 "$APP_DIR/backend/data/attendance/$TODAY.json"
else
  printf 'No local attendance file found: %s\n' "$APP_DIR/backend/data/attendance/$TODAY.json" | tee "$OUT_DIR/today-attendance.txt" | tee -a "$OUT_DIR/summary.txt"
fi

if [[ -n "$SEARCH_TERM" ]]; then
  run_capture search-journal bash -c 'journalctl -u "$1" -n "$2" --no-pager | grep -i -- "$3"' _ "$SERVICE_NAME" "$LINES" "$SEARCH_TERM"
  if [[ -f "$APP_DIR/backend/data/attendance/$TODAY.json" ]]; then
    run_capture search-local-attendance grep -i -- "$SEARCH_TERM" "$APP_DIR/backend/data/attendance/$TODAY.json"
  fi
fi

printf '\nDiagnostics written to %s\n' "$OUT_DIR" | tee -a "$OUT_DIR/summary.txt"