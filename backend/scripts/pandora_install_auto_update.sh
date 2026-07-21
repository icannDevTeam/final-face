#!/usr/bin/env bash
# pandora_install_auto_update.sh — one-time installer for the pull-based
# auto-deploy timer on Pandora.
#
# After this runs, every `git push origin main` from any machine is picked up
# automatically: the timer polls GitHub every 2 minutes, fast-forwards
# /opt/final-face, installs changed backend requirements, and restarts the
# listener service. No inbound ports are opened.
#
# Usage (as root):
#   sudo bash /opt/final-face/backend/scripts/pandora_install_auto_update.sh
#
# Optional overrides (env vars):
#   APP_DIR=/opt/final-face          repo checkout
#   SERVICE_USER=final-face          unix user that owns the checkout
#   SERVICE_NAME=final-face-listeners.service
#   REMOTE=origin
#   BRANCH=                          empty = follow currently checked-out branch
#   POLL_INTERVAL=2min               how often to poll GitHub
set -Eeuo pipefail

APP_DIR="${APP_DIR:-/opt/final-face}"
SERVICE_USER="${SERVICE_USER:-final-face}"
SERVICE_NAME="${SERVICE_NAME:-final-face-listeners.service}"
REMOTE="${REMOTE:-origin}"
BRANCH="${BRANCH:-}"
POLL_INTERVAL="${POLL_INTERVAL:-2min}"

UPDATE_SCRIPT="$APP_DIR/backend/scripts/pandora_auto_update.sh"
UNIT_SERVICE=/etc/systemd/system/final-face-auto-update.service
UNIT_TIMER=/etc/systemd/system/final-face-auto-update.timer

log() { printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

[[ "$(id -u)" -eq 0 ]] || die "run as root:  sudo bash $0"
[[ -d "$APP_DIR/.git" ]] || die "$APP_DIR is not a git checkout (set APP_DIR=... if the repo lives elsewhere)"
[[ -f "$UPDATE_SCRIPT" ]] || die "$UPDATE_SCRIPT not found — run 'git pull' in $APP_DIR first"
id "$SERVICE_USER" >/dev/null 2>&1 || die "service user '$SERVICE_USER' does not exist (set SERVICE_USER=... to override)"

# ── Safety: never enable auto-merge over uncommitted production changes ──────
cd "$APP_DIR"
if [[ -n "$(runuser -u "$SERVICE_USER" -- git status --porcelain)" ]]; then
  runuser -u "$SERVICE_USER" -- git status --short
  die "uncommitted/untracked files in $APP_DIR — commit & push them first (see backend/scripts/pandora_preflight.sh)"
fi

chmod +x "$UPDATE_SCRIPT"

log "Writing $UNIT_SERVICE"
cat > "$UNIT_SERVICE" <<EOF
[Unit]
Description=Pull latest final-face code and restart listeners
Wants=network-online.target
After=network-online.target

[Service]
Type=oneshot
Environment=APP_DIR=$APP_DIR
Environment=SERVICE_USER=$SERVICE_USER
Environment=REMOTE=$REMOTE
Environment=SERVICE_NAME=$SERVICE_NAME
${BRANCH:+Environment=BRANCH=$BRANCH}
ExecStart=$UPDATE_SCRIPT
EOF

log "Writing $UNIT_TIMER (poll every $POLL_INTERVAL)"
cat > "$UNIT_TIMER" <<EOF
[Unit]
Description=Poll GitHub for final-face updates

[Timer]
OnBootSec=$POLL_INTERVAL
OnUnitActiveSec=$POLL_INTERVAL
AccuracySec=15s
Unit=final-face-auto-update.service

[Install]
WantedBy=timers.target
EOF

log "Enabling timer"
systemctl daemon-reload
systemctl enable --now final-face-auto-update.timer

log "Running one update right now"
systemctl start final-face-auto-update.service || true
journalctl -u final-face-auto-update.service -n 25 --no-pager || true

echo
systemctl list-timers final-face-auto-update.timer --no-pager || true
echo
log "Done. Every push to $REMOTE is now deployed automatically within ~$POLL_INTERVAL."
log "Watch deployments with:  journalctl -u final-face-auto-update.service -f"
