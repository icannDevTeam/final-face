#!/usr/bin/env bash
# pandora_install_pickup_health_check.sh — one-time installer for the pickup
# device health check timer on Pandora.
#
# Installs pickup-health-check.service + .timer:
#   • 10 min after every boot        (catches the cold-boot stream race —
#                                     workers alive but streams never attached)
#   • Mon–Fri 06:10 Asia/Jakarta     (post power-on, before morning attendance)
#   • Mon–Fri 09:00 Asia/Jakarta     (mid-morning sanity)
#   • Mon–Fri 12:30 Asia/Jakarta     (~1h before main pickup windows)
#
# The check runs as root with --kill-stalled so stalled workers are killed and
# respawned automatically (the manager reconnects streams in ~10s). Clock/TZ
# drift on the terminals is auto-fixed. Device reboots stay manual.
#
# Usage (as root):
#   sudo bash backend/scripts/pandora_install_pickup_health_check.sh
#
# Optional overrides (env vars):
#   APP_DIR=/home/pandora/Downloads/final-face   # checkout containing the tool + .venv
set -Eeuo pipefail

APP_DIR="${APP_DIR:-/home/pandora/Downloads/final-face}"
TOOL="$APP_DIR/backend/tools/device_health_check.py"
PYTHON="$APP_DIR/.venv/bin/python"

UNIT_SERVICE=/etc/systemd/system/pickup-health-check.service
UNIT_TIMER=/etc/systemd/system/pickup-health-check.timer

log() { printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

[[ "$(id -u)" -eq 0 ]] || die "run as root:  sudo bash $0"
[[ -f "$TOOL" ]] || die "health check tool not found: $TOOL"
[[ -x "$PYTHON" ]] || die "venv python not found: $PYTHON"

log "Writing $UNIT_SERVICE"
cat > "$UNIT_SERVICE" <<EOF
[Unit]
Description=Pickup device health check (clocks, streams, auto-remediation)
Wants=network-online.target
After=network-online.target final-face-listeners.service

[Service]
Type=oneshot
# root: --kill-stalled must signal workers owned by the final-face user
User=root
WorkingDirectory=$APP_DIR
ExecStart=$PYTHON $TOOL --kill-stalled
TimeoutStartSec=300
EOF

log "Writing $UNIT_TIMER"
cat > "$UNIT_TIMER" <<EOF
[Unit]
Description=Run pickup health check after boot and before each window

[Timer]
OnBootSec=10min
OnCalendar=Mon..Fri 06:10 Asia/Jakarta
OnCalendar=Mon..Fri 09:00 Asia/Jakarta
OnCalendar=Mon..Fri 12:30 Asia/Jakarta
Persistent=true

[Install]
WantedBy=timers.target
EOF

log "Reloading systemd + enabling timer"
systemctl daemon-reload
systemctl enable --now pickup-health-check.timer

log "Installed. Next runs:"
systemctl list-timers pickup-health-check.timer --no-pager
log "Manual run:  sudo systemctl start pickup-health-check.service && journalctl -u pickup-health-check.service -n 30 --no-pager"
