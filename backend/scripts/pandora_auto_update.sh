#!/usr/bin/env bash
set -Eeuo pipefail

APP_DIR="${APP_DIR:-/opt/final-face}"
SERVICE_USER="${SERVICE_USER:-final-face}"
REMOTE="${REMOTE:-origin}"
SERVICE_NAME="${SERVICE_NAME:-final-face-listeners.service}"
VENV_DIR="${VENV_DIR:-$APP_DIR/backend/.venv}"
LOCK_FILE="${LOCK_FILE:-/tmp/final-face-pandora-update.lock}"

log() {
  printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*"
}

as_service_user() {
  if [[ "$(id -u)" -eq 0 && -n "$SERVICE_USER" ]]; then
    runuser -u "$SERVICE_USER" -- "$@"
  else
    "$@"
  fi
}

cd "$APP_DIR"
BRANCH="${BRANCH:-$(as_service_user git branch --show-current)}"
BRANCH="${BRANCH:-main}"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  log "Another update is already running; exiting."
  exit 0
fi

if [[ -n "$(as_service_user git status --porcelain)" ]]; then
  log "Local changes or untracked files detected in $APP_DIR; refusing to auto-merge."
  as_service_user git status --short
  exit 1
fi

current_head="$(as_service_user git rev-parse HEAD)"
log "Fetching $REMOTE/$BRANCH..."
as_service_user git fetch --prune "$REMOTE" "$BRANCH"
remote_head="$(as_service_user git rev-parse FETCH_HEAD)"

if [[ "$current_head" == "$remote_head" ]]; then
  log "Already up to date at $current_head."
  exit 0
fi

changed_files="$(as_service_user git diff --name-only "$current_head" "$remote_head")"
log "Updating $current_head -> $remote_head"

as_service_user git merge --ff-only FETCH_HEAD
as_service_user git submodule update --init --recursive

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  log "Creating Python virtualenv at $VENV_DIR"
  as_service_user python3 -m venv "$VENV_DIR"
fi

if grep -q '^backend/requirements\.txt$' <<<"$changed_files" || [[ ! -f "$VENV_DIR/.requirements-installed" ]]; then
  log "Installing backend Python requirements..."
  as_service_user "$VENV_DIR/bin/python" -m pip install --upgrade pip
  as_service_user "$VENV_DIR/bin/python" -m pip install -r "$APP_DIR/backend/requirements.txt"
  as_service_user bash -c 'date --iso-8601=seconds > "$1"' _ "$VENV_DIR/.requirements-installed"
fi

log "Restarting $SERVICE_NAME"
systemctl restart "$SERVICE_NAME"
systemctl --no-pager --full status "$SERVICE_NAME" | sed -n '1,20p'

log "Pandora update complete."