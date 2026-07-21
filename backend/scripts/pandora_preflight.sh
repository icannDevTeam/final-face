#!/usr/bin/env bash
set -Eeuo pipefail

APP_DIR="${APP_DIR:-/opt/final-face}"
REMOTE="${REMOTE:-origin}"
BRANCH="${BRANCH:-}"

log() {
  printf '\n[%s] %s\n' "$(date --iso-8601=seconds)" "$*"
}

cd "$APP_DIR"
BRANCH="${BRANCH:-$(git branch --show-current)}"

log "Pandora repository preflight"
printf 'App dir: %s\n' "$APP_DIR"
printf 'Branch:  %s\n' "$BRANCH"
printf 'Remote:  %s\n' "$REMOTE"

log "Remotes"
git remote -v

log "Local status"
git status --short --branch

log "Latest local commits"
git --no-pager log --oneline --decorate -n 8

log "Fetching remote comparison target"
git fetch --prune "$REMOTE" "$BRANCH"

log "Ahead/behind compared with $REMOTE/$BRANCH"
git status --short --branch
git --no-pager log --oneline --left-right --cherry-pick HEAD..."$REMOTE/$BRANCH" -n 30 || true

if [[ -n "$(git status --porcelain)" ]]; then
  cat <<'EOF'

WARNING: Pandora has uncommitted or untracked files.
Do not enable auto-update yet.

Recommended next step on Pandora:

  cd /opt/final-face
  git status --short
  git diff > /tmp/pandora-uncommitted.patch
  git diff --cached > /tmp/pandora-staged.patch
  git ls-files --others --exclude-standard > /tmp/pandora-untracked-files.txt

Then review those files, commit the real production changes, and push them to GitHub before enabling the timer.
EOF
else
  cat <<'EOF'

OK: Working tree is clean.
If Pandora is not ahead of GitHub, it is ready for auto-update.
If Pandora is ahead, push those commits first or intentionally choose which branch should be source of truth.
EOF
fi