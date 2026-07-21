# Pandora Auto Deploy

Use this on the office Pandora box so code pushed from home is picked up automatically.

This setup is pull-based: Pandora polls GitHub, fast-forwards the checked-out branch, installs changed backend requirements, then restarts the listener systemd service. It does not open an inbound port on the office network.

Do not enable the timer until Pandora's current production changes are committed and pushed. If Pandora has the newest uncommitted code, make Pandora the source of truth first, then let home machines pull from GitHub after that.

## Assumptions

- Repo checkout on Pandora: `/opt/final-face`
- Python listener command: `backend/run_listeners.py`
- Branch to follow: current checked-out branch by default, or `BRANCH` if set
- Git remote: `origin`
- Service user: `final-face`

Adjust the paths or names below if Pandora uses different ones.

## 1. Prepare Pandora

First run the preflight check on Pandora:

```bash
cd /opt/final-face
chmod +x backend/scripts/pandora_preflight.sh
APP_DIR=/opt/final-face backend/scripts/pandora_preflight.sh
```

If it reports uncommitted or untracked files, stop here. Review and commit Pandora's production changes before enabling auto-update:

```bash
cd /opt/final-face
git status --short
git diff > /tmp/pandora-uncommitted.patch
git diff --cached > /tmp/pandora-staged.patch
git ls-files --others --exclude-standard > /tmp/pandora-untracked-files.txt
```

After review, commit the real production changes on Pandora and push them:

```bash
git add <files-you-verified>
git commit -m "Capture Pandora production changes"
git push origin main   # or master, if Pandora follows master
```

Only after that should home machines pull the updated branch and continue making fixes.

Then prepare the service user and directories if they do not already exist:

```bash
sudo adduser --system --group --home /opt/final-face final-face
sudo mkdir -p /opt/final-face /etc/final-face
sudo chown -R final-face:final-face /opt/final-face
```

Clone the repo as the service user, or move the existing checkout into `/opt/final-face`:

```bash
sudo -u final-face git clone git@github.com:YOUR_ORG/YOUR_REPO.git /opt/final-face
cd /opt/final-face
sudo -u final-face git checkout main   # or master, if Pandora follows master
sudo -u final-face git submodule update --init --recursive
```

Create the backend virtualenv once:

```bash
cd /opt/final-face
sudo -u final-face python3 -m venv backend/.venv
sudo -u final-face backend/.venv/bin/python -m pip install --upgrade pip
sudo -u final-face backend/.venv/bin/python -m pip install -r backend/requirements.txt
sudo chmod +x backend/scripts/pandora_auto_update.sh
```

Put production environment variables and secrets in `/etc/final-face/backend.env`, not in the unit file:

```bash
sudo install -o root -g final-face -m 0640 /dev/null /etc/final-face/backend.env
```

## 2. Listener systemd service

Create `/etc/systemd/system/final-face-listeners.service`:

```ini
[Unit]
Description=Final Face Hikvision listener manager
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=final-face
Group=final-face
WorkingDirectory=/opt/final-face/backend
EnvironmentFile=-/etc/final-face/backend.env
ExecStart=/opt/final-face/backend/.venv/bin/python /opt/final-face/backend/run_listeners.py
Restart=always
RestartSec=10
KillSignal=SIGTERM
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
```

Enable it:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now final-face-listeners.service
sudo systemctl status final-face-listeners.service
```

## 3. Auto-update service and timer

Create `/etc/systemd/system/final-face-auto-update.service`:

```ini
[Unit]
Description=Pull latest final-face code and restart listeners
Wants=network-online.target
After=network-online.target

[Service]
Type=oneshot
Environment=APP_DIR=/opt/final-face
Environment=SERVICE_USER=final-face
# Set this only if Pandora must follow a specific branch, for example master.
# Environment=BRANCH=main
Environment=REMOTE=origin
Environment=SERVICE_NAME=final-face-listeners.service
ExecStart=/opt/final-face/backend/scripts/pandora_auto_update.sh
```

Create `/etc/systemd/system/final-face-auto-update.timer`:

```ini
[Unit]
Description=Poll GitHub for final-face updates

[Timer]
OnBootSec=2min
OnUnitActiveSec=2min
AccuracySec=15s
Unit=final-face-auto-update.service

[Install]
WantedBy=timers.target
```

Enable it:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now final-face-auto-update.timer
systemctl list-timers final-face-auto-update.timer
```

## 4. Test a deployment

From home:

```bash
git push origin main   # or master, if Pandora follows master
```

On Pandora:

```bash
journalctl -u final-face-auto-update.service -n 80 --no-pager
journalctl -u final-face-listeners.service -n 80 --no-pager
```

The update script refuses to run if Pandora has uncommitted local changes. That is intentional: Pandora should be production runtime, while source changes are committed and pushed from your working machine.

## Emergency commands

Collect logs when a scan happens at school but does not appear on the iPad/dashboard:

```bash
cd /opt/final-face
chmod +x backend/scripts/collect_pandora_logs.sh
sudo APP_DIR=/opt/final-face SERVICE_NAME=final-face-listeners.service backend/scripts/collect_pandora_logs.sh "STUDENT_NAME_OR_ID"
```

Start with these files from the output directory:

- `summary.txt` — quick view of service status, listener logs, and local attendance
- `listener-journal.txt` — raw systemd logs from `run_listeners.py` and `attendance_listener.py`
- `today-attendance.txt` — Pandora's local `backend/data/attendance/YYYY-MM-DD.json`

If `today-attendance.txt` has the test face but the iPad does not, the listener wrote locally and the next check is Firestore/browser refresh. If the journal shows the scan but local attendance does not, check duplicate suppression, blocklist, or employee number mapping. If the journal never shows the scan, check terminal/network/listener connectivity.

Check a manual gate override on Pandora:

```bash
cd /opt/final-face
python3 backend/scripts/check_gate_override.py --query "GRADE_OR_GATE_NAME"
python3 backend/scripts/check_gate_override.py --query "GRADE_OR_GATE_NAME" --set open
sudo journalctl -u final-face-listeners.service -n 120 --no-pager | grep -Ei 'Gate|alwaysOpen|alwaysClose|manual-open|manual-closed'
```

Use `--set auto` after testing to clear the manual override and return to the configured pickup window / normal face-match behavior.

Seed demo pickup cards to exactly one paired iPad/tablet release group:

```bash
cd /opt/final-face/backend
python3 seed_one_paired_device_demo.py --list
python3 seed_one_paired_device_demo.py --query "GRADE_OR_TABLET_NAME"
```

If more than one group matches, copy the exact group id from `--list`:

```bash
python3 seed_one_paired_device_demo.py --release-group RELEASE_GROUP_ID
```

Clear that one group's seeded rows after testing:

```bash
python3 seed_one_paired_device_demo.py --release-group RELEASE_GROUP_ID --clear-only
```

Pause automatic updates:

```bash
sudo systemctl disable --now final-face-auto-update.timer
```

Restart listeners manually:

```bash
sudo systemctl restart final-face-listeners.service
```

Run one update immediately:

```bash
sudo systemctl start final-face-auto-update.service
```