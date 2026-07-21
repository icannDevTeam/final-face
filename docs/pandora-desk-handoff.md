# Pandora Desk Handoff

Use this when someone is physically at the desk/Pandora machine and the home workspace is not a Git clone.

## 1. Confirm the real Git clone

On the desk machine or Pandora:

```bash
cd /opt/final-face
git status --short --branch
git remote -v
```

If Pandora has uncommitted production changes, do not pull over them. Preserve them first:

```bash
git diff > /tmp/pandora-uncommitted.patch
git diff --cached > /tmp/pandora-staged.patch
git ls-files --others --exclude-standard > /tmp/pandora-untracked-files.txt
```

## 2. Commit and push the update

Copy the changed files into the real clone, then run:

```bash
git status --short
git add \
  backend/scripts/pandora_auto_update.sh \
  backend/scripts/pandora_preflight.sh \
  backend/scripts/collect_pandora_logs.sh \
  backend/scripts/check_gate_override.py \
  backend/scripts/normalize_chaperone_pathway.py \
  backend/pickup_event_writer.py \
  backend/seed_one_paired_device_demo.py \
  backend/tests/test_pickup_event_writer.py \
  backend/tests/test_gate_controller.py \
  docs/pandora-auto-deploy.md \
  docs/system-architecture.md \
  docs/pandora-desk-handoff.md \
  docs/pickupguard-operations-manual-draft.md \
  CLAUDE.md
git commit -m "Add Pandora deployment and pickup diagnostics tools"
git push origin main   # or master, if that is Pandora's branch
```

## 3. Pull on Pandora

```bash
ssh USER@PANDORA_HOST
cd /opt/final-face
git status --short --branch
git pull --ff-only origin main   # or master
chmod +x backend/scripts/*.sh
sudo systemctl restart final-face-listeners.service
sudo journalctl -u final-face-listeners.service -n 120 --no-pager
```

## 4. Verify the current issue

Check live scan logs:

```bash
sudo APP_DIR=/opt/final-face SERVICE_NAME=final-face-listeners.service backend/scripts/collect_pandora_logs.sh "STUDENT_NAME_OR_ID"
```

Check manual gate override:

```bash
python3 backend/scripts/check_gate_override.py --query "GRADE_OR_GATE_NAME"
python3 backend/scripts/check_gate_override.py --query "GRADE_OR_GATE_NAME" --set open
sudo journalctl -u final-face-listeners.service -n 120 --no-pager | grep -Ei 'Gate|alwaysOpen|alwaysClose|manual-open|manual-closed'
python3 backend/scripts/check_gate_override.py --query "GRADE_OR_GATE_NAME" --set auto
```

Check paired iPad filtering for a terminal:

```bash
python3 backend/scripts/check_ipad_pairing.py --query "TERMINAL_NAME_OR_IP"
python3 backend/scripts/inspect_latest_pickup_event.py --limit 5
```

This check is read-only. If it says the terminal is not bound to a release
group, do not patch Firestore from a script. Pair that terminal to the iPad
release group from the admin UI only. Authorized pairing users should include
owner/admin accounts and assigned IT/pairing operators; the admin UI/API should
gate the action by that permission, not by direct Firestore writes. Otherwise
cards can arrive over the live socket and disappear on the iPad's Firestore
refresh.

Seed one paired iPad/tablet demo:

```bash
cd /opt/final-face/backend
python3 seed_one_paired_device_demo.py --list
python3 seed_one_paired_device_demo.py --query "GRADE_OR_TABLET_NAME"
```

Fix chaperone enrollment when `pathway` was made too strict:

```bash
cd /opt/final-face
python3 backend/scripts/normalize_chaperone_pathway.py --dry-run
python3 backend/scripts/normalize_chaperone_pathway.py
```

Also update the chaperone approval/enrollment validator in the web/admin code so `pathway` is optional. The accepted values should be `undefined`, `null`, or a non-empty string; do not reject a chaperone only because `pathway` is missing.