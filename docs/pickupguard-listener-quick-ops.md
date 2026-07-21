# PickupGuard Listener Quick Ops (Tech Support)

Audience: Tech Support  
Version: 2026-07-17  
Timezone: WIB (UTC+7)

## 1) Start of Shift

Run this from project root:

```bash
cd /home/pandora/Downloads/final-face
source .venv/bin/activate
python3 backend/run_listeners.py
```

Expected result:
- Listener manager starts.
- Coverage shows all enabled terminals are included (example: `3/3 enabled terminals covered`).
- Each enabled terminal shows as running.

Important:
- This command is now strict by default.
- If any enabled terminal is missing credentials, startup is blocked so no terminal is silently skipped.

## 2) End of Shift

In the same terminal window:
- Press `Ctrl + C`

Expected result:
- All listener processes stop gracefully.

## 3) Quick Pre-Check (Optional)

```bash
cd /home/pandora/Downloads/final-face
source .venv/bin/activate
python3 backend/run_listeners.py --dry-run
```

Use this to verify devices are detected before starting real listeners.

If dry-run fails with coverage check:
- Fix missing terminal password mapping first.
- Temporary bypass (not recommended for daily operations):

```bash
python3 backend/run_listeners.py --allow-partial
```

## 4) Emergency Mode (No Firebase)

If Firebase is down, keep local listener operations running:

```bash
cd /home/pandora/Downloads/final-face
source .venv/bin/activate
python3 backend/run_listeners.py --no-firebase
```

## 5) Minimal Diagnostics

### A) Listener does not start
Possible causes:
- Terminal password not resolved.
- Device config issue.

Check:

```bash
cd /home/pandora/Downloads/final-face
source .venv/bin/activate
python3 backend/run_listeners.py --dry-run
```

### B) Listener keeps restarting
Possible causes:
- Terminal unreachable on network.
- Wrong terminal password.

Action:
- Verify terminal is online.
- Verify password and environment mapping.
- Keep listener manager running for all enabled terminals (online and offline).
- Offline terminals are retried automatically; when they come back online they are picked up by the running listener process.
- If needed, run one listener manually for targeted check.

### C) iPad shows no cards
Possible causes:
- iPad token not paired.
- Terminal not in the iPad release group.
- Event decision is intentionally hidden on iPad.

## 6) Add User for RBAC (Quick SOP)

Use this when a new staff member needs dashboard access.

Steps:
- Open web admin and go to User Management.
- Add the user account (email + display name).
- Assign role: owner, admin, viewer, or guard (as approved by operations lead).
- Save user.
- Open RBAC page and verify the user has required module permissions.

Minimum check before handover:
- User can sign in.
- User can open only the pages they are allowed to see.

## 7) Create User Group / Role (Quick SOP)

Use this when multiple users need the same permission set.

Steps:
- Open RBAC page.
- Create a new role/group with clear name (example: Pickup Tech Support).
- Enable only required permissions.
- Save role/group.
- Assign users to this role/group.

Recommended minimal permissions for tech support:
- pickup_admin.view
- pickup_admin.manage_terminals
- pickup_admin.manage_groups

## 8) Pair iPad Device to Release Group (Quick SOP)

Steps:
- Open Release Groups page.
- Select the target release group.
- Click Start Pair.
- Note the 6-character pairing code (valid for a short time only).
- On iPad teacher page, enter pairing code.
- Confirm status changes to paired.

If pairing fails:
- Generate a new code and retry.
- Confirm iPad has network access.

## 9) Enroll Chaperone to Terminal Device (Quick SOP)

Preferred flow (UI):
- Open Pickup Admin > Chaperones.
- Create or open chaperone profile.
- Link authorized students.
- Upload chaperone face photo.
- Use enroll/sync action to push chaperone face to selected terminal(s).

Fallback flow (script, one terminal):

```bash
cd /home/pandora/Downloads/final-face
source .venv/bin/activate
python3 backend/scripts/quick_terminal_enroll.py \
	--ip <TERMINAL_IP> \
	--password '<TERMINAL_PASSWORD>' \
	--employee-no <CHAPERONE_ID_STARTS_WITH_9> \
	--name '<CHAPERONE_NAME>' \
	--image /absolute/path/to/photo.jpg
```

Post-enroll validation:
- Run one test face scan at the target terminal.
- Confirm listener log receives event.
- Confirm iPad gets card (if terminal is in the paired release group).

## 10) Escalate to Engineering When

- Listener fails repeatedly after password and network checks.
- All iPads fail pairing or token auth.
- No events are written despite successful terminal scans.

Include in escalation:
- Time (WIB)
- Terminal name and IP
- Screenshot of listener terminal logs
- Command used
