#!/usr/bin/env python3
"""
run_listeners.py — Multi-device Hikvision Listener Manager
============================================================
Spawns one attendance_listener.py subprocess per enabled device,
monitors all processes, and auto-restarts any that crash.

Usage:
  python3 run_listeners.py                # Start all enabled devices
  python3 run_listeners.py --no-firebase  # Disable Firebase for all
  python3 run_listeners.py --dry-run      # Show what would be launched

Config: devices.json (same directory)
"""

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import firebase_admin
from firebase_admin import credentials, firestore

import tenancy
from devices_config import load_devices as _load_devices_resolved

WIB = timezone(timedelta(hours=7))
SCRIPT_DIR = Path(__file__).parent
DEVICES_FILE = SCRIPT_DIR / "devices.json"
LISTENER_SCRIPT = SCRIPT_DIR / "attendance_listener.py"

# ANSI colors for per-device log tagging
COLORS = [
    "\033[96m",   # cyan
    "\033[93m",   # yellow
    "\033[95m",   # magenta
    "\033[92m",   # green
    "\033[91m",   # red
]
RESET = "\033[0m"
BOLD = "\033[1m"

RESTART_DELAY = 5        # seconds before restarting a crashed listener
MAX_RESTART_DELAY = 60   # backoff cap
HEALTH_CHECK_INTERVAL = 30  # seconds between health checks


class DeviceListener:
    """Manages a single listener subprocess."""

    def __init__(self, device: dict, index: int, extra_args: list):
        self.name = device["name"]
        self.ip = device["ip"]
        self.password = device["password"]
        self.terminal_id = device.get("terminalId")
        self.index = index
        self.color = COLORS[index % len(COLORS)]
        self.extra_args = extra_args
        self.process: subprocess.Popen | None = None
        self.restart_count = 0
        self.restart_delay = RESTART_DELAY
        self.started_at = None
        self.last_restart = None

    @property
    def tag(self):
        return f"{self.color}[{self.name}]{RESET}"

    def start(self):
        """Start the listener subprocess."""
        env = os.environ.copy()
        env["HIKVISION_IP"] = self.ip
        env["HIKVISION_PASS"] = self.password
        env["HIKVISION_DEVICE_NAME"] = self.name
        if self.terminal_id:
            env["HIKVISION_TERMINAL_ID"] = self.terminal_id
        env["PYTHONUNBUFFERED"] = "1"

        cmd = [sys.executable, str(LISTENER_SCRIPT)] + self.extra_args

        self.process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        self.started_at = datetime.now(WIB)
        print(f"{self.tag} Started (PID {self.process.pid}) — {self.ip}")

    def stop(self):
        """Stop the listener subprocess."""
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            print(f"{self.tag} Stopped")

    def is_alive(self):
        return self.process is not None and self.process.poll() is None

    def restart(self):
        """Restart with backoff."""
        self.restart_count += 1
        self.last_restart = datetime.now(WIB)
        delay = min(self.restart_delay * (1.5 ** min(self.restart_count - 1, 5)), MAX_RESTART_DELAY)

        returncode = self.process.returncode if self.process else "?"
        print(f"{self.tag} ⚠ Exited (code {returncode}). Restarting in {delay:.0f}s... (restart #{self.restart_count})")
        time.sleep(delay)

        # Reset backoff if the process ran for more than 5 minutes
        if self.started_at and (datetime.now(WIB) - self.started_at).total_seconds() > 300:
            self.restart_delay = RESTART_DELAY

        self.start()


def _load_enabled_from_devices_file() -> list[dict]:
    """Load enabled devices directly from devices.json (raw, without password resolution)."""
    if not DEVICES_FILE.exists():
        print(f"✗ {DEVICES_FILE} not found")
        sys.exit(1)
    try:
        raw = json.loads(DEVICES_FILE.read_text())
    except Exception as e:
        print(f"✗ Failed to read {DEVICES_FILE.name}: {e}")
        sys.exit(1)
    return [d for d in raw if isinstance(d, dict) and d.get("enabled", True)]


def _init_firestore_client():
    """Initialize Firebase Admin and return Firestore client, or None."""
    try:
        if not firebase_admin._apps:
            cred_path = os.getenv(
                "FIREBASE_CREDENTIALS",
                str(SCRIPT_DIR / "facial-attendance-binus-firebase-adminsdk.json"),
            )
            firebase_admin.initialize_app(credentials.Certificate(cred_path))
        return firestore.client()
    except Exception as e:
        print(f"⚠ Firestore init failed: {e}")
        return None


def _load_terminals_from_firestore() -> list[dict]:
    """Load enabled terminals from Firestore tenant registry."""
    db = _init_firestore_client()
    if db is None:
        return []

    tid = tenancy.get_tenant_id()
    col = db.collection(tenancy.terminals_path(tid))
    try:
        # Prefer new filter API to avoid positional-arg warning.
        from google.cloud.firestore_v1.base_query import FieldFilter
        docs = list(col.where(filter=FieldFilter("enabled", "==", True)).stream())
    except Exception:
        docs = list(col.where("enabled", "==", True).stream())

    terminals = []
    for doc in docs:
        d = doc.to_dict() or {}
        ip = str(d.get("ip") or "").strip()
        name = str(d.get("name") or d.get("deviceName") or doc.id).strip()
        if not ip:
            continue
        terminals.append(
            {
                "terminalId": doc.id,
                "name": name,
                "ip": ip,
                "enabled": d.get("enabled", True),
            }
        )

    terminals.sort(key=lambda d: d.get("name", ""))
    return terminals


def _resolve_password_for_terminal(term: dict, resolved_local: list[dict]) -> str | None:
    """Resolve password for a Firestore terminal using local/env credential sources."""
    t_name = str(term.get("name") or "").strip()
    t_ip = str(term.get("ip") or "").strip()

    # 1) exact name match in resolved local config
    by_name = next((d for d in resolved_local if str(d.get("name") or "").strip() == t_name), None)
    if by_name and by_name.get("password"):
        return by_name["password"]

    # 2) exact IP match in resolved local config
    by_ip = next((d for d in resolved_local if str(d.get("ip") or "").strip() == t_ip), None)
    if by_ip and by_ip.get("password"):
        return by_ip["password"]

    # 3) generic env fallback (shared device password setups)
    env_pw = os.getenv("HIKVISION_PASS")
    if env_pw:
        return env_pw

    # 4) if all local entries share one password, reuse it
    pw_set = {str(d.get("password")) for d in resolved_local if d.get("password")}
    if len(pw_set) == 1:
        return pw_set.pop()

    return None


def _build_devices_from_firestore(allow_partial: bool = False) -> tuple[list[dict], list[dict]]:
    """Build listener launch list from Firestore terminals + locally resolved credentials."""
    fs_terms = _load_terminals_from_firestore()
    if not fs_terms:
        return [], []

    # Single-password mode: one shared secret for all Hikvision terminals.
    # Useful when terminal names are renamed in Firestore and local per-name
    # overrides lag behind.
    shared_pw = os.getenv("HIKVISION_PASS")
    if shared_pw:
        devices = [
            {
                "terminalId": t.get("terminalId"),
                "name": t["name"],
                "ip": t["ip"],
                "password": shared_pw,
                "enabled": True,
            }
            for t in fs_terms
        ]
        return devices, []

    resolved_local = _load_devices_resolved()
    devices: list[dict] = []
    unresolved: list[dict] = []

    for t in fs_terms:
        pw = _resolve_password_for_terminal(t, resolved_local)
        if not pw:
            unresolved.append(t)
            continue
        devices.append(
            {
                "terminalId": t.get("terminalId"),
                "name": t["name"],
                "ip": t["ip"],
                "password": pw,
                "enabled": True,
            }
        )

    if unresolved and not allow_partial:
        print("✗ Coverage check failed: some Firestore terminals have no resolved password.")
        print("  Refusing to start partially. Resolve credentials first or use --allow-partial.")
        for d in unresolved:
            print(f"    - {d.get('name', '?')} ({d.get('ip', '?')})")
        sys.exit(1)

    if unresolved and allow_partial:
        print("⚠ Partial coverage: starting only Firestore terminals with resolved credentials.")
        for d in unresolved:
            print(f"    - Skipped: {d.get('name', '?')} ({d.get('ip', '?')})")

    return devices, unresolved


def load_devices(allow_partial: bool = False) -> tuple[list[dict], list[dict]]:
    """Load enabled devices with passwords resolved; Firestore is source of truth."""
    devices_fs, unresolved_fs = _build_devices_from_firestore(allow_partial=allow_partial)
    if devices_fs:
        return devices_fs, unresolved_fs

    print("⚠ No Firestore terminals available; falling back to backend/devices.json")
    enabled_raw = _load_enabled_from_devices_file()
    resolved = _load_devices_resolved()
    if not resolved:
        print("✗ No enabled devices with resolvable passwords. "
              "Check devices.json + devices.local.json / env vars.")
        sys.exit(1)

    resolved_names = {str(d.get("name", "")).strip() for d in resolved}
    unresolved = [d for d in enabled_raw if str(d.get("name", "")).strip() not in resolved_names]

    if unresolved and not allow_partial:
        print("✗ Coverage check failed: some enabled terminals have no resolved password.")
        print("  Refusing to start partially. Resolve credentials first or use --allow-partial.")
        for d in unresolved:
            print(f"    - {d.get('name', '?')} ({d.get('ip', '?')})")
        sys.exit(1)

    if unresolved and allow_partial:
        print("⚠ Partial coverage: starting listeners only for terminals with resolved credentials.")
        for d in unresolved:
            print(f"    - Skipped: {d.get('name', '?')} ({d.get('ip', '?')})")

    return resolved, unresolved


def print_status(listeners: list[DeviceListener]):
    """Print status summary of all listeners."""
    now = datetime.now(WIB)
    print(f"\n{'=' * 60}")
    print(f"  {BOLD}Listener Manager Status{RESET} — {now.strftime('%Y-%m-%d %H:%M:%S')} WIB")
    print(f"{'=' * 60}")
    for l in listeners:
        alive = "✓ Running" if l.is_alive() else "✗ Stopped"
        pid = l.process.pid if l.process and l.is_alive() else "-"
        uptime = ""
        if l.started_at and l.is_alive():
            delta = now - l.started_at
            mins = int(delta.total_seconds() // 60)
            uptime = f" (up {mins}m)"
        restarts = f" [{l.restart_count} restarts]" if l.restart_count else ""
        print(f"  {l.tag} {alive} | PID {pid}{uptime}{restarts}")
    print(f"{'=' * 60}\n")


def read_output(listener: DeviceListener):
    """Non-blocking read of subprocess output, tag with device name."""
    if not listener.process or not listener.process.stdout:
        return
    import select
    while True:
        ready, _, _ = select.select([listener.process.stdout], [], [], 0)
        if not ready:
            break
        line = listener.process.stdout.readline()
        if not line:
            break
        text = line.decode("utf-8", errors="replace").rstrip()
        if text:
            print(f"{listener.tag} {text}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Multi-device Hikvision listener manager")
    parser.add_argument("--no-firebase", action="store_true", help="Disable Firebase for all listeners")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be launched")
    parser.add_argument("--allow-partial", action="store_true", help="Allow startup even when some enabled terminals have unresolved credentials")
    args = parser.parse_args()

    extra_args = []
    if args.no_firebase:
        extra_args.append("--no-firebase")

    devices, unresolved = load_devices(allow_partial=args.allow_partial)
    expected = len(devices) + len(unresolved)

    print(f"\n{BOLD}╔══════════════════════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}║       Hikvision Multi-Device Listener Manager           ║{RESET}")
    print(f"{BOLD}╚══════════════════════════════════════════════════════════╝{RESET}\n")
    print(f"  Devices:  {len(devices)}/{expected} enabled terminals covered")
    print(f"  Firebase: {'disabled' if args.no_firebase else 'enabled'}")
    print(f"  Source:   Firestore terminals registry (fallback: {DEVICES_FILE.name})")
    print()

    if args.dry_run:
        for i, d in enumerate(devices):
            color = COLORS[i % len(COLORS)]
            print(f"  {color}[{d['name']}]{RESET} → HIKVISION_IP={d['ip']} python3 attendance_listener.py {' '.join(extra_args)}")
        print("\n  (dry run — no processes started)")
        return

    # Create listeners
    listeners = [DeviceListener(d, i, extra_args) for i, d in enumerate(devices)]

    # Graceful shutdown
    shutting_down = False

    def shutdown(signum, frame):
        nonlocal shutting_down
        if shutting_down:
            return
        shutting_down = True
        print(f"\n\n⏹  Shutting down all listeners...")
        for l in listeners:
            l.stop()
        print("  All listeners stopped. Goodbye.\n")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Start all listeners
    for l in listeners:
        l.start()
        time.sleep(1)  # stagger startup slightly

    print_status(listeners)

    # Main loop: monitor + read output + auto-restart
    last_health = time.time()

    while not shutting_down:
        # Read output from all listeners (non-blocking)
        for l in listeners:
            read_output(l)

        # Check for crashed processes and restart
        for l in listeners:
            if not l.is_alive() and not shutting_down:
                l.restart()

        # Periodic health check
        if time.time() - last_health > HEALTH_CHECK_INTERVAL:
            print_status(listeners)
            last_health = time.time()

        time.sleep(0.1)  # small sleep to prevent busy-wait


if __name__ == "__main__":
    main()
