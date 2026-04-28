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


def load_devices() -> list[dict]:
    """Load enabled devices with passwords resolved from env/devices.local.json."""
    enabled = _load_devices_resolved()
    if not enabled:
        print("✗ No enabled devices with resolvable passwords. "
              "Check devices.json + devices.local.json / env vars.")
        sys.exit(1)
    return enabled


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
    args = parser.parse_args()

    extra_args = []
    if args.no_firebase:
        extra_args.append("--no-firebase")

    devices = load_devices()

    print(f"\n{BOLD}╔══════════════════════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}║       Hikvision Multi-Device Listener Manager           ║{RESET}")
    print(f"{BOLD}╚══════════════════════════════════════════════════════════╝{RESET}\n")
    print(f"  Devices:  {len(devices)} enabled")
    print(f"  Firebase: {'disabled' if args.no_firebase else 'enabled'}")
    print(f"  Config:   {DEVICES_FILE}")
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
