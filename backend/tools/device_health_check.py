#!/usr/bin/env python3
"""Pre-window device health check for the pickup system.

Run this ~30 min before every pickup window (systemd timer). It:
  1. Pings every terminal
  2. Compares each device clock against the host (auto-fixes TZ + time on drift)
  3. Verifies each listener worker logged a recent stream connection
  4. Exits non-zero if anything needs human attention

Auto-fix scope: clock/timezone only. Reboots are NOT automatic — a stalled
device is reported so an operator can run --reboot <ip> explicitly.

Usage:
  python3 device_health_check.py                  # check + auto-fix clocks
  python3 device_health_check.py --kill-stalled   # also kill workers with no
                                                  # stream connect (root only;
                                                  # manager respawns → reconnect)
  python3 device_health_check.py --reboot 10.26.30.67
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from requests.auth import HTTPDigestAuth
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
load_dotenv()

WIB = timezone(timedelta(hours=7))
USER = os.getenv("HIKVISION_USER", "admin")
PASS = os.getenv("HIKVISION_PASS")
DRIFT_LIMIT_SEC = 120

TERMINALS = {
    "Terminal 01": "10.26.30.41",
    "Terminal 02": "10.26.30.67",
    "Terminal 03": "10.26.30.74",
    "Terminal 04": "10.26.30.75",
    "Terminal 05": "10.26.30.76",
    "Terminal 06": "10.26.30.92",
    "Terminal 07": "10.26.30.81",
    "Terminal 08": "10.26.30.82",
    "Terminal 09": "10.26.30.83",
    "Terminal 10": "10.26.30.201",
}


def _auth() -> HTTPDigestAuth:
    if not PASS:
        raise SystemExit("FATAL: HIKVISION_PASS not set")
    return HTTPDigestAuth(USER, PASS)


def ping(ip: str) -> bool:
    return subprocess.run(["ping", "-c", "1", "-W", "2", ip],
                          capture_output=True).returncode == 0


def get_device_time(ip: str):
    r = requests.get(f"http://{ip}/ISAPI/System/time", auth=_auth(), timeout=8)
    m = re.search(r"<localTime>([^<]+)</localTime>", r.text)
    tz = re.search(r"<timeZone>([^<]+)</timeZone>", r.text)
    return (m.group(1) if m else None), (tz.group(1) if tz else None)


def fix_clock(ip: str) -> bool:
    now = datetime.now(WIB).strftime("%Y-%m-%dT%H:%M:%S")
    body = (f"<Time><timeMode>manual</timeMode><localTime>{now}+07:00</localTime>"
            f"<timeZone>CST-7:00:00</timeZone></Time>")
    r = requests.put(f"http://{ip}/ISAPI/System/time", data=body.encode(),
                     headers={"Content-Type": "application/xml"}, auth=_auth(), timeout=8)
    return r.ok


def reboot(ip: str) -> bool:
    r = requests.put(f"http://{ip}/ISAPI/System/reboot", auth=_auth(), timeout=8)
    return r.ok


def _service_start() -> str | None:
    """ISO-ish timestamp of the last final-face-listeners start, for journalctl --since."""
    try:
        out = subprocess.run(
            ["systemctl", "show", "final-face-listeners.service",
             "--property=ActiveEnterTimestamp", "--value"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        m = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", out)
        return m.group(1) if m else None
    except Exception:
        return None


def _journal_grep(pattern: str) -> str:
    """Matching journal lines since service start (full window, not last-N)."""
    since = _service_start()
    cmd = "journalctl -u final-face-listeners --no-pager"
    cmd += f" --since '{since}'" if since else " -n 5000"
    cmd += f" | grep -F {pattern!r} || true"
    try:
        return subprocess.run(cmd, shell=True, capture_output=True,
                              text=True, timeout=30).stdout
    except Exception:
        return ""


def recent_stream_connections() -> set[str]:
    """Terminal names whose worker logged a stream connect since last service start."""
    return set(re.findall(r"\[(Terminal \d+)\]\s+✓ Connected",
                          _journal_grep("✓ Connected")))


def worker_pids() -> dict[str, int]:
    """Latest heartbeat PID per terminal from the manager's Running lines."""
    pids: dict[str, int] = {}
    for m in re.finditer(r"\[(Terminal \d+)\] ✓ Running \| PID (\d+)",
                         _journal_grep("✓ Running | PID")):
        pids[m.group(1)] = int(m.group(2))  # later lines overwrite → latest wins
    return pids


def kill_stalled_workers(stalled: list[str]) -> set[str]:
    """Kill workers with no stream connect so the manager respawns them.

    Returns the terminals that reconnected after remediation.
    """
    import time
    pids = worker_pids()
    for name in stalled:
        pid = pids.get(name)
        if not pid or not Path(f"/proc/{pid}").exists():
            print(f"  {name}: no live worker PID found — skipping")
            continue
        try:
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace")
            if "python" not in cmdline:
                print(f"  {name}: PID {pid} is not a python worker — skipping")
                continue
            os.kill(pid, 15)
            print(f"  {name}: killed worker PID {pid} — manager will respawn")
        except PermissionError:
            print(f"  {name}: PID {pid} kill denied (run as root for --kill-stalled)")
        except ProcessLookupError:
            print(f"  {name}: PID {pid} already gone")
    print("  waiting 45s for respawn + stream reconnect...")
    time.sleep(45)
    return recent_stream_connections()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reboot", metavar="IP", help="explicitly reboot one device and exit")
    ap.add_argument("--kill-stalled", action="store_true",
                    help="kill workers with no stream connect so they respawn (root)")
    args = ap.parse_args()

    if args.reboot:
        ok = reboot(args.reboot)
        print(f"reboot {args.reboot}: {'OK' if ok else 'FAILED'}")
        return 0 if ok else 1

    problems = 0
    stalled: list[str] = []
    connected = recent_stream_connections()
    host_now = datetime.now(WIB)
    for name, ip in TERMINALS.items():
        line = [f"{name} ({ip})"]
        if not ping(ip):
            line.append("✗ UNREACHABLE")
            problems += 1
            print("  ".join(line))
            continue
        try:
            local, tz = get_device_time(ip)
            dev_dt = datetime.fromisoformat(local) if local else None
            drift = abs((dev_dt - host_now).total_seconds()) if dev_dt else None
            tz_ok = tz == "CST-7:00:00"
            if drift is not None and drift <= DRIFT_LIMIT_SEC and tz_ok:
                line.append(f"clock OK (drift {drift:.0f}s)")
            else:
                line.append(f"⚠ clock drift {drift:.0f}s tz={tz} → fixing")
                if fix_clock(ip):
                    line.append("fixed ✓")
                else:
                    line.append("FIX FAILED")
                    problems += 1
        except Exception as e:
            line.append(f"✗ time check failed: {e}")
            problems += 1
        if name not in connected:
            line.append("⚠ no stream connect logged since service start")
            stalled.append(name)
            problems += 1
        print("  ".join(line))

    if stalled and args.kill_stalled:
        print(f"\nRemediating {len(stalled)} stalled worker(s): {', '.join(stalled)}")
        reconnected = kill_stalled_workers(stalled)
        for name in stalled:
            if name in reconnected:
                print(f"  {name}: ✓ reconnected")
                problems -= 1
            else:
                print(f"  {name}: ✗ still no stream connect — needs operator")

    print(f"\n{'ALL HEALTHY' if problems == 0 else f'{problems} PROBLEM(S) — see above'}")
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
