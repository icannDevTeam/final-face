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
  python3 device_health_check.py               # check + auto-fix clocks
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


def recent_stream_connections() -> set[str]:
    """Terminal names whose worker logged a stream connect since last service start."""
    try:
        out = subprocess.run(
            ["journalctl", "-u", "final-face-listeners", "--no-pager", "-n", "5000"],
            capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return set()
    return set(re.findall(r"\[(Terminal \d+)\]\s+✓ Connected", out))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reboot", metavar="IP", help="explicitly reboot one device and exit")
    args = ap.parse_args()

    if args.reboot:
        ok = reboot(args.reboot)
        print(f"reboot {args.reboot}: {'OK' if ok else 'FAILED'}")
        return 0 if ok else 1

    problems = 0
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
            problems += 1
        print("  ".join(line))

    print(f"\n{'ALL HEALTHY' if problems == 0 else f'{problems} PROBLEM(S) — see above'}")
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
