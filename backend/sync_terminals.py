#!/usr/bin/env python3
"""
sync_terminals.py — Multi-device Hikvision Catch-up Sync
==========================================================
One-shot orchestrator that pulls attendance events sitting on each
enabled Hikvision terminal (devices.json) and backfills them into:
  1. Local JSON      → data/attendance/YYYY-MM-DD.json
  2. Firebase        → attendance/{date}/records
  3. BINUS School API

This is essentially run_listeners.py + --catchup, but runs each device
serially to completion and exits (no live streaming).

Use this when:
  - Listeners were down and you want to backfill missed scans
  - You powered a device on after off-hours and want to grab the data
  - You want to verify Firestore has everything from the device
  - Daily cron: pull yesterday's full day to catch anything missed

Note: Only DS-K1T342MFX devices support event search. DS-K1T341AMF
will be skipped (it has no on-device event log retrievable via ISAPI).

Usage:
  python3 sync_terminals.py                    # Today across all devices
  python3 sync_terminals.py --days 3           # Last 3 days (today + 2 prior)
  python3 sync_terminals.py --date 2026-04-25  # Specific date (repeatable)
  python3 sync_terminals.py --date 2026-04-25 --date 2026-04-26
  python3 sync_terminals.py --no-firebase      # Local JSON only
  python3 sync_terminals.py --dry-run          # Show what would run
  python3 sync_terminals.py --device "MYP Tower"  # Substring match, single device
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

WIB = timezone(timedelta(hours=7))
SCRIPT_DIR = Path(__file__).parent
DEVICES_FILE = SCRIPT_DIR / "devices.json"
LISTENER_SCRIPT = SCRIPT_DIR / "attendance_listener.py"

# ANSI colors (match run_listeners.py)
COLORS = ["\033[96m", "\033[93m", "\033[95m", "\033[92m", "\033[91m"]
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"


def load_devices(filter_substr: str | None = None) -> list[dict]:
    from devices_config import load_devices as _resolved
    enabled = _resolved(filter_substr=filter_substr)
    if not enabled:
        print("✗ No matching enabled devices with resolvable passwords")
        sys.exit(1)
    return enabled


def resolve_dates(date_args: list[str] | None, days: int | None) -> list[str]:
    if date_args:
        # Validate format
        for d in date_args:
            try:
                datetime.strptime(d, "%Y-%m-%d")
            except ValueError:
                print(f"✗ Invalid date format: {d} (expected YYYY-MM-DD)")
                sys.exit(1)
        return sorted(set(date_args))
    if days and days > 0:
        today = datetime.now(WIB)
        return [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)][::-1]
    return [datetime.now(WIB).strftime("%Y-%m-%d")]


def run_device_catchup(device: dict, dates: list[str], use_firebase: bool, color: str) -> tuple[int, str]:
    """Run catchup for a single device. Returns (exit_code, tag)."""
    tag = f"{color}[{device['name']}]{RESET}"
    env = os.environ.copy()
    env["HIKVISION_IP"] = device["ip"]
    env["HIKVISION_PASS"] = device["password"]
    env["HIKVISION_DEVICE_NAME"] = device["name"]
    env["PYTHONUNBUFFERED"] = "1"

    # Use system python3 (where backend deps live), matching run_listeners.py setup
    python_bin = os.environ.get("PYTHON_BIN", "/usr/bin/python3")
    cmd = [python_bin, str(LISTENER_SCRIPT), "--catchup"]
    if not use_firebase:
        cmd.append("--no-firebase")
    for d in dates:
        cmd.extend(["--date", d])

    print(f"\n{tag} ▶ Starting catch-up for {len(dates)} day(s): {', '.join(dates)}")
    print(f"{tag} {DIM}{device['ip']} → {' '.join(cmd[1:])}{RESET}")

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )

    # Stream output with device tag prefix
    if proc.stdout is not None:
        for raw in proc.stdout:
            text = raw.decode("utf-8", errors="replace").rstrip()
            if text:
                print(f"{tag} {text}")

    proc.wait()
    return proc.returncode, tag


def main():
    parser = argparse.ArgumentParser(
        description="Pull attendance data from all Hikvision terminals and sync to Firestore + BINUS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1] if "Usage:" in (__doc__ or "") else "",
    )
    parser.add_argument("--no-firebase", action="store_true", help="Disable Firebase sync (local JSON only)")
    parser.add_argument("--date", action="append", help="Specific date YYYY-MM-DD (repeatable)")
    parser.add_argument("--days", type=int, help="Sync the last N days including today")
    parser.add_argument("--device", help="Filter to one device (substring match on name or IP)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run, don't execute")
    args = parser.parse_args()

    devices = load_devices(args.device)
    dates = resolve_dates(args.date, args.days)

    print(f"\n{BOLD}╔══════════════════════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}║       Hikvision Multi-Device Catch-up Sync              ║{RESET}")
    print(f"{BOLD}╚══════════════════════════════════════════════════════════╝{RESET}\n")
    print(f"  Devices:  {len(devices)} enabled")
    for i, d in enumerate(devices):
        color = COLORS[i % len(COLORS)]
        print(f"            {color}● {d['name']}{RESET}  ({d['ip']})")
    print(f"  Dates:    {', '.join(dates)}")
    print(f"  Firebase: {'disabled' if args.no_firebase else 'enabled'}")
    print()

    if args.dry_run:
        for i, d in enumerate(devices):
            color = COLORS[i % len(COLORS)]
            cmd_preview = f"python3 attendance_listener.py --catchup {' '.join(f'--date {x}' for x in dates)}"
            if args.no_firebase:
                cmd_preview += " --no-firebase"
            print(f"  {color}[{d['name']}]{RESET} → HIKVISION_IP={d['ip']} {cmd_preview}")
        print("\n  (dry run — no processes started)")
        return

    results: list[tuple[str, int]] = []
    for i, device in enumerate(devices):
        color = COLORS[i % len(COLORS)]
        try:
            code, tag = run_device_catchup(device, dates, not args.no_firebase, color)
            results.append((device["name"], code))
        except KeyboardInterrupt:
            print(f"\n\n⏹  Interrupted by user. {len(results)}/{len(devices)} device(s) processed.")
            sys.exit(130)
        except Exception as e:
            print(f"\n{color}[{device['name']}]{RESET} ✗ Failed: {e}")
            results.append((device["name"], -1))

    # Summary
    print(f"\n{BOLD}═══════════════════════════════════════════════════════════{RESET}")
    print(f"{BOLD}  Sync Summary{RESET}")
    print(f"{BOLD}═══════════════════════════════════════════════════════════{RESET}")
    ok = 0
    for name, code in results:
        if code == 0:
            print(f"  ✓ {name}")
            ok += 1
        else:
            print(f"  ✗ {name}  (exit code {code})")
    print(f"\n  {ok}/{len(results)} device(s) succeeded.\n")
    sys.exit(0 if ok == len(results) else 1)


if __name__ == "__main__":
    main()
