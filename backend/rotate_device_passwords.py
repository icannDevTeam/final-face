#!/usr/bin/env python3
"""
rotate_device_passwords.py — Rotate Hikvision admin password via ISAPI

For every enabled device in devices.json:
  1. Authenticate with the CURRENT password (resolved via devices_config)
  2. GET /ISAPI/Security/users  → find admin user's numeric id
  3. PUT /ISAPI/Security/users/<id>  with the new password
  4. Re-authenticate to confirm rotation worked
  5. Rewrite devices.local.json with the new password (timestamped backup first)

Usage:
  python3 rotate_device_passwords.py --dry-run
  NEW_HIK_PASS='Str0ng!Pass#2026' python3 rotate_device_passwords.py
  python3 rotate_device_passwords.py            # interactive prompt
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
from requests.auth import HTTPDigestAuth

from devices_config import DEVICES_LOCAL_FILE, load_devices

ADMIN_USER = os.environ.get("HIKVISION_USER", "admin")
TIMEOUT = 15


def _tag(xml: str, t: str) -> str:
    s, e = xml.find(f"<{t}>"), xml.find(f"</{t}>")
    return "" if s == -1 or e == -1 else xml[s + len(t) + 2 : e].strip()


def hik_get_users(ip: str, auth: HTTPDigestAuth) -> list[dict]:
    """List users — we need the numeric id of the admin account."""
    r = requests.get(
        f"http://{ip}/ISAPI/Security/users",
        auth=auth, timeout=TIMEOUT,
        headers={"Accept": "application/xml"},
    )
    r.raise_for_status()
    out = []
    for chunk in r.text.split("<User>")[1:]:
        chunk = chunk.split("</User>")[0]
        uid, uname = _tag(chunk, "id"), _tag(chunk, "userName")
        if uid:
            out.append({"id": uid, "userName": uname})
    return out


def hik_change_password(ip: str, auth: HTTPDigestAuth, user_id: str,
                        username: str, new_password: str) -> tuple[bool, str]:
    # Hikvision access terminals (DS-K1T*) require the OLD password as
    # <loginPassword> in the body, otherwise firmware returns
    # statusCode 6 / MessageParametersLack. Namespace must be isapi.org.
    old_password = auth.password
    body = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<User version="2.0" xmlns="http://www.isapi.org/ver20/XMLSchema">'
        f'<id>{user_id}</id>'
        f'<userName>{username}</userName>'
        '<userLevel>Administrator</userLevel>'
        f'<password>{new_password}</password>'
        f'<loginPassword>{old_password}</loginPassword>'
        '</User>'
    )
    try:
        r = requests.put(
            f"http://{ip}/ISAPI/Security/users/{user_id}",
            data=body, auth=auth, timeout=TIMEOUT,
            headers={"Content-Type": "application/xml"},
        )
    except requests.RequestException as e:
        return False, f"Network error: {e}"
    # Hikvision returns 200 with body statusCode=1 ("OK") on success.
    if r.status_code in (200, 204) and ("<statusCode>1<" in r.text or not r.text.strip()):
        return True, "OK"
    return False, f"HTTP {r.status_code}: {r.text[:300]}"


def verify_new_password(ip: str, new_password: str) -> bool:
    try:
        r = requests.get(
            f"http://{ip}/ISAPI/System/deviceInfo",
            auth=HTTPDigestAuth(ADMIN_USER, new_password),
            timeout=TIMEOUT,
            headers={"Accept": "application/xml"},
        )
        return r.status_code == 200
    except requests.RequestException:
        return False


def backup_local_file() -> Path | None:
    if not DEVICES_LOCAL_FILE.exists():
        return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = DEVICES_LOCAL_FILE.with_suffix(f".json.bak.{ts}")
    shutil.copy2(DEVICES_LOCAL_FILE, backup)
    return backup


def write_local_file(devices: list[dict]) -> None:
    payload = [{"name": d["name"], "password": d["password"]} for d in devices]
    DEVICES_LOCAL_FILE.write_text(json.dumps(payload, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Rotate Hikvision admin passwords via ISAPI.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show plan without changing anything.")
    ap.add_argument("--password",
                    help="New password (or set NEW_HIK_PASS env, or enter interactively).")
    args = ap.parse_args()

    devices = load_devices()
    if not devices:
        print("✗ No devices resolved.")
        sys.exit(1)

    print(f"▸ Will rotate admin password on {len(devices)} device(s):")
    for d in devices:
        print(f"   - {d['name']}  ({d['ip']})")
    print()

    new_password = args.password or os.environ.get("NEW_HIK_PASS")

    if args.dry_run:
        # Dry-run shouldn't require a real password.
        pwd_len = len(new_password) if new_password else 0
        print(f"⚙  DRY RUN — no password change will occur"
              + (f" (would use {pwd_len}-char password)" if pwd_len else ""))
        for d in devices:
            print(f"   would PUT http://{d['ip']}/ISAPI/Security/users/<adminId>")
        return

    if not new_password:
        new_password = getpass.getpass("Enter NEW admin password (input hidden): ")
        confirm = getpass.getpass("Confirm NEW admin password: ")
        if new_password != confirm:
            print("✗ Passwords did not match.")
            sys.exit(1)

    if len(new_password) < 8:
        print("✗ Password too short (Hikvision needs 8+ chars w/ mixed case + digits).")
        sys.exit(1)

    backup = backup_local_file()
    if backup:
        print(f"📦 Backed up {DEVICES_LOCAL_FILE.name} → {backup.name}")

    results = []
    for d in devices:
        ip = d["ip"]
        old_password = d["password"]
        print(f"\n▸ {d['name']} ({ip})")
        auth = HTTPDigestAuth(ADMIN_USER, old_password)

        try:
            users = hik_get_users(ip, auth)
        except (requests.RequestException, requests.HTTPError) as e:
            print(f"   ✗ Cannot list users: {e}")
            results.append((d, False, "list users failed"))
            continue

        admin = next((u for u in users if u["userName"] == ADMIN_USER), None)
        if not admin:
            print(f"   ✗ Admin user {ADMIN_USER!r} not found. Users: {[u['userName'] for u in users]}")
            results.append((d, False, "admin user not found"))
            continue

        user_id = admin["id"]
        print(f"   • admin user id = {user_id}")

        ok, msg = hik_change_password(ip, auth, user_id, ADMIN_USER, new_password)
        if not ok:
            print(f"   ✗ Change failed: {msg}")
            results.append((d, False, msg))
            continue
        print(f"   ✓ Password change accepted by device")

        time.sleep(1.5)
        if verify_new_password(ip, new_password):
            print(f"   ✓ Re-auth with new password OK")
            d["password"] = new_password
            results.append((d, True, "rotated"))
        else:
            print(f"   ⚠️  PUT accepted but re-auth FAILED. Investigate manually before discarding backup.")
            results.append((d, False, "verify failed"))

    write_local_file(devices)
    print(f"\n📝 Wrote {DEVICES_LOCAL_FILE.name}")

    print("\n" + "=" * 60)
    print("  Rotation summary")
    print("=" * 60)
    for d, ok, msg in results:
        marker = "✓" if ok else "✗"
        print(f"  {marker} {d['name']:50s}  {msg}")
    fail_count = sum(1 for _, ok, _ in results if not ok)
    if fail_count:
        print(f"\n⚠️  {fail_count} device(s) failed. Backup at: {backup}")
        sys.exit(2)
    print("\n✅ All devices rotated successfully.")
    print("\nNext steps:")
    print("  1. Restart the listener manager so it picks up new passwords:")
    print("     pkill -f 'attendance_listener.py' && python3 run_listeners.py &")
    print("  2. Update Vercel envs (HIKVISION_PASS_*) for web-dataset-collector if used")


if __name__ == "__main__":
    main()
