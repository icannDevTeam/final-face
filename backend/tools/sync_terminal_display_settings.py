#!/usr/bin/env python3
"""Audit and synchronize display settings across all Hikvision terminals.

Ensures all 10 terminals match Terminal 01 (EY device) display behavior:
  - showPicture: True            (display face thumbnail image on terminal screen upon scan)
  - showEmployeeNo: True         (display employee / chaperone ID)
  - showName: True               (display full name)
  - desensitiseEmployeeNo: False (unmasked ID)
  - desensitiseName: False       (unmasked name)
  - voicePrompt: True            (audio prompt enabled)
  - uploadCapPic: True           (upload captured photo in event payload)

Usage:
  python3 sync_terminal_display_settings.py          # check & apply if mismatched
  python3 sync_terminal_display_settings.py --check  # read-only check
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests
from requests.auth import HTTPDigestAuth

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import devices_config

TARGET_SETTINGS = {
    "showPicture": True,
    "showEmployeeNo": True,
    "showName": True,
    "desensitiseEmployeeNo": False,
    "desensitiseName": False,
    "uploadCapPic": True,
    "saveCapPic": True,
    "voicePrompt": True,
    "uploadVerificationPic": True,
    "saveVerificationPic": True,
    "saveFacePic": True,
}


def audit_and_sync(read_only: bool = False) -> int:
    devices = devices_config.load_devices()
    print(f"Loaded {len(devices)} terminal configuration(s).")
    print(f"{'Name':<15} {'IP':<15} {'Status':<10} {'showPic':<8} {'showEmp':<8} {'showName':<9} {'desensEmp':<10} {'desensName':<10}")
    print("=" * 95)

    mismatches = 0
    errors = 0

    for d in devices:
        ip = d["ip"]
        name = d["name"]
        auth = HTTPDigestAuth(d.get("user", "admin"), d["password"])

        try:
            r = requests.get(f"http://{ip}/ISAPI/AccessControl/AcsCfg?format=json", auth=auth, timeout=5)
            if r.status_code != 200:
                print(f"{name:<15} {ip:<15} GET HTTP {r.status_code}")
                errors += 1
                continue
            acs = r.json().get("AcsCfg", {})
        except Exception as e:
            print(f"{name:<15} {ip:<15} GET ERR: {e}")
            errors += 1
            continue

        needs_update = False
        for k, target_val in TARGET_SETTINGS.items():
            if acs.get(k) != target_val:
                needs_update = True
                break

        if needs_update:
            mismatches += 1
            if read_only:
                status = "MISMATCH"
            else:
                acs.update(TARGET_SETTINGS)
                try:
                    r_put = requests.put(
                        f"http://{ip}/ISAPI/AccessControl/AcsCfg?format=json",
                        auth=auth,
                        json={"AcsCfg": acs},
                        headers={"Content-Type": "application/json"},
                        timeout=5,
                    )
                    status = "UPDATED" if r_put.status_code == 200 else f"HTTP {r_put.status_code}"
                except Exception as e:
                    status = f"PUT ERR: {e}"
                    errors += 1
        else:
            status = "MATCH"

        # Re-verify
        try:
            r_verify = requests.get(f"http://{ip}/ISAPI/AccessControl/AcsCfg?format=json", auth=auth, timeout=5)
            if r_verify.status_code == 200:
                v_acs = r_verify.json().get("AcsCfg", {})
                sp = str(v_acs.get("showPicture"))
                sen = str(v_acs.get("showEmployeeNo"))
                sn = str(v_acs.get("showName"))
                d_emp = str(v_acs.get("desensitiseEmployeeNo"))
                d_name = str(v_acs.get("desensitiseName"))
                print(f"{name:<15} {ip:<15} {status:<10} {sp:<8} {sen:<8} {sn:<9} {d_emp:<10} {d_name:<10}")
            else:
                print(f"{name:<15} {ip:<15} {status:<10} VERIFY HTTP {r_verify.status_code}")
        except Exception as e:
            print(f"{name:<15} {ip:<15} {status:<10} VERIFY ERR: {e}")

    print("\nSummary:")
    print(f"  Mismatched terminals: {mismatches}")
    print(f"  Errors: {errors}")
    return 0 if errors == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit and sync Hikvision terminal display settings.")
    parser.add_argument("--check", action="store_true", help="Read-only check without applying changes")
    args = parser.parse_args()
    return audit_and_sync(read_only=args.check)


if __name__ == "__main__":
    sys.exit(main())
