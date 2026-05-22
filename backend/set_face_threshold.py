#!/usr/bin/env python3
"""
set_face_threshold.py — Probe and tighten the device-side face-match threshold
================================================================================

Hikvision terminals (DS-K1T341AMF / DS-K1T342MFX) run 1:N face matching on the
device itself. When an unenrolled person scans, the matcher will still return
its closest neighbor unless the similarity threshold is high enough to reject.

This script:
  1. Probes ISAPI capabilities to discover which threshold field the firmware
     exposes (`FaceRecognizeMode`, `captureFaceMode`, etc.).
  2. Reads the current value.
  3. Optionally writes a new value (only if --set is provided).

Defaults to probing every enabled device in devices.json.

Usage:
  python3 set_face_threshold.py                     # Probe all enabled devices (read-only)
  python3 set_face_threshold.py --device "PYP Lobby"  # Probe one device
  python3 set_face_threshold.py --set 85            # Try to raise threshold to 85 on all
  python3 set_face_threshold.py --device "PYP Lobby" --set 85

Notes:
  • Higher threshold (e.g. 85–95) → fewer false positives but more "scan again" prompts.
  • Vendor default is typically 65–75.
  • Always probe first to confirm the field name your firmware uses before --set.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import requests
from requests.auth import HTTPDigestAuth

from devices_config import load_devices

SCRIPT_DIR = Path(__file__).parent

# Candidate ISAPI endpoints that expose face-match thresholds across firmwares.
# We probe each and report what's there.
PROBE_ENDPOINTS = [
    "/ISAPI/AccessControl/FaceRecognizeMode/capabilities?format=json",
    "/ISAPI/AccessControl/FaceRecognizeMode?format=json",
    "/ISAPI/AccessControl/captureFaceMode/capabilities?format=json",
    "/ISAPI/AccessControl/captureFaceMode?format=json",
    "/ISAPI/Intelligent/FDLib/capabilities?format=json",
    "/ISAPI/AccessControl/Configuration/FaceContrastConditions?format=json",
    "/ISAPI/SecurityCP/AccessControl/FaceContrastConditions?format=json",
    "/ISAPI/Intelligent/analysisEngines/face/capabilities?format=json",
]

# Field names known to carry the 1:N match threshold across Hikvision firmwares.
THRESHOLD_KEYS = (
    "faceContrastThreshold",
    "faceRecognitionThreshold",
    "FaceContrastThreshold",
    "similarityThreshold",
    "threshold",
)


def _session(device: dict) -> tuple[requests.Session, str]:
    user = os.getenv("HIKVISION_USER", "admin")
    s = requests.Session()
    s.auth = HTTPDigestAuth(user, device["password"])
    return s, f"http://{device['ip']}"


def probe(device: dict) -> dict:
    """Probe candidate endpoints. Returns a dict of {endpoint: response}."""
    s, base = _session(device)
    found = {}
    for ep in PROBE_ENDPOINTS:
        try:
            r = s.get(base + ep, timeout=8)
            if r.status_code == 200:
                try:
                    body = r.json()
                except Exception:
                    body = {"raw": r.text[:300]}
                found[ep] = {"status": 200, "body": body}
            elif r.status_code not in (404, 403):
                found[ep] = {"status": r.status_code, "body": r.text[:200]}
        except requests.RequestException as e:
            found[ep] = {"status": "error", "body": str(e)[:200]}
    return found


def find_current_threshold(probe_result: dict) -> list[tuple[str, str, int]]:
    """Walk probe results and pluck out anything that looks like a threshold.
    Returns list of (endpoint, key, value)."""
    hits = []

    def _walk(node, ep, path=""):
        if isinstance(node, dict):
            for k, v in node.items():
                if k in THRESHOLD_KEYS and isinstance(v, (int, float)):
                    hits.append((ep, f"{path}.{k}".lstrip("."), int(v)))
                _walk(v, ep, f"{path}.{k}")
        elif isinstance(node, list):
            for i, item in enumerate(node):
                _walk(item, ep, f"{path}[{i}]")

    for ep, res in probe_result.items():
        if isinstance(res, dict) and res.get("status") == 200:
            _walk(res.get("body"), ep)
    return hits


# Endpoints known to wipe the FDLib face index as a side-effect of any PUT.
# Root cause of the 2026-05-19 incident: writing FaceRecognizeMode silently
# blanked every face on PYP + Basement, leaving UserInfo intact.
DANGEROUS_ENDPOINTS = (
    "/ISAPI/AccessControl/FaceRecognizeMode",
    "/ISAPI/AccessControl/captureFaceMode",
)


def set_threshold(device: dict, endpoint: str, key: str, value: int,
                  allow_rewipe: bool = False) -> tuple[int, str]:
    """PUT the new threshold value back. Reads, mutates, writes."""
    write_ep_check = endpoint.replace("/capabilities", "")
    if any(write_ep_check.startswith(d) for d in DANGEROUS_ENDPOINTS) and not allow_rewipe:
        return 0, (f"REFUSED: {write_ep_check} is a face-library-wiping endpoint. "
                   f"Re-run with --allow-mode-rewipe and be prepared to "
                   f"_rebind_pyp_faces.py every device immediately after.")
    s, base = _session(device)

    # Read current
    r = s.get(base + endpoint, timeout=8)
    if r.status_code != 200:
        return r.status_code, f"GET failed: {r.text[:200]}"
    try:
        body = r.json()
    except Exception:
        return r.status_code, "Response was not JSON; cannot mutate safely"

    # Mutate the matching key wherever it sits in the tree
    mutated = [False]

    def _mutate(node):
        if isinstance(node, dict):
            for k in list(node.keys()):
                if k == key and isinstance(node[k], (int, float)):
                    node[k] = value
                    mutated[0] = True
                else:
                    _mutate(node[k])
        elif isinstance(node, list):
            for item in node:
                _mutate(item)

    _mutate(body)
    if not mutated[0]:
        return 0, f"Key '{key}' not found in body — refusing to PUT"

    # PUT it back (strip /capabilities suffix if present)
    write_endpoint = endpoint.replace("/capabilities", "")
    r2 = s.put(
        base + write_endpoint,
        json=body,
        headers={"Content-Type": "application/json"},
        timeout=10,
    )
    return r2.status_code, r2.text[:300]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", help="Substring match on device name (default: all enabled)")
    ap.add_argument("--set", type=int, dest="set_value",
                    help="New threshold value to write (e.g. 85). Omit to probe only.")
    ap.add_argument("--allow-mode-rewipe", action="store_true",
                    help="Permit writes to FaceRecognizeMode / captureFaceMode "
                         "endpoints (these WIPE the FDLib face index on V4.38). "
                         "You must re-bind every device immediately after.")
    args = ap.parse_args()

    devices = [d for d in load_devices() if d.get("enabled", True)]
    if args.device:
        needle = args.device.lower()
        devices = [d for d in devices if needle in d["name"].lower()]

    if not devices:
        print("No matching enabled devices.")
        sys.exit(1)

    for d in devices:
        print(f"\n{'=' * 70}")
        print(f"  {d['name']}  ({d['ip']})")
        print(f"{'=' * 70}")

        result = probe(d)
        if not result:
            print("  ⚠ No probe endpoint responded — device may be offline.")
            continue

        for ep, res in result.items():
            print(f"\n  → {ep}  [{res['status']}]")
            if res["status"] == 200:
                pretty = json.dumps(res["body"], indent=4)
                # truncate long bodies
                if len(pretty) > 1500:
                    pretty = pretty[:1500] + "\n    ... (truncated)"
                print(pretty)

        hits = find_current_threshold(result)
        if hits:
            print("\n  ✓ Threshold-like fields found:")
            for ep, k, v in hits:
                print(f"     • {ep}  →  {k} = {v}")
        else:
            print("\n  ⚠ No threshold-like field detected. "
                  "Inspect the probe output above to find the firmware-specific key.")

        if args.set_value is not None:
            if not hits:
                print("\n  ✗ Cannot --set: no threshold field discovered to write to.")
                continue
            # Use the first hit by default
            ep, key, old = hits[0]
            print(f"\n  ⚙  Writing {key}: {old} → {args.set_value}  via {ep}")
            status, body = set_threshold(d, ep, key.split(".")[-1], args.set_value,
                                         allow_rewipe=args.allow_mode_rewipe)
            print(f"     PUT status: {status}")
            print(f"     Response  : {body}")


if __name__ == "__main__":
    main()
