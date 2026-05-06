"""
sync_terminal_registry.py — upsert backend/devices.json into the Firestore
tenants/{tid}/terminals collection on listener startup.

Each terminal gets a stable id derived from sha1(device_name)[:12] so the
Pandora backend, the Next.js admin, and the iPad can all agree without a
side lookup table.

Idempotent: safe to call on every start. Existing fields edited by an admin
in the dashboard (gradeLabel, gateLabel, releaseGroupId, gateOverride) are
preserved across syncs because they are only set on first creation.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional

import firebase_admin
from firebase_admin import firestore

import tenancy

DEVICES_FILE = Path(__file__).parent / "devices.json"


def stable_terminal_id(name: str) -> str:
    return hashlib.sha1((name or "").encode("utf-8")).hexdigest()[:12]


def _load_devices_file(path: Optional[Path] = None) -> list[dict]:
    p = path or DEVICES_FILE
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text())
    except Exception:
        return []


def sync_terminal_registry(tenant_id: Optional[str] = None) -> int:
    """Upsert each entry from devices.json into the terminals collection.

    Returns the number of terminals processed (created or merged).
    Silently no-ops when Firebase is not initialized.
    """
    if not firebase_admin._apps:
        return 0
    devices = _load_devices_file()
    if not devices:
        return 0
    db = firestore.client()
    tid = tenancy.get_tenant_id(tenant_id)
    now = firestore.SERVER_TIMESTAMP
    count = 0
    for d in devices:
        name = (d.get("name") or "").strip()
        if not name:
            continue
        ip = (d.get("ip") or "").strip() or None
        enabled = bool(d.get("enabled", True))
        terminal_id = stable_terminal_id(name)
        ref = db.document(f"{tenancy.terminals_path(tid)}/{terminal_id}")
        existing = ref.get()
        patch = {
            "terminalId": terminal_id,
            "name": name,
            "ip": ip,
            "deviceName": name,
            "enabled": enabled,
            "lastSeenAt": now,
            "updatedAt": now,
        }
        if not existing.exists:
            patch["createdAt"] = now
            patch.setdefault("releaseGroupId", None)
            patch.setdefault("gateOverride", None)
            patch.setdefault("gradeLabel", None)
            patch.setdefault("gateLabel", None)
        try:
            ref.set(patch, merge=True)
            count += 1
        except Exception as e:
            print(f"  ⚠ sync_terminal_registry: failed to upsert {name}: {e}")
    return count


if __name__ == "__main__":
    cred_path = os.getenv("FIREBASE_CRED_PATH") or str(
        Path(__file__).parent / "facial-attendance-binus-firebase-adminsdk.json"
    )
    if not firebase_admin._apps:
        from firebase_admin import credentials
        firebase_admin.initialize_app(credentials.Certificate(cred_path))
    n = sync_terminal_registry()
    print(f"✓ sync_terminal_registry: upserted {n} terminals")
