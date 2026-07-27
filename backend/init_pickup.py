#!/usr/bin/env python3
"""
Bootstrap pickup_settings + chaperone-counter for a tenant. Idempotent.

Usage:
    python3 init_pickup.py --tenant binus-simprug
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

import firebase_admin
from firebase_admin import credentials, firestore

import tenancy

SERVICE_ACCOUNT_PATH = os.path.join(
    os.path.dirname(__file__),
    "facial-attendance-binus-firebase-adminsdk.json",
)

DEFAULT_PICKUP_SETTINGS = {
    "holdSeconds": 60,
    "queueSize": 5,
    "requireLiveness": True,            # decision #1 — always on
    "requireOfficerConfirm": False,
    "pickupWindow": {"start": "14:00", "end": "17:30"},  # tenant-default; per-terminal overrides
    "cooldownSeconds": 600,             # same chap@same terminal silently skipped within this window
    "warmupMinutes": 30,                # grace minutes BEFORE pickupWindow.start where scans still count
    "enforceWindow": True,              # outside window → recorded as audit-only 'outside_window'
    "interParentSeconds": 2,            # global per-terminal throttle between any two scans (anti-collision)
    "outOfWindowAction": "warn",        # 'warn' | 'reject' (decision #5)
    "reEnrollMonths": 12,               # decision #6
    "reminderMonths": 11,
    "gates": [],                        # populated when first SSE display registers
    "denormalizeStudentPhotos": True,   # bake student thumbs onto pickup_events
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tenant", default=None)
    parser.add_argument("--reset", action="store_true",
                        help="Overwrite existing pickup settings (keeps counter)")
    args = parser.parse_args()

    if not firebase_admin._apps:
        firebase_admin.initialize_app(credentials.Certificate(SERVICE_ACCOUNT_PATH))
    db = firestore.client()
    tid = tenancy.get_tenant_id(args.tenant)

    # 1) Pickup settings
    settings_ref = db.document(tenancy.pickup_settings_doc(tid))
    snap = settings_ref.get()
    if snap.exists and not args.reset:
        print(f"[skip]  {tenancy.pickup_settings_doc(tid)} already exists "
              f"(use --reset to overwrite)")
    else:
        payload = dict(DEFAULT_PICKUP_SETTINGS)
        payload["tenantId"] = tid
        payload["updatedAt"] = datetime.now(timezone.utc).isoformat()
        settings_ref.set(payload, merge=False)
        print(f"[ok]    {tenancy.pickup_settings_doc(tid)} written")

    # 2) Chaperone counter (don't overwrite existing — preserves allocations)
    counter_ref = db.document(tenancy.id_allocations_doc("chaperone-counter", tid))
    if counter_ref.get().exists:
        print(f"[skip]  {tenancy.id_allocations_doc('chaperone-counter', tid)} exists")
    else:
        counter_ref.set({
            "last": 9_000_000_000,
            "prefix": tenancy.CHAPERONE_EMPLOYEENO_PREFIX,
            "tenantId": tid,
            "updatedAt": datetime.now(timezone.utc).isoformat(),
        })
        print(f"[ok]    chaperone-counter initialized at 9_000_000_000")

    return 0


if __name__ == "__main__":
    sys.exit(main())
