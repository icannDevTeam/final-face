#!/usr/bin/env python3
"""
drop_tv_collections.py — One-shot cleanup of TV-display Firestore collections.

After dropping the TV display feature, these per-tenant collections are dead:
  - tenants/{tid}/tv_devices
  - tenants/{tid}/kiosk_profiles            (NOT touched — still used for gate hours)
  - (root, if pre-tenancy)  tv_devices

Run AFTER you've deployed the code change and confirmed nothing still reads
these paths.

Safe to re-run — uses delete_batched().

Usage:
    cd backend && python3 scripts/drop_tv_collections.py [--dry-run] [--yes]

Env: needs facial-attendance-binus-firebase-adminsdk.json in cwd
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import firebase_admin
from firebase_admin import credentials, firestore

# Collections to wipe per tenant.
TV_ONLY_COLLECTIONS = ["tv_devices"]
# Root-level legacy paths (pre-tenancy installs).
ROOT_LEGACY = ["tv_devices"]

HERE = Path(__file__).resolve().parent.parent
CRED = HERE / "facial-attendance-binus-firebase-adminsdk.json"


def init_db() -> firestore.Client:
    if not CRED.exists():
        sys.exit(f"missing service account at {CRED}")
    if not firebase_admin._apps:
        firebase_admin.initialize_app(credentials.Certificate(str(CRED)))
    return firestore.client()


def list_tenants(db: firestore.Client) -> list[str]:
    out = []
    for doc in db.collection("tenants").stream():
        out.append(doc.id)
    return out


def delete_collection(coll_ref, dry_run: bool, batch_size: int = 200) -> int:
    """Delete all docs in coll_ref. Returns count deleted."""
    total = 0
    while True:
        docs = list(coll_ref.limit(batch_size).stream())
        if not docs:
            break
        for d in docs:
            total += 1
            if dry_run:
                print(f"    would delete: {d.reference.path}")
            else:
                d.reference.delete()
        if dry_run:
            break
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="show what would be deleted")
    ap.add_argument("--yes", action="store_true", help="skip confirmation")
    args = ap.parse_args()

    db = init_db()

    print("Scanning tenants…")
    tenants = list_tenants(db)
    print(f"  found {len(tenants)} tenant(s): {tenants}")

    targets = []
    for tid in tenants:
        for c in TV_ONLY_COLLECTIONS:
            targets.append(("tenant", f"tenants/{tid}/{c}", db.collection(f"tenants/{tid}/{c}")))
    for c in ROOT_LEGACY:
        targets.append(("root", c, db.collection(c)))

    print(f"\nTargets ({len(targets)}):")
    for kind, path, _ in targets:
        print(f"  [{kind}] {path}")

    if not args.dry_run and not args.yes:
        ans = input("\nDelete these collections permanently? [y/N] ").strip().lower()
        if ans != "y":
            print("Aborted.")
            return

    total = 0
    for kind, path, ref in targets:
        n = delete_collection(ref, dry_run=args.dry_run)
        total += n
        print(f"  {path}: {'(would delete)' if args.dry_run else 'deleted'} {n} docs")

    print(f"\nDone. {'Would have deleted' if args.dry_run else 'Deleted'} {total} docs.")


if __name__ == "__main__":
    main()
