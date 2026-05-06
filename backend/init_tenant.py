#!/usr/bin/env python3
"""
Bootstrap a tenant — Phase A1.

Creates `tenants/{tenantId}` and `tenants/{tenantId}/settings/config` if they
don't exist yet. Idempotent: safe to re-run.

Usage:
    python3 init_tenant.py                              # uses TENANT_ID env or 'binus-simprug'
    python3 init_tenant.py --tenant-id acme-school
    python3 init_tenant.py --tenant-id acme-school --name "ACME Elementary"
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


def init_firebase() -> firestore.Client:
    if not firebase_admin._apps:
        cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
        firebase_admin.initialize_app(cred)
    return firestore.client()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tenant-id", default=None, help="Tenant slug (default: env TENANT_ID or binus-simprug)")
    parser.add_argument("--name", default=None, help="Display name")
    parser.add_argument("--owner-email", default=None, help="Email of initial owner user")
    parser.add_argument("--force", action="store_true", help="Overwrite existing settings/config")
    args = parser.parse_args()

    tenant_id = tenancy.get_tenant_id(args.tenant_id)
    db = init_firebase()
    now = datetime.now(timezone.utc).isoformat()

    # ── Tenant root doc ────────────────────────────────────────────
    tenant_ref = db.collection("tenants").document(tenant_id)
    tenant_snap = tenant_ref.get()
    if tenant_snap.exists and not args.force:
        print(f"[skip] tenants/{tenant_id} already exists (use --force to overwrite)")
    else:
        tenant_ref.set({
            "name": args.name or tenancy.DEFAULT_TENANT_CONFIG["name"],
            "slug": tenant_id,
            "status": "active",
            "plan": "pilot",
            "createdAt": now,
        }, merge=True)
        print(f"[ok]   tenants/{tenant_id} written")

    # ── settings/config ────────────────────────────────────────────
    cfg_ref = tenant_ref.collection("settings").document("config")
    cfg_snap = cfg_ref.get()
    if cfg_snap.exists and not args.force:
        print(f"[skip] tenants/{tenant_id}/settings/config already exists")
    else:
        cfg = dict(tenancy.DEFAULT_TENANT_CONFIG)
        if args.name:
            cfg["name"] = args.name
            cfg["branding"] = dict(cfg["branding"])
            cfg["branding"]["schoolName"] = args.name
        cfg["slug"] = tenant_id
        cfg["updatedAt"] = now
        cfg_ref.set(cfg, merge=True)
        print(f"[ok]   tenants/{tenant_id}/settings/config written")

    # ── Owner user ─────────────────────────────────────────────────
    if args.owner_email:
        owner_ref = tenant_ref.collection("users").document(args.owner_email.lower())
        owner_ref.set({
            "email": args.owner_email.lower(),
            "role": "owner",
            "createdAt": now,
            "active": True,
        }, merge=True)
        print(f"[ok]   tenants/{tenant_id}/users/{args.owner_email.lower()} (role=owner)")

    print(f"\nDone. Tenant '{tenant_id}' is ready.")
    print("Next step: run `python3 migrate_to_tenants.py --dry-run` to preview the data migration.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
