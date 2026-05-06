#!/usr/bin/env python3
"""
Manage tenant user RBAC + custom claims — Phase A4.

Creates / updates the `tenants/{tid}/users/{email}` doc AND sets matching
Firebase Auth custom claims (tenantId + role) so Firestore rules will accept
the user's requests.

Usage:
    python3 manage_user_claims.py add --tenant binus-simprug --email a@b.com --role owner
    python3 manage_user_claims.py add --email a@b.com --role admin
    python3 manage_user_claims.py remove --email a@b.com
    python3 manage_user_claims.py list --tenant binus-simprug
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

import firebase_admin
from firebase_admin import auth, credentials, firestore

import tenancy

SERVICE_ACCOUNT_PATH = os.path.join(
    os.path.dirname(__file__),
    "facial-attendance-binus-firebase-adminsdk.json",
)
VALID_ROLES = ("owner", "admin", "viewer")


def init() -> firestore.Client:
    if not firebase_admin._apps:
        cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
        firebase_admin.initialize_app(cred)
    return firestore.client()


def cmd_add(args) -> int:
    if args.role not in VALID_ROLES:
        print(f"error: role must be one of {VALID_ROLES}", file=sys.stderr)
        return 2
    db = init()
    tid = tenancy.get_tenant_id(args.tenant)
    email = args.email.lower()

    # ── Find or create the auth user ───────────────────────────────
    try:
        user = auth.get_user_by_email(email)
        print(f"[ok]   found auth user {user.uid}")
    except auth.UserNotFoundError:
        user = auth.create_user(email=email)
        print(f"[ok]   created auth user {user.uid}")

    # ── Set claims ─────────────────────────────────────────────────
    existing = user.custom_claims or {}
    new_claims = {**existing, "tenantId": tid, "role": args.role}
    auth.set_custom_user_claims(user.uid, new_claims)
    print(f"[ok]   claims set: tenantId={tid} role={args.role}")
    print("       (user must refresh ID token before claims take effect)")

    # ── Mirror to tenant users collection ──────────────────────────
    now = datetime.now(timezone.utc).isoformat()
    db.document(f"{tenancy.tenant_doc(tid)}/users/{email}").set({
        "email": email,
        "uid": user.uid,
        "role": args.role,
        "active": True,
        "updatedAt": now,
    }, merge=True)
    print(f"[ok]   tenants/{tid}/users/{email} written")
    return 0


def cmd_remove(args) -> int:
    db = init()
    email = args.email.lower()

    try:
        user = auth.get_user_by_email(email)
    except auth.UserNotFoundError:
        print(f"[skip] no auth user for {email}")
        return 0

    existing = user.custom_claims or {}
    tid = existing.get("tenantId")
    cleaned = {k: v for k, v in existing.items() if k not in ("tenantId", "role")}
    auth.set_custom_user_claims(user.uid, cleaned)
    print(f"[ok]   claims cleared on {user.uid}")

    if tid:
        ref = db.document(f"{tenancy.tenant_doc(tid)}/users/{email}")
        if ref.get().exists:
            ref.update({"active": False, "deactivatedAt": datetime.now(timezone.utc).isoformat()})
            print(f"[ok]   tenants/{tid}/users/{email} marked inactive")
    return 0


def cmd_list(args) -> int:
    db = init()
    tid = tenancy.get_tenant_id(args.tenant)
    print(f"Users in tenant '{tid}':")
    print(f"{'email':40s} {'role':10s} {'active':6s} uid")
    print("-" * 80)
    for snap in db.collection(f"{tenancy.tenant_doc(tid)}/users").stream():
        d = snap.to_dict() or {}
        print(f"{d.get('email', snap.id):40s} {d.get('role', '?'):10s} {str(d.get('active', '?')):6s} {d.get('uid', '?')}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_add = sub.add_parser("add", help="Add or update a tenant user with role")
    p_add.add_argument("--tenant", default=None)
    p_add.add_argument("--email", required=True)
    p_add.add_argument("--role", required=True, choices=VALID_ROLES)
    p_add.set_defaults(func=cmd_add)

    p_rm = sub.add_parser("remove", help="Remove a user's tenant access (clears claims)")
    p_rm.add_argument("--email", required=True)
    p_rm.set_defaults(func=cmd_remove)

    p_ls = sub.add_parser("list", help="List users in a tenant")
    p_ls.add_argument("--tenant", default=None)
    p_ls.set_defaults(func=cmd_list)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
