#!/usr/bin/env python3
"""
Seed an immutable Privacy Policy version into a tenant — Phase B1+B2.

Creates `tenants/{tid}/policy_versions/{versionId}` with the supplied
markdown body, and (unless --no-activate) updates
`tenants/{tid}/settings/config.currentPolicyVersionId` to point at it.

Once a policy version is referenced by any consent record, it becomes
historically immutable — re-running this script always creates a NEW
version doc rather than mutating an existing one.

Usage:
    python3 seed_policy_version.py --tenant binus-simprug --file privacy_policy.md
    python3 seed_policy_version.py --version v1.0.0 --file policy.md --effective 2026-05-01
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import firebase_admin
from firebase_admin import credentials, firestore

import tenancy

SERVICE_ACCOUNT_PATH = os.path.join(
    os.path.dirname(__file__),
    "facial-attendance-binus-firebase-adminsdk.json",
)


def slugify_version(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "-", s.strip())
    return s.strip("-") or "v1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tenant", default=None)
    parser.add_argument("--file", required=True, help="Path to Privacy Policy markdown")
    parser.add_argument("--version", default=None, help="Version label (default: derived from timestamp)")
    parser.add_argument("--effective", default=None, help="Effective date YYYY-MM-DD (default: today UTC)")
    parser.add_argument("--no-activate", action="store_true", help="Don't set as currentPolicyVersionId")
    args = parser.parse_args()

    if not firebase_admin._apps:
        firebase_admin.initialize_app(credentials.Certificate(SERVICE_ACCOUNT_PATH))
    db = firestore.client()
    tid = tenancy.get_tenant_id(args.tenant)

    body = Path(args.file).read_text(encoding="utf-8")
    if not body.strip():
        print("error: policy file is empty", file=sys.stderr)
        return 2

    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    version_id = slugify_version(args.version or f"v-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}")
    effective = args.effective or datetime.now(timezone.utc).date().isoformat()
    now_iso = datetime.now(timezone.utc).isoformat()

    # Refuse to overwrite an existing version doc — versions are immutable
    pv_ref = db.document(f"{tenancy.policy_versions_path(tid)}/{version_id}")
    if pv_ref.get().exists:
        print(f"error: tenants/{tid}/policy_versions/{version_id} already exists. "
              f"Pick a new --version label.", file=sys.stderr)
        return 1

    pv_ref.set({
        "versionId": version_id,
        "tenantId": tid,
        "body": body,
        "bodyFormat": "markdown",
        "sha256": digest,
        "createdAt": now_iso,
        "effectiveDate": effective,
        "byteLength": len(body.encode("utf-8")),
    })
    print(f"[ok]   tenants/{tid}/policy_versions/{version_id}  ({len(body)} chars, sha256={digest[:12]}…)")

    if not args.no_activate:
        cfg_ref = db.document(f"{tenancy.tenant_doc(tid)}/settings/config")
        cfg_ref.set({"currentPolicyVersionId": version_id, "currentPolicyActivatedAt": now_iso}, merge=True)
        print(f"[ok]   tenants/{tid}/settings/config.currentPolicyVersionId = {version_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
