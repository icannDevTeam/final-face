#!/usr/bin/env python3
"""
Mint a guardian consent token for the consent capture page.

Generates an HMAC-signed token compatible with
`web-dataset-collector/lib/consent-token.js`. Email the resulting URL to
the guardian; they can record or withdraw consent without an account.

Requires CONSENT_SIGNING_SECRET (or SESSION_SECRET / DASHBOARD_API_KEY)
to be set in the environment — must match the server's secret.

Usage:
    python3 issue_consent_token.py --student 2270005673 --base-url https://attend.example.com
    python3 issue_consent_token.py --student 2270005673 --tenant binus-simprug --ttl-days 30
"""
from __future__ import annotations

import argparse
import base64
import hmac
import hashlib
import json
import os
import sys
import time

import tenancy


def _b64url(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


def sign(tenant_id: str, student_id: str, ttl_seconds: int, secret: str) -> str:
    payload = {
        "tid": tenant_id,
        "sid": student_id,
        "exp": int(time.time()) + int(ttl_seconds),
    }
    body = _b64url(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    sig = hmac.new(secret.encode("utf-8"), body.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{body}.{sig}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", required=True, help="Student ID (employeeNo)")
    parser.add_argument("--tenant", default=None, help="Tenant ID (default: env TENANT_ID or binus-simprug)")
    parser.add_argument("--ttl-days", type=int, default=30, help="Token validity window in days (default 30)")
    parser.add_argument("--base-url", default=None, help="Optional base URL to print the full link")
    args = parser.parse_args()

    secret = (
        os.environ.get("CONSENT_SIGNING_SECRET")
        or os.environ.get("SESSION_SECRET")
        or os.environ.get("DASHBOARD_API_KEY")
    )
    if not secret:
        print("error: set CONSENT_SIGNING_SECRET (or SESSION_SECRET) in env", file=sys.stderr)
        return 2

    tid = tenancy.get_tenant_id(args.tenant)
    token = sign(tid, args.student, args.ttl_days * 86400, secret)
    print(f"tenant:    {tid}")
    print(f"student:   {args.student}")
    print(f"ttl:       {args.ttl_days} day(s)")
    print(f"token:     {token}")
    if args.base_url:
        print(f"link:      {args.base_url.rstrip('/')}/consent/{token}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
