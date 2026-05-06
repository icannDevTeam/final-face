#!/usr/bin/env python3
"""
Mint a parent onboarding token for the PickupGuard onboarding page.

Compatible with web-dataset-collector/lib/pickup-token.js. The token
carries `purpose:'pickup-onboarding'` so it cannot be re-used in the
consent flow.

Usage:
    python3 issue_pickup_onboarding_token.py --tenant binus-simprug --student 2270005673
    python3 issue_pickup_onboarding_token.py --tenant binus-simprug --ttl-days 14 \
        --base-url https://attend.example.com
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

PURPOSE = "pickup-onboarding"


def _b64url(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


def sign(tenant_id: str, student_id: str | None, ttl_seconds: int, secret: str) -> str:
    payload = {
        "tid": tenant_id,
        "sid": student_id if student_id else None,
        "exp": int(time.time()) + int(ttl_seconds),
        "p": PURPOSE,
    }
    body = _b64url(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    sig = hmac.new(secret.encode("utf-8"), body.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{body}.{sig}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", default=None,
                        help="Optional primary student ID (siblings can be added in the form)")
    parser.add_argument("--tenant", default=None)
    parser.add_argument("--ttl-days", type=int, default=14)
    parser.add_argument("--base-url", default=None)
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
    print(f"student:   {args.student or '(none — parent picks in form)'}")
    print(f"ttl:       {args.ttl_days} day(s)")
    print(f"purpose:   {PURPOSE}")
    print(f"token:     {token}")
    if args.base_url:
        print(f"link:      {args.base_url.rstrip('/')}/pickup/onboarding/{token}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
