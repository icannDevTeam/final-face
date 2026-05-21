#!/usr/bin/env python3
"""
Mint a parent onboarding token for the PickupGuard onboarding page.

Compatible with web-dataset-collector/lib/pickup-token.js. The token
carries `purpose:'pickup-onboarding'` so it cannot be re-used in the
consent flow.

By default the script also mints a 6-character short code and writes it
to Firestore `pickupShortLinks/{CODE}`, so you can hand parents a friendly
URL like  https://dataset-sigma.vercel.app/p/K7M3Q9  instead of the
200-char signed-token URL. Disable with --no-short.

Usage:
    python3 issue_pickup_onboarding_token.py --tenant binus-simprug --student 2270005673
    python3 issue_pickup_onboarding_token.py --tenant binus-simprug --ttl-days 14 \
        --base-url https://dataset-sigma.vercel.app
    python3 issue_pickup_onboarding_token.py --tenant binus-simprug --no-short
"""
from __future__ import annotations

import argparse
import base64
import hmac
import hashlib
import json
import os
import secrets
import sys
import time

import tenancy

PURPOSE = "pickup-onboarding"

# Crockford-style alphabet — no 0/O, no 1/I/L, no U (lookalike or rude).
SHORT_CODE_ALPHABET = "ABCDEFGHJKMNPQRSTVWXYZ23456789"
SHORT_CODE_LEN = 6


def _b64url(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


def sign(tenant_id: str, student_id: str | None, ttl_seconds: int, secret: str) -> tuple[str, int]:
    exp = int(time.time()) + int(ttl_seconds)
    payload = {
        "tid": tenant_id,
        "sid": student_id if student_id else None,
        "exp": exp,
        "p": PURPOSE,
    }
    body = _b64url(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    sig = hmac.new(secret.encode("utf-8"), body.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{body}.{sig}", exp


def _new_short_code() -> str:
    return "".join(secrets.choice(SHORT_CODE_ALPHABET) for _ in range(SHORT_CODE_LEN))


def _init_firestore():
    """Lazy-init firebase-admin. Returns the firestore client."""
    import firebase_admin
    from firebase_admin import credentials, firestore as _fs

    sa_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "facial-attendance-binus-firebase-adminsdk.json",
    )
    if not firebase_admin._apps:
        firebase_admin.initialize_app(credentials.Certificate(sa_path))
    return _fs.client()


def _allocate_short_code(db, token: str, tid: str, sid: str | None,
                         exp_seconds: int, label: str | None,
                         max_attempts: int = 8) -> str | None:
    """
    Allocate a unique short code in Firestore. Retries on collision.
    Returns the code, or None on persistent failure.
    """
    payload = {
        "token": token,
        "tid": tid,
        "sid": sid,
        "exp": int(exp_seconds) * 1000,  # ms epoch — matches JS Date.now()
        "createdAt": int(time.time() * 1000),
        "hits": 0,
    }
    if label:
        payload["label"] = label

    coll = db.collection("pickupShortLinks")
    for _ in range(max_attempts):
        code = _new_short_code()
        ref = coll.document(code)
        try:
            ref.create(payload)
            return code
        except Exception as e:
            if "already exists" in str(e).lower():
                continue
            raise
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", default=None,
                        help="Optional primary student ID (siblings can be added in the form)")
    parser.add_argument("--tenant", default=None)
    parser.add_argument("--ttl-days", type=int, default=14)
    parser.add_argument("--base-url", default=None,
                        help="Public base URL (e.g. https://dataset-sigma.vercel.app)")
    parser.add_argument("--no-short", action="store_true",
                        help="Skip Firestore short-code allocation; print only the long URL")
    parser.add_argument("--label", default=None,
                        help="Optional human label stored alongside the short code (e.g. 'Carter S. - Grade 5A')")
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
    token, exp = sign(tid, args.student, args.ttl_days * 86400, secret)

    short_code = None
    if not args.no_short:
        try:
            db = _init_firestore()
            short_code = _allocate_short_code(
                db, token, tid, args.student, exp, args.label,
            )
            if not short_code:
                print("warning: could not allocate a unique short code after retries — "
                      "use --no-short to skip", file=sys.stderr)
        except Exception as e:
            print(f"warning: short-code allocation failed ({e}) — falling back to long URL",
                  file=sys.stderr)

    base = args.base_url.rstrip("/") if args.base_url else None

    print(f"tenant:    {tid}")
    print(f"student:   {args.student or '(none — parent picks in form)'}")
    print(f"ttl:       {args.ttl_days} day(s)")
    print(f"purpose:   {PURPOSE}")
    if short_code:
        print(f"code:      {short_code}")
    print(f"token:     {token}")
    if base:
        if short_code:
            print()
            print("  Short link (give this to parents):")
            print(f"    {base}/p/{short_code}")
            print()
            print("  Long link (fallback / direct):")
            print(f"    {base}/pickup/onboarding/{token}")
        else:
            print(f"link:      {base}/pickup/onboarding/{token}")
    elif short_code:
        print()
        print(f"  Short code: {short_code}  →  share as /p/{short_code} on your public domain")
    return 0


if __name__ == "__main__":
    sys.exit(main())
