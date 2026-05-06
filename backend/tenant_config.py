"""
Tenant config loader (Python side) — Phase A2.

Mirrors web-dataset-collector/lib/tenant-config.js. Used by all backend
listeners / scripts so they pick up tenant-specific settings (timezone,
late cutoff, geofence) instead of hardcoded BINUS values.
"""
from __future__ import annotations

import os
import time
from typing import Any, Optional

import firebase_admin
from firebase_admin import credentials, firestore

import tenancy

SERVICE_ACCOUNT_PATH = os.path.join(
    os.path.dirname(__file__),
    "facial-attendance-binus-firebase-adminsdk.json",
)

CACHE_TTL_SEC = 60
_cache: dict[str, tuple[dict, float]] = {}


def _ensure_admin() -> firestore.Client:
    if not firebase_admin._apps:
        cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
        firebase_admin.initialize_app(cred)
    return firestore.client()


def load_tenant_config(tenant_id: Optional[str] = None, force: bool = False) -> dict[str, Any]:
    tid = tenancy.get_tenant_id(tenant_id)
    now = time.time()
    cached = _cache.get(tid)
    if not force and cached and now - cached[1] < CACHE_TTL_SEC:
        return cached[0]
    db = _ensure_admin()
    snap = db.document(f"{tenancy.tenant_doc(tid)}/settings/config").get()
    if snap.exists:
        merged = {**tenancy.DEFAULT_TENANT_CONFIG, **(snap.to_dict() or {}), "slug": tid}
    else:
        merged = {**tenancy.DEFAULT_TENANT_CONFIG, "slug": tid}
    _cache[tid] = (merged, now)
    return merged


def get_late_cutoff_minute(tenant_id: Optional[str] = None) -> tuple[int, int]:
    """Returns (hour, minute) for the configured late cutoff (default 07:30)."""
    cfg = load_tenant_config(tenant_id)
    raw = cfg.get("lateCutoffHHmm", "07:30")
    try:
        hh, mm = raw.split(":")
        return int(hh), int(mm)
    except Exception:
        return 7, 30


def clear_cache(tenant_id: Optional[str] = None) -> None:
    if tenant_id:
        _cache.pop(tenancy.get_tenant_id(tenant_id), None)
    else:
        _cache.clear()
