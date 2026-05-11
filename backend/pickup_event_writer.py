"""
pickup_event_writer.py — PickupGuard event sink.

Called by attendance_listener.py whenever a face scan's employeeNo is
classified as a chaperone (employeeNo[0] == '9'). Responsibilities:

  1. Look up the chaperone doc to get authorized students + denormalized
     student photos.
  2. Persist the live face JPEG (if the device sent one) to Storage
     under tenants/{tid}/pickup_captures/{eventId}.jpg.
  3. Write a pickup_events/{eventId} doc with everything the TV display
     needs to render a card without further reads.
  4. Publish a tiny JSON line to a local UNIX socket so the (P3) SSE
     server can fan it out to connected gate displays in <100 ms.
     Failure to publish is non-fatal.

Pickup events are FULLY ISOLATED from attendance/* — chaperones never
flip a student's attendance status. (Decision #4.)
"""
from __future__ import annotations

import json
import os
import socket
import threading
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from firebase_admin import firestore
import firebase_admin

import tenancy

WIB = timezone(timedelta(hours=7))

# Local UNIX socket the SSE fan-out server (P3) listens on. Best-effort —
# absent socket means we just skip the publish, doc still lives in Firestore
# and the TV display can fall back to onSnapshot.
PICKUP_EVENT_SOCKET = os.environ.get(
    "PICKUP_EVENT_SOCKET", "/tmp/pickupguard.sock"
)

# HTTP fan-out to the Next.js SSE bus (Phase 1 of latency optimization).
# When set, every successful pickup_event write is also POSTed to
# {INTERNAL_NOTIFY_URL} so connected iPad teacher PWAs get the new card via
# Server-Sent Events in <1s instead of waiting for the next 2.5s poll.
# Both env vars must be set for the push to fire; missing config = silent skip
# (the polling fallback on the iPad still hydrates new events).
INTERNAL_NOTIFY_URL = os.environ.get("INTERNAL_NOTIFY_URL", "").strip()
INTERNAL_PUSH_SECRET = os.environ.get("INTERNAL_PUSH_SECRET", "").strip()
_NOTIFY_TIMEOUT_SEC = 1.5

# Cached pickup_settings per tenant (small TTL — admin changes are rare).
_settings_cache: dict[str, tuple[float, dict]] = {}
_SETTINGS_TTL = 60.0  # seconds


def _now() -> datetime:
    return datetime.now(WIB)


def _get_db():
    if not firebase_admin._apps:
        return None
    return firestore.client()


_terminal_group_cache: dict[str, tuple[float, Optional[str]]] = {}
_TERMINAL_CACHE_TTL = 60.0


def _resolve_release_group_id(db, tid: str, terminal_id: Optional[str]) -> Optional[str]:
    """Lookup tenants/{t}/terminals/{terminalId}.releaseGroupId, with a 60s cache.

    Returns None when terminal_id is missing or has no group binding yet.
    """
    if not terminal_id or not db:
        return None
    import time as _time
    key = f"{tid}|{terminal_id}"
    cached = _terminal_group_cache.get(key)
    if cached and _time.time() - cached[0] < _TERMINAL_CACHE_TTL:
        return cached[1]
    try:
        snap = db.document(f"{tenancy.terminals_path(tid)}/{terminal_id}").get()
        rg = (snap.to_dict() or {}).get("releaseGroupId") if snap.exists else None
    except Exception:
        rg = None
    _terminal_group_cache[key] = (_time.time(), rg)
    return rg


def _get_pickup_settings(tid: str) -> dict:
    """Read tenants/{tid}/settings/pickup with a 60s in-memory cache."""
    import time as _time
    cached = _settings_cache.get(tid)
    if cached and _time.time() - cached[0] < _SETTINGS_TTL:
        return cached[1]
    db = _get_db()
    if not db:
        return {}
    snap = db.document(tenancy.pickup_settings_doc(tid)).get()
    data = snap.to_dict() if snap.exists else {}
    _settings_cache[tid] = (_time.time(), data or {})
    return data or {}


def _resolve_chaperone(db, tid: str, employee_no: str) -> Optional[dict]:
    """Find chaperones doc for a 9XXX employeeNo. Returns None if unknown."""
    chap_id = f"chap-{employee_no}"
    snap = db.document(f"{tenancy.chaperones_path(tid)}/{chap_id}").get()
    if snap.exists:
        d = snap.to_dict()
        d["_id"] = chap_id
        return d
    # Fallback: query by employeeNo field (handles legacy id schemes)
    q = (db.collection(tenancy.chaperones_path(tid))
           .where("employeeNo", "==", employee_no).limit(1).stream())
    for s in q:
        d = s.to_dict()
        d["_id"] = s.id
        return d
    return None


def _resolve_students(db, tid: str, student_ids: list[str]) -> list[dict]:
    """Pull denormalized student summaries for the TV card."""
    out = []
    for sid in (student_ids or [])[:10]:  # safety cap
        snap = db.document(f"{tenancy.students_path(tid)}/{sid}").get()
        if not snap.exists:
            # legacy fallback during dual-read window
            snap = db.document(f"students/{sid}").get()
        if snap.exists:
            d = snap.to_dict() or {}
            out.append({
                "id": sid,
                "name": d.get("name") or d.get("fullName") or sid,
                "homeroom": d.get("homeroom") or d.get("className"),
                "photoUrl": d.get("photoUrl"),
            })
        else:
            out.append({"id": sid, "name": sid, "homeroom": None, "photoUrl": None})
    return out


def _is_reenroll_overdue(due_at, now: datetime) -> bool:
    """True if a chaperone's reEnrollDueAt has passed."""
    if not due_at:
        return False
    try:
        if hasattr(due_at, "timestamp"):  # Firestore Timestamp / datetime
            ts = due_at if isinstance(due_at, datetime) else due_at.to_datetime()
        elif isinstance(due_at, str):
            ts = datetime.fromisoformat(due_at.replace("Z", "+00:00"))
        else:
            return False
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts <= now
    except Exception:
        return False


def _log_security_incident(db, tid: str, *, kind: str, event_id: str,
                           employee_no: str, gate: str,
                           chaperone_name: Optional[str]) -> None:
    """Append-only record for unknown/suspended/reenroll-overdue scans. Best effort."""
    try:
        incident_id = uuid.uuid4().hex
        db.document(f"{tenancy.security_incidents_path(tid)}/{incident_id}").set({
            "incidentId": incident_id,
            "kind": kind,                       # 'unknown_chaperone' | 'suspended' | 'reenroll_overdue'
            "eventId": event_id,
            "employeeNo": employee_no,
            "gate": gate,
            "chaperoneName": chaperone_name,
            "createdAt": _now().isoformat(),
            "resolved": False,
        })
    except Exception as e:
        print(f"  ⚠ Security incident log failed ({kind}): {e}")


def _save_jpeg(bucket, tid: str, event_id: str, jpeg_bytes: bytes) -> Optional[str]:
    if not jpeg_bytes:
        return None
    try:
        path = tenancy.storage_pickup_capture_path(event_id, tid)
        blob = bucket.blob(path)
        blob.upload_from_string(jpeg_bytes, content_type="image/jpeg")
        return path
    except Exception as e:
        print(f"  ⚠ Pickup JPEG save failed: {e}")
        return None


def _publish_socket(payload: dict, gate: Optional[str] = None) -> None:
    """Best-effort UNIX socket publish for SSE fan-out. Silent on failure."""
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        sock.settimeout(0.2)
        line = json.dumps({"gate": gate, "event": payload}, separators=(",", ":")) + "\n"
        sock.sendto(line.encode("utf-8"), PICKUP_EVENT_SOCKET)
        sock.close()
    except Exception:
        pass  # SSE server may not be running yet — this is OK


def _publish_http_notify(tenant_id: str, event_id: str) -> None:
    """Fire-and-forget POST to the Next.js SSE bus so iPads see the new card
    in <1s instead of waiting for the polling fallback. Runs in a daemon
    thread so the writer's hot path never blocks on network IO."""
    if not INTERNAL_NOTIFY_URL or not INTERNAL_PUSH_SECRET:
        return
    if not tenant_id or not event_id:
        return

    def _do_post() -> None:
        try:
            body = json.dumps({
                "tenantId": tenant_id,
                "eventId": event_id,
            }).encode("utf-8")
            req = urllib.request.Request(
                INTERNAL_NOTIFY_URL,
                data=body,
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "X-Internal-Push-Secret": INTERNAL_PUSH_SECRET,
                },
            )
            with urllib.request.urlopen(req, timeout=_NOTIFY_TIMEOUT_SEC) as resp:
                resp.read()  # drain
        except urllib.error.HTTPError as e:
            print(f"  ⚠ SSE notify HTTP {e.code} for {event_id}")
        except Exception as e:
            # Non-fatal — polling fallback covers this.
            print(f"  ⚠ SSE notify failed for {event_id}: {e}")

    threading.Thread(target=_do_post, daemon=True).start()


def record_pickup_event(
    *,
    tenant_id: str,
    employee_no: str,
    chaperone_name_hint: Optional[str],
    scanned_at: datetime,
    device_name: str,
    gate: Optional[str] = None,
    terminal_id: Optional[str] = None,
    jpeg_bytes: Optional[bytes] = None,
    # ── Facial-recognition signals (all optional; the listener fills these
    # in when it has them — older Hikvision-only callers leave them null).
    fr_confidence: Optional[float] = None,    # 0.0 – 1.0 match similarity
    fr_distance: Optional[float] = None,      # raw embedding distance
    liveness_score: Optional[float] = None,   # 0.0 – 1.0 (1.0 = clearly live)
    liveness_passed: Optional[bool] = None,   # threshold result from engine
    spoof_flag: Optional[bool] = None,        # True ⇒ spoof attempt detected
    fr_retries: Optional[int] = None,         # # of attempts before lock-on
    fr_engine: Optional[str] = None,          # 'hikvision' | 'edge_dlib' | 'cloud'
) -> Optional[dict]:
    """
    Persist a pickup event. Returns the written doc dict (with id) or None
    on failure. Pure side-effect; caller does not need to do anything else.
    """
    db = _get_db()
    if not db:
        print("  ⚠ Pickup event skipped — Firestore not initialized")
        return None

    tid = tenant_id or tenancy.get_tenant_id()
    settings = _get_pickup_settings(tid)
    chaperone = _resolve_chaperone(db, tid, employee_no)
    reenroll_overdue = False

    # If the chaperone is unknown (Hikvision enroll happened outside our
    # app, or doc was deleted) we still record the event as 'unknown' so
    # the gate officer can deal with it manually.
    if not chaperone:
        chap_summary = {
            "id": None,
            "name": chaperone_name_hint or f"Unknown ({employee_no})",
            "relation": None,
            "photoUrl": None,
            "suspended": False,
        }
        student_ids = []
        decision = "unknown_chaperone"
    else:
        if chaperone.get("suspendedAt"):
            decision = "suspended"
        else:
            reenroll_overdue = _is_reenroll_overdue(chaperone.get("reEnrollDueAt"), _now())
            decision = "reenroll_overdue" if reenroll_overdue else "ok"
        chap_summary = {
            "id": chaperone.get("_id"),
            "name": chaperone.get("name") or chaperone_name_hint or employee_no,
            "relation": chaperone.get("relation"),
            "photoUrl": (chaperone.get("facePaths") or [None])[0],
            "phone": chaperone.get("phone"),
            "suspended": bool(chaperone.get("suspendedAt")),
            "reEnrollDueAt": chaperone.get("reEnrollDueAt"),
            "reEnrollOverdue": reenroll_overdue,
        }
        student_ids = chaperone.get("authorizedStudentIds") or []

    students = _resolve_students(db, tid, student_ids)

    # Compute card color hint for TV display. Gate-window classification is
    # now a pure display concern (handled per-profile in the TV feed) so we
    # do NOT colour by time-of-day here — a single source of truth lives in
    # web-dataset-collector/lib/kiosk-profiles.gateStatus().
    if decision in ("unknown_chaperone", "suspended"):
        card_state = "red"
    elif decision == "reenroll_overdue":
        card_state = "yellow"
    elif not students:
        card_state = "yellow"  # chaperone has no authorized students yet
    else:
        card_state = "green"

    event_id = uuid.uuid4().hex
    now_iso = _now().isoformat()

    # 6-digit override code for officers — only meaningful for flagged events.
    # Officer enters this in /v2/officer-override to flip the card to approved.
    override_code = None
    if decision != "ok":
        # uuid hex first 6 chars → digits only via int conversion mod 1e6
        override_code = f"{int(event_id[:8], 16) % 1_000_000:06d}"

    # Save JPEG to Storage (best-effort)
    capture_path = None
    if jpeg_bytes:
        try:
            from firebase_admin import storage as _storage
            bucket = _storage.bucket()
            capture_path = _save_jpeg(bucket, tid, event_id, jpeg_bytes)
        except Exception as e:
            print(f"  ⚠ Pickup capture skipped: {e}")

    doc = {
        "eventId": event_id,
        "tenantId": tid,
        "employeeNo": employee_no,
        "scannedAt": scanned_at.isoformat(),
        "recordedAt": now_iso,
        "deviceName": device_name,
        "gate": gate or device_name,
        "terminalId": terminal_id,
        "releaseGroupId": _resolve_release_group_id(db, tid, terminal_id),
        "chaperone": chap_summary,
        "students": students,
        "decision": decision,
        "cardState": card_state,
        "capturePath": capture_path,
        "officerOverride": None,    # set later if officer page approves
        "overrideCode": override_code,
        "holdSeconds": int(settings.get("holdSeconds") or 60),
        # ── FR signals (None when not provided) ──────────────────────
        "fr": {
            "confidence": float(fr_confidence) if fr_confidence is not None else None,
            "distance":   float(fr_distance)   if fr_distance   is not None else None,
            "liveness":   float(liveness_score) if liveness_score is not None else None,
            "livenessPassed": bool(liveness_passed) if liveness_passed is not None else None,
            "spoof":      bool(spoof_flag) if spoof_flag is not None else None,
            "retries":    int(fr_retries) if fr_retries is not None else None,
            "engine":     fr_engine or "hikvision",
            "enrolledPhotoPath": (chap_summary.get("photoUrl") if isinstance(chap_summary, dict) else None),
        },
    }

    try:
        db.document(f"{tenancy.pickup_events_path(tid)}/{event_id}").set(doc)
    except Exception as e:
        print(f"  ⚠ Pickup event write failed: {e}")
        return None

    # Log a security incident for any non-ok decision so admins/Prometheus
    # can act on them. unknown_chaperone is the most important case.
    if decision in ("unknown_chaperone", "suspended", "reenroll_overdue"):
        _log_security_incident(
            db, tid,
            kind=decision,
            event_id=event_id,
            employee_no=employee_no,
            gate=gate or device_name,
            chaperone_name=chap_summary.get("name"),
        )

    # Bump chaperone lastSeenAt (best-effort)
    if chaperone and chap_summary["id"]:
        try:
            db.document(f"{tenancy.chaperones_path(tid)}/{chap_summary['id']}").set(
                {"lastSeenAt": now_iso, "lastSeenGate": gate or device_name},
                merge=True,
            )
        except Exception:
            pass

    _publish_socket(doc, gate=gate)
    _publish_http_notify(tid, event_id)

    pretty = ", ".join(s["name"] for s in students) or "(no students)"
    color = {"green": "🟢", "yellow": "🟡", "red": "🔴"}.get(card_state, "⚪")
    print(f"{color} PICKUP [{scanned_at.strftime('%H:%M:%S')}] "
          f"{chap_summary['name']} → {pretty}  ({decision})")

    return doc
