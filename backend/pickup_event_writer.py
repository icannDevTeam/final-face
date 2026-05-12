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


# ─── Cooldown + window helpers ────────────────────────────────────────────────
# Same chaperone re-tapping the same terminal within `cooldownSeconds` is
# treated as a duplicate and silently skipped (no Firestore write, no iPad
# card). Unknown / suspended chaperones bypass the cooldown so security
# events always reach staff.
_recent_scan_cache: dict[str, float] = {}  # key=tid|empNo|terminalId → epoch sec
_recent_terminal_scan: dict[str, float] = {}  # key=tid|terminalId → epoch sec (any parent)
_RECENT_LOCAL_TTL = 600.0                  # cache up to 10 min in-process


def _hhmm_to_minutes(s: Optional[str]) -> Optional[int]:
    if not s or not isinstance(s, str) or ":" not in s:
        return None
    try:
        h, m = s.split(":", 1)
        return int(h) * 60 + int(m)
    except Exception:
        return None


def _resolve_window(db, tid: str, terminal_id: Optional[str]) -> Optional[dict]:
    """Per-terminal window first, tenant-default second, else None.

    Returns {"open": HH:MM, "close": HH:MM} or None for always-open.
    """
    if terminal_id:
        try:
            tsnap = db.document(f"{tenancy.terminals_path(tid)}/{terminal_id}").get()
            if tsnap.exists:
                td = tsnap.to_dict() or {}
                w_open = td.get("windowOpen")
                w_close = td.get("windowClose")
                if w_open and w_close:
                    return {"open": w_open, "close": w_close}
        except Exception:
            pass
    settings = _get_pickup_settings(tid)
    pw = settings.get("pickupWindow") or {}
    if pw.get("start") and pw.get("end"):
        return {"open": pw["start"], "close": pw["end"]}
    return None


def _in_window(now_dt: datetime, window: dict, warmup_min: int) -> bool:
    """True if now_dt is within [open-warmup, close]. Handles overnight too."""
    open_min = _hhmm_to_minutes(window.get("open"))
    close_min = _hhmm_to_minutes(window.get("close"))
    if open_min is None or close_min is None:
        return True
    cur = now_dt.hour * 60 + now_dt.minute
    start = max(0, open_min - max(0, int(warmup_min)))
    if close_min >= start:
        return start <= cur <= close_min
    # Overnight (e.g. 22:00 → 02:00): wraps midnight
    return cur >= start or cur <= close_min


def _cooldown_active(db, tid: str, employee_no: str, terminal_id: Optional[str],
                     cooldown_sec: int, now_dt: datetime) -> bool:
    """Check in-process cache first, then Firestore for cross-process safety."""
    if cooldown_sec <= 0 or not terminal_id:
        return False
    import time as _time
    key = f"{tid}|{employee_no}|{terminal_id}"
    last = _recent_scan_cache.get(key)
    now_epoch = _time.time()
    # Drop very old cache entries
    if last and now_epoch - last > _RECENT_LOCAL_TTL:
        _recent_scan_cache.pop(key, None)
        last = None
    if last and (now_epoch - last) < cooldown_sec:
        return True
    # Cross-process: query Firestore for the latest event from this chap@terminal
    try:
        q = (db.collection(tenancy.pickup_events_path(tid))
               .where("employeeNo", "==", employee_no)
               .where("terminalId", "==", terminal_id)
               .order_by("recordedAt", direction="DESCENDING")
               .limit(1).stream())
        for s in q:
            d = s.to_dict() or {}
            ra = d.get("recordedAt")
            ts = None
            if hasattr(ra, "timestamp"):
                ts = ra.timestamp()
            elif isinstance(ra, str):
                try:
                    ts = datetime.fromisoformat(ra).timestamp()
                except Exception:
                    ts = None
            if ts and (now_dt.timestamp() - ts) < cooldown_sec:
                _recent_scan_cache[key] = ts
                return True
    except Exception:
        # Missing composite index, etc — fail open (don't block legitimate scans)
        return False
    return False


def _mark_recent(tid: str, employee_no: str, terminal_id: Optional[str], now_dt: datetime) -> None:
    if not terminal_id:
        return
    ts = now_dt.timestamp()
    _recent_scan_cache[f"{tid}|{employee_no}|{terminal_id}"] = ts
    _recent_terminal_scan[f"{tid}|{terminal_id}"] = ts


def _inter_parent_throttle_active(tid: str, terminal_id: Optional[str],
                                  min_gap_sec: float, now_dt: datetime) -> bool:
    """True if ANY parent scanned this terminal within `min_gap_sec`.

    Prevents two parents tapping back-to-back from colliding on the iPad.
    Same-parent re-taps are caught by the longer per-chaperone cooldown.
    """
    if not terminal_id or min_gap_sec <= 0:
        return False
    key = f"{tid}|{terminal_id}"
    last = _recent_terminal_scan.get(key)
    if not last:
        return False
    gap = now_dt.timestamp() - last
    if gap < 0 or gap > _RECENT_LOCAL_TTL:
        _recent_terminal_scan.pop(key, None)
        return False
    return gap < min_gap_sec


def _terminal_grade_scopes(db, tid: str, terminal_id: Optional[str]) -> list[str]:
    """Return gradeScopes array for a terminal. [] means 'all grades'."""
    if not terminal_id:
        return []
    try:
        snap = db.document(f"{tenancy.terminals_path(tid)}/{terminal_id}").get()
        if snap.exists:
            d = snap.to_dict() or {}
            scopes = d.get("gradeScopes")
            if isinstance(scopes, list):
                return [str(x).strip() for x in scopes if str(x).strip()]
    except Exception:
        pass
    return []


def _student_grade(student: dict) -> Optional[str]:
    """Pull leading digits from homeroom (e.g. '5C' → '5', '4A' → '4')."""
    hr = (student or {}).get("homeroom") or ""
    if not hr:
        return None
    digits = ""
    for ch in str(hr):
        if ch.isdigit():
            digits += ch
        else:
            break
    return digits or None


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

    now_dt = _now()

    # ── Wrong-terminal detection ─────────────────────────────────────────
    # If the terminal is grade-scoped and this chaperone has students but
    # NONE of them belong to that grade, treat as audit-only "wrong_terminal"
    # so the bound iPad doesn't fire. (e.g. Grade 2 parent walking past the
    # GRADE 5 gate.) Unknown chaperones bypass this check.
    if decision == "ok" and students:
        scopes = _terminal_grade_scopes(db, tid, terminal_id)
        if scopes:
            student_grades = {g for g in (_student_grade(s) for s in students) if g}
            if student_grades and not (student_grades & set(scopes)):
                decision = "wrong_terminal"

    # ── Cooldown + window guards ─────────────────────────────────────────
    # `silent` decisions are written for audit but suppressed from iPad feed.
    silent_decisions = {"unknown_chaperone", "wrong_terminal"}
    is_silent = decision in silent_decisions
    is_suspended = decision == "suspended"
    cooldown_sec = int(settings.get("cooldownSeconds") or 300)
    warmup_min = int(settings.get("warmupMinutes") or 30)
    enforce_window = bool(settings.get("enforceWindow", True))
    inter_parent_sec = float(settings.get("interParentSeconds") or 2)
    outside_window = False

    # Inter-parent throttle — applies to ALL scans (including silent + suspended)
    # to prevent terminal log spam and iPad collisions when two parents tap
    # back-to-back. Tunable via tenants/{tid}/settings/pickup.interParentSeconds
    # (default 2s; set 0 to disable).
    if _inter_parent_throttle_active(tid, terminal_id, inter_parent_sec, now_dt):
        print(f"  ⏸  Inter-parent throttle — {device_name} "
              f"(another scan <{inter_parent_sec}s ago) — skipped")
        return None

    # Cooldown applies to silent + ok decisions to prevent log spam.
    # Suspended bypasses cooldown so security flags always reach staff.
    if not is_suspended:
        if _cooldown_active(db, tid, employee_no, terminal_id, cooldown_sec, now_dt):
            print(f"  ⏸  Pickup cooldown — {chap_summary['name']} @ {device_name} "
                  f"(within {cooldown_sec}s) — skipped")
            return None
        # Window enforcement only for normal "ok" — silent decisions stay silent.
        if decision == "ok" and enforce_window:
            window = _resolve_window(db, tid, terminal_id)
            if window and not _in_window(now_dt, window, warmup_min):
                outside_window = True
                decision = "outside_window"
                card_state = "info"

    # Mark this scan as recent so the next one within cooldown is skipped.
    _mark_recent(tid, employee_no, terminal_id, now_dt)

    # now a pure display concern (handled per-profile in the TV feed) so we
    # do NOT colour by time-of-day here — a single source of truth lives in
    # web-dataset-collector/lib/kiosk-profiles.gateStatus().
    if outside_window:
        card_state = "info"   # preserve info-card state for outside-window
    elif decision in ("unknown_chaperone", "wrong_terminal"):
        card_state = "silent" # written for audit but hidden from iPad
    elif decision == "suspended":
        card_state = "red"
    elif decision == "reenroll_overdue":
        card_state = "yellow"
    elif not students:
        card_state = "yellow"  # chaperone has no authorized students yet
    else:
        card_state = "green"

    event_id = uuid.uuid4().hex
    now_iso = now_dt.isoformat()

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
        "scannedAt": scanned_at,
        "recordedAt": now_dt,
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
