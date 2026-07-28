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
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from firebase_admin import firestore
import firebase_admin

import tenancy

WIB = timezone(timedelta(hours=7))

# Local UNIX socket the SSE fan-out server (P3) listens on. Best-effort —
# absent socket means we just skip the publish, doc still lives in Firestore
# and the iPad teacher PWA falls back to the Firestore onSnapshot path in
# pages/api/pickup/tablet/stream.js (SSE) for sub-second delivery.
PICKUP_EVENT_SOCKET = os.environ.get(
    "PICKUP_EVENT_SOCKET", "/tmp/pickupguard.sock"
)

# Cached pickup_settings per tenant (small TTL — admin changes are rare).
_settings_cache: dict[str, tuple[float, dict]] = {}
_SETTINGS_TTL = 60.0  # seconds

# Hot-path caches — keep the scan→Firestore latency in the low-100ms range.
# All have short TTLs so admin changes (suspending a chaperone, moving a
# student, changing terminal scopes) propagate within a minute.
_chaperone_cache: dict[str, tuple[float, Optional[dict]]] = {}
_student_cache: dict[str, tuple[float, dict]] = {}
_terminal_doc_cache: dict[str, tuple[float, dict]] = {}
_HOT_CACHE_TTL = 300.0  # seconds (5 min) — real-time watcher invalidates on edits

# Active Firestore real-time watches (one per tenant) so admin edits to
# chaperones/students propagate instantly. Without this, an admin
# suspending a parent would take up to _HOT_CACHE_TTL seconds to apply.
_watch_handles: dict[str, list] = {}

# Background executor for non-critical post-write side effects (JPEG upload,
# security incident log, chaperone lastSeenAt bump, http notify). The hot
# path returns to the caller as soon as the pickup_events doc is written.
import concurrent.futures as _futures
_post_write_pool = _futures.ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="pickup-postwrite"
)

# Distance above which the software FR audit marks an event as suspicious.
# face_recognition distances: ~0.0 (identical) → ~0.6 (threshold) → 1.0+ (different).
# 0.55 is intentionally tighter than the library default (0.6) so borderline
# matches surface for human review without blocking legitimate scans.
_SW_FR_FLAG_THRESHOLD = float(os.environ.get("SW_FR_FLAG_THRESHOLD", "0.55"))


def _now() -> datetime:
    return datetime.now(WIB)


def _get_terminal_doc_cached(db, tid: str, terminal_id: Optional[str]) -> dict:
    """Single cached fetch of tenants/{t}/terminals/{terminal_id}. Both
    `_terminal_grade_scopes` and `_resolve_window` need this — hit Firestore
    once per minute, not twice per scan."""
    if not terminal_id:
        return {}
    import time as _time
    key = f"{tid}|{terminal_id}"
    cached = _terminal_doc_cache.get(key)
    if cached and _time.time() - cached[0] < _HOT_CACHE_TTL:
        return cached[1]
    try:
        snap = db.document(f"{tenancy.terminals_path(tid)}/{terminal_id}").get()
        data = snap.to_dict() if snap.exists else {}
    except Exception:
        data = {}
    _terminal_doc_cache[key] = (_time.time(), data or {})
    return data or {}


def _get_db():
    if not firebase_admin._apps:
        return None
    return firestore.client()


_terminal_group_cache: dict[str, tuple[float, Optional[str]]] = {}
# Keep pairing resolution fresh: a just-paired/re-paired iPad should stop
# dropping cards after the next scan, not after a minute-old cache expires.
_TERMINAL_CACHE_TTL = 5.0


def _resolve_release_group_id(db, tid: str, terminal_id: Optional[str]) -> Optional[str]:
    """Lookup the release group paired to a terminal, with a 60s cache.

    The admin pairing model may store the binding either on the terminal doc
    (`terminals/{terminalId}.releaseGroupId`) or on release group docs
    (`release_groups/{groupId}.terminalIds`). Check both so iPad group feeds do
    not drop live pickup events after their Firestore refresh.
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
    if not rg:
        try:
            try:
                from google.cloud.firestore_v1.base_query import FieldFilter
                docs = (db.collection(tenancy.release_groups_path(tid))
                          .where(filter=FieldFilter("terminalIds", "array_contains", terminal_id))
                          .limit(1)
                          .stream())
            except Exception:
                docs = (db.collection(tenancy.release_groups_path(tid))
                          .where("terminalIds", "array_contains", terminal_id)
                          .limit(1)
                          .stream())
            for doc in docs:
                rg = doc.id
                break
        except Exception:
            rg = None
    if not rg:
        try:
            for doc in db.collection(tenancy.release_groups_path(tid)).stream():
                data = doc.to_dict() or {}
                terminal_ids = data.get("terminalIds") or data.get("terminals") or []
                if isinstance(terminal_ids, str):
                    terminal_ids = [terminal_ids]
                normalized = []
                for item in terminal_ids:
                    if isinstance(item, str):
                        normalized.append(item)
                    elif isinstance(item, dict):
                        value = item.get("terminalId") or item.get("id")
                        if value:
                            normalized.append(str(value))
                if terminal_id in normalized or data.get("terminalId") == terminal_id:
                    rg = doc.id
                    break
        except Exception:
            rg = None
    # Cache positive bindings only. During live setup, admins may pair a
    # terminal while the listener is already running; caching None would keep
    # writing pickup_events without releaseGroupId until the TTL expires.
    if rg:
        _terminal_group_cache[key] = (_time.time(), rg)
    else:
        _terminal_group_cache.pop(key, None)
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
_RECENT_LOCAL_TTL = 60.0 * 10              # 10-min system cooldown (== cooldownSeconds=600)

# ─── Cross-terminal dedup ────────────────────────────────────────────────────
# The pole terminals face the same walkway: a parent walking past can be
# recognized by 3-6 devices within seconds, firing duplicate green cards on
# several iPads. Each listener is a separate OS process, so in-memory caches
# can't catch this — but all listeners run on one host, so a tiny flock'd
# JSON file is the shared source of truth. Only decision == "ok" is deduped:
# security decisions must always surface, and an earlier outside_window scan
# must never suppress the later legitimate "ok".
# Window: settings.crossTerminalCooldownSeconds (default = cooldownSeconds).
import fcntl

_XT_DEDUP_FILE = os.environ.get(
    "PICKUP_XT_DEDUP_FILE", "/tmp/pickupguard_xt_dedup.json"
)


def _xt_check_and_mark(tid: str, employee_no: str,
                       window_sec: float, now_ts: float) -> bool:
    """Atomic check-and-set across listener processes.

    Returns True if this chaperone already produced an 'ok' card at ANY
    terminal within `window_sec` (caller should skip). Otherwise records
    now_ts and returns False. We mark BEFORE the Firestore write so two
    terminals scanning in the same second can't both pass. Fail-open on
    any I/O error — a duplicate card beats a missing card.
    """
    if window_sec <= 0:
        return False
    key = f"{tid}|{employee_no}"
    try:
        fd = os.open(_XT_DEDUP_FILE, os.O_RDWR | os.O_CREAT, 0o666)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            raw = os.read(fd, 1 << 20).decode("utf-8", "replace")
            try:
                data = json.loads(raw) if raw.strip() else {}
            except Exception:
                data = {}
            last = data.get(key)
            if last and (now_ts - float(last)) < window_sec:
                return True
            # Prune stale entries, then record this scan.
            data = {k: v for k, v in data.items()
                    if now_ts - float(v) < window_sec}
            data[key] = now_ts
            payload = json.dumps(data, separators=(",", ":")).encode("utf-8")
            os.lseek(fd, 0, os.SEEK_SET)
            os.ftruncate(fd, 0)
            os.write(fd, payload)
            return False
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
    except Exception:
        return False


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
    td = _get_terminal_doc_cached(db, tid, terminal_id)
    if td:
        w_open = td.get("windowOpen")
        w_close = td.get("windowClose")
        if w_open and w_close:
            return {"open": w_open, "close": w_close}
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
    d = _get_terminal_doc_cached(db, tid, terminal_id)
    scopes = d.get("gradeScopes") if d else None
    if isinstance(scopes, list):
        return [str(x).strip() for x in scopes if str(x).strip()]
    return []


def _student_grade_tokens(student: dict) -> set[str]:
    """Extract scope-match tokens from homeroom/class fields.

    Examples:
      5C   -> {"5"}
      EY1  -> {"EY1", "EY"}
      EY2A -> {"EY2", "EY"}
    """
    hr = (student or {}).get("homeroom") or (student or {}).get("className") or ""
    if not hr:
        return set()

    raw = str(hr).strip().upper()
    if not raw:
        return set()

    # Numeric primary-school style homerooms: "5C", "4A" -> "5", "4"
    digits = ""
    for ch in raw:
        if ch.isdigit():
            digits += ch
        else:
            break
    if digits:
        return {digits}

    # Early Years style homerooms: "EY1", "EY2A" -> "EY1"/"EY2" + broad "EY"
    if raw.startswith("EY"):
        ey_token = "EY"
        for ch in raw[2:]:
            if ch.isdigit():
                ey_token += ch
            else:
                break
        if ey_token == "EY":
            return {"EY"}
        return {ey_token, "EY"}

    return set()


def _resolve_chaperone(db, tid: str, employee_no: str) -> Optional[dict]:
    """Find chaperones doc for a 9XXX employeeNo. Returns None if unknown.

    60s in-memory cache — chaperones are essentially static during a pickup
    window (admin suspends/edits propagate within a minute)."""
    import time as _time
    cache_key = f"{tid}|{employee_no}"
    cached = _chaperone_cache.get(cache_key)
    if cached and _time.time() - cached[0] < _HOT_CACHE_TTL:
        return cached[1]

    chap_id = f"chap-{employee_no}"
    snap = db.document(f"{tenancy.chaperones_path(tid)}/{chap_id}").get()
    if snap.exists:
        d = snap.to_dict()
        d["_id"] = chap_id
        _chaperone_cache[cache_key] = (_time.time(), d)
        return d
    # Fallback: query by employeeNo field (handles legacy id schemes)
    q = (db.collection(tenancy.chaperones_path(tid))
           .where("employeeNo", "==", employee_no).limit(1).stream())
    for s in q:
        d = s.to_dict()
        d["_id"] = s.id
        _chaperone_cache[cache_key] = (_time.time(), d)
        return d
    # Cache the negative result too so unknown-card spam doesn't keep hitting Firestore
    _chaperone_cache[cache_key] = (_time.time(), None)
    return None


def _resolve_students(db, tid: str, student_ids: list[str]) -> list[dict]:
    """Pull denormalized student summaries for the TV card.

    60s in-memory cache per student — student rosters are stable during a
    pickup window. Cuts ~150ms per student per scan."""
    import time as _time
    out = []
    now_ts = _time.time()
    for sid in (student_ids or [])[:10]:  # safety cap
        cache_key = f"{tid}|{sid}"
        cached = _student_cache.get(cache_key)
        if cached and now_ts - cached[0] < _HOT_CACHE_TTL:
            out.append(cached[1])
            continue

        snap = db.document(f"{tenancy.students_path(tid)}/{sid}").get()
        if not snap.exists:
            # legacy fallback during dual-read window
            snap = db.document(f"students/{sid}").get()
        if snap.exists:
            d = snap.to_dict() or {}
            entry = {
                "id": sid,
                "name": d.get("name") or d.get("fullName") or sid,
                "homeroom": d.get("homeroom") or d.get("className"),
                "photoUrl": d.get("photoUrl"),
            }
        else:
            entry = {"id": sid, "name": sid, "homeroom": None, "photoUrl": None}
        _student_cache[cache_key] = (now_ts, entry)
        out.append(entry)
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


# ─── Cache pre-warm + real-time invalidation ────────────────────────────────
def prewarm_caches(tenant_id: Optional[str] = None) -> dict:
    """Bulk-fetch all chaperones + their authorized students once at startup.

    First scan of the day = warm cache (~50ms instead of ~250ms cold).
    Returns counts for logging. Best-effort — failures degrade gracefully
    to lazy per-scan reads.
    """
    import time as _time
    db = _get_db()
    if not db:
        return {"chaperones": 0, "students": 0}
    tid = tenant_id or tenancy.get_tenant_id()
    now_ts = _time.time()
    chap_count = 0
    student_ids: set[str] = set()

    try:
        for s in db.collection(tenancy.chaperones_path(tid)).stream():
            d = s.to_dict() or {}
            d["_id"] = s.id
            employee_no = d.get("employeeNo") or s.id.replace("chap-", "")
            _chaperone_cache[f"{tid}|{employee_no}"] = (now_ts, d)
            chap_count += 1
            for sid in (d.get("authorizedStudentIds") or []):
                student_ids.add(sid)
    except Exception as e:
        print(f"  ⚠ Chaperone prewarm failed: {e}")

    student_count = 0
    try:
        # Firestore SDK supports get_all() — bulk fetch by doc refs.
        if student_ids:
            refs = [db.document(f"{tenancy.students_path(tid)}/{sid}")
                    for sid in student_ids]
            for snap in db.get_all(refs):
                if snap.exists:
                    d = snap.to_dict() or {}
                    sid = snap.id
                    entry = {
                        "id": sid,
                        "name": d.get("name") or d.get("fullName") or sid,
                        "homeroom": d.get("homeroom") or d.get("className"),
                        "photoUrl": d.get("photoUrl"),
                    }
                    _student_cache[f"{tid}|{sid}"] = (now_ts, entry)
                    student_count += 1
    except Exception as e:
        print(f"  ⚠ Student prewarm failed: {e}")

    return {"chaperones": chap_count, "students": student_count}


def start_realtime_invalidation(tenant_id: Optional[str] = None) -> bool:
    """Attach Firestore on_snapshot watchers so admin edits to chaperones/
    students immediately invalidate the in-process cache. Combined with the
    5-minute TTL, this gives near-instant propagation of suspensions, photo
    updates, etc., without re-reading on every scan.
    """
    db = _get_db()
    if not db:
        return False
    tid = tenant_id or tenancy.get_tenant_id()
    if tid in _watch_handles:
        return True  # already watching

    handles = []

    def _on_chap(snapshots, changes, _read_time):
        import time as _time
        now_ts = _time.time()
        for change in changes:
            doc = change.document
            d = doc.to_dict() or {}
            employee_no = d.get("employeeNo") or doc.id.replace("chap-", "")
            key = f"{tid}|{employee_no}"
            if change.type.name == "REMOVED":
                _chaperone_cache.pop(key, None)
            else:
                d["_id"] = doc.id
                _chaperone_cache[key] = (now_ts, d)

    def _on_student(snapshots, changes, _read_time):
        import time as _time
        now_ts = _time.time()
        for change in changes:
            doc = change.document
            key = f"{tid}|{doc.id}"
            if change.type.name == "REMOVED":
                _student_cache.pop(key, None)
            else:
                d = doc.to_dict() or {}
                _student_cache[key] = (now_ts, {
                    "id": doc.id,
                    "name": d.get("name") or d.get("fullName") or doc.id,
                    "homeroom": d.get("homeroom") or d.get("className"),
                    "photoUrl": d.get("photoUrl"),
                })

    try:
        h1 = db.collection(tenancy.chaperones_path(tid)).on_snapshot(_on_chap)
        handles.append(h1)
        h2 = db.collection(tenancy.students_path(tid)).on_snapshot(_on_student)
        handles.append(h2)
        _watch_handles[tid] = handles
        return True
    except Exception as e:
        print(f"  ⚠ Realtime cache invalidation failed: {e}")
        # Roll back any partial handles so we don't leak streams.
        for h in handles:
            try: h.unsubscribe()
            except Exception: pass
        return False


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
    import time as _time
    _t0 = _time.perf_counter()
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
            normalized_scopes = {str(s).strip().upper() for s in scopes if str(s).strip()}
            student_grades: set[str] = set()
            for s in students:
                student_grades.update(_student_grade_tokens(s))
            if student_grades and not (student_grades & normalized_scopes):
                decision = "wrong_terminal"

    # ── Cooldown + window guards ─────────────────────────────────────────
    # `silent` decisions are written for audit but suppressed from iPad feed.
    silent_decisions = {"unknown_chaperone", "wrong_terminal"}
    is_silent = decision in silent_decisions
    is_suspended = decision == "suspended"
    cooldown_sec = int(settings.get("cooldownSeconds") or 600)
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

    # Cross-terminal dedup — one green card per parent per pickup pass.
    # Poles all face the same walkway, so 3-6 terminals can recognize the
    # same parent within seconds; only the FIRST terminal fires the card.
    if decision == "ok":
        xt_window = float(
            settings.get("crossTerminalCooldownSeconds") or cooldown_sec
        )
        if _xt_check_and_mark(tid, employee_no, xt_window, now_dt.timestamp()):
            _mark_recent(tid, employee_no, terminal_id, now_dt)
            print(f"  ⏸  Cross-terminal dedup — {chap_summary['name']} @ "
                  f"{device_name} (ok card already fired at another terminal "
                  f"within {int(xt_window)}s) — skipped")
            return None

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

    # The iPad feed refreshes from Firestore after the socket push. Demo rows
    # have status='pending'; real rows need the same active status or they can
    # appear briefly from SSE and disappear on the next filtered poll.
    status = "hidden" if card_state == "silent" else "pending"

    event_id = uuid.uuid4().hex
    now_iso = now_dt.isoformat()

    # 6-digit override code for officers — only meaningful for flagged events.
    # Officer enters this in /v2/officer-override to flip the card to approved.
    override_code = None
    if decision != "ok":
        # uuid hex first 6 chars → digits only via int conversion mod 1e6
        override_code = f"{int(event_id[:8], 16) % 1_000_000:06d}"

    # Save JPEG to Storage (deferred to background — we set the deterministic
    # capturePath upfront so the doc write doesn't wait for the upload).
    # Worst case: iPad sees the card with placeholder photo for ~1s before
    # the JPEG lands at the path.
    capture_path = None
    if jpeg_bytes:
        try:
            capture_path = tenancy.storage_pickup_capture_path(event_id, tid)
        except Exception:
            capture_path = None

    release_group_id = _resolve_release_group_id(db, tid, terminal_id)
    if status == "pending" and not release_group_id:
        print(f"  ⚠ Pickup event has no releaseGroupId — iPad group feed may hide it ({device_name}, terminalId={terminal_id})")

    doc = {
        "eventId": event_id,
        "tenantId": tid,
        "employeeNo": employee_no,
        "scannedAt": scanned_at,
        "recordedAt": now_dt,
        "deviceName": device_name,
        "gate": gate or device_name,
        "terminalId": terminal_id,
        "releaseGroupId": release_group_id,
        "chaperone": chap_summary,
        "students": students,
        "decision": decision,
        "cardState": card_state,
        "status": status,
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

    # ---- Hot path ends here. Everything below is fire-and-forget. ----
    def _post_write():
        # 1. JPEG upload — path was already baked into the doc.
        if jpeg_bytes and capture_path:
            try:
                from firebase_admin import storage as _storage
                bucket_name = os.getenv(
                    "FIREBASE_STORAGE_BUCKET",
                    "facial-attendance-binus.firebasestorage.app",
                )
                bucket = _storage.bucket(bucket_name)
                blob = bucket.blob(capture_path)
                blob.upload_from_string(jpeg_bytes, content_type="image/jpeg")
            except Exception as e:
                print(f"  ⚠ Pickup capture upload failed: {e}")

        # 2. Security incident log for non-ok decisions.
        if decision in ("unknown_chaperone", "suspended", "reenroll_overdue"):
            _log_security_incident(
                db, tid,
                kind=decision,
                event_id=event_id,
                employee_no=employee_no,
                gate=gate or device_name,
                chaperone_name=chap_summary.get("name"),
            )

        # 2b. Mirror the live face into the shared unknown_faces review
        # queue so an admin can compare it against the enrolled chaperone
        # photo and re-enroll / unsuspend / clear, using the same dashboard
        # that handles unknown students. We only do this when we actually
        # captured a frame from the device — no JPEG means nothing to review.
        if jpeg_bytes and decision in ("unknown_chaperone", "suspended", "reenroll_overdue"):
            try:
                import unknown_faces as _uf
                _uf.capture_unknown_face(
                    jpeg_bytes=jpeg_bytes,
                    kind={
                        "unknown_chaperone":   "unknown_chaperone",
                        "suspended":           "suspended_chaperone",
                        "reenroll_overdue":    "reenroll_overdue_chaperone",
                    }[decision],
                    device_ip="",                # not known here; can be added later
                    device_name=device_name,
                    terminal_id=terminal_id or "",
                    suspected_emp_no=employee_no,
                    suspected_name=chap_summary.get("name") or "",
                    reason=f"pickup decision={decision}",
                    when=scanned_at,
                    tenant_id=tid,
                    extra={
                        "pickupEventId": event_id,
                        "decision": decision,
                        "gate": gate or device_name,
                        "students": [s.get("name") for s in (students or []) if s.get("name")],
                        "enrolledPhotoUrl": chap_summary.get("photoUrl"),
                    },
                )
            except Exception as e:
                print(f"  ⚠ unknownFaces mirror skipped: {e}")

        # 3. Bump chaperone lastSeenAt.
        if chaperone and chap_summary["id"]:
            try:
                db.document(f"{tenancy.chaperones_path(tid)}/{chap_summary['id']}").set(
                    {"lastSeenAt": now_iso, "lastSeenGate": gate or device_name},
                    merge=True,
                )
            except Exception:
                pass

        # 4. Push notify so iPads on long-poll wake immediately.
        try:
            _publish_socket(doc, gate=gate)
        except Exception:
            pass

        # 5. Software-side FR confidence audit.
        # Hikvision tells us the terminal identified employeeNo, but gives no
        # confidence score. We download the enrolled photo + the capture JPEG
        # and run an independent face_recognition distance check. Results are
        # written back to fr.software* — they never block the pickup flow.
        # Only runs for "ok" events with an enrolled photo and a capture frame.
        enrolled_photo_path = (
            chap_summary.get("photoUrl")
            if isinstance(chap_summary, dict) else None
        )
        if decision == "ok" and jpeg_bytes and enrolled_photo_path:
            try:
                import io as _io
                import face_recognition as _fr_lib
                from firebase_admin import storage as _fr_storage

                bucket_name = os.getenv(
                    "FIREBASE_STORAGE_BUCKET",
                    "facial-attendance-binus.firebasestorage.app",
                )
                _fr_bucket = _fr_storage.bucket(bucket_name)

                # Download enrolled photo.
                if enrolled_photo_path.startswith("http"):
                    import requests as _req
                    r = _req.get(enrolled_photo_path, timeout=10)
                    r.raise_for_status()
                    enrolled_bytes = r.content
                else:
                    enrolled_bytes = _fr_bucket.blob(enrolled_photo_path).download_as_bytes()

                enrolled_img = _fr_lib.load_image_file(_io.BytesIO(enrolled_bytes))
                capture_img  = _fr_lib.load_image_file(_io.BytesIO(jpeg_bytes))
                enrolled_encs = _fr_lib.face_encodings(enrolled_img)
                capture_encs  = _fr_lib.face_encodings(capture_img)

                if enrolled_encs and capture_encs:
                    dist = float(_fr_lib.face_distance(enrolled_encs, capture_encs[0])[0])
                    sw_conf = max(0.0, 1.0 - dist)  # 0–1 (1 = identical)
                    sw_flagged = dist > _SW_FR_FLAG_THRESHOLD
                    db.document(
                        f"{tenancy.pickup_events_path(tid)}/{event_id}"
                    ).update({
                        "fr.softwareDistance":   dist,
                        "fr.softwareConfidence": sw_conf,
                        "fr.softwareFlagged":    sw_flagged,
                        "fr.softwareCheckedAt":  _now().isoformat(),
                    })
                    if sw_flagged:
                        print(f"  ⚠ [FR-audit] FLAGGED {chap_summary.get('name')} "
                              f"dist={dist:.3f} threshold={_SW_FR_FLAG_THRESHOLD}")
                else:
                    db.document(
                        f"{tenancy.pickup_events_path(tid)}/{event_id}"
                    ).update({
                        "fr.softwareCheckedAt": _now().isoformat(),
                        "fr.softwareError": "no_face_detected",
                    })
            except Exception as _fr_err:
                print(f"  ⚠ FR software audit failed for {event_id}: {_fr_err}")

    _post_write_pool.submit(_post_write)

    pretty = ", ".join(s["name"] for s in students) or "(no students)"
    color = {"green": "🟢", "yellow": "🟡", "red": "🔴"}.get(card_state, "⚪")
    _elapsed_ms = int((_time.perf_counter() - _t0) * 1000)
    print(f"{color} PICKUP [{scanned_at.strftime('%H:%M:%S')}] "
          f"{chap_summary['name']} → {pretty}  ({decision}) "
          f"[{_elapsed_ms}ms]")

    return doc
