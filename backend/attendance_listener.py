#!/usr/bin/env python3
"""
attendance_listener.py — Hikvision Live Event Stream Listener
==============================================================
Connects to Hikvision face terminals via alertStream (multipart MIME
over HTTP), parses face-recognition events in real time, and stores
attendance records to:
  1. Local JSON files  → data/attendance/YYYY-MM-DD.json
  2. Firebase Firestore → collection "attendance/{date}/records"

Catch-up sync:
  On connect/reconnect, if the device supports AcsEvent search
  (DS-K1T342MFX does, DS-K1T341AMF does NOT), the listener queries
  missed events since the last recorded attendance and backfills them
  before resuming the live stream. This means no attendance is lost
  even if the server was down for hours.

Event stream format:
  --MIME_boundary
  Content-Type: application/json; charset="UTF-8"
  { "eventType": "AccessControllerEvent", "AccessControllerEvent": { ... } }
  --MIME_boundary
  Content-Type: image/jpeg         ← face capture photo (optional)

Face verification events:
  majorEventType = 5, subEventType = 76

Usage:
  python attendance_listener.py                  # Run listener
  python attendance_listener.py --no-firebase    # Local JSON only
"""

import os
import sys
import json
import re
import time
import fcntl
import tempfile
import hashlib
import signal
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

# Multi-tenancy helpers (Phase A1) — dual-write to tenant-scoped paths
import tenancy
import pickup_event_writer
import unknown_faces

load_dotenv()

# ─── Configuration ───────────────────────────────────────────────────────────

HIKVISION_IP   = os.getenv("HIKVISION_IP", "10.26.30.200")
HIKVISION_USER = os.getenv("HIKVISION_USER", "admin")
HIKVISION_PASS = os.getenv("HIKVISION_PASS")
_HIKVISION_DEVICE_NAME_ENV = os.getenv("HIKVISION_DEVICE_NAME")
_HIKVISION_TERMINAL_ID_ENV = os.getenv("HIKVISION_TERMINAL_ID")
HIKVISION_DEVICE_NAME = _HIKVISION_DEVICE_NAME_ENV or HIKVISION_IP
# Stable terminalId derived from the device name (matches
# web-dataset-collector/pages/api/pickup/admin/terminals.js stableTerminalId).
HIKVISION_TERMINAL_ID = _HIKVISION_TERMINAL_ID_ENV or (
    hashlib.sha1(HIKVISION_DEVICE_NAME.encode("utf-8")).hexdigest()[:12]
)
if not HIKVISION_PASS:
    raise SystemExit("FATAL: HIKVISION_PASS environment variable is required")

DATA_DIR = Path(__file__).parent / "data" / "attendance"
CUTOFF_HOUR = 7
CUTOFF_MINUTE = 30  # 07:30 = late threshold
DUPLICATE_WINDOW = 28800  # 8 hours — one-time attendance per session

WIB = timezone(timedelta(hours=7))  # UTC+7

USE_FIREBASE = True
PICKUP_ONLY = os.getenv("PICKUP_ONLY", "true").strip().lower() in {"1", "true", "yes", "on"}

# ─── Manual HTTP Digest Auth ─────────────────────────────────────────────────

_digest_challenge = None  # Cached challenge for device
_nc_counter = 0  # Nonce counter

def parse_digest_header(header):
    """Parse WWW-Authenticate: Digest header."""
    obj = {}
    parts = header.replace('Digest ', '', 1)
    pattern = r'(\w+)=(?:"([^"]*)"|(\w+))'
    for match in re.finditer(pattern, parts):
        key = match.group(1)
        value = match.group(2) if match.group(2) else match.group(3)
        obj[key] = value
    return obj

def get_digest_challenge():
    """Get digest challenge from device (cached)."""
    global _digest_challenge
    if _digest_challenge:
        return _digest_challenge
    
    # Probe with lightweight GET
    resp = requests.get(
        f"http://{HIKVISION_IP}/ISAPI/System/deviceInfo",
        timeout=10
    )
    if resp.status_code == 401:
        auth_header = resp.headers.get('WWW-Authenticate', '')
        if auth_header.lower().startswith('digest'):
            _digest_challenge = parse_digest_header(auth_header)
            return _digest_challenge
    raise Exception("Device did not return digest challenge")

def build_digest_auth(method, uri, challenge):
    """Build Authorization: Digest header."""
    global _nc_counter
    realm = challenge.get('realm', '')
    nonce = challenge.get('nonce', '')
    qop = challenge.get('qop', 'auth')
    opaque = challenge.get('opaque', '')
    
    _nc_counter += 1
    nc = f'{_nc_counter:08x}'
    cnonce = hashlib.md5(str(time.time()).encode()).hexdigest()[:16]
    
    ha1 = hashlib.md5(f"{HIKVISION_USER}:{realm}:{HIKVISION_PASS}".encode()).hexdigest()
    ha2 = hashlib.md5(f"{method}:{uri}".encode()).hexdigest()
    response = hashlib.md5(f"{ha1}:{nonce}:{nc}:{cnonce}:{qop}:{ha2}".encode()).hexdigest()
    
    return (
        f'Digest username="{HIKVISION_USER}", '
        f'realm="{realm}", '
        f'nonce="{nonce}", '
        f'uri="{uri}", '
        f'qop={qop}, '
        f'nc={nc}, '
        f'cnonce="{cnonce}", '
        f'response="{response}", '
        f'opaque="{opaque}"'
    )

def invalidate_challenge():
    """Invalidate cached challenge (on stale nonce)."""
    global _digest_challenge, _nc_counter
    _digest_challenge = None
    _nc_counter = 0

# ─── Firebase (optional) ─────────────────────────────────────────────────────

_firestore_client = None


def _is_app_label(name: str) -> bool:
    n = (name or "").strip()
    return n.startswith("Terminal ") or n.startswith("Grade ") or n.startswith("EY")


def _resolve_terminal_identity(db):
    """Align this listener's terminal name/id to the app registry by IP.

    This prevents legacy names in local config (e.g. old lobby/tower labels)
    from diverging from the labels used by web/iPad apps.
    """
    global HIKVISION_DEVICE_NAME, HIKVISION_TERMINAL_ID
    tid = tenancy.get_tenant_id()
    try:
        col = db.collection(tenancy.terminals_path(tid))
        try:
            from google.cloud.firestore_v1.base_query import FieldFilter
            q = col.where(filter=FieldFilter("ip", "==", HIKVISION_IP))
        except Exception:
            q = col.where("ip", "==", HIKVISION_IP)
        docs = list(q.limit(10).stream())
    except Exception as e:
        print(f"  ⚠ Terminal identity lookup failed: {e}")
        return

    if not docs:
        return

    chosen = None
    if _HIKVISION_TERMINAL_ID_ENV:
        chosen = next((d for d in docs if d.id == _HIKVISION_TERMINAL_ID_ENV), None)
    if chosen is None and _HIKVISION_DEVICE_NAME_ENV:
        chosen = next(
            (d for d in docs if (d.to_dict() or {}).get("name") == _HIKVISION_DEVICE_NAME_ENV),
            None,
        )
    if chosen is None and len(docs) == 1:
        chosen = docs[0]
    if chosen is None:
        def _score(doc):
            data = doc.to_dict() or {}
            name = str(data.get("name") or "")
            score = 0
            if data.get("enabled", True):
                score += 4
            if _is_app_label(name):
                score += 6
            if "Lobby" in name or "Tower" in name:
                score -= 1
            return score

        docs.sort(key=_score, reverse=True)
        chosen = docs[0]

    data = chosen.to_dict() or {}
    resolved_name = (data.get("name") or "").strip()
    resolved_id = chosen.id

    if resolved_name and not _HIKVISION_DEVICE_NAME_ENV:
        HIKVISION_DEVICE_NAME = resolved_name
    if resolved_id and not _HIKVISION_TERMINAL_ID_ENV:
        HIKVISION_TERMINAL_ID = resolved_id

    print(f"  ✓ Terminal identity: {HIKVISION_DEVICE_NAME} ({HIKVISION_TERMINAL_ID})")

def get_firestore():
    global _firestore_client
    if _firestore_client is not None:
        return _firestore_client

    try:
        import firebase_admin
        from firebase_admin import credentials, firestore

        cred_path = os.getenv(
            "FIREBASE_CREDENTIALS",
            str(Path(__file__).parent / "facial-attendance-binus-firebase-adminsdk.json"),
        )
        if not firebase_admin._apps:
            cred = credentials.Certificate(cred_path)
            bucket_name = os.getenv("FIREBASE_STORAGE_BUCKET", "facial-attendance-binus.firebasestorage.app")
            firebase_admin.initialize_app(cred, {"storageBucket": bucket_name})

        _firestore_client = firestore.client()
        print("  ✓ Firebase Firestore connected")
        # Align local listener identity with app-facing terminal registry.
        _resolve_terminal_identity(_firestore_client)
        # Optional bootstrap: upsert local devices.json into Firestore.
        # Disabled by default because Firestore is now the source of truth.
        if str(os.getenv("SYNC_TERMINAL_REGISTRY_ON_START", "0")).lower() in {"1", "true", "yes", "on"}:
            try:
                from sync_terminal_registry import sync_terminal_registry
                n = sync_terminal_registry()
                if n:
                    print(f"  ✓ Terminal registry synced ({n} entries)")
            except Exception as _sync_e:
                print(f"  ⚠ Terminal registry sync skipped: {_sync_e}")
        else:
            print("  ℹ️  Terminal registry sync from devices.json is disabled (Firestore is source of truth)")
        # Pre-warm the pickup hot-path caches + start real-time invalidation
        # so the first scan of the day is already warm (~50ms not ~250ms).
        try:
            counts = pickup_event_writer.prewarm_caches()
            if counts.get("chaperones") or counts.get("students"):
                print(f"  ✓ Pickup cache prewarmed ({counts['chaperones']} chaperones, {counts['students']} students)")
            if pickup_event_writer.start_realtime_invalidation():
                print("  ✓ Pickup cache realtime watch active")
        except Exception as _pw_e:
            print(f"  ⚠ Pickup cache prewarm skipped: {_pw_e}")
        return _firestore_client
    except Exception as e:
        print(f"  ⚠ Firebase unavailable: {e}")
        return None


# ─── Pickup gate enforcer (Firestore schedule + admin override → relay) ──────

_gate_enforcer = None

def _start_gate_enforcer():
    """Spawn a daemon thread that pushes door state to this Hikvision terminal
    based on the schedule + manual override stored in Firestore.
    """
    global _gate_enforcer
    if _gate_enforcer is not None:
        return
    db = get_firestore()
    if db is None:
        return
    import gate_controller
    tid = tenancy.get_tenant_id()
    term_path = f"{tenancy.terminals_path(tid)}/{HIKVISION_TERMINAL_ID}"

    def _read_doc():
        try:
            snap = db.document(term_path).get()
            return snap.to_dict() if snap.exists else None
        except Exception as e:
            print(f"  ⚠ Gate enforcer firestore read failed: {e}")
            return None

    def _send(cmd):
        # Refresh challenge on 401 so subsequent calls succeed.
        # Suppress per-call error logs — GateEnforcer throttles repeats itself.
        ok = gate_controller.hik_remote_control_door(
            HIKVISION_IP,
            lambda method, uri: build_digest_auth(method, uri, get_digest_challenge()),
            cmd,
            log=lambda *a, **k: None,
        )
        if not ok:
            invalidate_challenge()
        return ok

    _gate_enforcer = gate_controller.GateEnforcer(
        terminal_id=HIKVISION_TERMINAL_ID,
        hik_ip=HIKVISION_IP,
        send_cmd=_send,
        get_terminal_doc=_read_doc,
    )
    _gate_enforcer.start()


# ─── Binus School API (optional) ─────────────────────────────────────────────

try:
    import api_integrate
    API_INTEGRATE_ENABLED = True
except Exception:
    API_INTEGRATE_ENABLED = False

# Student metadata mapping (employeeNo → BINUS IDs)
try:
    import student_metadata
    METADATA_ENABLED = True
except Exception:
    METADATA_ENABLED = False


def resolve_employee_no(emp_no: str) -> str:
    """Resolve HEX card employeeNo to the real student employeeNo.

    If the metadata entry for emp_no has a 'linkedTo' field, return the
    linked (real) employeeNo instead.  This collapses HEX card events
    to the same identity as face-match events.
    """
    if not METADATA_ENABLED:
        return emp_no
    meta = student_metadata.get_student(emp_no)
    if meta:
        linked = meta.get("linkedTo", "")
        if linked:
            return linked
    return emp_no


def upload_to_binus_api(name: str, emp_no: str, class_name: str, timestamp: str, status: str, is_late: bool):
    """Upload attendance record to Binus School API.

    Looks up the student's IdStudent and IdBinusian from the metadata mapping
    (populated during enrollment), then calls the B.2 attendance insert API.
    Follows linkedTo chain for HEX card entries.
    """
    if not API_INTEGRATE_ENABLED:
        print(f"  ⚠ Binus API module not available")
        return False

    # Look up student metadata to get BINUS IDs
    id_student = ""
    id_binusian = ""

    if METADATA_ENABLED:
        meta = student_metadata.get_student(emp_no)
        if meta:
            # Follow linkedTo chain for HEX card entries
            linked = meta.get("linkedTo", "")
            if linked:
                linked_meta = student_metadata.get_student(linked)
                if linked_meta:
                    id_student = linked_meta.get("idStudent", "") or meta.get("idStudent", "")
                    id_binusian = linked_meta.get("idBinusian", "") or meta.get("idBinusian", "")
                    print(f"  📎 Metadata found (via linkedTo {linked}): IdStudent={id_student}, IdBinusian={id_binusian}")
                else:
                    id_student = meta.get("idStudent", "")
                    id_binusian = meta.get("idBinusian", "")
                    print(f"  📎 Metadata found (linkedTo {linked} missing): IdStudent={id_student}, IdBinusian={id_binusian}")
            else:
                id_student = meta.get("idStudent", "")
                id_binusian = meta.get("idBinusian", "")
                print(f"  📎 Metadata found: IdStudent={id_student}, IdBinusian={id_binusian}")
        else:
            # Try lookup by name as fallback
            meta = student_metadata.find_by_name(name)
            if meta:
                id_student = meta.get("idStudent", "")
                id_binusian = meta.get("idBinusian", "")
                print(f"  📎 Metadata found (by name): IdStudent={id_student}, IdBinusian={id_binusian}")

    if not id_student and not id_binusian:
        print(f"  ⚠ Binus API: No IdStudent or IdBinusian found for {name} (emp#{emp_no}). Skipping API upload.")
        print(f"    → Re-enroll this student to populate metadata, or add manually to data/student_metadata.json")
        return False

    try:
        payload = {
            "IdStudent": id_student,
            "IdBinusian": id_binusian,
            "ImageDesc": "-",
            "UserAction": os.getenv("USER_ACTION", "TEACHER7"),
        }
        primary_id = id_student or id_binusian
        id_label = "ID" if id_student else "BinusianID"
        print(f"  ☁️  Binus API: Sending attendance for {name} ({id_label}:{primary_id})...")
        success = api_integrate.insert_student_attendance(payload)
        if success:
            print(f"  ☁️  Binus API: ✓ Attendance recorded for {name}")
            return True
        else:
            print(f"  ⚠ Binus API: upload returned failure for {name}")
            return False
    except Exception as e:
        print(f"  ⚠ Binus API error: {e}")
        return False


# ─── Persistent dedup (survives restarts) ────────────────────────────────────

def load_logged_today(date_str: str) -> dict:
    """Load today's attendance from local JSON to restore dedup state.
    Returns dict of {employeeNo: timestamp_epoch}."""
    filepath = DATA_DIR / f"{date_str}.json"
    result = {}
    if not filepath.exists():
        return result
    try:
        records = json.loads(filepath.read_text())
        now_ts = datetime.now(WIB).timestamp()
        for name, rec in records.items():
            emp_no = rec.get("employeeNo", "")
            ts_str = rec.get("timestamp", "")
            if not emp_no:
                continue
            try:
                rec_ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S").replace(
                    tzinfo=WIB
                ).timestamp()
            except Exception:
                rec_ts = now_ts
            if now_ts - rec_ts < DUPLICATE_WINDOW:
                result[emp_no] = rec_ts
        if result:
            print(f"  🔄 Restored {len(result)} attendance records from {date_str}.json")
    except Exception as e:
        print(f"  ⚠ Could not restore today's attendance: {e}")
    return result


# ─── Attendance blocklist (suppress known false-positive events) ─────────────

_BLOCKLIST_FILE = DATA_DIR.parent / "attendance_blocklist.json"
_blocklist_cache: list[dict] | None = None


def _load_blocklist() -> list[dict]:
    """Load false-positive blocklist (cached). Each entry: employeeNo, timestamp, window_seconds."""
    global _blocklist_cache
    if _blocklist_cache is not None:
        return _blocklist_cache
    try:
        if _BLOCKLIST_FILE.exists():
            _blocklist_cache = json.loads(_BLOCKLIST_FILE.read_text()) or []
        else:
            _blocklist_cache = []
    except Exception as e:
        print(f"  ⚠ Failed to read attendance blocklist: {e}")
        _blocklist_cache = []
    return _blocklist_cache


def is_event_blocklisted(emp_no: str, timestamp_str: str) -> tuple[bool, str]:
    """Return (True, reason) if (emp_no, timestamp) is a known false positive.
    Entry forms:
      - {employeeNo, timestamp, window_seconds}  → matches within ±window of that ts
      - {employeeNo, date}                        → blocks ALL events for emp_no on that YYYY-MM-DD
    """
    try:
        event_dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return False, ""
    event_date = event_dt.strftime("%Y-%m-%d")
    for entry in _load_blocklist():
        if str(entry.get("employeeNo", "")) != str(emp_no):
            continue
        # Full-day block
        if entry.get("date") and entry["date"] == event_date:
            return True, entry.get("reason", "blocklisted (full day)")
        # Timestamp-window block
        if entry.get("timestamp"):
            try:
                blk_dt = datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S")
            except Exception:
                continue
            window = int(entry.get("window_seconds", 5))
            if abs((event_dt - blk_dt).total_seconds()) <= window:
                return True, entry.get("reason", "blocklisted")
    return False, ""


# ─── Catch-up sync (pull missed events from device) ──────────────────────────

_event_search_supported = None  # None = unknown, True/False after first check

def check_event_search_support():
    """Probe whether this device supports AcsEvent search.
    DS-K1T342MFX supports it, DS-K1T341AMF does NOT."""
    global _event_search_supported
    if _event_search_supported is not None:
        return _event_search_supported

    try:
        now = datetime.now(WIB)
        body = {
            "AcsEventCond": {
                "searchID": "probe",
                "searchResultPosition": 0,
                "maxResults": 1,
                "major": 5,
                "minor": 0,
                "startTime": now.strftime("%Y-%m-%dT00:00:00+07:00"),
                "endTime": now.strftime("%Y-%m-%dT23:59:59+07:00"),
            }
        }
        challenge = get_digest_challenge()
        uri = "/ISAPI/AccessControl/AcsEvent?format=json"
        auth_header = build_digest_auth("POST", uri, challenge)
        r = requests.post(
            f"http://{HIKVISION_IP}{uri}",
            json=body,
            headers={"Authorization": auth_header},
            timeout=10,
        )
        if r.status_code == 401:
            invalidate_challenge()
            challenge = get_digest_challenge()
            auth_header = build_digest_auth("POST", uri, challenge)
            r = requests.post(
                f"http://{HIKVISION_IP}{uri}",
                json=body,
                headers={"Authorization": auth_header},
                timeout=10,
            )
        if r.status_code == 200:
            data = r.json()
            if "AcsEvent" in data:
                _event_search_supported = True
                return True
        _event_search_supported = False
        return False
    except Exception:
        _event_search_supported = False
        return False


def get_last_recorded_time(date_str: str) -> str | None:
    """Get the latest recorded attendance timestamp for a given date.
    Returns ISO 8601 string or None if no records exist."""
    filepath = DATA_DIR / f"{date_str}.json"
    if not filepath.exists():
        return None
    try:
        records = json.loads(filepath.read_text())
        latest = None
        for name, rec in records.items():
            ts = rec.get("timestamp", "")
            if ts and (latest is None or ts > latest):
                latest = ts
        if latest:
            # Convert "YYYY-MM-DD HH:MM:SS" to ISO with timezone
            dt = datetime.strptime(latest, "%Y-%m-%d %H:%M:%S").replace(tzinfo=WIB)
            return dt.strftime("%Y-%m-%dT%H:%M:%S+07:00")
    except Exception:
        pass
    return None


def catchup_sync(name_map: dict, logged_today: dict, today: str, use_firebase: bool, push_binus: bool = True, full_day: bool = False):
    """Pull missed face-verification events from the device since last recorded
    attendance and backfill them into local JSON + Firestore + BINUS API.

    Only runs on devices that support AcsEvent search (DS-K1T342MFX).
    Skips silently on unsupported devices (DS-K1T341AMF)."""

    if not check_event_search_support():
        print("  ℹ️  Event search not supported on this device — catch-up sync skipped")
        return 0

    now = datetime.now(WIB)
    today_str = now.strftime("%Y-%m-%d")
    is_today = (today == today_str)

    # Determine start time: last recorded event, or start of `today` (the
    # date being recovered).  When `full_day=True` (multi-device catch-up of
    # a past date) we ignore the cursor — otherwise the second device run
    # would resume from the first device's last record and miss its morning.
    last_ts = None if full_day else get_last_recorded_time(today)
    if last_ts:
        # Start 1 second after last recorded event to avoid re-processing it
        try:
            last_dt = datetime.fromisoformat(last_ts)
            start_dt = last_dt + timedelta(seconds=1)
            start_time = start_dt.strftime("%Y-%m-%dT%H:%M:%S+07:00")
        except Exception:
            start_time = f"{today}T00:00:00+07:00"
        print(f"  📡 Catch-up sync: searching events since {last_ts}")
    else:
        start_time = f"{today}T00:00:00+07:00"
        print(f"  📡 Catch-up sync: searching events for full day {today}")

    # End-time: end-of-day for past dates, "now" for today.  Without this
    # clamp a single-date catch-up would silently span into the present and
    # drag in unrelated events from later dates.
    if is_today:
        end_time = now.strftime("%Y-%m-%dT%H:%M:%S+07:00")
    else:
        end_time = f"{today}T23:59:59+07:00"

    # Paginate through all missed events
    pos = 0
    batch_size = 30
    synced = 0
    seen_emp = set()

    while True:
        body = {
            "AcsEventCond": {
                "searchID": "catchup",
                "searchResultPosition": pos,
                "maxResults": batch_size,
                "major": 5,
                "minor": 0,
                "startTime": start_time,
                "endTime": end_time,
            }
        }
        try:
            challenge = get_digest_challenge()
            uri = "/ISAPI/AccessControl/AcsEvent?format=json"
            auth_header = build_digest_auth("POST", uri, challenge)
            r = requests.post(
                f"http://{HIKVISION_IP}{uri}",
                json=body,
                headers={"Authorization": auth_header},
                timeout=15,
            )
            if r.status_code == 401:
                invalidate_challenge()
                challenge = get_digest_challenge()
                auth_header = build_digest_auth("POST", uri, challenge)
                r = requests.post(
                    f"http://{HIKVISION_IP}{uri}",
                    json=body,
                    headers={"Authorization": auth_header},
                    timeout=15,
                )
            if r.status_code != 200:
                print(f"  ⚠ Catch-up: HTTP {r.status_code}")
                break

            data = r.json()
            acs = data.get("AcsEvent", {})
            total = int(acs.get("totalMatches", 0))
            events = acs.get("InfoList", [])

            if not events:
                break

            for evt in events:
                minor = evt.get("minor", 0)
                # Only face verification successes
                if minor not in (75, 76, 104):
                    continue

                emp_no = evt.get("employeeNoString", "") or str(evt.get("employeeNo", ""))
                name = evt.get("name", "") or name_map.get(emp_no, "")

                if not emp_no or not name:
                    continue

                # Resolve HEX card → real student ID for dedup
                real_emp_no = resolve_employee_no(emp_no)

                # PickupGuard: chaperones (9XXX) are not retroactively recorded
                # via catch-up — pickup events without a live capture image are
                # not useful for the TV display, and they must never enter the
                # attendance pipeline.
                if tenancy.is_chaperone_employee_no(real_emp_no) or \
                   tenancy.is_chaperone_employee_no(emp_no):
                    continue

                # Only first occurrence per student (dedup within catch-up)
                if real_emp_no in seen_emp:
                    continue
                seen_emp.add(real_emp_no)
                seen_emp.add(emp_no)

                # Skip if already in today's dedup set (check both HEX and real)
                if real_emp_no in logged_today or emp_no in logged_today:
                    continue

                # Parse device timestamp
                event_time_str = evt.get("time", "")
                try:
                    event_dt = datetime.fromisoformat(event_time_str)
                    if event_dt.tzinfo is None:
                        event_dt = event_dt.replace(tzinfo=WIB)
                except Exception:
                    event_dt = now

                event_date = event_dt.strftime("%Y-%m-%d")
                timestamp = event_dt.strftime("%Y-%m-%d %H:%M:%S")
                status = determine_status(event_dt)

                # Skip known false-positive events (manual blocklist)
                blocked, reason = is_event_blocklisted(real_emp_no, timestamp)
                if not blocked:
                    blocked, reason = is_event_blocklisted(emp_no, timestamp)
                if blocked:
                    print(f"  🚫 [catch-up] [{timestamp}] {name} — BLOCKED ({reason})")
                    # Mark as logged so future catch-ups in the same run skip it too
                    logged_today[emp_no] = event_dt.timestamp()
                    logged_today[real_emp_no] = event_dt.timestamp()
                    continue

                # Record it (mark both HEX and real as logged)
                logged_today[emp_no] = event_dt.timestamp()
                logged_today[real_emp_no] = event_dt.timestamp()
                is_late = status == "Late"
                icon = "🔄⏰" if is_late else "🔄✅"
                if emp_no != real_emp_no:
                    print(f"  {icon} [catch-up] [{timestamp}] {name} — {status} (HEX {emp_no} → {real_emp_no})")
                else:
                    print(f"  {icon} [catch-up] [{timestamp}] {name} — {status}")

                save_local(name, real_emp_no, timestamp, status, event_date, scanned_emp_no=emp_no)
                fb_result = None
                if use_firebase:
                    fb_result = save_firebase(name, real_emp_no, timestamp, status, event_date, scanned_emp_no=emp_no)
                # Push to BINUS unless already confirmed pushed (and only when caller allowed it)
                if push_binus and fb_result != "already_exists":
                    binus_ok = upload_to_binus_api(name, real_emp_no, "", timestamp, status, is_late)
                    if binus_ok:
                        mark_binus_pushed(real_emp_no, event_date)
                elif not push_binus:
                    print(f"  ⏭️  BINUS push skipped (--no-binus) for {name}")
                synced += 1

            pos += len(events)
            status_str = acs.get("responseStatusStrg", "")
            if status_str != "MORE" or pos >= total:
                break

        except Exception as e:
            print(f"  ⚠ Catch-up error: {e}")
            break

    if synced > 0:
        print(f"  ✅ Catch-up sync complete: {synced} missed record(s) backfilled")
    else:
        print(f"  ✅ Catch-up sync: no missed events")
    return synced


# ─── Name lookup from device ─────────────────────────────────────────────────

def build_name_map():
    """Fetch enrolled users from device and build employeeNo → name map."""
    name_map = {}
    pos = 0
    batch = 30

    while True:
        body = {
            "UserInfoSearchCond": {
                "searchID": "listener",
                "searchResultPosition": pos,
                "maxResults": batch,
            }
        }
        try:
            challenge = get_digest_challenge()
            uri = "/ISAPI/AccessControl/UserInfo/Search?format=json"
            auth_header = build_digest_auth("POST", uri, challenge)
            
            r = requests.post(
                f"http://{HIKVISION_IP}{uri}",
                json=body,
                headers={"Authorization": auth_header},
                timeout=15
            )
            
            if r.status_code == 401:
                # Stale nonce, retry once
                invalidate_challenge()
                challenge = get_digest_challenge()
                auth_header = build_digest_auth("POST", uri, challenge)
                r = requests.post(
                    f"http://{HIKVISION_IP}{uri}",
                    json=body,
                    headers={"Authorization": auth_header},
                    timeout=15
                )
            
            if r.status_code != 200:
                print(f"  ⚠ User fetch failed: HTTP {r.status_code}")
                break
                
            data = r.json()
            info = data.get("UserInfoSearch", {})
            users = info.get("UserInfo", [])
            if isinstance(users, dict):
                users = [users]
            for u in users:
                eno = u.get("employeeNo", "")
                name = u.get("name", "")
                if eno and name:
                    name_map[eno] = name
            total = int(info.get("totalMatches", 0))
            pos += len(users)
            if pos >= total or not users:
                break
        except Exception as e:
            print(f"  ⚠ Failed to fetch users: {e}")
            break

    return name_map


# ─── Unknown / suspicious face capture ───────────────────────────────────────
#
# Delegates to the shared unknown_faces module so the same review queue is
# used by both attendance and PickupGuard pathways.

def _save_unknown_snapshot(
    jpeg_bytes: bytes,
    kind: str,                    # "unmatched" | "suspected_false_match"
    ace: dict,
    now: datetime,
    suspected_emp_no: str = "",
    suspected_name: str = "",
    reason: str = "",
) -> str | None:
    """Thin wrapper over unknown_faces.capture_unknown_face for the listener."""
    # Ensure Firebase app is initialized before the mirror attempt
    get_firestore()
    return unknown_faces.capture_unknown_face(
        jpeg_bytes=jpeg_bytes,
        kind=kind,
        device_ip=HIKVISION_IP,
        device_name=HIKVISION_DEVICE_NAME,
        terminal_id=HIKVISION_TERMINAL_ID,
        suspected_emp_no=suspected_emp_no,
        suspected_name=suspected_name,
        reason=reason,
        when=now,
        raw_event=ace,
    )


# ─── Attendance storage ──────────────────────────────────────────────────────

def determine_status(dt: datetime) -> str:
    cutoff = dt.replace(hour=CUTOFF_HOUR, minute=CUTOFF_MINUTE, second=0, microsecond=0)
    return "Late" if dt > cutoff else "Present"


def save_local(name: str, emp_no: str, timestamp: str, status: str, date_str: str, scanned_emp_no: str | None = None):
    """Append attendance record to local JSON file.

    Uses fcntl.flock() for exclusive locking so multiple listener processes
    can safely write to the same date file without corruption.
    Writes to a temp file first, then atomically renames.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    filepath = DATA_DIR / f"{date_str}.json"
    lockpath = DATA_DIR / f".{date_str}.lock"

    with open(lockpath, "w") as lockf:
        fcntl.flock(lockf, fcntl.LOCK_EX)
        try:
            records = {}
            if filepath.exists():
                try:
                    records = json.loads(filepath.read_text())
                except json.JSONDecodeError:
                    records = {}

            records[name] = {
                "employeeNo": emp_no,
                "scannedEmployeeNo": scanned_emp_no or emp_no,
                "timestamp": timestamp,
                "status": status,
                "late": status == "Late",
            }

            # Write to temp file then atomic rename to prevent partial writes
            fd, tmp = tempfile.mkstemp(dir=DATA_DIR, suffix=".tmp")
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(records, f, indent=2, ensure_ascii=False)
                os.replace(tmp, filepath)  # atomic on POSIX
            except BaseException:
                os.unlink(tmp)
                raise
        finally:
            fcntl.flock(lockf, fcntl.LOCK_UN)


def save_firebase(name: str, emp_no: str, timestamp: str, status: str, date_str: str, scanned_emp_no: str | None = None):
    """Write attendance record to Firestore, enriched with class/grade metadata.
    
    DEDUP: If a record already exists for this student today (e.g. from the
    mobile app), skip overwriting and return 'already_exists'.
    """
    db = get_firestore()
    if not db:
        return None

    # Look up class/grade from student metadata
    homeroom = ""
    grade = ""
    if METADATA_ENABLED:
        meta = student_metadata.get_student(emp_no)
        if meta:
            homeroom = meta.get("homeroom", "")
            grade = meta.get("grade", "")

    try:
        doc_ref = db.collection("attendance").document(date_str).collection("records").document(emp_no)

        # ── Dedup: check if record already exists from ANY source ──────
        existing = doc_ref.get()
        if existing.exists:
            existing_data = existing.to_dict()
            existing_source = existing_data.get("source", "unknown")
            print(f"  ℹ️  {name} already clocked in via {existing_source} — skipping overwrite")
            return "already_exists"

        record_payload = {
            "name": name,
            "employeeNo": emp_no,
            "scannedEmployeeNo": scanned_emp_no or emp_no,
            "timestamp": timestamp,
            "status": status,
            "late": status == "Late",
            "homeroom": homeroom,
            "grade": grade,
            "source": "hikvision_terminal",
            "deviceName": HIKVISION_DEVICE_NAME,
            "deviceIp": HIKVISION_IP,
            "binusPushed": False,
            "updatedAt": datetime.now(WIB).isoformat(),
        }
        if tenancy.legacy_paths_enabled():
            doc_ref.set(record_payload)
            db.collection("attendance").document(date_str).set(
                {"lastUpdated": datetime.now(WIB).isoformat()}, merge=True
            )
        # Tenant-scoped dual-write (Phase A6 — always on)
        try:
            tenant_record_ref = db.document(tenancy.attendance_record_path(date_str, emp_no))
            tenant_record_ref.set({**record_payload, "tenantId": tenancy.get_tenant_id()})
            db.document(tenancy.attendance_day_doc(date_str)).set(
                {"lastUpdated": datetime.now(WIB).isoformat(), "tenantId": tenancy.get_tenant_id()},
                merge=True,
            )
        except Exception as te:
            print(f"  ⚠ Tenant dual-write failed (non-fatal): {te}")
        return "created"
    except Exception as e:
        print(f"  ⚠ Firestore write failed: {e}")
        return None


def mark_binus_pushed(emp_no: str, date_str: str):
    """Mark a Firestore attendance record as successfully pushed to BINUS API."""
    db = get_firestore()
    if not db:
        return
    try:
        push_payload = {"binusPushed": True, "binusPushedAt": datetime.now(WIB).isoformat()}
        if tenancy.legacy_paths_enabled():
            db.collection("attendance").document(date_str).collection("records").document(emp_no).update(push_payload)
        # Tenant-scoped dual-write
        try:
            db.document(tenancy.attendance_record_path(date_str, emp_no)).update(push_payload)
        except Exception as te:
            # Tenant doc may not exist yet if backfill hasn't run — ignore quietly
            if "NOT_FOUND" not in str(te) and "No document to update" not in str(te):
                print(f"  ⚠ Tenant binusPushed update failed (non-fatal): {te}")
    except Exception as e:
        print(f"  ⚠ Failed to mark binusPushed for {emp_no}: {e}")


# ─── Event stream parser ─────────────────────────────────────────────────────

def parse_event_stream(resp, name_map: dict, logged_today: set, today: str, pickup_only: bool = PICKUP_ONLY):
    """
    Parse multipart MIME event stream from the device.
    Yields tuples for each new attendance.

    Yield shape:
        (name, real_emp_no, scanned_emp_no, timestamp, status, date_str,
         jpeg_bytes_or_none)

    The JPEG, when present, is the face-capture image the device sends
    in the part *immediately following* the JSON event. PickupGuard
    (chaperone branch) needs it; normal attendance ignores it.
    """
    buffer = b""
    boundary = b"--MIME_boundary"
    pending_event = None  # JSON event waiting to pair with its JPEG

    for chunk in resp.iter_content(chunk_size=4096):
        if not chunk:
            continue
        buffer += chunk

        # Split on boundary
        while boundary in buffer:
            idx = buffer.index(boundary)
            part = buffer[:idx]
            buffer = buffer[idx + len(boundary):]

            # Skip empty or tiny parts
            if len(part) < 20:
                continue

            # Detect content-type early (work on raw bytes — JPEG isn't UTF-8)
            ct_match = re.search(rb"Content-Type:\s*([\w/+\-.]+)", part, re.IGNORECASE)
            content_type = (ct_match.group(1).decode("ascii", "ignore").lower()
                            if ct_match else "")

            if content_type.startswith("image/"):
                # Strip MIME headers (terminated by blank line) to leave raw image bytes
                hdr_end = part.find(b"\r\n\r\n")
                if hdr_end == -1:
                    hdr_end = part.find(b"\n\n")
                    sep_len = 2
                else:
                    sep_len = 4
                jpeg_bytes = part[hdr_end + sep_len:].rstrip(b"\r\n-") if hdr_end > 0 else b""
                if pending_event is not None and jpeg_bytes:
                    pending_event["_jpeg"] = jpeg_bytes
                    # Flush immediately — we now have both the JSON and the JPEG.
                    # No need to wait for the next event or the 1.5 s timeout.
                    yield from _emit_event(pending_event, name_map, logged_today, today, pickup_only=pickup_only)
                    today = pending_event.get("_today", today)
                    pending_event = None
                continue

            if "json" not in content_type:
                continue

            try:
                text = part.decode("utf-8", errors="ignore")
            except Exception:
                continue

            # If we had a pending event without a JPEG, flush it now
            if pending_event is not None:
                yield from _emit_event(pending_event, name_map, logged_today, today, pickup_only=pickup_only)
                today = pending_event.get("_today", today)
                pending_event = None

            # Extract JSON object from the text
            json_start = text.find("{")
            json_end = text.rfind("}") + 1
            if json_start < 0 or json_end <= json_start:
                continue

            try:
                event = json.loads(text[json_start:json_end])
            except json.JSONDecodeError:
                continue

            # Only process AccessControllerEvent face verifications
            if event.get("eventType") != "AccessControllerEvent":
                continue

            ace = event.get("AccessControllerEvent", {})
            major = ace.get("majorEventType", 0)
            sub = ace.get("subEventType", 0)

            # Keep all face events (major==5). Known success sub-events
            # (75/76/104) flow through normal attendance. Other face sub-
            # events (e.g. faceAuthFailed) are kept so we can capture the
            # snapshot for unenrolled students — see _emit_event.
            if major != 5:
                continue

            pending_event = {
                "ace": ace,
                "_today": today,
                "_jpeg": None,
                "_is_known_sub": sub in (75, 76, 104),
            }

        # End of available parts in buffer; if pending event is older than
        # ~1.5s we can flush it without its image (device may have skipped
        # the JPEG for this scan).
        if pending_event is not None:
            t0 = pending_event.get("_seen_at", time.time())
            if "_seen_at" not in pending_event:
                pending_event["_seen_at"] = time.time()
            elif time.time() - t0 > 1.5:
                yield from _emit_event(pending_event, name_map, logged_today, today, pickup_only=pickup_only)
                today = pending_event.get("_today", today)
                pending_event = None


def _emit_event(pending: dict, name_map: dict, logged_today: dict, today: str, pickup_only: bool = PICKUP_ONLY):
    """Convert a buffered ACE event (+ optional JPEG) into a yieldable tuple."""
    ace = pending["ace"]
    emp_no = ace.get("employeeNoString", "") or str(ace.get("employeeNo", ""))
    name = ace.get("name", "") or name_map.get(emp_no, "")
    jpeg = pending.get("_jpeg")
    is_known_sub = pending.get("_is_known_sub", True)
    now_capture = datetime.now(WIB)

    # Face seen but device couldn't match it to any enrolled user.
    # Previously we captured a JPG + JSON for each of these so an admin
    # could enrol the unknown person; in practice the device fires bursts
    # of identity-less events (one every ~2s) when an unenrolled person
    # stands at the gate, flooding data/unknown_faces/ with hundreds of
    # near-duplicates that never lead to action. We just drop them now.
    # (The blocklist branch below still saves snapshots — those are
    # actionable misidentifications, not anonymous strangers.)
    if not emp_no or not name or not is_known_sub:
        return

    now = now_capture
    current_date = now.strftime("%Y-%m-%d")
    if current_date != today:
        today = current_date
        logged_today.clear()
        print(f"\n📅 New day: {today}")
        logged_today.update(load_logged_today(today))

    real_emp_no = resolve_employee_no(emp_no)

    # PickupGuard: chaperone events bypass attendance dedup entirely so a
    # parent doing two pickups in a day still produces two cards.
    is_chaperone = tenancy.is_chaperone_employee_no(real_emp_no) or \
                   tenancy.is_chaperone_employee_no(emp_no)

    # Pickup-first runtime: ignore all non-chaperone scans.
    if pickup_only and not is_chaperone:
        return

    if not is_chaperone:
        already_logged = False
        for check_no in (emp_no, real_emp_no):
            if check_no in logged_today:
                elapsed = now.timestamp() - logged_today[check_no]
                if elapsed < DUPLICATE_WINDOW:
                    hrs = (DUPLICATE_WINDOW - elapsed) / 3600
                    print(f"  ℹ️  {name} — Already Logged ✓ (next allowed in {hrs:.1f}h)")
                    already_logged = True
                    break
        if already_logged:
            return
        logged_today[emp_no] = now.timestamp()
        logged_today[real_emp_no] = now.timestamp()

    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    status = determine_status(now)
    pending["_today"] = current_date

    # Suppress known false positives (manual blocklist — e.g. mis-recognitions)
    blocked, reason = is_event_blocklisted(real_emp_no, timestamp)
    if not blocked:
        blocked, reason = is_event_blocklisted(emp_no, timestamp)
    if blocked:
        print(f"  🚫 [{timestamp}] {name} — BLOCKED ({reason})")
        # Save the snapshot — a blocklisted match is almost always the
        # device misidentifying an unenrolled student. The picture is
        # exactly what we need to enroll the real person.
        _save_unknown_snapshot(
            jpeg or b"",
            kind="suspected_false_match",
            ace=ace,
            now=now,
            suspected_emp_no=real_emp_no,
            suspected_name=name,
            reason=reason,
        )
        return

    yield (name, real_emp_no, emp_no, timestamp, status, current_date, jpeg)


# ─── Main listener ───────────────────────────────────────────────────────────

def run_listener(use_firebase=True):
    print("=" * 60)
    print("  Hikvision Attendance Listener")
    print("=" * 60)
    print(f"  Device:    {HIKVISION_IP}")
    print(f"  Late after: {CUTOFF_HOUR:02d}:{CUTOFF_MINUTE:02d}")
    print(f"  Data dir:  {DATA_DIR}")
    print(f"  Firebase:  {'enabled' if use_firebase else 'disabled'}")
    print(f"  Mode:      {'pickup-only' if PICKUP_ONLY else 'attendance+pickup'}")
    print(f"  Binus API: {'disabled (pickup-only mode)' if PICKUP_ONLY else ('enabled' if API_INTEGRATE_ENABLED else 'disabled')}")
    print()

    # Build name lookup
    print("📋 Fetching enrolled users from device...")
    name_map = build_name_map()
    print(f"   {len(name_map)} user(s) enrolled")
    for eno, name in name_map.items():
        print(f"   • {name} (ID: {eno})")
    print()

    if not PICKUP_ONLY:
        # Load student metadata mapping (employeeNo → BINUS IDs)
        student_meta = {}
        if METADATA_ENABLED:
            print("📝 Loading student metadata mapping...")
            student_meta = student_metadata.load_from_firebase()
            if student_meta:
                mapped = sum(1 for v in student_meta.values() if v.get("idStudent"))
                print(f"   {len(student_meta)} student(s) in metadata, {mapped} with BINUS IDs")
                unmapped = [v.get('name', '?') for v in student_meta.values() if not v.get('idStudent')]
                if unmapped:
                    print(f"   ⚠ Missing BINUS IDs: {', '.join(unmapped[:5])}{'...' if len(unmapped) > 5 else ''}")
            else:
                print("   ⚠ No student metadata found. BINUS API uploads will be skipped.")
                print("     → Run enrollment (hikvision_attendance.py enroll-live/enroll-class) to populate.")
        else:
            print("⚠ Student metadata module not available — BINUS API uploads disabled.")
        print()

    if use_firebase:
        get_firestore()
    print()

    # ── Start per-terminal gate enforcer (Firestore → Hikvision relay) ────
    if use_firebase:
        try:
            _start_gate_enforcer()
        except Exception as _ge:
            print(f"⚠ Gate enforcer not started: {_ge}")
    print()

    today = datetime.now(WIB).strftime("%Y-%m-%d")
    logged_today = load_logged_today(today)
    count = 0
    retry_delay = 5
    first_connect = True

    while True:
        try:
            # ── Catch-up sync on (re)connect (attendance mode only) ──
            if not PICKUP_ONLY:
                if first_connect:
                    print("📡 Running catch-up sync for missed events...")
                else:
                    print("📡 Reconnected — running catch-up sync...")

                # Refresh today in case of date rollover during downtime
                today = datetime.now(WIB).strftime("%Y-%m-%d")
                caught = catchup_sync(name_map, logged_today, today, use_firebase)
                count += caught
            else:
                if first_connect:
                    print("📡 Pickup-only mode: catch-up attendance sync is disabled")
                else:
                    print("📡 Reconnected (pickup-only mode)")
            first_connect = False

            print(f"🔗 Connecting to event stream...")
            
            # Get digest challenge and build auth header
            challenge = get_digest_challenge()
            uri = "/ISAPI/Event/notification/alertStream"
            auth_header = build_digest_auth("GET", uri, challenge)
            
            resp = requests.get(
                f"http://{HIKVISION_IP}{uri}",
                headers={"Authorization": auth_header},
                stream=True,
                timeout=(10, None),  # 10s connect, no read timeout
            )

            if resp.status_code == 401:
                # Stale nonce, retry with new challenge
                print(f"   ✗ HTTP 401 (stale nonce) — refreshing...")
                invalidate_challenge()
                challenge = get_digest_challenge()
                auth_header = build_digest_auth("GET", uri, challenge)
                resp = requests.get(
                    f"http://{HIKVISION_IP}{uri}",
                    headers={"Authorization": auth_header},
                    stream=True,
                    timeout=(10, None),
                )
            
            if resp.status_code != 200:
                print(f"   ✗ HTTP {resp.status_code} — retrying in {retry_delay}s")
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 60)
                continue

            print(f"   ✓ Connected! Listening for face recognition events...\n")
            retry_delay = 5  # reset on success

            for name, emp_no, scanned_emp_no, timestamp, status, date_str, jpeg_bytes in parse_event_stream(
                resp, name_map, logged_today, today, pickup_only=PICKUP_ONLY
            ):
                today = date_str
                count += 1

                # ── PickupGuard branch ────────────────────────────────
                # Chaperones (employeeNo prefix '9') never touch attendance.
                if tenancy.is_chaperone_employee_no(emp_no) or \
                   tenancy.is_chaperone_employee_no(scanned_emp_no):
                    if use_firebase:
                        try:
                            pickup_event_writer.record_pickup_event(
                                tenant_id=tenancy.get_tenant_id(),
                                employee_no=emp_no,
                                chaperone_name_hint=name,
                                scanned_at=datetime.now(WIB),
                                device_name=HIKVISION_DEVICE_NAME,
                                gate=HIKVISION_DEVICE_NAME,
                                terminal_id=HIKVISION_TERMINAL_ID,
                                jpeg_bytes=jpeg_bytes,
                            )
                        except Exception as e:
                            print(f"  ⚠ Pickup event handler failed: {e}")
                    else:
                        print(f"🟦 PICKUP (firebase disabled) [{timestamp}] {name}")
                    continue

                if PICKUP_ONLY:
                    continue

                # ── Normal attendance flow ───────────────────────────
                is_late = status == "Late"
                icon = "⏰" if is_late else "✅"
                print(f"{icon} [{timestamp}] {name} — {status}")

                save_local(name, emp_no, timestamp, status, date_str, scanned_emp_no=scanned_emp_no)
                fb_result = None
                if use_firebase:
                    fb_result = save_firebase(name, emp_no, timestamp, status, date_str, scanned_emp_no=scanned_emp_no)
                # Push to BINUS unless already confirmed pushed
                if fb_result != "already_exists":
                    binus_ok = upload_to_binus_api(name, emp_no, "", timestamp, status, is_late)
                    if binus_ok:
                        mark_binus_pushed(emp_no, date_str)

        except requests.exceptions.ConnectionError:
            print(f"\n⚠ Connection lost — reconnecting in {retry_delay}s...")
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 60)
        except KeyboardInterrupt:
            if PICKUP_ONLY:
                print(f"\n\n⏹  Stopped. {count} pickup event(s) processed.")
            else:
                print(f"\n\n⏹  Stopped. {count} attendance record(s) logged today.")
            sys.exit(0)
        except Exception as e:
            print(f"\n⚠ Error: {e} — retrying in {retry_delay}s...")
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 60)


def run_catchup(dates: list[str], use_firebase: bool = True, push_binus: bool = True, full_day: bool = False):
    """One-shot catch-up sync for specified dates, then exit."""
    print("=" * 60)
    print("  Hikvision Catch-up Sync")
    print("=" * 60)
    print(f"  Device:    {HIKVISION_IP}")
    print(f"  Dates:     {', '.join(dates)}")
    print(f"  Firebase:  {'enabled' if use_firebase else 'disabled'}")
    print(f"  BINUS API: {'enabled' if push_binus else 'disabled (--no-binus)'}")
    print()

    name_map = build_name_map()
    print(f"  {len(name_map)} user(s) enrolled")

    if not check_event_search_support():
        print("\n❌ This device does not support event search. Catch-up sync unavailable.")
        sys.exit(1)

    if METADATA_ENABLED:
        student_metadata.load_from_firebase()
    if use_firebase:
        get_firestore()
    print()

    total = 0
    for date_str in dates:
        print(f"── {date_str} ──")
        logged = load_logged_today(date_str)
        caught = catchup_sync(name_map, logged, date_str, use_firebase, push_binus=push_binus, full_day=full_day)
        total += caught
        print()

    print(f"═══ Done. {total} record(s) backfilled across {len(dates)} day(s). ═══")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hikvision attendance event listener")
    parser.add_argument("--no-firebase", action="store_true", help="Disable Firebase sync")
    parser.add_argument("--no-binus", action="store_true", help="Disable BINUS API push (catch-up only)")
    parser.add_argument("--catchup", action="store_true", help="One-shot catch-up sync (pull missed events), then exit")
    parser.add_argument("--full-day", action="store_true", help="Catch-up: scan from 00:00 of each date, ignoring any cursor (use when running multiple devices against the same date)")
    parser.add_argument("--date", type=str, help="Specific date to catch up (YYYY-MM-DD). Can be repeated.", action="append")
    parser.add_argument("--days", type=int, help="Catch up the last N days (default: 1 = today only)")
    args = parser.parse_args()

    if args.catchup:
        if args.date:
            dates = args.date
        elif args.days:
            today = datetime.now(WIB)
            dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(args.days)]
        else:
            dates = [datetime.now(WIB).strftime("%Y-%m-%d")]
        run_catchup(dates, use_firebase=not args.no_firebase, push_binus=not args.no_binus, full_day=args.full_day)
    else:
        run_listener(use_firebase=not args.no_firebase)
