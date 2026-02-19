#!/usr/bin/env python3
"""
attendance_listener.py — Hikvision Live Event Stream Listener
==============================================================
Connects to the DS-K1T341AMF alertStream (multipart MIME over HTTP),
parses face-recognition events in real time, and stores attendance
records to:
  1. Local JSON files  → data/attendance/YYYY-MM-DD.json
  2. Firebase Firestore → collection "attendance/{date}/records"

The device's AcsEventSearch endpoint is NOT supported on this firmware,
so we use the persistent /ISAPI/Event/notification/alertStream instead.

Event stream format:
  --MIME_boundary
  Content-Type: application/json; charset="UTF-8"
  { "eventType": "AccessControllerEvent", "AccessControllerEvent": { ... } }
  --MIME_boundary
  Content-Type: image/jpeg         ← face capture photo (optional)

Face verification events:
  majorEventType = 5, subEventType = 76

NOTE: Device event timestamps are broken (show 1970-01-01) even though
the system clock is correct. We use server-side timestamps instead.

Usage:
  python attendance_listener.py                  # Run listener
  python attendance_listener.py --no-firebase    # Local JSON only
"""

import os
import sys
import json
import re
import time
import hashlib
import signal
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

# ─── Configuration ───────────────────────────────────────────────────────────

HIKVISION_IP   = os.getenv("HIKVISION_IP", "10.26.30.200")
HIKVISION_USER = os.getenv("HIKVISION_USER", "admin")
HIKVISION_PASS = os.getenv("HIKVISION_PASS", "password.123")

DATA_DIR = Path(__file__).parent / "data" / "attendance"
CUTOFF_HOUR = 8
CUTOFF_MINUTE = 15  # 08:15 = late threshold
DUPLICATE_WINDOW = 28800  # 8 hours — one-time attendance per session

WIB = timezone(timedelta(hours=7))  # UTC+7

USE_FIREBASE = True

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
            firebase_admin.initialize_app(cred)

        _firestore_client = firestore.client()
        print("  ✓ Firebase Firestore connected")
        return _firestore_client
    except Exception as e:
        print(f"  ⚠ Firebase unavailable: {e}")
        return None


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


def upload_to_binus_api(name: str, emp_no: str, class_name: str, timestamp: str, status: str, is_late: bool):
    """Upload attendance record to Binus School API.

    Looks up the student's IdStudent and IdBinusian from the metadata mapping
    (populated during enrollment), then calls the B.2 attendance insert API.
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

    if not id_student:
        print(f"  ⚠ Binus API: No IdStudent found for {name} (emp#{emp_no}). Skipping API upload.")
        print(f"    → Re-enroll this student to populate metadata, or add manually to data/student_metadata.json")
        return False

    try:
        payload = {
            "IdStudent": id_student,
            "IdBinusian": id_binusian,
            "ImageDesc": "-",
            "UserAction": os.getenv("USER_ACTION", "TEACHER7"),
        }
        print(f"  ☁️  Binus API: Sending attendance for {name} (ID:{id_student})...")
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


# ─── Attendance storage ──────────────────────────────────────────────────────

def determine_status(dt: datetime) -> str:
    cutoff = dt.replace(hour=CUTOFF_HOUR, minute=CUTOFF_MINUTE, second=0, microsecond=0)
    return "Late" if dt > cutoff else "Present"


def save_local(name: str, emp_no: str, timestamp: str, status: str, date_str: str):
    """Append attendance record to local JSON file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    filepath = DATA_DIR / f"{date_str}.json"

    records = {}
    if filepath.exists():
        try:
            records = json.loads(filepath.read_text())
        except json.JSONDecodeError:
            records = {}

    records[name] = {
        "employeeNo": emp_no,
        "timestamp": timestamp,
        "status": status,
        "late": status == "Late",
    }

    filepath.write_text(json.dumps(records, indent=2, ensure_ascii=False))


def save_firebase(name: str, emp_no: str, timestamp: str, status: str, date_str: str):
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

        doc_ref.set({
            "name": name,
            "employeeNo": emp_no,
            "timestamp": timestamp,
            "status": status,
            "late": status == "Late",
            "homeroom": homeroom,
            "grade": grade,
            "source": "hikvision_terminal",
            "updatedAt": datetime.now(WIB).isoformat(),
        })
        # Also update the day summary
        day_ref = db.collection("attendance").document(date_str)
        day_ref.set({"lastUpdated": datetime.now(WIB).isoformat()}, merge=True)
        return "created"
    except Exception as e:
        print(f"  ⚠ Firestore write failed: {e}")
        return None


# ─── Event stream parser ─────────────────────────────────────────────────────

def parse_event_stream(resp, name_map: dict, logged_today: set, today: str):
    """
    Parse multipart MIME event stream from the device.
    Yields (name, emp_no, timestamp_str, status, date_str) for each new attendance.
    """
    buffer = b""
    boundary = b"--MIME_boundary"

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

            # We only care about JSON parts (not image parts)
            try:
                text = part.decode("utf-8", errors="ignore")
            except Exception:
                continue

            if "application/json" not in text:
                continue

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

            # majorEventType=5: access control events
            # subEventType=75: face comparison success
            # subEventType=76: face verification success  
            # subEventType=104: face recognition (alternate)
            if major != 5 or sub not in (75, 76, 104):
                continue

            emp_no = ace.get("employeeNoString", "") or str(ace.get("employeeNo", ""))
            name = ace.get("name", "") or name_map.get(emp_no, "")

            if not emp_no or not name:
                continue

            # Check date rollover
            now = datetime.now(WIB)
            current_date = now.strftime("%Y-%m-%d")
            if current_date != today:
                today = current_date
                logged_today.clear()
                print(f"\n📅 New day: {today}")
                logged_today.update(load_logged_today(today))

            # Skip if already logged within window
            if emp_no in logged_today:
                elapsed = now.timestamp() - logged_today[emp_no]
                if elapsed < DUPLICATE_WINDOW:
                    hrs = (DUPLICATE_WINDOW - elapsed) / 3600
                    print(f"  ℹ️  {name} — Already Logged ✓ (next allowed in {hrs:.1f}h)")
                    continue

            logged_today[emp_no] = now.timestamp()
            timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
            status = determine_status(now)

            yield name, emp_no, timestamp, status, current_date
            today = current_date  # update for next iteration


# ─── Main listener ───────────────────────────────────────────────────────────

def run_listener(use_firebase=True):
    print("=" * 60)
    print("  Hikvision Attendance Listener")
    print("=" * 60)
    print(f"  Device:    {HIKVISION_IP}")
    print(f"  Late after: {CUTOFF_HOUR:02d}:{CUTOFF_MINUTE:02d}")
    print(f"  Data dir:  {DATA_DIR}")
    print(f"  Firebase:  {'enabled' if use_firebase else 'disabled'}")
    print(f"  Binus API: {'enabled' if API_INTEGRATE_ENABLED else 'disabled'}")
    print()

    # Build name lookup
    print("📋 Fetching enrolled users from device...")
    name_map = build_name_map()
    print(f"   {len(name_map)} user(s) enrolled")
    for eno, name in name_map.items():
        print(f"   • {name} (ID: {eno})")
    print()

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

    today = datetime.now(WIB).strftime("%Y-%m-%d")
    logged_today = load_logged_today(today)
    count = 0
    retry_delay = 5

    while True:
        try:
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

            for name, emp_no, timestamp, status, date_str in parse_event_stream(
                resp, name_map, logged_today, today
            ):
                today = date_str
                count += 1
                is_late = status == "Late"
                icon = "⏰" if is_late else "✅"
                print(f"{icon} [{timestamp}] {name} — {status}")

                save_local(name, emp_no, timestamp, status, date_str)
                fb_result = None
                if use_firebase:
                    fb_result = save_firebase(name, emp_no, timestamp, status, date_str)
                # Only push to BINUS API if this is a NEW record (not already captured by mobile)
                if fb_result != "already_exists":
                    upload_to_binus_api(name, emp_no, "", timestamp, status, is_late)

        except requests.exceptions.ConnectionError:
            print(f"\n⚠ Connection lost — reconnecting in {retry_delay}s...")
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 60)
        except KeyboardInterrupt:
            print(f"\n\n⏹  Stopped. {count} attendance record(s) logged today.")
            sys.exit(0)
        except Exception as e:
            print(f"\n⚠ Error: {e} — retrying in {retry_delay}s...")
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hikvision attendance event listener")
    parser.add_argument("--no-firebase", action="store_true", help="Disable Firebase sync")
    args = parser.parse_args()

    run_listener(use_firebase=not args.no_firebase)
