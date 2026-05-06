#!/usr/bin/env python3
"""
upload_and_enroll.py — Upload student photos to Firebase Storage, then enroll on Hikvision
==========================================================================================
Two-phase approach:
  Phase 1: Download photos from BINUS blob storage → upload to Firebase Storage
  Phase 2: Serve photos from local disk → enroll faces on Hikvision device

Firebase Storage layout:  face_dataset/{homeroom}/{name}/{emp_no}.jpg
This matches the convention used by the web dashboard and existing enrollments.

Usage:
    python3 upload_and_enroll.py                    # Upload + enroll all new students
    python3 upload_and_enroll.py --upload-only       # Phase 1 only (Firebase upload)
    python3 upload_and_enroll.py --enroll-only       # Phase 2 only (device enrollment)
    python3 upload_and_enroll.py --dry-run           # Preview only
    python3 upload_and_enroll.py --grade 4           # Only Grade 4
"""

import os
import sys
import json
import time
import hashlib
import socket
import threading
import argparse
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from datetime import datetime, timezone, timedelta
from urllib.parse import quote

import requests
from requests.auth import HTTPDigestAuth
from dotenv import load_dotenv

import tenancy

load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))
import student_metadata

WIB = timezone(timedelta(hours=7))
DATA_DIR = Path(__file__).parent / "data"
REPORT_FILE = DATA_DIR / "binus_students_all.json"
FACES_DIR = DATA_DIR / "faces_batch"
FACES_RESIZED_DIR = DATA_DIR / "faces_batch_resized"
FACE_SERVER_PORT = 8899

FIREBASE_CREDENTIALS = os.getenv(
    "FIREBASE_CREDENTIALS",
    str(Path(__file__).parent / "facial-attendance-binus-firebase-adminsdk.json"),
)
FIREBASE_BUCKET = os.getenv(
    "FIREBASE_STORAGE_BUCKET",
    "facial-attendance-binus.firebasestorage.app",
)


# ─── Helpers ────────────────────────────────────────────────────────────────

def emp_no(name):
    """Deterministic 8-char employee number from name (matches hikvision_attendance.py)."""
    return hashlib.md5(name.encode()).hexdigest()[:8].upper()


def load_new_students(grade_filter=None):
    """Load students that need enrollment (enrollable, not yet in metadata)."""
    if not REPORT_FILE.exists():
        print("❌ No student report. Run fetch_all_students.py first.")
        sys.exit(1)

    report = json.loads(REPORT_FILE.read_text())
    all_students = report["students"]
    enrollable = [s for s in all_students if s.get("enrollable")]
    if grade_filter:
        enrollable = [s for s in enrollable if str(s.get("grade")) == str(grade_filter)]

    meta = student_metadata._load_local()
    existing_enos = set(meta.keys())

    new = [s for s in enrollable if emp_no(s["name"]) not in existing_enos]
    return new, meta


# ─── Phase 1: Download from BINUS + Upload to Firebase ─────────────────────

def phase1_upload_to_firebase(students, dry_run=False):
    """Download photos from BINUS blob storage, upload to Firebase Storage."""
    import firebase_admin
    from firebase_admin import credentials, storage as fb_storage

    FACES_DIR.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  PHASE 1: Upload {len(students)} photos to Firebase Storage")
    print(f"{'='*60}\n")

    if dry_run:
        for s in students:
            eno = emp_no(s["name"])
            fb_path = f"face_dataset/{s['homeroom']}/{s['name']}/{eno}.jpg"
            print(f"  Would upload: {fb_path}")
        print(f"\nTotal: {len(students)} photos")
        return

    # Init Firebase
    cred_path = os.path.abspath(FIREBASE_CREDENTIALS)
    if not os.path.isfile(cred_path):
        print(f"❌ Firebase credentials not found: {cred_path}")
        return

    try:
        app = firebase_admin.get_app()
    except ValueError:
        cred = credentials.Certificate(cred_path)
        app = firebase_admin.initialize_app(cred, {"storageBucket": FIREBASE_BUCKET})

    bucket = fb_storage.bucket(app=app)

    uploaded = 0
    skipped = 0
    failed = 0

    for i, s in enumerate(students, 1):
        name = s["name"]
        homeroom = s.get("homeroom", "unknown")
        eno = emp_no(name)
        photo_url = (s.get("filePath") or "").strip()
        local_path = FACES_DIR / f"{eno}.jpg"
        rel = f"{homeroom}/{name}/{eno}.jpg"
        fb_path = f"face_dataset/{rel}"
        tenant_path = f"{tenancy.storage_face_dataset_prefix()}/{rel}"

        print(f"[{i}/{len(students)}] {name} → {fb_path}")

        # Step 1: Download from BINUS if not cached locally
        if local_path.exists() and local_path.stat().st_size > 1000:
            print(f"  📁 Using cached local photo ({local_path.stat().st_size // 1024}KB)")
        elif photo_url:
            try:
                r = requests.get(photo_url, timeout=15)
                if r.status_code == 200 and len(r.content) > 1000:
                    local_path.write_bytes(r.content)
                    print(f"  📥 Downloaded from BINUS ({len(r.content) // 1024}KB)")
                else:
                    print(f"  ❌ BINUS download failed (HTTP {r.status_code})")
                    failed += 1
                    continue
            except Exception as e:
                print(f"  ❌ Download error: {e}")
                failed += 1
                continue
        else:
            print(f"  ⏭️  No photo URL")
            skipped += 1
            continue

        # Step 2: Check if already in Firebase
        blob = bucket.blob(fb_path)
        if blob.exists():
            print(f"  ☁️  Already in Firebase Storage")
            uploaded += 1
            # Mirror to tenant path if missing
            try:
                tblob = bucket.blob(tenant_path)
                if not tblob.exists():
                    bucket.copy_blob(blob, bucket, tenant_path)
                    print(f"  ☁️  Tenant copy: {tenant_path}")
            except Exception as te:
                print(f"  ⚠️  Tenant copy failed (non-fatal): {te}")
            continue

        # Step 3: Upload to Firebase Storage
        try:
            if tenancy.legacy_paths_enabled():
                blob.upload_from_filename(str(local_path), content_type="image/jpeg")
                blob.make_public()
                print(f"  ☁️  Uploaded to Firebase Storage ✅")
            try:
                tblob = bucket.blob(tenant_path)
                tblob.upload_from_filename(str(local_path), content_type="image/jpeg")
                tblob.make_public()
                print(f"  ☁️  Tenant copy: {tenant_path}")
            except Exception as te:
                print(f"  ⚠️  Tenant storage dual-write failed (non-fatal): {te}")
            uploaded += 1
        except Exception as e:
            print(f"  ❌ Firebase upload failed: {e}")
            failed += 1

    print(f"\n  Phase 1 Summary: ✅ {uploaded} uploaded, ⏭️ {skipped} skipped, ❌ {failed} failed")
    return uploaded


# ─── Phase 2: Enroll on Hikvision Device ───────────────────────────────────

class DeviceClient:
    """Hikvision ISAPI wrapper with auto-reconnect on auth failures."""

    def __init__(self, ip, user="admin", password=None):
        if not password:
            raise ValueError("Device password is required")
        self.ip = ip
        self.user = user
        self.password = password
        self.base = f"http://{ip}"
        self._new_session()

    def _new_session(self):
        self.session = requests.Session()
        self.session.auth = HTTPDigestAuth(self.user, self.password)

    def api(self, method, path, **kwargs):
        kwargs.setdefault("timeout", 15)
        r = getattr(self.session, method)(f"{self.base}{path}", **kwargs)
        # Detect stale auth / userCheck lockout
        if r.status_code == 401 or (r.status_code == 200 and b"userCheck" in r.content[:200]):
            self._new_session()
            time.sleep(1)
            r = getattr(self.session, method)(f"{self.base}{path}", **kwargs)
        return r

    def api_json(self, method, path, **kwargs):
        r = self.api(method, path, **kwargs)
        try:
            return r.status_code, r.json()
        except Exception:
            return r.status_code, {"raw": r.text[:500]}

    def get_enrolled_users(self):
        users = {}
        pos = 0
        while True:
            body = {
                "UserInfoSearchCond": {
                    "searchID": "batch",
                    "searchResultPosition": pos,
                    "maxResults": 30,
                }
            }
            status, data = self.api_json(
                "post", "/ISAPI/AccessControl/UserInfo/Search?format=json", json=body
            )
            if status != 200:
                break
            info = data.get("UserInfoSearch", {})
            user_list = info.get("UserInfo", [])
            if isinstance(user_list, dict):
                user_list = [user_list]
            for u in user_list:
                eno = u.get("employeeNo", "")
                if eno:
                    users[eno] = {
                        "name": u.get("name", ""),
                        "numOfFace": u.get("numOfFace", 0),
                    }
            total = int(info.get("totalMatches", 0))
            pos += len(user_list)
            if pos >= total or not user_list:
                break
        return users

    def create_user(self, employee_no, name):
        user_data = {
            "UserInfo": {
                "employeeNo": employee_no,
                "name": name,
                "userType": "normal",
                "gender": "unknown",
                "Valid": {
                    "enable": True,
                    "beginTime": "2024-01-01T00:00:00",
                    "endTime": "2037-12-31T23:59:59",
                    "timeType": "local",
                },
                "doorRight": "1",
                "RightPlan": [{"doorNo": 1, "planTemplateNo": "1"}],
            }
        }
        status, data = self.api_json(
            "post", "/ISAPI/AccessControl/UserInfo/Record?format=json", json=user_data
        )
        ok = status == 200 and data.get("statusCode") == 1
        return ok, data

    def upload_face(self, employee_no, name, face_url):
        payload = {
            "faceLibType": "blackFD",
            "FDID": "1",
            "FPID": employee_no,
            "name": name,
            "faceURL": face_url,
        }
        status, data = self.api_json(
            "put", "/ISAPI/Intelligent/FDLib/FDSetUp?format=json", json=payload
        )
        ok = status == 200 and data.get("statusCode") == 1
        return ok, data


def _start_face_server(serve_dir, port=FACE_SERVER_PORT):
    """Start background HTTP server to serve face images to the device."""

    class SilentHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=serve_dir, **kwargs)

        def log_message(self, fmt, *args):
            pass

    server = HTTPServer(("0.0.0.0", port), SilentHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect((os.getenv("HIKVISION_IP", "10.26.80.65"), 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "127.0.0.1"

    return server, f"http://{local_ip}:{port}"


def phase2_enroll_on_device(students, device_ip, device_pass, dry_run=False):
    """Enroll students on Hikvision device using locally cached photos."""
    # Use resized photos if available, fall back to originals
    serve_dir = FACES_RESIZED_DIR if FACES_RESIZED_DIR.exists() else FACES_DIR
    photo_dir = serve_dir
    print(f"\n{'='*60}")
    print(f"  PHASE 2: Enroll {len(students)} faces on device {device_ip}")
    print(f"{'='*60}\n")

    device = DeviceClient(device_ip, "admin", device_pass)

    # Verify device reachable
    try:
        r = device.api("get", "/ISAPI/System/deviceInfo")
        if r.status_code != 200:
            print(f"❌ Device returned HTTP {r.status_code}")
            return
        print(f"✅ Connected to device {device_ip}")
    except Exception as e:
        print(f"❌ Cannot reach device: {e}")
        return

    existing = device.get_enrolled_users()
    print(f"📊 Currently on device: {len(existing)} users")

    # Filter to students with local photos and not already enrolled with face
    to_enroll = []
    skip = 0
    for s in students:
        eno = emp_no(s["name"])
        local_photo = photo_dir / f"{eno}.jpg"
        if not local_photo.exists() or local_photo.stat().st_size < 1000:
            continue
        if eno in existing and existing[eno].get("numOfFace", 0) > 0:
            skip += 1
            continue
        to_enroll.append(s)

    print(f"⏭️  Already on device with face: {skip}")
    print(f"🆕 To enroll: {len(to_enroll)}")

    if not to_enroll:
        print("✅ All done!")
        return

    if dry_run:
        for s in to_enroll:
            eno = emp_no(s["name"])
            print(f"  Would enroll: {eno} {s['name']} (Grade {s['grade']} {s['homeroom']})")
        return

    # Start face server
    server, base_url = _start_face_server(str(photo_dir))
    print(f"🌐 Face server: {base_url}\n")

    enrolled = 0
    failed = 0

    for i, s in enumerate(to_enroll, 1):
        name = s["name"]
        eno = emp_no(name)
        sid = s["idStudent"]
        bid = s.get("idBinusian", "")
        homeroom = s.get("homeroom", "")
        grade = s.get("grade", "")

        print(f"[{i}/{len(to_enroll)}] {name} (Grade {grade} {homeroom})")

        # Create user if needed
        if eno not in existing:
            ok, resp = device.create_user(eno, name)
            if not ok:
                if "exist" in str(resp).lower():
                    print(f"  ℹ️  User exists")
                else:
                    err = resp.get("subStatusCode", "unknown")
                    print(f"  ❌ Create user failed: {err}")
                    failed += 1
                    continue
            else:
                print(f"  ✅ User created")

        # Upload face (with retries)
        face_url = f"{base_url}/{eno}.jpg"
        ok_face = False
        for attempt in range(3):
            ok, resp = device.upload_face(eno, name, face_url)
            if ok:
                print(f"  ✅ Face enrolled!")
                enrolled += 1
                ok_face = True

                student_metadata.save_student(
                    employee_no=eno,
                    name=name,
                    id_student=sid,
                    id_binusian=bid,
                    homeroom=homeroom,
                    grade=grade,
                )
                existing[eno] = {"name": name, "numOfFace": 1}
                break
            else:
                err = resp.get("subStatusCode", resp.get("errorMsg", str(resp)[:120]))
                if attempt < 2:
                    print(f"  ⚠️  Retry {attempt+1}/2 — {err}")
                    time.sleep(3)
                else:
                    print(f"  ❌ Face failed: {err}")
                    failed += 1

        # Pace the device
        if i % 10 == 0:
            print(f"\n  ⏸️  Progress: {enrolled} enrolled, {failed} failed ({i}/{len(to_enroll)})\n")
            time.sleep(3)
        else:
            time.sleep(1.5)

    server.shutdown()

    # Final report
    print(f"\n{'='*60}")
    print(f"  ENROLLMENT COMPLETE")
    print(f"{'='*60}")
    print(f"  ✅ Enrolled: {enrolled}")
    print(f"  ❌ Failed:   {failed}")

    final = device.get_enrolled_users()
    faces = sum(1 for u in final.values() if u.get("numOfFace", 0) > 0)
    print(f"  📊 Device: {len(final)} users, {faces} with faces\n")


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Upload photos to Firebase + enroll on device")
    parser.add_argument("--device", default=os.getenv("HIKVISION_IP", "10.26.80.65"))
    parser.add_argument("--password", default=os.getenv("HIKVISION_PASS"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--grade", default=None)
    parser.add_argument("--upload-only", action="store_true", help="Phase 1 only")
    parser.add_argument("--enroll-only", action="store_true", help="Phase 2 only")
    args = parser.parse_args()

    if not args.password:
        parser.error("Device password required: use --password or set HIKVISION_PASS env var")

    students, meta = load_new_students(args.grade)
    print(f"📋 {len(students)} new students to process")

    if not students:
        print("✅ All enrollable students already in metadata!")
        return

    if not args.enroll_only:
        phase1_upload_to_firebase(students, dry_run=args.dry_run)

    if not args.upload_only:
        phase2_enroll_on_device(students, args.device, args.password, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
