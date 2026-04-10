#!/usr/bin/env python3
"""
batch_enroll.py — Batch enroll students from BINUS API photos onto Hikvision devices
====================================================================================
Downloads student photos from BINUS blob storage, serves them to the device,
and enrolls each student's face on the Hikvision terminal.

Usage:
    python3 batch_enroll.py                        # Enroll all new students on default device
    python3 batch_enroll.py --device 10.26.80.65   # Specify device IP
    python3 batch_enroll.py --dry-run              # Preview only, don't enroll
    python3 batch_enroll.py --grade 4              # Only Grade 4
    python3 batch_enroll.py --resume               # Skip already-enrolled students
"""

import os
import sys
import json
import time
import hashlib
import tempfile
import socket
import threading
import argparse
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from datetime import datetime, timezone, timedelta

import requests
from requests.auth import HTTPDigestAuth
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))
import student_metadata
import api_integrate

WIB = timezone(timedelta(hours=7))
DATA_DIR = Path(__file__).parent / "data"
REPORT_FILE = DATA_DIR / "binus_students_all.json"
FACE_SERVER_PORT = 8899


# ─── Hikvision Device Helpers ────────────────────────────────────────────────

class DeviceClient:
    """Wrapper for Hikvision ISAPI calls."""

    def __init__(self, ip, user="admin", password="password.123"):
        self.ip = ip
        self.base = f"http://{ip}"
        self.session = requests.Session()
        self.session.auth = HTTPDigestAuth(user, password)

    def api(self, method, path, **kwargs):
        kwargs.setdefault("timeout", 15)
        r = getattr(self.session, method)(f"{self.base}{path}", **kwargs)
        if r.status_code == 401:
            time.sleep(0.5)
            self.session.auth = HTTPDigestAuth(
                self.session.auth.username, self.session.auth.password
            )
            r = getattr(self.session, method)(f"{self.base}{path}", **kwargs)
        return r

    def api_json(self, method, path, **kwargs):
        r = self.api(method, path, **kwargs)
        try:
            return r.status_code, r.json()
        except Exception:
            return r.status_code, {"raw": r.text[:500]}

    def get_enrolled_users(self):
        """Get dict of enrolled users: {employeeNo: {name, numOfFace, ...}}."""
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

    def create_user(self, employee_no, name, student_id=""):
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
        """Upload face to device FDLib. Device downloads from face_url."""
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

    def upload_face_data(self, employee_no, image_bytes):
        """Upload face image directly to device via multipart form.

        This is more reliable than faceURL because WE push the image
        to the device instead of asking the device to download it.
        """
        import io

        metadata = json.dumps({
            "faceLibType": "blackFD",
            "FDID": "1",
            "FPID": employee_no,
        })

        # Build multipart manually for Hikvision ISAPI compatibility
        boundary = "----HikvisionBatchEnroll"
        body = io.BytesIO()

        # JSON metadata part
        body.write(f"--{boundary}\r\n".encode())
        body.write(b'Content-Disposition: form-data; name="FaceDataRecord"; filename="FaceDataRecord.json"\r\n')
        body.write(b"Content-Type: application/json\r\n\r\n")
        body.write(metadata.encode())
        body.write(b"\r\n")

        # Image part
        body.write(f"--{boundary}\r\n".encode())
        body.write(b'Content-Disposition: form-data; name="img"; filename="face.jpg"\r\n')
        body.write(b"Content-Type: image/jpeg\r\n\r\n")
        body.write(image_bytes)
        body.write(b"\r\n")

        body.write(f"--{boundary}--\r\n".encode())

        headers = {
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        }

        r = self.api(
            "put",
            "/ISAPI/Intelligent/FDLib/FaceDataRecord?format=json",
            data=body.getvalue(),
            headers=headers,
            timeout=30,
        )
        try:
            data = r.json()
        except Exception:
            data = {"raw": r.text[:500]}
        ok = r.status_code == 200 and data.get("statusCode") == 1
        return ok, data


# ─── Local HTTP Server ──────────────────────────────────────────────────────

class _QuietHandler(partial(lambda d, *a, **k: None, "").__class__):
    pass


def start_face_server(serve_dir, port=FACE_SERVER_PORT):
    """Start a background HTTP server to serve face images to the device."""
    handler = partial(
        type("H", (HTTPServer,), {"log_message": lambda *a: None}).__class__,
    )

    # Simple quiet handler
    class QuietHandler(HTTPServer):
        pass

    from http.server import SimpleHTTPRequestHandler as SHTH

    class SilentHandler(SHTH):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=serve_dir, **kwargs)

        def log_message(self, fmt, *args):
            pass

    server = HTTPServer(("0.0.0.0", port), SilentHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    # Get local IP on same network as device
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect((os.getenv("HIKVISION_IP", "10.26.80.65"), 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "127.0.0.1"

    base_url = f"http://{local_ip}:{port}"
    return server, base_url


def stop_face_server(server):
    if server:
        server.shutdown()


# ─── Helpers ────────────────────────────────────────────────────────────────

def emp_no_from_name(name):
    """Generate deterministic 8-char employee number from name (matches hikvision_attendance.py)."""
    return hashlib.md5(name.encode()).hexdigest()[:8].upper()


def download_photo(url, dest_path):
    """Download a student photo from BINUS blob storage."""
    try:
        r = requests.get(url.strip(), timeout=15)
        if r.status_code == 200 and len(r.content) > 1000:
            with open(dest_path, "wb") as f:
                f.write(r.content)
            return True
        return False
    except Exception:
        return False


# ─── Main Enrollment Logic ──────────────────────────────────────────────────

def run_batch_enroll(device_ip, device_pass, dry_run=False, grade_filter=None, resume=True):
    # Load the report
    if not REPORT_FILE.exists():
        print("❌ No student report found. Run fetch_all_students.py first.")
        return

    report = json.loads(REPORT_FILE.read_text())
    all_students = report["students"]
    print(f"📋 Loaded {len(all_students)} students from report")

    # Filter to enrollable only
    enrollable = [s for s in all_students if s.get("enrollable")]
    if grade_filter:
        enrollable = [s for s in enrollable if s.get("grade") == grade_filter]
    print(f"🎯 Enrollable students: {len(enrollable)}")

    # Connect to device
    device = DeviceClient(device_ip, "admin", device_pass)

    # Check device is reachable
    try:
        r = device.api("get", "/ISAPI/System/deviceInfo")
        if r.status_code != 200:
            print(f"❌ Device {device_ip} returned HTTP {r.status_code}")
            return
        print(f"✅ Connected to device {device_ip}")
    except Exception as e:
        print(f"❌ Cannot reach device {device_ip}: {e}")
        return

    # Get currently enrolled users
    existing = device.get_enrolled_users()
    print(f"📊 Currently enrolled on device: {len(existing)} users")

    # Load local metadata
    meta = student_metadata._load_local()

    # Determine who needs enrollment
    to_enroll = []
    already = 0
    for s in enrollable:
        name = s.get("name", "")
        if not name:
            continue
        emp_no = emp_no_from_name(name)

        # Check if already on device with a face
        if emp_no in existing and existing[emp_no].get("numOfFace", 0) > 0:
            already += 1
            continue

        # Check by studentId in metadata too
        if resume:
            found_on_device = False
            for eno, m in meta.items():
                if m.get("idStudent") == s["idStudent"] and eno in existing:
                    if existing[eno].get("numOfFace", 0) > 0:
                        found_on_device = True
                        break
            if found_on_device:
                already += 1
                continue

        to_enroll.append(s)

    print(f"⏭️  Already enrolled: {already}")
    print(f"🆕 To enroll: {len(to_enroll)}")
    print()

    if not to_enroll:
        print("✅ All enrollable students are already on the device!")
        return

    if dry_run:
        print("🏷️  DRY RUN — would enroll:")
        for s in to_enroll:
            emp = emp_no_from_name(s["name"])
            print(f"  + {s['name']:40s} Grade {s['grade']} {s['homeroom']:4s} emp={emp} photo={bool(s.get('filePath'))}")
        print(f"\nTotal: {len(to_enroll)} students")
        return

    # Pre-download all photos
    faces_dir = DATA_DIR / "faces_batch"
    faces_dir.mkdir(exist_ok=True)

    print(f"📥 Downloading {len(to_enroll)} photos...")
    download_ok = 0
    download_fail = 0
    for s in to_enroll:
        name = s["name"]
        photo_url = (s.get("filePath") or "").strip()
        emp_no = emp_no_from_name(name)
        dest = faces_dir / f"{emp_no}.jpg"
        if dest.exists() and dest.stat().st_size > 1000:
            download_ok += 1
            continue
        if not photo_url:
            download_fail += 1
            continue
        if download_photo(photo_url, str(dest)):
            download_ok += 1
        else:
            download_fail += 1
    print(f"  ✅ Downloaded: {download_ok}  ❌ Failed: {download_fail}")
    print()

    # Start local HTTP server for face images
    server, base_url = start_face_server(str(faces_dir))
    print(f"🌐 Face server started at {base_url}")
    print()

    enrolled = 0
    failed = 0
    skipped = 0
    batch_count = 0

    for i, s in enumerate(to_enroll, 1):
        name = s["name"]
        sid = s["idStudent"]
        bid = s.get("idBinusian", "")
        homeroom = s.get("homeroom", "")
        grade = s.get("grade", "")
        emp_no = emp_no_from_name(name)

        print(f"[{i}/{len(to_enroll)}] {name} (Grade {grade} {homeroom})")

        photo_path = faces_dir / f"{emp_no}.jpg"
        if not photo_path.exists() or photo_path.stat().st_size < 1000:
            print(f"  ⏭️  No photo file, skipping")
            skipped += 1
            continue

        print(f"  📸 Photo: {photo_path.stat().st_size // 1024}KB")

        # Create user on device
        if emp_no not in existing:
            ok, resp = device.create_user(emp_no, name, sid)
            if not ok:
                if "exist" in str(resp).lower():
                    print(f"  ℹ️  User already exists on device")
                else:
                    err = resp.get("subStatusCode", resp.get("statusString", "unknown"))
                    print(f"  ❌ Failed to create user: {err}")
                    failed += 1
                    continue
            else:
                print(f"  ✅ User created")

        # Upload face via URL (device downloads from our server)
        face_url = f"{base_url}/{emp_no}.jpg"
        face_ok = False
        for attempt in range(3):
            ok, resp = device.upload_face(emp_no, name, face_url)
            if ok:
                print(f"  ✅ Face enrolled!")
                enrolled += 1
                face_ok = True

                student_metadata.save_student(
                    employee_no=emp_no,
                    name=name,
                    id_student=sid,
                    id_binusian=bid,
                    homeroom=homeroom,
                    grade=grade,
                )
                existing[emp_no] = {"name": name, "numOfFace": 1}
                break
            else:
                err = resp.get("subStatusCode", resp.get("errorMsg", str(resp)[:150]))
                if attempt < 2:
                    print(f"  ⚠️  Retry {attempt+1}/2 — {err}")
                    time.sleep(3)
                else:
                    print(f"  ❌ Face upload failed: {err}")
                    failed += 1

        batch_count += 1
        # Every 20 enrollments, pause to let the device breathe
        if batch_count % 20 == 0:
            print(f"\n  ⏸️  Batch pause (20 done, total {enrolled} enrolled)...\n")
            time.sleep(5)
        else:
            time.sleep(1.0)

    stop_face_server(server)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  BATCH ENROLLMENT COMPLETE")
    print(f"{'=' * 60}")
    print(f"  ✅ Enrolled: {enrolled}")
    print(f"  ❌ Failed:   {failed}")
    print(f"  ⏭️  Skipped:  {skipped}")
    print(f"  📊 Total on device: {len(existing)}")
    print()

    # Verify final count
    final_users = device.get_enrolled_users()
    faces = sum(1 for u in final_users.values() if u.get("numOfFace", 0) > 0)
    print(f"  📊 Final device state: {len(final_users)} users, {faces} with faces")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch enroll students on Hikvision device")
    parser.add_argument("--device", default=os.getenv("HIKVISION_IP", "10.26.80.65"), help="Device IP")
    parser.add_argument("--password", default=os.getenv("HIKVISION_PASS", "password.123"), help="Device password")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    parser.add_argument("--grade", default=None, help="Only enroll specific grade (1-6)")
    parser.add_argument("--no-resume", action="store_true", help="Don't skip already-enrolled")
    args = parser.parse_args()

    run_batch_enroll(
        device_ip=args.device,
        device_pass=args.password,
        dry_run=args.dry_run,
        grade_filter=args.grade,
        resume=not args.no_resume,
    )
