#!/usr/bin/env python3
"""
re_enroll.py — Re-enroll students with outdated photos using the device camera
===============================================================================
Students whose enrollment photos are 3-4+ years old will have poor face
recognition accuracy. This script lets you re-enroll them by capturing a
fresh face directly from the Hikvision device camera.

Modes:
    python3 re_enroll.py --list                 # Show all students ranked by photo age risk
    python3 re_enroll.py --grade 5              # Re-enroll all Grade 5 students interactively
    python3 re_enroll.py --student "John Doe"   # Re-enroll one specific student
    python3 re_enroll.py --high-risk            # Re-enroll all HIGH risk students (4+ year photos)
    python3 re_enroll.py --all                  # Re-enroll ALL students (fresh capture for everyone)

The device camera captures a 352x432 face-optimized JPEG. This replaces
the old face embedding on the device and backs up to Firebase Storage.
"""

import os
import sys
import json
import time
import hashlib
import re
import socket
import tempfile
import threading
import argparse
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from datetime import datetime, timezone, timedelta

import requests
from requests.auth import HTTPDigestAuth

sys.path.insert(0, os.path.dirname(__file__))
import student_metadata

WIB = timezone(timedelta(hours=7))
DATA_DIR = Path(__file__).parent / "data"
BINUS_REPORT = DATA_DIR / "binus_students_all.json"

FIREBASE_CREDENTIALS = os.getenv(
    "FIREBASE_CREDENTIALS",
    str(Path(__file__).parent / "facial-attendance-binus-firebase-adminsdk.json"),
)
FIREBASE_BUCKET = os.getenv(
    "FIREBASE_STORAGE_BUCKET",
    "facial-attendance-binus.firebasestorage.app",
)


def emp_no(name):
    return hashlib.md5(name.encode()).hexdigest()[:8].upper()


# ─── Device Client ──────────────────────────────────────────────────────────

class DeviceClient:
    def __init__(self, ip, user="admin", password="password.123"):
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
            body = {"UserInfoSearchCond": {"searchID": "re", "searchResultPosition": pos, "maxResults": 30}}
            status, data = self.api_json("post", "/ISAPI/AccessControl/UserInfo/Search?format=json", json=body)
            if status != 200:
                break
            info = data.get("UserInfoSearch", {})
            ul = info.get("UserInfo", [])
            if isinstance(ul, dict):
                ul = [ul]
            for u in ul:
                eno = u.get("employeeNo", "")
                if eno:
                    users[eno] = {"name": u.get("name", ""), "numOfFace": u.get("numOfFace", 0)}
            pos += len(ul)
            if pos >= int(info.get("totalMatches", 0)) or not ul:
                break
        return users

    def capture_face(self, output_path, timeout=60):
        """Capture face from device camera. Blocks until face detected."""
        xml_body = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<CaptureFaceDataCond xmlns="http://www.isapi.org/ver20/XMLSchema" version="2.0">'
            '<captureInfrared>false</captureInfrared>'
            '<dataType>binary</dataType>'
            '</CaptureFaceDataCond>'
        )
        try:
            r = self.api("post", "/ISAPI/AccessControl/CaptureFaceData",
                         data=xml_body,
                         headers={"Content-Type": "application/xml"},
                         timeout=timeout)
        except requests.exceptions.Timeout:
            return False, "timeout"
        except requests.exceptions.RequestException as e:
            return False, str(e)

        if r.status_code != 200:
            return False, f"HTTP {r.status_code}"

        content = r.content
        jpeg_start = content.find(b"\xff\xd8")
        jpeg_end = content.rfind(b"\xff\xd9")
        if jpeg_start == -1 or jpeg_end == -1:
            return False, "no JPEG in response"

        jpeg_data = content[jpeg_start:jpeg_end + 2]
        with open(output_path, "wb") as f:
            f.write(jpeg_data)
        return True, f"{len(jpeg_data)} bytes"

    def delete_face(self, employee_no):
        payload = {"faceLibType": "blackFD", "FDID": "1", "FPID": employee_no}
        status, data = self.api_json("put", "/ISAPI/Intelligent/FDLib/FDDelete?format=json", json=payload)
        return status == 200

    def upload_face(self, employee_no, name, face_url):
        payload = {
            "faceLibType": "blackFD", "FDID": "1",
            "FPID": employee_no, "name": name, "faceURL": face_url,
        }
        status, data = self.api_json("put", "/ISAPI/Intelligent/FDLib/FDSetUp?format=json", json=payload)
        return status == 200 and data.get("statusCode") == 1, data

    def create_user(self, employee_no, name):
        user_data = {
            "UserInfo": {
                "employeeNo": employee_no, "name": name,
                "userType": "normal", "gender": "unknown",
                "Valid": {"enable": True, "beginTime": "2024-01-01T00:00:00",
                          "endTime": "2037-12-31T23:59:59", "timeType": "local"},
                "doorRight": "1",
                "RightPlan": [{"doorNo": 1, "planTemplateNo": "1"}],
            }
        }
        status, data = self.api_json("post", "/ISAPI/AccessControl/UserInfo/Record?format=json", json=user_data)
        return status == 200 and data.get("statusCode") == 1, data


# ─── Firebase Backup ────────────────────────────────────────────────────────

def upload_to_firebase(local_path, student_name, homeroom):
    try:
        import firebase_admin
        from firebase_admin import credentials, storage as fb_storage
        try:
            app = firebase_admin.get_app()
        except ValueError:
            cred = credentials.Certificate(os.path.abspath(FIREBASE_CREDENTIALS))
            app = firebase_admin.initialize_app(cred, {"storageBucket": FIREBASE_BUCKET})
        bucket = fb_storage.bucket(app=app)
        ts = datetime.now(WIB).strftime("%Y%m%d_%H%M%S")
        blob_path = f"face_dataset/{homeroom}/{student_name}/{ts}_device_capture.jpg"
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(local_path, content_type="image/jpeg")
        blob.make_public()
        print(f"    ☁️  Firebase: {blob_path}")
    except Exception as e:
        print(f"    ⚠️  Firebase backup failed: {e}")


# ─── Face Server ────────────────────────────────────────────────────────────

def start_face_server(serve_dir, port=8899):
    class SilentHandler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=serve_dir, **kw)
        def log_message(self, *a):
            pass

    server = HTTPServer(("0.0.0.0", port), SilentHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect((os.getenv("HIKVISION_IP", "10.26.80.65"), 80))
        local_ip = sock.getsockname()[0]
        sock.close()
    except Exception:
        local_ip = "127.0.0.1"
    return server, f"http://{local_ip}:{port}"


# ─── Risk Assessment ────────────────────────────────────────────────────────

def build_risk_list():
    """Build list of enrolled students with photo age risk."""
    if not BINUS_REPORT.exists():
        print("❌ Run fetch_all_students.py first.")
        sys.exit(1)

    binus = json.loads(BINUS_REPORT.read_text())
    binus_by_name = {s["name"]: s for s in binus["students"]}
    meta = student_metadata._load_local()

    results = []
    for eno, m in meta.items():
        name = m.get("name", "")
        if not name:
            continue
        bs = binus_by_name.get(name, {})
        fp = bs.get("filePath", "")
        grade = m.get("grade", bs.get("grade", "?"))
        homeroom = m.get("homeroom", bs.get("homeroom", "?"))
        sid = m.get("idStudent", bs.get("idStudent", ""))
        bid = m.get("idBinusian", bs.get("idBinusian", ""))

        match = re.search(r"SIMPRUG(\d{4})", fp)
        photo_year = int(match.group(1)) if match else None
        gap = (2026 - photo_year) if photo_year else None

        risk = "unknown"
        if gap is not None:
            if gap >= 4:
                risk = "high"
            elif gap >= 3:
                risk = "medium"
            else:
                risk = "low"

        results.append({
            "name": name, "emp": eno, "grade": grade, "homeroom": homeroom,
            "idStudent": sid, "idBinusian": bid,
            "photoYear": photo_year, "gap": gap, "risk": risk,
        })

    # Sort: high risk first, then by grade descending (older students first)
    risk_order = {"high": 0, "medium": 1, "unknown": 2, "low": 3}
    results.sort(key=lambda x: (risk_order.get(x["risk"], 9), -(int(x["grade"]) if x["grade"].isdigit() else 0)))
    return results


def show_risk_list(students, grade_filter=None):
    if grade_filter:
        students = [s for s in students if str(s["grade"]) == str(grade_filter)]

    risk_icons = {"high": "🔴", "medium": "🟡", "low": "🟢", "unknown": "⚪"}
    from collections import Counter
    counts = Counter(s["risk"] for s in students)

    print(f"\n{'='*70}")
    print(f"  PHOTO AGE RISK ASSESSMENT — {len(students)} enrolled students")
    print(f"{'='*70}")
    print(f"  🔴 HIGH (4+ years):   {counts.get('high', 0)}")
    print(f"  🟡 MEDIUM (3 years):  {counts.get('medium', 0)}")
    print(f"  🟢 LOW (≤2 years):    {counts.get('low', 0)}")
    print(f"  ⚪ Unknown:           {counts.get('unknown', 0)}")
    print()

    current_grade = None
    for s in students:
        g = s["grade"]
        if g != current_grade:
            current_grade = g
            grade_students = [x for x in students if x["grade"] == g]
            print(f"\n  --- Grade {g} ({len(grade_students)} students) ---")

        icon = risk_icons.get(s["risk"], "?")
        year = s["photoYear"] or "?"
        gap = f"{s['gap']}yr" if s["gap"] else "?"
        print(f"  {icon} {s['name']:<42s} {s['homeroom']:<4s} photo={year} gap={gap}")


# ─── Re-enrollment ──────────────────────────────────────────────────────────

def re_enroll_students(students, device_ip, device_pass):
    """Interactive re-enrollment using device camera."""
    if not students:
        print("No students to re-enroll.")
        return

    device = DeviceClient(device_ip, "admin", device_pass)

    try:
        r = device.api("get", "/ISAPI/System/deviceInfo")
        if r.status_code != 200:
            print(f"❌ Device unreachable (HTTP {r.status_code})")
            return
        print(f"✅ Connected to device {device_ip}")
    except Exception as e:
        print(f"❌ Cannot reach device: {e}")
        return

    enrolled_users = device.get_enrolled_users()

    with tempfile.TemporaryDirectory(prefix="hik_reenroll_") as tmp:
        serve_dir = Path(tmp) / "faces"
        serve_dir.mkdir()
        server, base_url = start_face_server(str(serve_dir))
        print(f"🌐 Face server: {base_url}\n")

        success = 0
        failed = 0
        skipped = 0

        for i, s in enumerate(students, 1):
            name = s["name"]
            eno = s["emp"]
            homeroom = s["homeroom"]
            grade = s["grade"]
            gap = s["gap"]

            risk_icon = {"high": "🔴", "medium": "🟡"}.get(s["risk"], "⚪")
            print(f"\n[{i}/{len(students)}] {risk_icon} {name} (Grade {grade} {homeroom}, photo {gap or '?'}yr old)")

            # Ask for confirmation
            try:
                choice = input(f"  Re-enroll? [y]es / [s]kip / [q]uit: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n  Aborted.")
                break

            if choice == "q":
                print("  Stopping.")
                break
            if choice != "y":
                print("  ⏭️  Skipped")
                skipped += 1
                continue

            # Ensure user exists on device
            if eno not in enrolled_users:
                ok, resp = device.create_user(eno, name)
                if not ok and "exist" not in str(resp).lower():
                    print(f"  ❌ Create user failed: {resp}")
                    failed += 1
                    continue

            # Delete old face
            device.delete_face(eno)
            print(f"  🗑️  Old face removed")
            time.sleep(0.5)

            # Capture new face from device camera
            print(f"  📷 {name}, please look at the device camera...")
            capture_path = str(serve_dir / f"{eno}_face.jpg")
            ok, info = device.capture_face(capture_path, timeout=60)
            if not ok:
                print(f"  ❌ Capture failed: {info}")
                failed += 1
                continue
            print(f"  📸 Captured ({info})")

            # Upload new face to device
            face_url = f"{base_url}/{eno}_face.jpg"
            ok, resp = device.upload_face(eno, name, face_url)
            if ok:
                print(f"  ✅ Re-enrolled!")
                success += 1

                # Update metadata timestamp
                student_metadata.save_student(
                    employee_no=eno, name=name,
                    id_student=s.get("idStudent", ""),
                    id_binusian=s.get("idBinusian", ""),
                    homeroom=homeroom, grade=grade,
                )

                # Backup to Firebase
                upload_to_firebase(capture_path, name, homeroom)
            else:
                err = resp.get("subStatusCode", resp.get("errorMsg", "unknown"))
                print(f"  ❌ Upload failed: {err}")
                failed += 1

            time.sleep(1)

        server.shutdown()

    print(f"\n{'='*50}")
    print(f"  RE-ENROLLMENT SUMMARY")
    print(f"{'='*50}")
    print(f"  ✅ Re-enrolled: {success}")
    print(f"  ❌ Failed:      {failed}")
    print(f"  ⏭️  Skipped:     {skipped}")


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Re-enroll students with outdated photos")
    parser.add_argument("--device", default=os.getenv("HIKVISION_IP", "10.26.80.65"))
    parser.add_argument("--password", default=os.getenv("HIKVISION_PASS", "password.123"))
    parser.add_argument("--list", action="store_true", help="Show risk assessment list")
    parser.add_argument("--grade", default=None, help="Filter by grade")
    parser.add_argument("--student", default=None, help="Re-enroll specific student by name")
    parser.add_argument("--high-risk", action="store_true", help="Re-enroll all HIGH risk (4+ years)")
    parser.add_argument("--all", action="store_true", help="Re-enroll ALL students")
    args = parser.parse_args()

    all_students = build_risk_list()

    if args.list:
        show_risk_list(all_students, args.grade)
        return

    # Filter
    targets = all_students
    if args.student:
        targets = [s for s in all_students if args.student.lower() in s["name"].lower()]
        if not targets:
            print(f"❌ No enrolled student matching '{args.student}'")
            return
    elif args.high_risk:
        targets = [s for s in all_students if s["risk"] == "high"]
    elif not args.all:
        # Default: high + medium risk
        targets = [s for s in all_students if s["risk"] in ("high", "medium")]

    if args.grade:
        targets = [s for s in targets if str(s["grade"]) == str(args.grade)]

    print(f"📋 {len(targets)} students selected for re-enrollment")
    if not targets:
        print("✅ Nothing to do.")
        return

    re_enroll_students(targets, args.device, args.password)


if __name__ == "__main__":
    main()
