#!/usr/bin/env python3
"""
capture_and_upload.py — Capture high-quality face images from the Hikvision
device and upload them to Firebase Storage along with the student metadata
(name, ID, grade) retrieved from the Binus School API.

Usage:
    # Single student — enter ID interactively
    python capture_and_upload.py

    # Single student — pass ID on command line
    python capture_and_upload.py 2470006173

    # Multiple students — pass several IDs
    python capture_and_upload.py 2470006173 2570010026 2470005555

    # Whole class by grade + homeroom
    python capture_and_upload.py --class "EL 4" 4C

Each captured image is saved under:
    gs://<bucket>/face_dataset/<Homeroom>/<StudentName>/<timestamp>_device_capture.jpg

Metadata (name, ID, grade, homeroom, capture time) is written alongside each
image in Firebase Firestore under:  students/{studentId}
"""

import os
import sys
import json
import hashlib
import socket
import argparse
from datetime import datetime
from pathlib import Path

import requests
from requests.auth import HTTPDigestAuth
from dotenv import load_dotenv

# ── Load environment ─────────────────────────────────────────────────────────
load_dotenv()

HIKVISION_IP   = os.getenv("HIKVISION_IP", "10.26.30.200")
HIKVISION_USER = os.getenv("HIKVISION_USER", "admin")
HIKVISION_PASS = os.getenv("HIKVISION_PASS", "password.123")
HIK_BASE       = f"http://{HIKVISION_IP}"
HIK_AUTH       = HTTPDigestAuth(HIKVISION_USER, HIKVISION_PASS)

BINUS_API_KEY    = os.getenv("API_KEY", "")
BINUS_AUTH_URL   = "http://binusian.ws/binusschool/auth/token"
BINUS_STUDENT_URL = "http://binusian.ws/binusschool/bss-student-enrollment"
BINUS_PHOTOS_URL  = "http://binusian.ws/binusschool/bss-get-simprug-studentphoto-fr"

FIREBASE_CREDENTIALS = os.getenv(
    "FIREBASE_CREDENTIALS",
    "facial-attendance-binus-firebase-adminsdk.json",
)
FIREBASE_BUCKET = os.getenv(
    "FIREBASE_STORAGE_BUCKET",
    "facial-attendance-binus.firebasestorage.app",
)

CAPTURES_PER_STUDENT = 3          # how many shots per student
CAPTURE_TIMEOUT      = 60         # seconds the device waits for a face
LOCAL_SAVE_DIR       = Path("captured_faces")  # local backup folder


# ── Binus API ────────────────────────────────────────────────────────────────

def binus_get_token() -> str | None:
    """Authenticate with the Binus School API and return a Bearer token."""
    if not BINUS_API_KEY:
        print("❌  API_KEY not set in .env — cannot call Binus API.")
        return None
    try:
        r = requests.get(
            BINUS_AUTH_URL,
            headers={"Authorization": f"Basic {BINUS_API_KEY}"},
            timeout=15,
        )
        r.raise_for_status()
        token = r.json().get("data", {}).get("token")
        if token:
            return token
        print("⚠️   Binus auth returned no token.")
        return None
    except Exception as e:
        print(f"❌  Binus auth failed: {e}")
        return None


def binus_lookup_student(student_id: str, token: str) -> dict | None:
    """Look up a student by ID. Returns the raw studentDataResponse dict."""
    try:
        r = requests.post(
            BINUS_STUDENT_URL,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json={"IdStudent": str(student_id)},
            timeout=15,
        )
        r.raise_for_status()
        result = r.json()
        if result.get("resultCode") == 200 and result.get("studentDataResponse"):
            return result["studentDataResponse"]
        print(f"⚠️   Student {student_id} not found: {result.get('errorMessage', '')}")
        return None
    except Exception as e:
        print(f"❌  Student lookup failed: {e}")
        return None


def binus_get_class_students(grade: str, homeroom: str, token: str) -> list[dict]:
    """Fetch the full student list for a grade/homeroom."""
    try:
        r = requests.post(
            BINUS_PHOTOS_URL,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json={"Grade": str(grade), "Homeroom": str(homeroom), "IdStudentList": None},
            timeout=30,
        )
        r.raise_for_status()
        result = r.json()
        if result.get("resultCode") == 200:
            students = result.get("studentPhotoResponse", {}).get("studentList") or []
            return students if isinstance(students, list) else []
        print(f"⚠️   Class lookup error: {result.get('errorMessage', '')}")
        return []
    except Exception as e:
        print(f"❌  Class fetch failed: {e}")
        return []


def parse_student(data: dict) -> dict:
    """Normalise a Binus API response into a clean student dict."""
    name = (
        data.get("studentName")
        or data.get("name")
        or data.get("fullName")
        or "Unknown"
    )
    homeroom = (
        data.get("homeroom")
        or data.get("class")
        or data.get("className")
        or "Unknown"
    )
    return {
        "studentId": str(data.get("idStudent", data.get("studentId", ""))),
        "name": name,
        "gradeCode": data.get("gradeCode", ""),
        "gradeName": data.get("gradeName", ""),
        "homeroom": homeroom,
    }


# ── Hikvision Device Capture ────────────────────────────────────────────────

def capture_face(output_path: str, timeout: int = CAPTURE_TIMEOUT) -> bool:
    """Trigger the Hikvision device to capture a 352×432 face JPEG.

    Blocks until a face is detected or timeout expires.
    """
    xml_body = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<CaptureFaceDataCond xmlns="http://www.isapi.org/ver20/XMLSchema" version="2.0">'
        '<captureInfrared>false</captureInfrared>'
        '<dataType>binary</dataType>'
        '</CaptureFaceDataCond>'
    )
    try:
        r = requests.post(
            f"{HIK_BASE}/ISAPI/AccessControl/CaptureFaceData",
            data=xml_body,
            headers={"Content-Type": "application/xml"},
            auth=HIK_AUTH,
            timeout=timeout,
        )
    except requests.exceptions.Timeout:
        print("      ⏰  Timed out — no face detected.")
        return False
    except requests.exceptions.RequestException as e:
        print(f"      ❌  Capture request failed: {e}")
        return False

    if r.status_code != 200:
        print(f"      ❌  Device returned HTTP {r.status_code}")
        return False

    content = r.content
    jpeg_start = content.find(b"\xff\xd8")
    jpeg_end = content.rfind(b"\xff\xd9")
    if jpeg_start == -1 or jpeg_end == -1:
        print("      ❌  No JPEG data in device response — retryable.")
        return False

    jpeg_data = content[jpeg_start : jpeg_end + 2]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(jpeg_data)

    print(f"      📸  Captured {len(jpeg_data):,} bytes → {output_path}")
    return True


# ── Firebase Upload ──────────────────────────────────────────────────────────

_firebase_app = None


def _init_firebase():
    """Initialise the Firebase Admin SDK (once)."""
    global _firebase_app
    if _firebase_app is not None:
        return _firebase_app

    import firebase_admin
    from firebase_admin import credentials

    cred_path = os.path.abspath(FIREBASE_CREDENTIALS)
    if not os.path.isfile(cred_path):
        raise FileNotFoundError(f"Firebase credentials not found: {cred_path}")

    cred = firebase_admin.credentials.Certificate(cred_path)
    _firebase_app = firebase_admin.initialize_app(cred, {
        "storageBucket": FIREBASE_BUCKET,
    })
    return _firebase_app


def upload_to_firebase(local_path: str, student: dict, capture_num: int) -> str | None:
    """Upload a face image to Firebase Storage and write metadata to Firestore.

    Storage path:  face_dataset/<homeroom>/<name>/<timestamp>_capture_N.jpg
    Firestore doc:  students/<studentId>/captures/<auto-id>

    Returns the public download URL (or None on failure).
    """
    try:
        from firebase_admin import storage as fb_storage
        from firebase_admin import firestore as fb_firestore

        app = _init_firebase()
        bucket = fb_storage.bucket(app=app)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        blob_path = (
            f"face_dataset/"
            f"{student['homeroom']}/"
            f"{student['name']}/"
            f"{ts}_capture_{capture_num}.jpg"
        )
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(local_path, content_type="image/jpeg")
        blob.make_public()
        url = blob.public_url
        print(f"      ☁️   Uploaded → gs://{FIREBASE_BUCKET}/{blob_path}")

        # Write metadata to Firestore
        try:
            db = fb_firestore.client(app=app)
            doc_ref = db.collection("students").document(student["studentId"])
            doc_ref.set(
                {
                    "studentId": student["studentId"],
                    "name": student["name"],
                    "gradeCode": student["gradeCode"],
                    "gradeName": student["gradeName"],
                    "homeroom": student["homeroom"],
                    "lastUpdated": datetime.now().isoformat(),
                },
                merge=True,
            )
            # Sub-collection for individual captures
            doc_ref.collection("captures").add({
                "storagePath": blob_path,
                "url": url,
                "captureNum": capture_num,
                "capturedAt": datetime.now().isoformat(),
                "source": "hikvision_device",
                "imageSize": os.path.getsize(local_path),
            })
        except Exception as e:
            print(f"      ⚠️   Firestore metadata write failed (non-fatal): {e}")

        return url

    except Exception as e:
        print(f"      ❌  Firebase upload failed: {e}")
        return None


# ── Main Pipeline ────────────────────────────────────────────────────────────

def process_student(student: dict, num_captures: int = CAPTURES_PER_STUDENT):
    """Capture N face images for a student and upload each to Firebase."""
    sid  = student["studentId"]
    name = student["name"]
    hr   = student["homeroom"]
    grade = student.get("gradeName", student.get("gradeCode", ""))

    print(f"\n{'═'*60}")
    print(f"  👤  {name}")
    print(f"      ID: {sid}  │  Grade: {grade}  │  Homeroom: {hr}")
    print(f"{'═'*60}")

    captured = 0
    for i in range(1, num_captures + 1):
        print(f"\n   📷  Capture {i}/{num_captures} — look at the device screen...")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = name.replace(" ", "_").replace("/", "_")
        local_path = str(LOCAL_SAVE_DIR / hr / safe_name / f"{ts}_cap{i}.jpg")

        ok = capture_face(local_path, timeout=CAPTURE_TIMEOUT)
        if not ok:
            print(f"      ⚠️   Capture {i} failed — retrying once...")
            ok = capture_face(local_path, timeout=CAPTURE_TIMEOUT)
        if not ok:
            print(f"      ❌  Capture {i} failed after retry, skipping.")
            continue

        captured += 1
        # Upload to Firebase
        upload_to_firebase(local_path, student, capture_num=i)

    print(f"\n   ✅  Done: {captured}/{num_captures} images captured & uploaded for {name}")
    return captured


def main():
    parser = argparse.ArgumentParser(
        description="Capture face images from Hikvision device → Firebase + Binus API metadata",
    )
    parser.add_argument(
        "student_ids",
        nargs="*",
        help="One or more Binus student IDs to capture",
    )
    parser.add_argument(
        "--class", "-c",
        nargs=2,
        metavar=("GRADE", "HOMEROOM"),
        dest="class_info",
        help='Capture an entire class, e.g. --class "EL 4" 4C',
    )
    parser.add_argument(
        "--captures", "-n",
        type=int,
        default=CAPTURES_PER_STUDENT,
        help=f"Number of captures per student (default: {CAPTURES_PER_STUDENT})",
    )
    args = parser.parse_args()

    # ── Verify device connectivity ────────────────────────────────────────
    print("🔌  Checking Hikvision device connectivity...")
    try:
        r = requests.get(
            f"{HIK_BASE}/ISAPI/System/deviceInfo",
            auth=HIK_AUTH,
            timeout=5,
        )
        if r.status_code == 200:
            print(f"   ✅  Device at {HIKVISION_IP} is online")
        else:
            print(f"   ⚠️   Device returned HTTP {r.status_code}")
    except Exception as e:
        print(f"   ❌  Cannot reach device at {HIKVISION_IP}: {e}")
        print("       Check the device is powered on and on the same network.")
        sys.exit(1)

    # ── Verify Firebase ───────────────────────────────────────────────────
    print("☁️   Initialising Firebase...")
    try:
        _init_firebase()
        print("   ✅  Firebase connected")
    except Exception as e:
        print(f"   ❌  Firebase init failed: {e}")
        print("       Check FIREBASE_CREDENTIALS path in .env")
        sys.exit(1)

    # ── Authenticate with Binus API ───────────────────────────────────────
    print("🔑  Authenticating with Binus School API...")
    token = binus_get_token()
    if not token:
        print("   ❌  Cannot get Binus API token. Check API_KEY in .env")
        sys.exit(1)
    print("   ✅  Binus API token acquired")

    # ── Build student list ────────────────────────────────────────────────
    students = []

    if args.class_info:
        grade, homeroom = args.class_info
        print(f"\n📋  Fetching student list for {grade} / {homeroom}...")
        raw_list = binus_get_class_students(grade, homeroom, token)
        if not raw_list:
            print("   ❌  No students found for that class.")
            sys.exit(1)
        for s in raw_list:
            students.append(parse_student(s))
        print(f"   ✅  Found {len(students)} students")

    elif args.student_ids:
        for sid in args.student_ids:
            print(f"\n🔍  Looking up student {sid}...")
            data = binus_lookup_student(sid, token)
            if data:
                s = parse_student(data)
                s["studentId"] = sid  # keep the original ID they typed
                students.append(s)
                print(f"   ✅  {s['name']} — {s['gradeName']} {s['homeroom']}")
            else:
                print(f"   ❌  Student {sid} not found, skipping.")

    else:
        # Interactive: prompt for IDs
        print("\n📝  Enter student IDs one per line. Empty line to finish.\n")
        while True:
            sid = input("   Student ID (or Enter to finish): ").strip()
            if not sid:
                break
            data = binus_lookup_student(sid, token)
            if data:
                s = parse_student(data)
                s["studentId"] = sid
                students.append(s)
                print(f"   ✅  {s['name']} — {s['gradeName']} {s['homeroom']}")
            else:
                print(f"   ❌  Not found, try again.")

    if not students:
        print("\n❌  No students to capture. Exiting.")
        sys.exit(0)

    # ── Summary before starting ───────────────────────────────────────────
    print(f"\n{'━'*60}")
    print(f"  📋  Ready to capture {len(students)} student(s)")
    print(f"      {args.captures} image(s) each → Firebase Storage + Firestore")
    print(f"      Device: {HIKVISION_IP}")
    print(f"      Bucket: {FIREBASE_BUCKET}")
    print(f"{'━'*60}")
    for i, s in enumerate(students, 1):
        print(f"   {i:>3}. {s['name']:30s}  {s['studentId']:15s}  {s['gradeName']} {s['homeroom']}")
    print()

    confirm = input("  ▶  Start capture? [Y/n] ").strip().lower()
    if confirm and confirm != "y":
        print("Aborted.")
        sys.exit(0)

    # ── Capture loop ──────────────────────────────────────────────────────
    LOCAL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    total_ok = 0
    total_fail = 0

    for student in students:
        n = process_student(student, num_captures=args.captures)
        if n > 0:
            total_ok += 1
        else:
            total_fail += 1

    # ── Final summary ─────────────────────────────────────────────────────
    print(f"\n{'━'*60}")
    print(f"  🏁  DONE")
    print(f"      ✅  {total_ok} student(s) captured successfully")
    if total_fail:
        print(f"      ❌  {total_fail} student(s) had no captures")
    print(f"      📁  Local copies: {LOCAL_SAVE_DIR}/")
    print(f"      ☁️   Firebase:    gs://{FIREBASE_BUCKET}/face_dataset/")
    print(f"{'━'*60}\n")


if __name__ == "__main__":
    main()
