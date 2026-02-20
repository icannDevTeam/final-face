#!/usr/bin/env python3
"""
Hikvision DS-K1T341AMF Attendance System
=========================================
Enrolls students from Firebase Storage directly onto the Hikvision face
recognition terminal via the ISAPI Intelligent FDLib (Face Database Library),
then monitors access events and syncs attendance back to Firebase Firestore.

The device's built-in DeepLearn algorithm handles all face recognition.
Its LCD displays the student's name and photo on match — no OpenCV needed.

Enrollment pipeline:
    Firebase Storage → download images → detect/crop face (dlib) →
    serve via temp HTTP → device downloads & extracts own embeddings

Usage:
    python hikvision_attendance.py enroll           # Enroll faces from Firebase → device
    python hikvision_attendance.py enroll-live      # Enroll one student (Binus ID → device camera)
    python hikvision_attendance.py enroll-class 1 1A  # Enroll entire class via Binus API
    python hikvision_attendance.py monitor           # Poll events → Firebase sync
    python hikvision_attendance.py status            # Show enrolled users on device
    python hikvision_attendance.py clear             # Remove all users from device
"""

import os
import sys
import json
import time
import glob
import hashlib
import argparse
import tempfile
import re
import socket
import threading
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from datetime import datetime, timezone, timedelta

import cv2
import dlib
import numpy as np
import requests
from requests.auth import HTTPDigestAuth
from dotenv import load_dotenv

load_dotenv()

# Student metadata mapping (employeeNo → BINUS IDs)
import student_metadata

# ─── Configuration ───────────────────────────────────────────────────────────

HIKVISION_IP   = os.getenv("HIKVISION_IP", "10.26.30.200")
HIKVISION_USER = os.getenv("HIKVISION_USER", "admin")
HIKVISION_PASS = os.getenv("HIKVISION_PASS", "password.123")
HIKVISION_BASE = f"http://{HIKVISION_IP}"
HIKVISION_AUTH = HTTPDigestAuth(HIKVISION_USER, HIKVISION_PASS)
POLL_INTERVAL  = int(os.getenv("POLL_INTERVAL", "3"))

# Persistent session for digest auth (avoids stale nonce issues)
_hik_session = requests.Session()
_hik_session.auth = HIKVISION_AUTH

FIREBASE_CREDENTIALS = os.getenv(
    "FIREBASE_CREDENTIALS",
    "facial-attendance-binus-firebase-adminsdk.json",
)
FIREBASE_BUCKET = os.getenv(
    "FIREBASE_STORAGE_BUCKET",
    "facial-attendance-binus.firebasestorage.app",
)

# Binus School API
BINUS_API_KEY = os.getenv("API_KEY", "")
BINUS_AUTH_URL = "http://binusian.ws/binusschool/auth/token"
BINUS_STUDENT_URL = "http://binusian.ws/binusschool/bss-student-enrollment"
BINUS_PHOTOS_URL = "http://binusian.ws/binusschool/bss-get-simprug-studentphoto-fr"

# Port for temporary HTTP server that serves face images to the device
FACE_SERVER_PORT = int(os.getenv("FACE_SERVER_PORT", "8888"))

# Face crop resize target (portrait, W×H)
FACE_TARGET_SIZE = (480, 640)

# ─── dlib models (loaded once) ───────────────────────────────────────────────

_face_detector = None
_shape_predictor = None

LANDMARK_MODEL = os.path.join(
    os.path.dirname(__file__), "shape_predictor_68_face_landmarks.dat"
)

def _get_face_detector():
    global _face_detector
    if _face_detector is None:
        _face_detector = dlib.get_frontal_face_detector()
    return _face_detector


def _get_shape_predictor():
    global _shape_predictor
    if _shape_predictor is None:
        if not os.path.isfile(LANDMARK_MODEL):
            raise FileNotFoundError(
                f"Landmark model not found: {LANDMARK_MODEL}\n"
                "Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
            )
        _shape_predictor = dlib.shape_predictor(LANDMARK_MODEL)
    return _shape_predictor


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _api(method, path, **kwargs):
    """Call the Hikvision ISAPI and return the response.
    
    Automatically retries once on 401 (stale digest nonce) by rebuilding
    the session auth.
    """
    url = f"{HIKVISION_BASE}{path}"
    kwargs.setdefault("timeout", 15)
    kwargs.pop("auth", None)
    
    r = getattr(_hik_session, method)(url, **kwargs)
    
    # Retry on 401 — rebuild auth (device may have invalidated nonce)
    if r.status_code == 401:
        import time as _time
        _time.sleep(1)
        _hik_session.auth = HTTPDigestAuth(HIKVISION_USER, HIKVISION_PASS)
        r = getattr(_hik_session, method)(url, **kwargs)
    
    return r


def _api_json(method, path, **kwargs):
    """Call ISAPI and parse JSON response."""
    r = _api(method, path, **kwargs)
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, {"raw": r.text[:500]}


def _employee_no_from_name(name: str) -> str:
    """Generate a deterministic 8-char employee number from a student name."""
    h = hashlib.md5(name.encode()).hexdigest()[:8].upper()
    return h


def _extract_student_id(filename: str) -> str | None:
    """Try to extract a numeric student ID from a filename like '2070003324_front_...'."""
    m = re.match(r"^(\d{7,})_", filename)
    return m.group(1) if m else None


def _get_local_ip() -> str:
    """Get the local IP address on the same network as the Hikvision device."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect((HIKVISION_IP, 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


# ─── Binus School API ────────────────────────────────────────────────────────

def _binus_get_token() -> str | None:
    """Get a Bearer token from the Binus School API."""
    if not BINUS_API_KEY:
        print("❌ API_KEY not set in .env — cannot call Binus API.")
        return None
    try:
        r = requests.get(
            BINUS_AUTH_URL,
            headers={"Authorization": f"Basic {BINUS_API_KEY}"},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        token = data.get("data", {}).get("token")
        if token:
            return token
        print(f"⚠️  Binus auth response has no token: {data}")
        return None
    except Exception as e:
        print(f"❌ Binus auth failed: {e}")
        return None


def binus_lookup_student(student_id: str, token: str | None = None) -> dict | None:
    """Look up a single student by ID using the C2 endpoint.

    Returns dict with keys: studentName, gradeCode, gradeName, homeroom/class.
    """
    if token is None:
        token = _binus_get_token()
    if token is None:
        return None
    try:
        r = requests.post(
            BINUS_STUDENT_URL,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"IdStudent": str(student_id)},
            timeout=15,
        )
        r.raise_for_status()
        result = r.json()
        if result.get("resultCode") == 200 and result.get("studentDataResponse"):
            return result["studentDataResponse"]
        print(f"⚠️  Student {student_id} not found: {result.get('errorMessage', 'unknown')}")
        return None
    except Exception as e:
        print(f"❌ Binus student lookup failed: {e}")
        return None


def binus_get_class_students(grade: str, homeroom: str, token: str | None = None) -> list[dict]:
    """Get the student list for a grade/homeroom from Binus API.

    Returns list of dicts from studentPhotoResponse.studentList.
    """
    if token is None:
        token = _binus_get_token()
    if token is None:
        return []
    try:
        r = requests.post(
            BINUS_PHOTOS_URL,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"Grade": str(grade), "Homeroom": str(homeroom), "IdStudentList": None},
            timeout=30,
        )
        r.raise_for_status()
        result = r.json()
        if result.get("resultCode") == 200:
            spr = result.get("studentPhotoResponse", {})
            students = spr.get("studentList") or []
            return students if isinstance(students, list) else []
        print(f"⚠️  Binus class lookup error: {result.get('errorMessage', 'unknown')}")
        return []
    except Exception as e:
        print(f"❌ Binus class fetch failed: {e}")
        return []


def _parse_student_name(data: dict) -> str:
    """Extract the best student name from a Binus API response."""
    return (
        data.get("studentName")
        or data.get("name")
        or data.get("fullName")
        or data.get("studentFullName")
        or "Unknown"
    )


def _parse_homeroom(data: dict) -> str:
    """Extract homeroom/class from a Binus API response."""
    return (
        data.get("homeroom")
        or data.get("class")
        or data.get("className")
        or "Unknown"
    )


# ─── Temporary HTTP Server for Face Images ───────────────────────────────────

class _QuietHTTPHandler(SimpleHTTPRequestHandler):
    """HTTP handler that suppresses log output."""
    def log_message(self, format, *args):
        pass  # silent


def _start_face_server(serve_dir: str, port: int = FACE_SERVER_PORT):
    """Start a background HTTP server to serve face images to the device.

    Returns (server, thread, base_url).
    """
    handler = partial(_QuietHTTPHandler, directory=serve_dir)
    server = HTTPServer(("0.0.0.0", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    local_ip = _get_local_ip()
    base_url = f"http://{local_ip}:{port}"
    return server, thread, base_url


def _stop_face_server(server):
    """Shut down the face image HTTP server."""
    if server:
        server.shutdown()


# ─── Face Image Processing ──────────────────────────────────────────────────

def score_frontality(image_path: str) -> float:
    """Score how frontal a face is using 68-point landmarks.

    Returns a float between 0.0 (profile / no face) and 1.0 (perfect frontal).
    Uses nose-to-jaw symmetry and eye centering to measure head pose.
    """
    img = cv2.imread(image_path)
    if img is None:
        return 0.0

    detector = _get_face_detector()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faces = detector(rgb, 1)
    if not faces:
        faces = detector(rgb, 2)
    if not faces:
        return 0.0

    try:
        predictor = _get_shape_predictor()
        shape = predictor(rgb, faces[0])

        # Nose tip (point 30)
        nose_x = shape.part(30).x
        # Jaw edges (point 0 = left, point 16 = right)
        jaw_left = shape.part(0).x
        jaw_right = shape.part(16).x
        face_width = jaw_right - jaw_left
        if face_width <= 0:
            return 0.0
        face_center = (jaw_left + jaw_right) / 2

        # Nose offset from jaw center (0 = centered = frontal)
        nose_offset = abs(nose_x - face_center) / face_width

        # Left/right face width symmetry around nose
        left_w = nose_x - jaw_left
        right_w = jaw_right - nose_x
        symmetry = min(left_w, right_w) / max(left_w, right_w) if max(left_w, right_w) > 0 else 0.0

        # Eye center offset
        left_eye = np.mean([shape.part(i).x for i in range(36, 42)])
        right_eye = np.mean([shape.part(i).x for i in range(42, 48)])
        eye_center = (left_eye + right_eye) / 2
        eye_offset = abs(eye_center - face_center) / face_width

        frontality = symmetry * (1.0 - nose_offset) * (1.0 - eye_offset)
        return max(0.0, min(1.0, frontality))
    except Exception:
        # If landmark model missing, fall back to basic detection ratio
        d = faces[0]
        fw = d.right() - d.left()
        fh = d.bottom() - d.top()
        ratio = fw / fh if fh > 0 else 0
        # Front faces have ratio ~1.0, side faces > 1.1
        return max(0.0, 1.0 - abs(ratio - 1.0))


def rank_images_by_frontality(image_paths: list[Path]) -> list[tuple[Path, float]]:
    """Sort images best-frontal-first.  Returns [(path, score), ...]."""
    scored = []
    for p in image_paths:
        s = score_frontality(str(p))
        scored.append((p, s))
    scored.sort(key=lambda x: -x[1])  # highest frontality first
    return scored


def crop_face(image_path: str, output_path: str,
              target_size=FACE_TARGET_SIZE, margin: float = 0.4) -> bool:
    """Detect a face in an image, crop with margin, and resize.

    Returns True if a face was found and cropped, False otherwise.
    If no face is detected, the original image is resized as fallback.
    """
    img = cv2.imread(image_path)
    if img is None:
        return False

    detector = _get_face_detector()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Try normal detection first, then with upsample
    faces = detector(rgb, 1)
    if not faces:
        faces = detector(rgb, 2)

    if faces:
        d = faces[0]
        h, w = img.shape[:2]
        margin_x = int((d.right() - d.left()) * margin)
        margin_y = int((d.bottom() - d.top()) * margin)
        x1 = max(0, d.left() - margin_x)
        y1 = max(0, d.top() - margin_y)
        x2 = min(w, d.right() + margin_x)
        y2 = min(h, d.bottom() + margin_y)
        face_crop = img[y1:y2, x1:x2]
    else:
        # Fallback: use the whole image
        face_crop = img

    resized = cv2.resize(face_crop, target_size)
    cv2.imwrite(output_path, resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return len(faces) > 0


# ─── Device Info ─────────────────────────────────────────────────────────────

def get_device_info():
    """Print device information."""
    r = _api("get", "/ISAPI/System/deviceInfo")
    if r.status_code != 200:
        print(f"⚠️  Could not reach device (HTTP {r.status_code})")
        return
    # Device returns XML for this endpoint
    try:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(r.text)
        ns = {"ns": "http://www.isapi.org/ver20/XMLSchema"}
        print(f"📱 Device: {root.findtext('ns:model', '?', ns)}")
        print(f"   Name:   {root.findtext('ns:deviceName', '?', ns)}")
        print(f"   FW:     {root.findtext('ns:firmwareVersion', '?', ns)} "
              f"({root.findtext('ns:firmwareReleasedDate', '?', ns)})")
        print(f"   Serial: {root.findtext('ns:serialNumber', '?', ns)}")
        print(f"   MAC:    {root.findtext('ns:macAddress', '?', ns)}")
    except Exception:
        print(f"📱 Device at {HIKVISION_IP} (could not parse info)")


# ─── User Management ────────────────────────────────────────────────────────

def get_enrolled_users() -> list[dict]:
    """Fetch all enrolled users from the device."""
    users = []
    pos = 0
    while True:
        body = {
            "UserInfoSearchCond": {
                "searchID": "enrollment_check",
                "searchResultPosition": pos,
                "maxResults": 30,
            }
        }
        status, data = _api_json(
            "post", "/ISAPI/AccessControl/UserInfo/Search?format=json", json=body
        )
        if status != 200:
            break
        info = data.get("UserInfoSearch", {})
        total = int(info.get("totalMatches", 0))
        records = info.get("UserInfo", [])
        if isinstance(records, dict):
            records = [records]
        users.extend(records)
        if len(users) >= total or not records:
            break
        pos += len(records)
    return users


def create_user(employee_no: str, name: str, student_id: str = ""):
    """Create a user record on the device."""
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
    if student_id:
        user_data["UserInfo"]["PersonInfoExtends"] = [{"value": student_id}]

    status, data = _api_json(
        "post", "/ISAPI/AccessControl/UserInfo/Record?format=json", json=user_data
    )
    ok = status == 200 and data.get("statusCode") == 1
    return ok, data


def upload_face_fdlib(employee_no: str, name: str, face_url: str) -> tuple[bool, dict]:
    """Upload a face image to the device's FDLib (Face Database Library).

    The device downloads the image from `face_url`, extracts its own DeepLearn
    embeddings, and links the face record to the user via FPID=employeeNo.

    Args:
        employee_no: The user's employee number (must match an existing user)
        name: Display name for the face record
        face_url: HTTP URL the device can fetch the face image from

    Returns:
        (success, response_data) tuple
    """
    payload = {
        "faceLibType": "blackFD",
        "FDID": "1",
        "FPID": employee_no,
        "name": name,
        "faceURL": face_url,
    }
    status, data = _api_json(
        "put", "/ISAPI/Intelligent/FDLib/FDSetUp?format=json", json=payload
    )
    ok = status == 200 and data.get("statusCode") == 1
    return ok, data


def delete_user(employee_no: str):
    """Delete a user from the device."""
    body = {"UserInfoDelCond": {"EmployeeNoList": [{"employeeNo": employee_no}]}}
    status, data = _api_json(
        "put", "/ISAPI/AccessControl/UserInfo/Delete?format=json", json=body
    )
    return status == 200 and data.get("statusCode") == 1


def delete_face_fdlib(employee_no: str):
    """Delete a face record from the FDLib."""
    payload = {
        "faceLibType": "blackFD",
        "FDID": "1",
        "FPID": employee_no,
    }
    status, data = _api_json(
        "put", "/ISAPI/Intelligent/FDLib/FDDelete?format=json", json=payload
    )
    return status == 200


def capture_face_from_device(output_path: str, timeout: int = 60) -> bool:
    """Capture a face image directly from the device's camera.

    Calls POST /ISAPI/AccessControl/CaptureFaceData which blocks until the
    device detects a face in front of its camera.  Returns a face-optimized
    352×432 JPEG image.

    Args:
        output_path: File path to save the captured JPEG.
        timeout: HTTP request timeout in seconds (device blocks until face detected).

    Returns:
        True if a face was captured and saved, False otherwise.
    """
    xml_body = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<CaptureFaceDataCond xmlns="http://www.isapi.org/ver20/XMLSchema" version="2.0">'
        '<captureInfrared>false</captureInfrared>'
        '<dataType>binary</dataType>'
        '</CaptureFaceDataCond>'
    )
    try:
        r = _api(
            "post",
            "/ISAPI/AccessControl/CaptureFaceData",
            data=xml_body,
            headers={"Content-Type": "application/xml"},
            timeout=timeout,
        )
    except requests.exceptions.Timeout:
        print("   ⏰ Capture timed out — no face detected within timeout period.")
        return False
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Capture request failed: {e}")
        return False

    if r.status_code != 200:
        print(f"   ❌ Device returned HTTP {r.status_code}")
        return False

    # Parse multipart response: boundary=MIME_boundary, contains XML + JPEG
    content = r.content
    jpeg_start = content.find(b"\xff\xd8")
    jpeg_end = content.rfind(b"\xff\xd9")
    if jpeg_start == -1 or jpeg_end == -1:
        print("   ❌ No JPEG data found in device response.")
        return False

    jpeg_data = content[jpeg_start : jpeg_end + 2]
    with open(output_path, "wb") as f:
        f.write(jpeg_data)

    # Verify it's a valid image
    img = cv2.imread(output_path)
    if img is None:
        print("   ❌ Captured image is corrupt.")
        return False

    h, w = img.shape[:2]
    print(f"   📸 Captured {w}×{h} face image ({len(jpeg_data):,} bytes)")
    return True


def _upload_face_to_firebase(local_path: str, student_name: str, homeroom: str):
    """Upload a captured face image to Firebase Storage as backup."""
    try:
        from firebase_dataset_sync import initialize_firebase_app
        import firebase_admin
        from firebase_admin import storage as fb_storage

        app = initialize_firebase_app(FIREBASE_CREDENTIALS, FIREBASE_BUCKET)
        bucket = fb_storage.bucket(app=app)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        blob_path = f"face_dataset/{homeroom}/{student_name}/{ts}_device_capture.jpg"
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(local_path, content_type="image/jpeg")
        print(f"   ☁️  Backed up to Firebase: {blob_path}")
    except Exception as e:
        print(f"   ⚠️  Firebase backup failed (non-fatal): {e}")


def _enroll_one_student(
    name: str,
    student_id: str,
    homeroom: str,
    employee_no: str,
    existing: dict,
    serve_dir: Path,
    base_url: str,
) -> bool:
    """Enroll a single student using the device camera. Returns True on success."""
    print(f"\n{'─'*50}")
    print(f"👤 {name}")
    print(f"   Student ID: {student_id or 'n/a'} │ Homeroom: {homeroom} │ Device ID: {employee_no}")

    # Create user if not exists
    if employee_no in existing:
        current_faces = existing[employee_no].get("numOfFace", 0)
        if current_faces > 0:
            print(f"   ℹ️  Already enrolled with {current_faces} face(s), skipping.")
            return True
        else:
            print(f"   ℹ️  User exists, capturing face...")
    else:
        ok, resp = create_user(employee_no, name, student_id)
        if not ok:
            print(f"   ❌ Failed to create user: {resp}")
            return False
        print(f"   ✅ User created on device.")

    # Capture face from device camera — starts immediately
    print(f"\n   📷 {name}, please look at the device camera now...")
    capture_path = str(serve_dir / f"{employee_no}_face.jpg")
    print(f"   ⏳ Waiting for face (up to 60s)...")
    ok = capture_face_from_device(capture_path, timeout=60)
    if not ok:
        print(f"   ❌ Face capture failed for {name}.")
        return False

    # Upload captured face to device FDLib
    face_url = f"{base_url}/{employee_no}_face.jpg"
    ok, resp = upload_face_fdlib(employee_no, name, face_url)
    if ok:
        print(f"   ✅ Face enrolled on device!")
        # Backup to Firebase
        _upload_face_to_firebase(capture_path, name, homeroom)
        return True
    else:
        err = resp.get("subStatusCode", resp.get("errorMsg", "unknown"))
        print(f"   ❌ Device rejected face: {err}")
        return False


def enroll_live():
    """Enroll a single student by Binus Student ID using the device camera.

    Flow:
        1. Ask for student ID
        2. Call Binus API → get name, grade, homeroom
        3. Create user on device
        4. Capture face via device camera (CaptureFaceData)
        5. Upload face to FDLib + Firebase backup
    """
    with tempfile.TemporaryDirectory(prefix="hik_live_") as tmp:
        serve_dir = Path(tmp) / "serve"
        serve_dir.mkdir(parents=True, exist_ok=True)

        print("🌐 Starting face image server...")
        server, thread, base_url = _start_face_server(str(serve_dir), FACE_SERVER_PORT)
        print(f"   Serving from: {base_url}\n")

        try:
            existing = {u["employeeNo"]: u for u in get_enrolled_users()}
            enrolled = 0
            failed = 0

            token = _binus_get_token()

            print("Enter student IDs to enroll (empty line to finish):")
            while True:
                sid = input("\n  📋 Student ID (Binus): ").strip()
                if not sid:
                    break

                # Lookup via Binus API
                print(f"   🔍 Looking up student {sid}...")
                data = binus_lookup_student(sid, token=token)
                if data is None:
                    print(f"   ❌ Student {sid} not found in Binus system.")
                    failed += 1
                    continue

                name = _parse_student_name(data)
                homeroom = _parse_homeroom(data)
                grade_code = data.get("gradeCode", "")
                grade_name = data.get("gradeName", "")
                employee_no = _employee_no_from_name(name)

                print(f"   ✅ Found: {name}")
                print(f"      Grade: {grade_name} ({grade_code}) │ Homeroom: {homeroom}")

                ok = _enroll_one_student(
                    name=name,
                    student_id=sid,
                    homeroom=homeroom,
                    employee_no=employee_no,
                    existing=existing,
                    serve_dir=serve_dir,
                    base_url=base_url,
                )
                if ok:
                    enrolled += 1
                    existing[employee_no] = {"employeeNo": employee_no, "numOfFace": 1}
                    # Save metadata mapping for attendance pipeline
                    # Try to get idBinusian from class list lookup
                    id_binusian = ""
                    try:
                        photos = binus_get_class_students(
                            grade_code or "1", homeroom, token=token
                        )
                        for p in (photos or []):
                            if str(p.get("idStudent", "")) == sid:
                                id_binusian = str(p.get("idBinusian", ""))
                                break
                    except Exception:
                        pass
                    student_metadata.save_student(
                        employee_no=employee_no,
                        name=name,
                        id_student=sid,
                        id_binusian=id_binusian,
                        homeroom=homeroom,
                        grade=grade_code or "",
                    )
                    print(f"   📝 Metadata saved (ID:{sid}, BN:{id_binusian or 'pending'})")
                else:
                    failed += 1

        finally:
            _stop_face_server(server)
            print(f"\n🌐 Face server stopped.")

        _print_enrollment_summary(enrolled, failed)


def enroll_class(grade: str, homeroom: str):
    """Enroll an entire class by grade and homeroom using the Binus API.

    Flow:
        1. Call Binus API → get full student list for the class
        2. For each student, create user → capture face → enroll
    """
    print(f"📚 Fetching student list for Grade {grade}, Homeroom {homeroom}...")
    token = _binus_get_token()
    students = binus_get_class_students(grade, homeroom, token=token)

    if not students:
        print("❌ No students found for this class.")
        return

    print(f"✅ Found {len(students)} student(s):\n")
    for i, s in enumerate(students, 1):
        sid = s.get("idStudent", "?")
        name = s.get("studentName", "") or s.get("fileName", "?")
        print(f"   {i:3d}. {name} (ID: {sid})")

    confirm = input(f"\n🎯 Enroll all {len(students)} students? [y/N]: ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        return

    with tempfile.TemporaryDirectory(prefix="hik_class_") as tmp:
        serve_dir = Path(tmp) / "serve"
        serve_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n🌐 Starting face image server...")
        server, thread, base_url = _start_face_server(str(serve_dir), FACE_SERVER_PORT)

        try:
            existing = {u["employeeNo"]: u for u in get_enrolled_users()}
            enrolled = 0
            failed = 0

            for i, s in enumerate(students, 1):
                sid = str(s.get("idStudent", ""))
                # Get full student info via C2 endpoint for accurate name
                full_data = binus_lookup_student(sid, token=token)
                if full_data:
                    name = _parse_student_name(full_data)
                    hr = _parse_homeroom(full_data)
                else:
                    # Fallback to class list data
                    name = s.get("studentName", "") or s.get("fileName", f"Student_{sid}")
                    hr = homeroom

                employee_no = _employee_no_from_name(name)

                print(f"\n[{i}/{len(students)}]")
                ok = _enroll_one_student(
                    name=name,
                    student_id=sid,
                    homeroom=hr,
                    employee_no=employee_no,
                    existing=existing,
                    serve_dir=serve_dir,
                    base_url=base_url,
                )
                if ok:
                    enrolled += 1
                    existing[employee_no] = {"employeeNo": employee_no, "numOfFace": 1}
                    # Save metadata mapping for attendance pipeline
                    id_binusian = str(s.get("idBinusian", ""))
                    student_metadata.save_student(
                        employee_no=employee_no,
                        name=name,
                        id_student=sid,
                        id_binusian=id_binusian,
                        homeroom=hr,
                        grade=grade,
                    )
                    print(f"   📝 Metadata saved (ID:{sid}, BN:{id_binusian or 'n/a'})")
                else:
                    failed += 1

        finally:
            _stop_face_server(server)
            print(f"\n🌐 Face server stopped.")

        _print_enrollment_summary(enrolled, failed)


def _print_enrollment_summary(enrolled: int, failed: int):
    """Print enrollment totals and FDLib counts."""
    print(f"\n{'='*50}")
    print(f"✅ Enrolled: {enrolled}  ❌ Failed: {failed}")
    counts = get_fdlib_count()
    for lib in counts:
        print(f"   FDLib {lib['FDID']} ({lib['faceLibType']}): "
              f"{lib['recordDataNumber']} face record(s)")
    print(f"\n🎉 Done! The device will recognise these students on its LCD screen!")


def clear_all_users():
    """Remove all users and their face data from the device."""
    users = get_enrolled_users()
    if not users:
        print("ℹ️  No users on device.")
        return

    print(f"🗑️  Removing {len(users)} users from device...")
    for u in users:
        eno = u.get("employeeNo", "")
        name = u.get("name", "?")
        # Delete face from FDLib first, then user record
        delete_face_fdlib(eno)
        ok = delete_user(eno)
        status_icon = "✅" if ok else "❌"
        print(f"  {status_icon} {name} ({eno})")
    print("Done.")


# ─── FDLib Status ────────────────────────────────────────────────────────────

def get_fdlib_count() -> list[dict]:
    """Get face record counts per FDLib."""
    status, data = _api_json("get", "/ISAPI/Intelligent/FDLib/Count?format=json")
    if status == 200:
        return data.get("FDRecordDataInfo", [])
    return []


def get_fdlib_records(fdid: str = "1") -> list[dict]:
    """Search all face records in a given FDLib."""
    payload = {
        "searchResultPosition": 0,
        "maxResults": 100,
        "FDID": fdid,
        "faceLibType": "blackFD",
    }
    status, data = _api_json(
        "post", "/ISAPI/Intelligent/FDLib/FDSearch?format=json", json=payload
    )
    if status == 200:
        return data.get("MatchList", [])
    return []


# ─── Firebase → Device Enrollment Pipeline ──────────────────────────────────

def enroll_from_firebase():
    """Download face images from Firebase Storage, crop faces, and enroll
    them on the Hikvision device via the FDLib API.

    Uses real BINUS Student IDs (from Firebase `students` collection) as the
    device employeeNo, so that attendance events directly carry the IdStudent
    needed for the BINUS e-Desk API.

    Pipeline:
        1. Load student metadata from Firebase Firestore (`students` collection)
        2. Batch-fetch IdBinusian from BINUS C.1 API per class
        3. Download face images from Firebase Storage
        4. Detect and crop faces using dlib
        5. Start temporary HTTP server to serve cropped faces
        6. Create user on device with IdStudent as employeeNo
        7. Upload face via FDLib → device downloads & extracts embeddings
        8. Save full metadata (IdStudent, IdBinusian, name, class) for attendance
    """
    try:
        from firebase_dataset_sync import sync_face_dataset_from_firebase, initialize_firebase_app
        from firebase_admin import firestore as fb_firestore
    except ImportError:
        print("❌ firebase_admin is required. Install it (pip install firebase-admin).")
        return

    # ── Step 1: Load student metadata from Firebase Firestore ──
    print("📋 Loading student metadata from Firebase Firestore...")
    try:
        app = initialize_firebase_app(FIREBASE_CREDENTIALS, FIREBASE_BUCKET)
        db = fb_firestore.client(app=app)
        docs = db.collection("students").stream()
        fb_students = {}  # name → {id, homeroom, gradeCode, ...}
        for doc in docs:
            d = doc.to_dict()
            sid = d.get("id", "")
            name = d.get("name", "")
            # Skip test/manual entries
            if not sid or not name or sid.startswith("TEST") or sid.startswith("MANUAL"):
                continue
            fb_students[name.strip()] = {
                "idStudent": str(sid),
                "homeroom": d.get("homeroom", ""),
                "gradeCode": d.get("gradeCode", ""),
                "gradeName": d.get("gradeName", ""),
            }
        print(f"   {len(fb_students)} student(s) with BINUS IDs in Firestore")
        for name, info in fb_students.items():
            print(f"   • {name}: ID={info['idStudent']}, Class={info['homeroom']}")
    except Exception as e:
        print(f"⚠️  Could not load student metadata from Firestore: {e}")
        fb_students = {}

    # ── Step 2: Batch-fetch IdBinusian from BINUS C.1 API per class ──
    print("\n🔑 Fetching IdBinusian from BINUS API...")
    id_binusian_map = {}  # idStudent → idBinusian
    _binus_token = None
    try:
        _binus_token = _binus_get_token()
    except Exception:
        pass

    if _binus_token and fb_students:
        # Group students by (gradeCode, homeroom) for batch lookup
        class_groups = {}
        for name, info in fb_students.items():
            grade = info.get("gradeCode", "")
            hr = info.get("homeroom", "")
            if grade and hr:
                key = (grade, hr)
                if key not in class_groups:
                    class_groups[key] = []
                class_groups[key].append((name, info["idStudent"]))

        for (grade, hr), members in class_groups.items():
            print(f"   📚 Grade {grade}, Homeroom {hr} ({len(members)} student(s))...")
            try:
                class_students = binus_get_class_students(grade, hr, token=_binus_token)
                for cs in (class_students or []):
                    cs_id = str(cs.get("idStudent", ""))
                    cs_bn = str(cs.get("idBinusian", ""))
                    if cs_id and cs_bn:
                        id_binusian_map[cs_id] = cs_bn
                found = sum(1 for _, sid in members if sid in id_binusian_map)
                print(f"      ✓ {found}/{len(members)} IdBinusian found")
            except Exception as e:
                print(f"      ⚠️  Lookup failed: {e}")
    else:
        print("   ⚠️  BINUS API unavailable — IdBinusian will be empty")

    print(f"   Total IdBinusian mapped: {len(id_binusian_map)}\n")

    # ── Step 3: Download face images from Firebase Storage ──
    with tempfile.TemporaryDirectory(prefix="hik_enroll_") as tmp:
        dataset_root = Path(tmp) / "face_dataset"
        serve_dir = Path(tmp) / "serve"
        dataset_root.mkdir(parents=True, exist_ok=True)
        serve_dir.mkdir(parents=True, exist_ok=True)

        print("📥 Downloading face images from Firebase Storage...")
        stats = sync_face_dataset_from_firebase(
            destination_root=dataset_root,
            credentials_path=FIREBASE_CREDENTIALS,
            storage_bucket=FIREBASE_BUCKET,
            skip_existing=True,
        )
        print(f"   {stats}")

        # ── Step 4: Start HTTP server ──
        print(f"\n🌐 Starting face image server on port {FACE_SERVER_PORT}...")
        server, thread, base_url = _start_face_server(str(serve_dir), FACE_SERVER_PORT)
        print(f"   Serving from: {base_url}")

        try:
            existing = {u["employeeNo"]: u for u in get_enrolled_users()}
            print(f"📋 Currently {len(existing)} user(s) on device.\n")

            enrolled = 0
            failed = 0

            for student_dir in sorted(dataset_root.iterdir()):
                if not student_dir.is_dir():
                    continue
                student_name = student_dir.name
                homeroom = ""

                # ── Resolve BINUS Student ID ──
                # Priority: Firebase Firestore metadata > filename extraction > skip
                fb_info = fb_students.get(student_name)
                if fb_info:
                    student_id = fb_info["idStudent"]
                    homeroom = fb_info.get("homeroom", "")
                else:
                    # Try to extract from image filename (e.g. 2070003324_front_...)
                    student_id = ""
                    for img_p in student_dir.iterdir():
                        sid = _extract_student_id(img_p.name)
                        if sid:
                            student_id = sid
                            break

                if not student_id:
                    print(f"⚠️  {student_name}: No BINUS Student ID found — skipping.")
                    print(f"     → Capture photos via the web app first (enter student ID)")
                    failed += 1
                    continue

                # Use the real BINUS Student ID as device employeeNo
                employee_no = str(student_id)
                id_binusian = id_binusian_map.get(student_id, "")

                # Gather and rank images
                raw_images = list(student_dir.glob("*.jp*")) + list(student_dir.glob("*.png"))
                if not raw_images:
                    print(f"⚠️  {student_name}: no images found, skipping.")
                    failed += 1
                    continue
                ranked = rank_images_by_frontality(raw_images)
                images = [p for p, _ in ranked]
                best_score = ranked[0][1] if ranked else 0

                print(f"👤 {student_name}")
                print(f"   IdStudent: {student_id} │ IdBinusian: {id_binusian or 'n/a'} │ Class: {homeroom or 'n/a'}")
                print(f"   Images: {len(images)} │ Best frontality: {best_score:.2f}")

                # ── Step 5: Create user on device ──
                if employee_no in existing:
                    current_faces = existing[employee_no].get("numOfFace", 0)
                    if current_faces > 0:
                        print(f"   ℹ️  Already enrolled with {current_faces} face(s), skipping.")
                        # Still save/update metadata
                        student_metadata.save_student(
                            employee_no=employee_no,
                            name=student_name,
                            id_student=student_id,
                            id_binusian=id_binusian,
                            homeroom=homeroom,
                        )
                        enrolled += 1
                        continue
                    print(f"   ℹ️  User exists, uploading face...")
                else:
                    ok, resp = create_user(employee_no, student_name, student_id)
                    if not ok:
                        print(f"   ❌ Failed to create user: {resp}")
                        failed += 1
                        continue
                    print(f"   ✅ User created on device (employeeNo={employee_no})")

                # ── Step 6: Crop face, serve via HTTP, upload to FDLib ──
                face_ok = False
                for img_path in images:
                    filename = f"{employee_no}_face.jpg"
                    out_path = serve_dir / filename
                    face_found = crop_face(str(img_path), str(out_path))

                    img_score = next((s for p, s in ranked if p == img_path), 0)
                    if face_found:
                        print(f"   📷 Face in {img_path.name} (frontality: {img_score:.2f}), uploading...")
                    else:
                        print(f"   ⚠️  No face in {img_path.name}, using fallback resize...")

                    face_url = f"{base_url}/{filename}"
                    ok, resp = upload_face_fdlib(employee_no, student_name, face_url)
                    if ok:
                        print(f"   ✅ Face enrolled on device!")
                        face_ok = True
                        break
                    else:
                        err = resp.get("subStatusCode", resp.get("errorMsg", "unknown"))
                        print(f"   ⚠️  Device rejected ({err}), trying next image...")

                if face_ok:
                    enrolled += 1
                    # ── Step 7: Save metadata for attendance pipeline ──
                    student_metadata.save_student(
                        employee_no=employee_no,
                        name=student_name,
                        id_student=student_id,
                        id_binusian=id_binusian,
                        homeroom=homeroom,
                        grade=fb_info.get("gradeCode", "") if fb_info else "",
                    )
                    print(f"   📝 Metadata: IdStudent={student_id}, IdBinusian={id_binusian or 'pending'}")
                else:
                    print(f"   ❌ None of {len(images)} images accepted by device.")
                    failed += 1

        finally:
            _stop_face_server(server)
            print(f"\n🌐 Face server stopped.")

        # Summary
        print(f"\n{'='*50}")
        print(f"✅ Enrolled: {enrolled}  ❌ Failed: {failed}")

        # Verify FDLib count
        counts = get_fdlib_count()
        for lib in counts:
            print(f"   FDLib {lib['FDID']} ({lib['faceLibType']}): "
                  f"{lib['recordDataNumber']} face record(s)")

        print(f"\n🎉 The device will now recognise these students on its LCD screen!")


# ─── Attendance Event Monitoring ─────────────────────────────────────────────

_FIREBASE_APP = None


def _get_firestore_client():
    """Get (or create) a Firestore client for attendance uploads."""
    global _FIREBASE_APP
    try:
        from firebase_dataset_sync import initialize_firebase_app
        from firebase_admin import firestore as fb_firestore

        if _FIREBASE_APP is None:
            _FIREBASE_APP = initialize_firebase_app(FIREBASE_CREDENTIALS, FIREBASE_BUCKET)
        return fb_firestore.client(app=_FIREBASE_APP)
    except Exception as e:
        print(f"⚠️  Firebase Firestore unavailable: {e}")
        return None


def _upload_attendance_record(db, name: str, employee_no: str,
                              event_time: str, status: str):
    """Upload a single attendance record to Firestore."""
    if db is None:
        return
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        doc_id = f"{employee_no}_{today}"
        record = {
            "name": name,
            "employeeNo": employee_no,
            "time": event_time,
            "date": today,
            "status": status,
            "source": "hikvision_terminal",
            "device": f"DS-K1T341AMF@{HIKVISION_IP}",
            "timestamp": datetime.now().isoformat(),
        }
        db.collection("attendance").document(doc_id).set(record, merge=True)
        print(f"   ☁️  Synced to Firebase: {name} → {status}")
    except Exception as e:
        print(f"   ⚠️  Firebase sync error: {e}")


def _determine_status(event_time_str: str) -> str:
    """Determine Present/Late based on time.  Cutoff is 08:15."""
    try:
        dt = datetime.fromisoformat(event_time_str.replace("Z", "+00:00"))
        cutoff = dt.replace(hour=8, minute=15, second=0, microsecond=0)
        return "Present" if dt <= cutoff else "Late"
    except Exception:
        return "Present"


def poll_events(since_serial: int = 0, max_results: int = 30):
    """Poll the latest access events from the device."""
    body = {
        "AcsEventCond": {
            "searchID": "attendance_poll",
            "searchResultPosition": 0,
            "maxResults": max_results,
            "major": 0,
            "minor": 0,
            "timeReverseOrder": True,
        }
    }
    if since_serial > 0:
        body["AcsEventCond"]["beginSerialNo"] = since_serial

    status, data = _api_json(
        "post", "/ISAPI/AccessControl/AcsEvent?format=json", json=body
    )
    if status != 200:
        return []

    info = data.get("AcsEvent", {})
    events = info.get("InfoList", [])
    if isinstance(events, dict):
        events = [events]
    return events


def monitor_attendance():
    """Continuously poll the device for face recognition events and sync to Firebase."""
    print(f"👁️  Monitoring attendance events from {HIKVISION_IP}...")
    print(f"   Poll interval: {POLL_INTERVAL}s")
    print(f"   Press Ctrl+C to stop.\n")

    db = _get_firestore_client()
    seen_serials = set()
    today = datetime.now().strftime("%Y-%m-%d")
    logged_today = set()

    # Build name lookup from device users
    users = get_enrolled_users()
    name_map = {u.get("employeeNo", ""): u.get("name", "Unknown") for u in users}
    print(f"   📋 {len(name_map)} enrolled user(s) on device.\n")

    try:
        while True:
            events = poll_events()
            for ev in events:
                serial = ev.get("serialNo", 0)
                if serial in seen_serials:
                    continue
                seen_serials.add(serial)

                emp_no = ev.get("employeeNoString", "") or str(ev.get("employeeNo", ""))
                name = ev.get("name", "") or name_map.get(emp_no, "")
                event_time = ev.get("time", "")

                if not emp_no or not name:
                    continue

                current_date = datetime.now().strftime("%Y-%m-%d")
                if current_date != today:
                    today = current_date
                    logged_today.clear()

                if emp_no in logged_today:
                    continue

                logged_today.add(emp_no)
                att_status = _determine_status(event_time)

                print(f"✅ {event_time} │ {name} │ {att_status}")
                _upload_attendance_record(db, name, emp_no, event_time, att_status)

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print(f"\n⏹️  Stopped. Logged {len(logged_today)} attendance records today.")


# ─── Status Display ──────────────────────────────────────────────────────────

def show_status():
    """Show device info, enrolled users, and FDLib face records."""
    get_device_info()
    print()

    # FDLib counts
    counts = get_fdlib_count()
    if counts:
        print("📚 Face Database Libraries:")
        for lib in counts:
            print(f"   FDID={lib['FDID']} ({lib['faceLibType']}): "
                  f"{lib['recordDataNumber']} record(s)")
        print()

    # Enrolled users
    users = get_enrolled_users()
    if not users:
        print("📋 No users enrolled on device.")
        return

    print(f"📋 {len(users)} user(s) enrolled:")
    for u in users:
        eno = u.get("employeeNo", "?")
        name = u.get("name", "?")
        faces = u.get("numOfFace", 0)
        face_icon = "🟢" if faces > 0 else "🔴"
        print(f"   {face_icon} {name:30s} │ ID: {eno} │ Faces: {faces}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Hikvision DS-K1T341AMF Attendance System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("enroll", help="Push Firebase faces to device")
    sub.add_parser("enroll-live", help="Enroll by student ID (Binus API → device camera)")

    ec = sub.add_parser("enroll-class", help="Enroll entire class (Binus API → device camera)")
    ec.add_argument("grade", help="Grade level, e.g. 1, 2, EY1, EY2")
    ec.add_argument("homeroom", help="Homeroom code, e.g. 1A, 2B")

    sub.add_parser("monitor", help="Poll attendance events → Firebase sync")
    sub.add_parser("status", help="Show enrolled users on device")
    sub.add_parser("clear", help="Remove all users from device")

    args = parser.parse_args()

    if args.command == "enroll":
        enroll_from_firebase()
    elif args.command == "enroll-live":
        enroll_live()
    elif args.command == "enroll-class":
        enroll_class(args.grade, args.homeroom)
    elif args.command == "monitor":
        monitor_attendance()
    elif args.command == "status":
        show_status()
    elif args.command == "clear":
        confirm = input(
            "⚠️  This will delete ALL users from the device. Type 'yes' to confirm: "
        )
        if confirm.strip().lower() == "yes":
            clear_all_users()
        else:
            print("Cancelled.")


if __name__ == "__main__":
    main()
