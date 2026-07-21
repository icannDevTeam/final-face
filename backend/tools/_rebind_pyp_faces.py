#!/usr/bin/env python3
"""
_rebind_pyp_faces.py — Plan A: re-push face data for users already on PYP device.

Reads the 99 existing users on PYP Lobby (10.26.30.59), looks up each
employeeNo in local student metadata + binus_students_all.json, downloads
the photo if needed, and pushes the face via FDLib/FDSetUp.

Does NOT create or delete any user records — only binds faces to existing users.
"""
import os, sys, json, time, socket, threading
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from batch_enroll import DeviceClient, download_photo, emp_no_from_name
from face_photo_utils import ensure_device_safe

# Firebase Storage fallback
import firebase_admin
from firebase_admin import credentials, storage as fb_storage
if not firebase_admin._apps:
    firebase_admin.initialize_app(
        credentials.Certificate(str(Path(__file__).parent.parent / "facial-attendance-binus-firebase-adminsdk.json")),
        {"storageBucket": "facial-attendance-binus.firebasestorage.app"},
    )
_bucket = fb_storage.bucket()

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
FACES_DIR = DATA_DIR / "faces_batch"
FACES_DIR.mkdir(exist_ok=True)

DEFAULT_DEVICE = "PYP Lobby Entrance (DS-K1T342MFX)"
FACE_SERVER_PORT = 8899

DRY = "--dry-run" in sys.argv

# --device <name-substring>  selects which device to rebind
sel = DEFAULT_DEVICE
for i, a in enumerate(sys.argv):
    if a == "--device" and i + 1 < len(sys.argv):
        sel = sys.argv[i + 1]

# Resolve password + IP
local_devs = json.loads((HERE / "devices.local.json").read_text())
config_devs = {d["name"]: d for d in json.loads((HERE / "devices.json").read_text())}
match = [d for d in local_devs if sel.lower() in d["name"].lower()]
if not match:
    raise SystemExit(f"❌ No device matches --device '{sel}'. Known: {[d['name'] for d in local_devs]}")
local_match = match[0]
DEVICE_NAME = local_match["name"]
pw = local_match["password"]
DEVICE_IP = config_devs.get(DEVICE_NAME, {}).get("ip")
if not DEVICE_IP:
    raise SystemExit(f"❌ No IP in devices.json for '{DEVICE_NAME}'")
print(f"🎯 Target device: {DEVICE_NAME} @ {DEVICE_IP}")

# Load metadata: employeeNo -> info  (also build name -> info for fallback)
meta = json.loads((DATA_DIR / "student_metadata.json").read_text())
by_emp = {}
by_name = {}
for emp, m in meta.items():
    by_emp[emp.upper()] = m
    n = (m.get("name") or "").strip().lower()
    if n:
        by_name[n] = (emp, m)

# Load BINUS report for photo URLs
report = json.loads((DATA_DIR / "binus_students_all.json").read_text())
binus_by_id = {s.get("idStudent"): s for s in report.get("students", []) if s.get("idStudent")}
binus_by_name = {(s.get("name") or "").strip().lower(): s
                 for s in report.get("students", []) if s.get("name")}


def find_photo_url(emp, name):
    """Return (url, source) or (None, reason). Tries BINUS report first, then Firebase Storage."""
    info = by_emp.get(emp.upper())
    if info:
        sid = info.get("idStudent") or info.get("studentId")
        if sid and sid in binus_by_id:
            url = (binus_by_id[sid].get("filePath") or "").strip()
            if url:
                return url, f"binus({sid})"
    # Fall back by name in BINUS report
    nlow = (name or "").strip().lower()
    if nlow and nlow in binus_by_name:
        url = (binus_by_name[nlow].get("filePath") or "").strip()
        if url:
            return url, "binus-by-name"
    return None, "no-binus-url"


def find_storage_blob(emp, name):
    """Look up Firebase Storage face_dataset/<homeroom>/<name>/photo_*.jpg. Returns blob or None."""
    info = by_emp.get(emp.upper())
    hr = (info or {}).get("homeroom", "")
    if not hr or not name:
        return None
    prefix = f"face_dataset/{hr}/{name}/"
    blobs = [b for b in _bucket.list_blobs(prefix=prefix, max_results=10)
             if b.size and b.size > 1000]
    if not blobs:
        return None
    # Prefer largest (usually highest quality)
    blobs.sort(key=lambda b: -b.size)
    return blobs[0]


def start_face_server(serve_dir):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=str(serve_dir), **k)
        def log_message(self, *a, **k): pass
    srv = HTTPServer(("0.0.0.0", FACE_SERVER_PORT), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect((DEVICE_IP, 80)); ip = s.getsockname()[0]; s.close()
    return srv, f"http://{ip}:{FACE_SERVER_PORT}"


def main():
    dev = DeviceClient(DEVICE_IP, "admin", pw)
    users = dev.get_enrolled_users()
    print(f"📊 PYP has {len(users)} users; {sum(1 for u in users.values() if u.get('numOfFace',0)>0)} already have faces")

    plan = []                # (emp, name, kind, ref, source) — Firebase Storage ONLY
    skip_no_photo = []
    skip_already_face = []
    for emp, u in users.items():
        if u.get("numOfFace", 0) > 0:
            skip_already_face.append((emp, u["name"]))
            continue
        blob = find_storage_blob(emp, u.get("name", ""))
        if blob:
            plan.append((emp, u["name"], "storage", blob.name, f"storage:{blob.name}"))
            continue
        skip_no_photo.append((emp, u.get("name", ""), "no-storage-photo"))

    print(f"\n🎯 To re-bind: {len(plan)}")
    print(f"⏭️  Already have face: {len(skip_already_face)}")
    print(f"⚠️  No photo URL found: {len(skip_no_photo)}")
    if skip_no_photo:
        print("\n  Missing-photo users (will be left without face):")
        for emp, name, reason in skip_no_photo:
            print(f"    - {emp:10s} {name:40s} ({reason})")

    if DRY:
        print("\n--- DRY RUN — preview only ---")
        for emp, name, kind, ref, src in plan[:30]:
            print(f"  + {emp:10s} {name:40s} [{kind}] {src}")
        if len(plan) > 30:
            print(f"  ... ({len(plan)-30} more)")
        print(f"\nTotal would-rebind: {len(plan)}")
        return

    # Download / cache all photos
    print(f"\n📥 Ensuring {len(plan)} photos cached...")
    dl_ok = dl_skip = dl_fail = 0
    for emp, name, kind, ref, _ in plan:
        dest = FACES_DIR / f"{emp.upper()}.jpg"
        if dest.exists() and dest.stat().st_size > 1000:
            ensure_device_safe(dest)
            dl_skip += 1; continue
        if kind == "url":
            ok = download_photo(ref, str(dest))
        else:  # storage
            try:
                _bucket.blob(ref).download_to_filename(str(dest))
                ok = dest.exists() and dest.stat().st_size > 1000
            except Exception as e:
                print(f"  ⚠ storage download failed for {emp}: {e}")
                ok = False
        if ok:
            ensure_device_safe(dest)
            dl_ok += 1
        else: dl_fail += 1
    print(f"  cached: ok={dl_ok} skip-already={dl_skip} fail={dl_fail}")

    srv, base = start_face_server(FACES_DIR)
    print(f"🌐 Face server: {base}")

    ok = fail = 0
    err = ""
    for i, (emp, name, kind, ref, src) in enumerate(plan, 1):
        emp_up = emp.upper()
        photo = FACES_DIR / f"{emp_up}.jpg"
        if not photo.exists():
            print(f"[{i}/{len(plan)}] {name:40s} ❌ photo file missing")
            fail += 1; continue
        face_url = f"{base}/{emp_up}.jpg"
        success = False
        for attempt in range(3):
            res, data = dev.upload_face(emp, name, face_url)
            if res:
                success = True; break
            err = data.get("subStatusCode") or data.get("errorMsg") or str(data)[:120]
            if attempt < 2:
                time.sleep(2)
        if success:
            print(f"[{i}/{len(plan)}] {name:40s} ✅ ({kind})")
            ok += 1
        else:
            print(f"[{i}/{len(plan)}] {name:40s} ❌ {err}")
            fail += 1
        time.sleep(0.8)
        if i % 20 == 0:
            print(f"  ⏸  batch pause ({ok} ok, {fail} fail so far)")
            time.sleep(4)

    srv.shutdown()

    print(f"\n=== Done: {ok} bound, {fail} failed ===")
    # Use FDSearch (authoritative) rather than UserInfo/Count (counter often stale).
    body = {"searchResultPosition": 0, "maxResults": 1,
            "faceLibType": "blackFD", "FDID": "1"}
    _, fd = dev.api_json("post", "/ISAPI/Intelligent/FDLib/FDSearch?format=json",
                         json=body, timeout=20)
    total = fd.get("totalMatches", "?")
    print(f"📊 Final FDSearch: {total} faces in library")


if __name__ == "__main__":
    main()
