"""
unknown_faces.py — Shared face-capture sink for review / re-enrollment.

Whenever a Hikvision face event is suspicious — either the device couldn't
match the face to an enrolled user, our blocklist intercepted a known false
match, or the chaperone/student record is missing/stale on our side — we
keep the JPEG so an admin can identify the person and (re-)enroll them.

Storage layout:
  Local : backend/data/unknown_faces/<YYYY-MM-DD>/<HHMMSS>_<device>_<kind>_NNNN.{jpg,json}
  Cloud : Storage : unknown_faces/<YYYY-MM-DD>/<file>.jpg
          Firestore: tenants/<tid>/unknownFaces/<id>  (status='pending_review')

Both attendance_listener.py and pickup_event_writer.py call into this
module so a chaperone whose face was mis-recognized, or an unenrolled
parent, surfaces in the same review queue used for unenrolled students.
"""
from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

WIB = timezone(timedelta(hours=7))
UNKNOWN_DIR = Path(__file__).parent / "data" / "unknown_faces"

_counter_lock = threading.Lock()
_counter = 0


def _next_seq() -> int:
    """Process-local sequence to disambiguate sub-second collisions."""
    global _counter
    with _counter_lock:
        _counter = (_counter + 1) % 10000
        return _counter


def capture_unknown_face(
    *,
    jpeg_bytes: Optional[bytes],
    kind: str,                       # 'unmatched' | 'suspected_false_match'
                                     # | 'unknown_chaperone' | 'suspended_chaperone'
                                     # | 'reenroll_overdue_chaperone'
    device_ip: str = "",
    device_name: str = "",
    terminal_id: str = "",
    suspected_emp_no: str = "",
    suspected_name: str = "",
    reason: str = "",
    when: Optional[datetime] = None,
    raw_event: Optional[dict] = None,
    extra: Optional[dict] = None,
    tenant_id: Optional[str] = None,
) -> Optional[str]:
    """
    Persist a face snapshot for offline review.

    Returns the local file path on success, or None when there's nothing
    useful to save (no/tiny JPEG) or on disk error. Firebase mirror is
    best-effort — failures are logged but never raised.
    """
    if not jpeg_bytes or len(jpeg_bytes) < 1000:
        return None

    now = when or datetime.now(WIB)
    try:
        seq = _next_seq()
        date_str = now.strftime("%Y-%m-%d")
        day_dir = UNKNOWN_DIR / date_str
        day_dir.mkdir(parents=True, exist_ok=True)

        dev_slug = re.sub(r"[^A-Za-z0-9]+", "_", device_name or device_ip or "device")
        dev_slug = dev_slug.strip("_")[:24] or "device"
        ts_slug = now.strftime("%H%M%S")
        base = f"{ts_slug}_{dev_slug}_{kind}_{seq:04d}"
        jpg_path = day_dir / f"{base}.jpg"
        json_path = day_dir / f"{base}.json"

        jpg_path.write_bytes(jpeg_bytes)

        sidecar = {
            "kind": kind,
            "deviceIp": device_ip,
            "deviceName": device_name,
            "terminalId": terminal_id,
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "capturedAt": now.isoformat(),
            "suspectedEmployeeNo": suspected_emp_no,
            "suspectedName": suspected_name,
            "reason": reason,
            "imageBytes": len(jpeg_bytes),
        }
        if raw_event:
            sidecar["rawEvent"] = {
                k: raw_event.get(k) for k in (
                    "majorEventType", "subEventType", "currentVerifyMode",
                    "employeeNoString", "employeeNo", "name", "time",
                    "serialNo", "cardNo", "userType",
                ) if raw_event.get(k) is not None
            }
        if extra:
            sidecar.update(extra)
        json_path.write_text(json.dumps(sidecar, indent=2))

        icon = {
            "unmatched": "❓",
            "suspected_false_match": "⚠️",
            "unknown_chaperone": "🚸",
            "suspended_chaperone": "🛑",
            "reenroll_overdue_chaperone": "🔁",
        }.get(kind, "👤")
        suffix = f"  (was: {suspected_name or suspected_emp_no})" if suspected_emp_no else ""
        print(f"  {icon} Captured face → {jpg_path.name}{suffix}")

        # Best-effort mirror to Firebase (Storage + Firestore)
        try:
            _mirror_to_firebase(jpg_path, sidecar, tenant_id=tenant_id)
        except Exception as e:
            print(f"     ⚠ Firebase mirror failed: {e}")

        return str(jpg_path)
    except Exception as e:
        print(f"  ⚠ Failed to save unknown face snapshot: {e}")
        return None


def _mirror_to_firebase(jpg_path: Path, sidecar: dict, tenant_id: Optional[str]) -> None:
    """Upload the JPEG to Storage + write a Firestore review doc."""
    try:
        import firebase_admin
        from firebase_admin import storage, firestore as _fs
    except Exception:
        return
    if not firebase_admin._apps:
        return  # Firebase not initialized in this process — silently skip

    try:
        import tenancy
        tid = tenant_id or tenancy.get_tenant_id()
    except Exception:
        tid = tenant_id or "default"

    date_str = sidecar["timestamp"][:10]
    storage_path = f"unknown_faces/{date_str}/{jpg_path.name}"
    bucket_name = os.environ.get(
        "FIREBASE_STORAGE_BUCKET",
        "facial-attendance-binus.firebasestorage.app",
    )
    public_url = None
    try:
        bucket = storage.bucket(bucket_name)
        blob = bucket.blob(storage_path)
        blob.upload_from_filename(str(jpg_path), content_type="image/jpeg")
        try:
            blob.make_public()
            public_url = blob.public_url
        except Exception:
            # Uniform bucket-level access; signed URL or rules will gate reads.
            pass
    except Exception as e:
        print(f"     ⚠ Storage upload failed: {e}")

    try:
        db = _fs.client()
        doc_id = jpg_path.stem
        payload = {
            **sidecar,
            "storagePath": storage_path,
            "imageUrl": public_url,
            "tenantId": tid,
            "status": "pending_review",
            "createdAt": _fs.SERVER_TIMESTAMP,
        }
        db.collection("tenants").document(tid).collection("unknownFaces") \
          .document(doc_id).set(payload)
    except Exception as e:
        print(f"     ⚠ Firestore mirror failed: {e}")
