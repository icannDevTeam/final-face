#!/usr/bin/env python3
"""
student_metadata.py — Student Metadata Mapping
================================================
Maps Hikvision device employeeNo → BINUS student IDs (IdStudent, IdBinusian).

This mapping is essential for the attendance pipeline:
  1. During enrollment: student IDs from BINUS API are saved alongside the
     device's employeeNo (MD5 hash of student name).
  2. During attendance: when the device fires a face recognition event,
     we look up the student's IdStudent/IdBinusian from this mapping
     to record attendance in the BINUS e-Desk system.

Storage:
  - Local JSON file: data/student_metadata.json (primary, survives restarts)
  - Firebase Firestore: collection "student_metadata" (cloud backup/sync)
"""

import json
import os
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

WIB = timezone(timedelta(hours=7))

METADATA_DIR = Path(__file__).parent / "data"
METADATA_FILE = METADATA_DIR / "student_metadata.json"


# ─── Local File Operations ──────────────────────────────────────────────────

def _load_local() -> dict:
    """Load student metadata from local JSON file."""
    if not METADATA_FILE.exists():
        return {}
    try:
        data = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"Failed to load student metadata: {e}")
    return {}


def _save_local(data: dict):
    """Save student metadata to local JSON file."""
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        METADATA_FILE.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception as e:
        logger.error(f"Failed to save student metadata: {e}")


def save_student(
    employee_no: str,
    name: str,
    id_student: str,
    id_binusian: str = "",
    homeroom: str = "",
    grade: str = "",
):
    """Save or update a single student's metadata.

    Called during enrollment so that the attendance listener can later
    map the device's employeeNo back to BINUS IdStudent/IdBinusian.
    """
    data = _load_local()
    now = datetime.now(WIB).isoformat()

    entry = data.get(employee_no, {})
    entry.update({
        "employeeNo": employee_no,
        "name": name,
        "idStudent": str(id_student) if id_student else "",
        "idBinusian": str(id_binusian) if id_binusian else "",
        "homeroom": homeroom,
        "grade": grade,
        "updatedAt": now,
    })
    if "enrolledAt" not in entry:
        entry["enrolledAt"] = now

    data[employee_no] = entry
    _save_local(data)
    logger.info(f"Metadata saved: {employee_no} → {name} (ID:{id_student}, BN:{id_binusian})")

    # Also sync to Firebase (non-blocking, best-effort)
    _sync_to_firebase(employee_no, entry)


def save_students_batch(students: list[dict]):
    """Save multiple students' metadata at once.

    Each dict should have keys: employeeNo, name, idStudent, idBinusian,
    homeroom, grade.
    """
    data = _load_local()
    now = datetime.now(WIB).isoformat()

    for s in students:
        eno = s.get("employeeNo", "")
        if not eno:
            continue
        entry = data.get(eno, {})
        entry.update({
            "employeeNo": eno,
            "name": s.get("name", ""),
            "idStudent": str(s.get("idStudent", "")) if s.get("idStudent") else "",
            "idBinusian": str(s.get("idBinusian", "")) if s.get("idBinusian") else "",
            "homeroom": s.get("homeroom", ""),
            "grade": s.get("grade", ""),
            "updatedAt": now,
        })
        if "enrolledAt" not in entry:
            entry["enrolledAt"] = now
        data[eno] = entry

    _save_local(data)
    logger.info(f"Batch metadata saved: {len(students)} student(s)")

    # Sync all to Firebase
    _sync_batch_to_firebase(data)


def get_student(employee_no: str) -> dict | None:
    """Look up student metadata by employeeNo.

    Returns dict with keys: employeeNo, name, idStudent, idBinusian, homeroom, grade.
    Returns None if not found.
    """
    data = _load_local()
    return data.get(employee_no)


def get_all_students() -> dict:
    """Get all student metadata. Returns dict keyed by employeeNo."""
    return _load_local()


def find_by_name(name: str) -> dict | None:
    """Find student metadata by exact name match.
    
    If multiple entries exist for the same name, prefer the one
    that has idStudent populated (from BINUS API enrollment).
    """
    data = _load_local()
    best = None
    for entry in data.values():
        if entry.get("name", "").strip().lower() == name.strip().lower():
            # Prefer entry with idStudent populated
            if entry.get("idStudent"):
                return entry
            if best is None:
                best = entry
    return best


def find_by_student_id(id_student: str) -> dict | None:
    """Find student metadata by BINUS IdStudent."""
    data = _load_local()
    for entry in data.values():
        if entry.get("idStudent") == str(id_student):
            return entry
    return None


# ─── Firebase Sync (best-effort, non-blocking) ──────────────────────────────

def _get_firestore():
    """Get Firestore client (lazy init)."""
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

        return firestore.client()
    except Exception as e:
        logger.debug(f"Firebase unavailable for metadata sync: {e}")
        return None


def _sync_to_firebase(employee_no: str, entry: dict):
    """Sync a single student entry to Firebase Firestore."""
    try:
        db = _get_firestore()
        if not db:
            return
        db.collection("student_metadata").document(employee_no).set(entry, merge=True)
    except Exception as e:
        logger.debug(f"Firebase metadata sync failed for {employee_no}: {e}")


def _sync_batch_to_firebase(data: dict):
    """Sync all student metadata to Firebase Firestore."""
    try:
        db = _get_firestore()
        if not db:
            return
        batch = db.batch()
        count = 0
        for eno, entry in data.items():
            ref = db.collection("student_metadata").document(eno)
            batch.set(ref, entry, merge=True)
            count += 1
            # Firestore batch limit is 500
            if count >= 400:
                batch.commit()
                batch = db.batch()
                count = 0
        if count > 0:
            batch.commit()
        logger.info(f"Firebase metadata sync complete: {len(data)} student(s)")
    except Exception as e:
        logger.debug(f"Firebase batch metadata sync failed: {e}")


def load_from_firebase() -> dict:
    """Load student metadata from Firebase Firestore.

    Useful for bootstrapping a new machine or after metadata file loss.
    Merges with local data (local takes precedence).
    """
    try:
        db = _get_firestore()
        if not db:
            return {}
        docs = db.collection("student_metadata").stream()
        remote = {}
        for doc in docs:
            remote[doc.id] = doc.to_dict()

        if remote:
            # Merge: remote first, then local overwrites
            local = _load_local()
            merged = {**remote, **local}
            _save_local(merged)
            logger.info(f"Loaded {len(remote)} student(s) from Firebase, merged total: {len(merged)}")
            return merged
    except Exception as e:
        logger.debug(f"Firebase metadata load failed: {e}")
    return _load_local()


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    data = _load_local()
    if not data:
        print("No student metadata found. Trying Firebase...")
        data = load_from_firebase()

    if not data:
        print("No student metadata available.")
        sys.exit(0)

    print(f"\n{'='*70}")
    print(f"  Student Metadata Mapping ({len(data)} students)")
    print(f"{'='*70}")
    print(f"  {'Employee#':<12} {'Name':<30} {'IdStudent':<12} {'IdBinusian':<12} {'Class'}")
    print(f"  {'─'*12} {'─'*30} {'─'*12} {'─'*12} {'─'*8}")
    for eno, entry in sorted(data.items(), key=lambda x: x[1].get("name", "")):
        print(
            f"  {eno:<12} {entry.get('name', '?'):<30} "
            f"{entry.get('idStudent', '-'):<12} "
            f"{entry.get('idBinusian', '-'):<12} "
            f"{entry.get('homeroom', '-')}"
        )
    print()
