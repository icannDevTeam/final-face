#!/usr/bin/env python3
"""
Seed iPad Teacher (Grades 4-5) pickup_events for the demo.

Writes events the iPad feed renders:
  - status='pending' (Active card slots) × 14
  - status='held'    (Held rail)         × 8
Spread across the Grade 5A / 4A / 4B terminals in release group
VF5ZcEkhBtVoJUyIwzCg (the group the paired "GRADE 5 iPad" is bound to).
Uses ~20 real Grade 4-5 students from the production `/students` collection
(Albert Arthur excluded). Chaperone faces come from pickupguard/parent photo/
uploaded once to Storage (cycled across chaperones).

Usage:
  python3 seed_tablet_demo.py                 # clear old + write demo events
  python3 seed_tablet_demo.py --preview       # route events to the isolated
                                              #   preview-demo group/terminal
  python3 seed_tablet_demo.py --clear-only    # remove any prior demo events
"""
from __future__ import annotations

import argparse
import os
import random
import uuid
from datetime import datetime, timedelta, timezone

import firebase_admin
from firebase_admin import credentials, firestore, storage

import tenancy

SERVICE_ACCOUNT_PATH = os.path.join(
    os.path.dirname(__file__),
    "facial-attendance-binus-firebase-adminsdk.json",
)

WIB = timezone(timedelta(hours=7))
DEMO_MARKER = "tablet-demo-seed"

# Terminals in the iPad's release group (verified against Firestore registry
# 2026-07-15). Students are routed to their grade's terminal.
TERMINALS = {
    "5":  ("30754945c9eb", "Grade 5A Terminal (10.26.30.83)"),
    "4A": ("455b597cf22f", "Grade 4A Terminal (10.26.30.74)"),
    "4B": ("0d847419cf16", "Grade 4B Terminal (10.26.30.82)"),
    "4C": ("455b597cf22f", "Grade 4A Terminal (10.26.30.74)"),  # 4C has no own terminal
}
RELEASE_GROUP_ID = "VF5ZcEkhBtVoJUyIwzCg"

PARENT_PHOTO_DIR = os.path.join(
    os.path.dirname(__file__), "..", "pickupguard", "parent photo")
DEMO_FACE_PREFIX = "demo_chaperone_faces"

# Real students who must NEVER appear in seeded demo data.
EXCLUDED_STUDENTS = {
    "albert arthur",  # 4C — explicitly excluded by request
}


def init_db():
    if not firebase_admin._apps:
        firebase_admin.initialize_app(
            credentials.Certificate(SERVICE_ACCOUNT_PATH),
            {"storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET")
             or "facial-attendance-binus.firebasestorage.app"},
        )
    return firestore.client()


def upload_demo_faces(tid: str) -> list[str]:
    """Upload local parent photos to Storage once; return storage paths."""
    bucket = storage.bucket()
    paths = []
    if not os.path.isdir(PARENT_PHOTO_DIR):
        print(f"  ! parent photo dir missing: {PARENT_PHOTO_DIR} — using initials")
        return paths
    for fname in sorted(os.listdir(PARENT_PHOTO_DIR)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".jpe", ".png")):
            continue
        dest = f"tenants/{tid}/{DEMO_FACE_PREFIX}/{os.path.splitext(fname)[0]}.jpg"
        blob = bucket.blob(dest)
        if not blob.exists():
            blob.upload_from_filename(
                os.path.join(PARENT_PHOTO_DIR, fname), content_type="image/jpeg")
            print(f"  ↑ uploaded {fname} → {dest}")
        paths.append(dest)
    return paths


def demo_students(db) -> dict[str, list[dict]]:
    """Real Grade 4-5 students from /students, keyed by terminal key."""
    buckets: dict[str, list[dict]] = {k: [] for k in TERMINALS}
    for d in db.collection("students").stream():
        s = d.to_dict()
        name = (s.get("name") or s.get("displayLabel") or "").strip()
        hr = (s.get("homeroom") or "").strip()
        if not name or name.lower() in EXCLUDED_STUDENTS:
            continue
        key = None
        if hr.startswith("5"):
            key = "5"
        elif hr in ("4A", "4B", "4C"):
            key = hr
        if key:
            buckets[key].append({
                "id": f"stu-{s.get('id') or d.id}",
                "name": name,
                "homeroom": hr,
                "photoUrl": None,
            })
    return buckets


def chap(name, relation, photo_path=None):
    return {
        "id": f"chap-demo-{abs(hash(name)) % 10**8}",
        "name": name,
        "relation": relation,
        "photoUrl": photo_path,
        "phone": f"+628{random.randint(100000000, 999999999)}",
        "suspended": False,
        "reEnrollDueAt": None,
        "reEnrollOverdue": False,
    }


def make_event(*, when, status, chaperone, students, terminal_key,
               decision="ok", card_state="green"):
    terminal_id, device_name = TERMINALS[terminal_key]
    event_id = f"tabdemo-{uuid.uuid4().hex[:10]}"
    return {
        "eventId": event_id,
        "_demo": DEMO_MARKER,
        "tenantId": None,
        "employeeNo": f"9{random.randint(100000000, 999999999)}",
        "scannedAt": when,
        "recordedAt": when,
        "deviceName": device_name,
        "gate": device_name,
        "terminalId": terminal_id,
        "releaseGroupId": RELEASE_GROUP_ID,
        "chaperone": chaperone,
        "students": students,
        "decision": decision,
        "cardState": card_state,
        "capturePath": None,
        "officerOverride": None,
        "overrideCode": None,
        "holdSeconds": 60,
        "status": status,
    }, event_id


def clear_demo(db, tid):
    coll = db.collection(tenancy.pickup_events_path(tid))
    n = 0
    while True:
        snap = list(coll.where("_demo", "==", DEMO_MARKER).limit(200).stream())
        if not snap:
            break
        batch = db.batch()
        for d in snap:
            batch.delete(d.reference)
        batch.commit()
        n += len(snap)
        if len(snap) < 200:
            break
    return n


def main():
    global RELEASE_GROUP_ID
    ap = argparse.ArgumentParser()
    ap.add_argument("--tenant", default=tenancy.get_tenant_id())
    ap.add_argument("--clear-only", action="store_true")
    ap.add_argument("--preview", action="store_true",
                    help="Route all events to the preview-demo release group "
                         "and terminal (never shows on real iPads)")
    args = ap.parse_args()

    if args.preview:
        RELEASE_GROUP_ID = "preview-demo"
        for k in TERMINALS:
            TERMINALS[k] = ("preview-demo-terminal", "Preview Demo Terminal")

    db = init_db()
    tid = args.tenant

    removed = clear_demo(db, tid)
    if removed:
        print(f"  – removed {removed} prior demo event(s)")
    if args.clear_only:
        return

    face_paths = upload_demo_faces(tid)

    buckets = demo_students(db)
    for k, v in buckets.items():
        random.shuffle(v)
    total_avail = sum(len(v) for v in buckets.values())
    print(f"  student pool: 5→{len(buckets['5'])}  4A→{len(buckets['4A'])}"
          f"  4B→{len(buckets['4B'])}  4C→{len(buckets['4C'])}  (total {total_avail})")
    if total_avail < 15:
        raise SystemExit(f"Need ≥15 Grade 4-5 students, found {total_avail}")

    idx = {k: 0 for k in buckets}
    def take(terminal_key, n):
        pool = buckets[terminal_key]
        out = []
        for _ in range(n):
            out.append(pool[idx[terminal_key] % len(pool)])
            idx[terminal_key] += 1
        return out

    def face(i):
        return face_paths[i % len(face_paths)] if face_paths else None

    now = datetime.now(timezone.utc)
    events = []

    # ── ACTIVE pickups (status='pending') ──
    actives = [
        (10,  "Ibu Sari Wijaya",    "Mother",      "5",  1),
        (38,  "Pak Hadi Kusuma",    "Father",      "5",  2),
        (72,  "Bu Maya Tanjung",    "Mother",      "4A", 1),
        (108, "Pak Dimas Halim",    "Father",      "4B", 3),
        (150, "Oma Ratna Sari",     "Grandmother", "4C", 1),
        (185, "Pak Agus Pranata",   "Driver",      "5",  2),
        (210, "Bu Dewi Santoso",    "Mother",      "4A", 2),
        (238, "Pak Rudi Wibowo",    "Father",      "4B", 1),
        (265, "Opa Hendra Gunawan", "Grandfather", "5",  1),
        (290, "Bu Fitri Anggraini", "Mother",      "4C", 2),
        (318, "Pak Yanto Salim",    "Driver",      "4A", 1),
        (342, "Bu Nina Hartanto",   "Mother",      "4B", 2),
        (370, "Pak Iwan Cahyono",   "Father",      "5",  1),
        (395, "Oma Lestari Wijaya", "Grandmother", "4A", 1),
    ]
    for i, (secs, name, rel, tkey, n) in enumerate(actives):
        e, _ = make_event(
            when=now - timedelta(seconds=secs),
            status="pending",
            chaperone=chap(name, rel, face(i)),
            students=take(tkey, n),
            terminal_key=tkey,
        )
        events.append(e)

    # ── HELD pickups (status='held') ──
    helds = [
        (2,  30, "Bu Linda Pranata",  "Mother",      "4A", 1),
        (4,  10, "Pak Eko Setiawan",  "Driver",      "4B", 2),
        (5,  50, "Oma Rina Halim",    "Grandmother", "5",  1),
        (7,  15, "Pak Budi Hartono",  "Father",      "4C", 2),
        (8,  40, "Bu Yuni Kusuma",    "Mother",      "4A", 1),
        (10, 25, "Pak Arif Lim",      "Driver",      "4B", 1),
        (12, 5,  "Bu Citra Wijaya",   "Mother",      "4C", 2),
        (14, 30, "Opa Joko Suryadi",  "Grandfather", "5",  1),
    ]
    for i, (mins, secs, name, rel, tkey, n) in enumerate(helds):
        e, _ = make_event(
            when=now - timedelta(minutes=mins, seconds=secs),
            status="held",
            chaperone=chap(name, rel, face(len(actives) + i)),
            students=take(tkey, n),
            terminal_key=tkey,
        )
        events.append(e)

    coll = db.collection(tenancy.pickup_events_path(tid))
    seen_students = set()
    for ev in events:
        ev["tenantId"] = tid
        coll.document(ev["eventId"]).set(ev)
        wib = ev["recordedAt"].astimezone(WIB).strftime("%H:%M:%S")
        ss = ", ".join(s["name"] for s in ev["students"])
        seen_students.update(s["name"] for s in ev["students"])
        print(f"  + [{ev['status']:7}] {wib}  {ev['chaperone']['name']:20} → {ss}")

    print(f"\n  ✓ Wrote {len(events)} demo event(s) covering {len(seen_students)} students")
    print("    Terminals: Grade 5A + 4A + 4B (release group VF5ZcEkhBtVoJUyIwzCg)")
    print("    Refresh the GRADE 5 iPad — active cards + held rail should populate.")


if __name__ == "__main__":
    main()
