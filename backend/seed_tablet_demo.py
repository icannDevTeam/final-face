#!/usr/bin/env python3
"""
Seed iPad Teacher (Grade 5) pickup_events for the demo.

Writes events that the iPad feed will actually render:
  - status='pending' (Active card slots) × 2
  - status='held'    (Held rail)         × 3
Bound to MYP Tower terminal (cf18e11f9d8e). Uses real Grade 5 student names
from the production `/students` collection. Chaperone faces are unset so
the avatar falls back to initials (no fake photos).

Usage:
  python3 seed_tablet_demo.py                 # write demo events
  python3 seed_tablet_demo.py --clear-only    # remove any prior demo events
"""
from __future__ import annotations

import argparse
import os
import random
import uuid
from datetime import datetime, timedelta, timezone

import firebase_admin
from firebase_admin import credentials, firestore

import tenancy

SERVICE_ACCOUNT_PATH = os.path.join(
    os.path.dirname(__file__),
    "facial-attendance-binus-firebase-adminsdk.json",
)

WIB = timezone(timedelta(hours=7))
DEMO_MARKER = "tablet-demo-seed"

# MYP Tower = the terminal the Grade 5 iPad is bound to.
TERMINAL_ID = "cf18e11f9d8e"
DEVICE_NAME = "MYP Tower (DS-K1T342MFX)"
RELEASE_GROUP_ID = "VF5ZcEkhBtVoJUyIwzCg"


def init_db():
    if not firebase_admin._apps:
        firebase_admin.initialize_app(credentials.Certificate(SERVICE_ACCOUNT_PATH))
    return firestore.client()


def grade5_students(db) -> list[dict]:
    out = []
    for d in db.collection("students").stream():
        s = d.to_dict()
        hr = (s.get("homeroom") or "").strip()
        gn = (s.get("gradeName") or "").strip()
        if hr.startswith("5") or "5" in gn:
            out.append({
                "id": f"stu-{s.get('id') or d.id}",
                "name": s.get("name") or s.get("displayLabel"),
                "homeroom": hr or "5A",
                "photoUrl": None,
            })
    return [s for s in out if s["name"]]


def chap(name, relation):
    return {
        "id": f"chap-demo-{abs(hash(name)) % 10**8}",
        "name": name,
        "relation": relation,
        "photoUrl": None,
        "phone": f"+628{random.randint(100000000, 999999999)}",
        "suspended": False,
        "reEnrollDueAt": None,
        "reEnrollOverdue": False,
    }


def make_event(*, when, status, chaperone, students, decision="ok", card_state="green"):
    event_id = f"tabdemo-{uuid.uuid4().hex[:10]}"
    return {
        "eventId": event_id,
        "_demo": DEMO_MARKER,
        "tenantId": None,
        "employeeNo": f"9{random.randint(100000000, 999999999)}",
        "scannedAt": when,
        "recordedAt": when,
        "deviceName": DEVICE_NAME,
        "gate": DEVICE_NAME,
        "terminalId": TERMINAL_ID,
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--tenant", default=tenancy.get_tenant_id())
    ap.add_argument("--clear-only", action="store_true")
    args = ap.parse_args()

    db = init_db()
    tid = args.tenant

    removed = clear_demo(db, tid)
    if removed:
        print(f"  – removed {removed} prior demo event(s)")
    if args.clear_only:
        return

    pool = grade5_students(db)
    if len(pool) < 4:
        raise SystemExit(f"Need ≥4 Grade 5 students, found {len(pool)}")
    random.shuffle(pool)
    take_i = [0]
    def take(n):
        out = []
        for _ in range(n):
            out.append(pool[take_i[0] % len(pool)])
            take_i[0] += 1
        return out

    now = datetime.now(timezone.utc)
    events = []

    # ── ACTIVE pickups (status='pending') — newest first slot ──
    actives = [
        (10,  "Ibu Sari Wijaya",      "Mother",      1),
        (38,  "Pak Hadi Kusuma",      "Father",      2),
        (72,  "Bu Maya Tanjung",      "Mother",      1),
        (108, "Pak Dimas Halim",      "Father",      3),
    ]
    for secs, name, rel, n in actives:
        e, _ = make_event(
            when=now - timedelta(seconds=secs),
            status="pending",
            chaperone=chap(name, rel),
            students=take(n),
        )
        events.append(e)

    # ── HELD pickups (status='held') — fills the held rail ──
    helds = [
        (2,  30,  "Bu Linda Pranata",   "Mother",       1),
        (4,  10,  "Pak Eko Setiawan",   "Driver",       2),
        (5,  50,  "Oma Rina Halim",     "Grandmother",  1),
        (7,  15,  "Pak Budi Hartono",   "Father",       2),
        (8,  40,  "Bu Yuni Kusuma",     "Mother",       1),
        (10, 25,  "Pak Arif Lim",       "Driver",       1),
        (12, 5,   "Bu Citra Wijaya",    "Mother",       2),
        (14, 30,  "Opa Joko Suryadi",   "Grandfather",  1),
    ]
    for mins, secs, name, rel, n in helds:
        e, _ = make_event(
            when=now - timedelta(minutes=mins, seconds=secs),
            status="held",
            chaperone=chap(name, rel),
            students=take(n),
        )
        events.append(e)

    coll = db.collection(tenancy.pickup_events_path(tid))
    for ev, _ in [(e, None) for e in events]:
        ev["tenantId"] = tid
        coll.document(ev["eventId"]).set(ev)
        wib = ev["recordedAt"].astimezone(WIB).strftime("%H:%M:%S")
        ss = ", ".join(s["name"] for s in ev["students"])
        print(f"  + [{ev['status']:7}] {wib}  {ev['chaperone']['name']:24} → {ss}")

    print(f"\n  ✓ Wrote {len(events)} demo event(s) for Grade 5 / MYP Tower")
    print("    Refresh the iPad — should render 2 active cards + 3 held cards.")


if __name__ == "__main__":
    main()
