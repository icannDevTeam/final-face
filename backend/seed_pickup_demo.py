#!/usr/bin/env python3
"""
Seed demo pickup_events for TV-display testing (Grades 3-5).

Two modes:

  perfect-demo (default)  — Curated 13-event scenario that exercises every
                            TV feature: featured-card success seal, all
                            decision/colour states, multi-student overflow,
                            all 3 gates, "● LIVE" badge, queue rows.

  random                  — Legacy random mix (use --random or --count).

Usage:
    python3 seed_pickup_demo.py                       # perfect demo (13 events)
    python3 seed_pickup_demo.py --clear               # clear, then perfect demo
    python3 seed_pickup_demo.py --random --count 20   # random scenario
    python3 seed_pickup_demo.py --clear-only          # wipe demo events, exit
    python3 seed_pickup_demo.py --tenant binus-simprug

Then open  http://localhost:3000/pickup/tv?token=$PICKUP_TV_TOKEN
        or http://localhost:3000/pickup/tv?profile=<kioskProfileId>
"""
from __future__ import annotations

import argparse
import os
import random
import sys
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
DEMO_MARKER = "demo-seed"

# Three real production gate names (must match what attendance_listener writes,
# so kiosk profile gate-filters work identically against demo + live data).
GATES = [
    "Basement 1 Terminal (DS-K1T341AMF)",
    "PYP Lobby Entrance (DS-K1T342MFX)",
    "MYP Tower (DS-K1T342MFX)",
]

CHAP_FIRST = ["Andi", "Sari", "Budi", "Maya", "Rudi", "Linda", "Eko", "Putri",
              "Tono", "Rina", "Hadi", "Dewi", "Arif", "Yuni", "Joko"]
CHAP_LAST = ["Wijaya", "Halim", "Setiawan", "Pranata", "Kusuma", "Tanjung",
             "Lim", "Sugianto", "Hartono", "Suryadi"]
RELATIONS = ["Father", "Mother", "Grandmother", "Grandfather",
             "Driver", "Aunt", "Uncle", "Nanny"]


def rand_chaperone():
    return f"{random.choice(CHAP_FIRST)} {random.choice(CHAP_LAST)}"


# Real students who must NEVER appear in seeded demo data (parents/staff
# would notice their child being "picked up" by a fake chaperone).
EXCLUDED_STUDENTS = {
    "albert arthur",          # 4C — explicitly excluded by request
}


# ─── Student loader ────────────────────────────────────────────────────────
def load_local_students(grades: set[str]) -> list[dict]:
    """Read backend/data/student_metadata.json and filter by grade prefix."""
    import json
    path = os.path.join(os.path.dirname(__file__), "data", "student_metadata.json")
    with open(path) as f:
        raw = json.load(f)
    students = list(raw.values()) if isinstance(raw, dict) else raw
    out = []
    for s in students:
        hr = (s.get("homeroom") or "").upper()
        if not hr:
            continue
        if hr[0] not in grades:
            continue
        name = s.get("name") or s.get("studentName") or s.get("fullName")
        if not name:
            continue
        if name.strip().lower() in EXCLUDED_STUDENTS:
            continue
        out.append({"name": name, "homeroom": hr})
    return out


# ─── Event factory ─────────────────────────────────────────────────────────
def _student_dict(s: dict) -> dict:
    """Match the shape pickup_event_writer._resolve_students produces."""
    return {
        "id": f"demo-stu-{abs(hash(s['name'])) % 10**8}",
        "name": s["name"],
        "homeroom": s["homeroom"],
        "photoUrl": None,   # let TV feed auto-resolve from face_dataset/{homeroom}/{name}
    }


def _chap_dict(*, name: str, relation,
               suspended: bool = False,
               reenroll_overdue: bool = False,
               unknown: bool = False) -> dict:
    return {
        "id": None if unknown else f"chap-9{random.randint(100000000, 999999999)}",
        "name": name,
        "relation": relation,
        "photoUrl": None,
        "phone": None if unknown else f"+628{random.randint(100000000, 999999999)}",
        "suspended": suspended,
        "reEnrollDueAt": None,
        "reEnrollOverdue": reenroll_overdue,
    }


def make_event(*, when: datetime, gate: str,
               chaperone: dict, students: list,
               decision: str, card_state: str,
               officer_override=None) -> dict:
    event_id = f"demo-{uuid.uuid4().hex[:10]}"
    # 6-digit override code for non-OK events (matches pickup_event_writer.py)
    override_code = None
    if decision != "ok":
        override_code = f"{random.randint(0, 999_999):06d}"
    return {
        "eventId": event_id,
        "tenantId": None,                 # filled by writer
        "employeeNo": f"9{random.randint(100000000, 999999999)}",
        "scannedAt": when,                # datetime → Firestore Timestamp
        "recordedAt": when,
        "deviceName": gate.split(" (")[0],
        "gate": gate,
        "chaperone": chaperone,
        "students": students,
        "decision": decision,
        "cardState": card_state,
        "capturePath": None,
        "officerOverride": officer_override,
        "overrideCode": override_code,
        "holdSeconds": 60,
        "_demo": DEMO_MARKER,
    }


# ─── Perfect demo scenario ─────────────────────────────────────────────────
def build_perfect_scenario(students_pool: list) -> list:
    """
    Curated scenario that exercises every TV feature.

    Wall (4 cards — featured + 3 supporting):
      1. FEATURED  · GREEN · 3 children → success-seal + dynamic kids-3 grid
      2. supporting · GREEN · 1 child   → kids-1 single-tile preview shape
      3. supporting · YELLOW (reenroll_overdue) · 2 children
      4. supporting · RED   (unknown_chaperone)   · 0 children

    Queue (14 rows): mixes all 3 gates, all decisions, varied child counts
    including a 4-kid pickup (tests queue +more) and an officer-override case.

    Decisions covered:  ok · suspended · unknown_chaperone · reenroll_overdue
    Card states:        green · yellow · red
    Gates:              all 3 production devices
    Time spread:        newest within 8s (LIVE badge) → up to 28 min ago
    """
    if len(students_pool) < 14:
        raise SystemExit(
            f"Need ≥14 students in selected grades, got {len(students_pool)}")

    now = datetime.now(timezone.utc)
    pool = students_pool.copy()
    random.shuffle(pool)
    state = {"pool": pool}

    def take(n):
        if len(state["pool"]) < n:
            state["pool"] = students_pool.copy()
            random.shuffle(state["pool"])
        out = state["pool"][:n]
        state["pool"] = state["pool"][n:]
        return [_student_dict(s) for s in out]

    events = []

    # ─── WALL 1 / FEATURED ─ GREEN, 3 children, LIVE badge, success-seal
    events.append(make_event(
        when=now - timedelta(seconds=8),         # < 25s → "● LIVE" badge
        gate=GATES[1],                           # PYP Lobby
        chaperone=_chap_dict(name="Maya Halim", relation="Mother"),
        students=take(3),
        decision="ok", card_state="green",
    ))

    # ─── WALL 2 ─ GREEN, single child (small-card layout)
    events.append(make_event(
        when=now - timedelta(minutes=1, seconds=30),
        gate=GATES[2],                           # MYP Tower
        chaperone=_chap_dict(name="Hadi Setiawan", relation="Father"),
        students=take(1),
        decision="ok", card_state="green",
    ))

    # ─── WALL 3 ─ YELLOW (re-enroll overdue)
    events.append(make_event(
        when=now - timedelta(minutes=2, seconds=10),
        gate=GATES[0],                           # Basement 1
        chaperone=_chap_dict(
            name="Rina Pranata", relation="Grandmother",
            reenroll_overdue=True),
        students=take(2),
        decision="reenroll_overdue", card_state="yellow",
    ))

    # ─── WALL 4 ─ RED (unknown chaperone, no resolved students)
    events.append(make_event(
        when=now - timedelta(minutes=3, seconds=45),
        gate=GATES[1],                           # PYP Lobby
        chaperone=_chap_dict(
            name="Unknown (9382194021)", relation=None, unknown=True),
        students=[],
        decision="unknown_chaperone", card_state="red",
    ))

    # ─── QUEUE rows — populate the side queue heavily.
    # Each tuple: (offset_min, gate_idx, name, relation, n_kids, decision, color, override?)
    queue_recipes = [
        (5,    2, "Eko Sugianto",     "Driver",       5, "ok",                "green",  False),  # auto-fill kids-many
        (6,    0, "Putri Lim",        "Aunt",         1, "suspended",         "yellow", True),   # officer override
        (7,    1, "Hendra Tanjung",   "Father",       2, "ok",                "green",  False),
        (8,    1, "Andi Wijaya",      "Father",       2, "ok",                "green",  False),
        (9,    0, "Maya Kusuma",      "Mother",       3, "ok",                "green",  False),
        (10,   2, "Linda Tanjung",    "Mother",       1, "ok",                "green",  False),
        (11,   1, "Bambang Halim",    "Grandfather",  2, "ok",                "green",  False),
        (12,   0, "Budi Kusuma",      "Grandfather",  3, "ok",                "green",  False),
        (13,   2, "Nadia Suryadi",    "Mother",       1, "ok",                "green",  False),
        (14,   1, "Yuni Hartono",     "Nanny",        2, "ok",                "green",  False),
        (15,   0, "Iwan Pranata",     "Father",       4, "ok",                "green",  False),  # queue +more
        (16,   2, "Tono Suryadi",     "Driver",       4, "ok",                "green",  False),  # queue +more
        (17,   1, "Wati Setiawan",    "Mother",       1, "ok",                "green",  False),
        (18,   0, "Sari Lim",         "Mother",       1, "reenroll_overdue",  "yellow", False),
        (19,   2, "Dimas Wijaya",     "Uncle",        2, "ok",                "green",  False),
        (20,   1, "Joko Halim",       "Uncle",        2, "ok",                "green",  False),
        (21,   0, "Reni Hartono",     "Aunt",         1, "ok",                "green",  False),
        (22,   2, "Dewi Wijaya",      "Mother",       1, "ok",                "green",  False),
        (23,   1, "Agus Sugianto",    "Father",       3, "ok",                "green",  False),
        (24,   0, "Rudi Halim",       "Father",       3, "ok",                "green",  False),
        (25,   1, "Arif Setiawan",    "Driver",       2, "ok",                "green",  False),
        (26,   2, "Citra Lim",        "Mother",       1, "reenroll_overdue",  "yellow", False),
        (27,   2, "Unknown Visitor",  "",             0, "unknown_chaperone", "red",    False),
        (28,   0, "Linda Pranata",    "Grandmother",  1, "ok",                "green",  False),
        (29,   1, "Yuli Kusuma",      "Mother",       2, "ok",                "green",  False),
    ]
    for mins, gi, name, rel, n, dec, color, override in queue_recipes:
        events.append(make_event(
            when=now - timedelta(minutes=mins),
            gate=GATES[gi],
            chaperone=_chap_dict(
                name=name, relation=rel or None,
                suspended=(dec == "suspended"),
                reenroll_overdue=(dec == "reenroll_overdue"),
                unknown=(dec == "unknown_chaperone"),
            ),
            students=take(n) if n else [],
            decision=dec, card_state=color,
            officer_override={
                "by": "officer-01",
                "note": "Verified via emergency contact list",
                "decision": "approved",
                "at": (now - timedelta(minutes=mins) + timedelta(seconds=10)).isoformat(),
            } if override else None,
        ))

    return events


# ─── Random scenario (legacy) ──────────────────────────────────────────────
def build_random_scenario(students_pool: list, count: int) -> list:
    now = datetime.now(timezone.utc)
    out = []
    for _ in range(count):
        offset = timedelta(seconds=random.randint(5, 25 * 60))
        when = now - offset
        roll = random.random()
        kind = "green" if roll < 0.70 else ("yellow" if roll < 0.92 else "red")
        gate = random.choice(GATES)
        if kind == "red":
            chap = _chap_dict(name="Unknown Visitor", relation=None, unknown=True)
            students = []
            decision = "unknown_chaperone"
        elif kind == "yellow":
            decision = random.choice(["suspended", "reenroll_overdue"])
            chap = _chap_dict(
                name=rand_chaperone(),
                relation=random.choice(RELATIONS),
                suspended=(decision == "suspended"),
                reenroll_overdue=(decision == "reenroll_overdue"),
            )
            n = random.randint(1, 2)
            picks = random.sample(students_pool, k=min(len(students_pool), n))
            students = [_student_dict(s) for s in picks]
        else:
            decision = "ok"
            chap = _chap_dict(name=rand_chaperone(),
                              relation=random.choice(RELATIONS))
            n = random.randint(1, 3)
            picks = random.sample(students_pool, k=min(len(students_pool), n))
            students = [_student_dict(s) for s in picks]
        out.append(make_event(
            when=when, gate=gate, chaperone=chap, students=students,
            decision=decision, card_state=kind,
        ))
    return out


# ─── Firestore write / clear ───────────────────────────────────────────────
def clear_demo_events(db, tid: str) -> int:
    coll = db.collection(tenancy.pickup_events_path(tid))
    n = 0
    while True:
        snap = list(coll.where("_demo", "==", DEMO_MARKER).limit(400).stream())
        if not snap:
            break
        batch = db.batch()
        for doc in snap:
            batch.delete(doc.reference)
        batch.commit()
        n += len(snap)
        if len(snap) < 400:
            break
    return n


def write_events(db, tid: str, events: list) -> int:
    coll = db.collection(tenancy.pickup_events_path(tid))
    written = 0
    for ev in events:
        ev = dict(ev)
        ev["tenantId"] = tid
        coll.add(ev)
        written += 1
        wib = ev["recordedAt"].astimezone(WIB).strftime("%H:%M:%S")
        sample = ev["students"][0]["name"] if ev["students"] else "(no student)"
        n_more = max(0, len(ev["students"]) - 1)
        more = f" +{n_more}" if n_more else ""
        print(f"  + [{ev['cardState']:6}] {wib} {ev['gate'][:35]:35}"
              f"  {ev['chaperone']['name']:24} → {sample}{more}")
    return written


# ─── Seed demo chaperones ──────────────────────────────────────────────────
DEMO_CHAPERONES = [
    dict(name="Maya Halim",     relationship="Mother",      phone="+6281200001001", suspended=False, photoUrl="/parent-photos/mom.jpe"),
    dict(name="Hadi Setiawan",  relationship="Father",      phone="+6281200001002", suspended=False, photoUrl="/parent-photos/dad.jpe"),
    dict(name="Rina Pranata",   relationship="Grandmother", phone="+6281200001003", suspended=False, photoUrl="/parent-photos/granny.jpe"),
    dict(name="Putri Lim",      relationship="Aunt",        phone="+6281200001004", suspended=True, photoUrl="/parent-photos/aunt.jpg"),
    dict(name="Eko Sugianto",   relationship="Driver",      phone="+6281200001005", suspended=False, photoUrl="/parent-photos/images.jpe"),
]


def clear_demo_chaperones(db, tid: str) -> int:
    coll = db.collection(tenancy.chaperones_path(tid))
    snap = list(coll.where("_demo", "==", DEMO_MARKER).limit(200).stream())
    if not snap:
        return 0
    batch = db.batch()
    for doc in snap:
        batch.delete(doc.reference)
    batch.commit()
    return len(snap)


def write_chaperones(db, tid: str, now: datetime) -> int:
    coll = db.collection(tenancy.chaperones_path(tid))
    for i, c in enumerate(DEMO_CHAPERONES):
        chap_id = f"demo-chap-{i+1:03d}"
        doc = {
            "_demo": DEMO_MARKER,
            "employeeNo": f"9{900_000_000 + i}",
            "name": c["name"],
            "relationship": c["relationship"],
            "phone": c["phone"],
            "suspended": c["suspended"],
            "authorizedStudentIds": [],
            "photoUrls": [c.get("photoUrl")] if c.get("photoUrl") else [],
            "enrollmentSummary": {"ok": random.randint(1, 8), "total": random.randint(8, 12)},
            "reenrollDueAt": (now + timedelta(days=random.choice([-5, 30, 60, 90]))).isoformat(),
            "lastSeenAt": (now - timedelta(minutes=random.randint(5, 40))).isoformat(),
            "lastSeenGate": random.choice(GATES),
            "createdAt": (now - timedelta(days=random.randint(30, 180))).isoformat(),
        }
        coll.document(chap_id).set(doc)
    return len(DEMO_CHAPERONES)


# ─── Seed demo security incidents ─────────────────────────────────────────
def clear_demo_incidents(db, tid: str) -> int:
    coll = db.collection(tenancy.security_incidents_path(tid))
    snap = list(coll.where("_demo", "==", DEMO_MARKER).limit(200).stream())
    if not snap:
        return 0
    batch = db.batch()
    for doc in snap:
        batch.delete(doc.reference)
    batch.commit()
    return len(snap)


def write_incidents(db, tid: str, events: list) -> int:
    coll = db.collection(tenancy.security_incidents_path(tid))
    written = 0
    for ev in events:
        if ev.get("decision") not in ("unknown_chaperone", "suspended", "reenroll_overdue"):
            continue
        incident_id = uuid.uuid4().hex
        coll.document(incident_id).set({
            "_demo": DEMO_MARKER,
            "incidentId": incident_id,
            "kind": ev["decision"],
            "eventId": ev["eventId"],
            "employeeNo": ev.get("employeeNo"),
            "gate": ev.get("gate"),
            "chaperoneName": ev.get("chaperone", {}).get("name"),
            "createdAt": ev["recordedAt"].isoformat(),
            "resolved": ev.get("officerOverride") is not None,
        })
        written += 1
    return written


# ─── CLI ───────────────────────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tenant", default=None)
    p.add_argument("--scenario", choices=["perfect", "random"], default="perfect",
                   help="perfect = curated 13-event demo (default); random = legacy")
    p.add_argument("--random", action="store_true",
                   help="Shortcut for --scenario random")
    p.add_argument("--count", type=int, default=12,
                   help="Random scenario only: number of events")
    p.add_argument("--grades", default="3,4,5",
                   help="Comma-separated grade prefixes (default: 3,4,5)")
    p.add_argument("--clear", action="store_true",
                   help="Delete previous demo events before seeding")
    p.add_argument("--clear-only", action="store_true",
                   help="Just delete demo events and exit")
    args = p.parse_args()

    grades = {g.strip() for g in args.grades.split(",") if g.strip()}

    if not firebase_admin._apps:
        firebase_admin.initialize_app(credentials.Certificate(SERVICE_ACCOUNT_PATH))
    db = firestore.client()
    tid = tenancy.get_tenant_id(args.tenant)

    if args.clear or args.clear_only:
        n = clear_demo_events(db, tid)
        print(f"[clear] removed {n} previous demo events")
        nc = clear_demo_chaperones(db, tid)
        print(f"[clear] removed {nc} previous demo chaperones")
        ni = clear_demo_incidents(db, tid)
        print(f"[clear] removed {ni} previous demo incidents")
        if args.clear_only:
            return 0

    students = load_local_students(grades)
    if not students:
        print(f"[fail] no students found for grades {sorted(grades)}", file=sys.stderr)
        return 1
    print(f"[load] {len(students)} students across grades {sorted(grades)}")

    scenario = "random" if args.random else args.scenario
    if scenario == "random":
        events = build_random_scenario(students, args.count)
    else:
        events = build_perfect_scenario(students)

    print(f"[seed] {scenario} scenario → {len(events)} events")
    written = write_events(db, tid, events)

    now = datetime.now(timezone.utc)
    nc = write_chaperones(db, tid, now)
    print(f"[seed] {nc} demo chaperones written to {tenancy.chaperones_path(tid)}")
    ni = write_incidents(db, tid, events)
    print(f"[seed] {ni} demo security incidents written to {tenancy.security_incidents_path(tid)}")

    print(f"\n[done] wrote {written} demo events to "
          f"{tenancy.pickup_events_path(tid)}")
    print("       Open: http://localhost:3000/pickup/tv?token=$PICKUP_TV_TOKEN")
    print("       (or pair a kiosk and use a kiosk profile to filter gates)")
    print("       Events auto-expire from the TV feed after 30 minutes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
