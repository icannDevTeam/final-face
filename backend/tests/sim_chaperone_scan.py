#!/usr/bin/env python3
"""
sim_chaperone_scan.py — End-to-end chaperone-scan simulation.

Drives `record_pickup_event()` with realistic Hikvision-style scan bursts
against a FAKE in-memory Firestore (no network, no service account). The
goal is to measure:

  * decision-classification correctness across the full matrix
  * hot-path latency (scan-arrived → pickup_events doc written)
  * cooldown + inter-parent throttle behavior under burst load
  * burst capacity (events/sec at the writer boundary)

We do NOT touch real Firestore — this is a load + behavior simulation,
not a benchmark of Firestore itself.

Run:
  cd backend && python3 tests/sim_chaperone_scan.py
  cd backend && SIM_SCANS=500 python3 tests/sim_chaperone_scan.py
"""
from __future__ import annotations

import os
import random
import sys
import time
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

# ── Stub firebase_admin BEFORE writer import ────────────────────────────────
if "firebase_admin" not in sys.modules:
    fa = types.ModuleType("firebase_admin")
    fa._apps = {}
    sys.modules["firebase_admin"] = fa
    fs = types.ModuleType("firebase_admin.firestore")
    fs.client = lambda: None
    sys.modules["firebase_admin.firestore"] = fs

import pickup_event_writer as pew  # noqa: E402

WIB = timezone(timedelta(hours=7))
SIM_SCANS = int(os.environ.get("SIM_SCANS", "200"))
SEED = int(os.environ.get("SIM_SEED", "42"))
rng = random.Random(SEED)

# ── Fake DB ─────────────────────────────────────────────────────────────────
class FakeDB:
    def __init__(self):
        self.writes: dict[str, dict] = {}
        self.write_count = 0

    def document(self, path):
        store = self.writes
        class _Ref:
            id = path.rsplit("/", 1)[-1]
            def set(self_inner, data, merge=False):
                if merge and path in store:
                    store[path] = {**store[path], **data}
                else:
                    store[path] = dict(data)
        return _Ref()

    def collection(self, _p):
        class _Q:
            def where(self, *a, **kw): return self
            def order_by(self, *a, **kw): return self
            def limit(self, *a, **kw): return self
            def stream(self): return iter([])
        return _Q()


# ── Fixtures: a population of chaperones, students, terminals ───────────────
TENANT = "binus-simprug"

TERMINALS = {
    "term-basement":   {"gradeScopes": [],          "windowOpen": "14:00", "windowClose": "16:00", "releaseGroupId": "rg-basement"},
    "term-myp-tower":  {"gradeScopes": ["7","8","9"],"windowOpen": "14:30", "windowClose": "16:00", "releaseGroupId": "rg-myp"},
    "term-gate-5":     {"gradeScopes": ["5","6"],   "windowOpen": "14:00", "windowClose": "16:00", "releaseGroupId": "rg-pyp-up"},
    "term-gate-4":     {"gradeScopes": ["3","4"],   "windowOpen": "14:00", "windowClose": "16:00", "releaseGroupId": "rg-pyp-low"},
}

CHAPERONES = {
    # employee_no → chaperone doc (None ⇒ unknown card)
    "9001": {"_id":"chap-9001","name":"Mr. Albert (4C dad)","relation":"father",
             "facePaths":["tenants/binus-simprug/chaperones/9001/face.jpg"],
             "authorizedStudentIds":["std-albert"]},
    "9002": {"_id":"chap-9002","name":"Ms. Bella (5C+6B mum)","relation":"mother",
             "facePaths":["tenants/binus-simprug/chaperones/9002/face.jpg"],
             "authorizedStudentIds":["std-bella1","std-bella2"]},
    "9003": {"_id":"chap-9003","name":"Mr. Cedric (8A dad)","relation":"father",
             "facePaths":["tenants/binus-simprug/chaperones/9003/face.jpg"],
             "authorizedStudentIds":["std-cedric"]},
    "9004": {"_id":"chap-9004","name":"Ms. Diana (suspended)","relation":"mother",
             "facePaths":["tenants/binus-simprug/chaperones/9004/face.jpg"],
             "authorizedStudentIds":["std-diana"],
             "suspendedAt":"2026-05-01T00:00:00Z"},
    "9005": {"_id":"chap-9005","name":"Mr. Edwin (re-enroll due)","relation":"father",
             "facePaths":["tenants/binus-simprug/chaperones/9005/face.jpg"],
             "authorizedStudentIds":["std-edwin"],
             "reEnrollDueAt": (datetime.now(WIB) - timedelta(days=10)).isoformat()},
    "9006": {"_id":"chap-9006","name":"Mr. Felix (no kids yet)","relation":"father",
             "facePaths":["tenants/binus-simprug/chaperones/9006/face.jpg"],
             "authorizedStudentIds":[]},
    # 9999 intentionally NOT in dict → unknown_chaperone case
}

STUDENTS = {
    "std-albert":  [{"id":"std-albert", "name":"Albert Arthur",  "homeroom":"4C"}],
    "std-bella1":  [{"id":"std-bella1", "name":"Bella One",      "homeroom":"5C"}],
    "std-bella2":  [{"id":"std-bella2", "name":"Bella Two",      "homeroom":"6B"}],
    "std-cedric":  [{"id":"std-cedric", "name":"Cedric Storm",   "homeroom":"8A"}],
    "std-diana":   [{"id":"std-diana",  "name":"Diana Park",     "homeroom":"5A"}],
    "std-edwin":   [{"id":"std-edwin",  "name":"Edwin Lim",      "homeroom":"6A"}],
}

SETTINGS = {
    "cooldownSeconds": 300,     # same parent re-tap window
    "interParentSeconds": 2,    # back-to-back guard
    "holdSeconds": 60,
    "warmupMinutes": 30,
    "enforceWindow": True,
}


def fake_resolve_chaperone(_db, _tid, employee_no):
    return CHAPERONES.get(employee_no)


def fake_resolve_students(_db, _tid, ids):
    out = []
    for sid in ids:
        out.extend(STUDENTS.get(sid, []))
    return out


def fake_terminal_grade_scopes(_db, _tid, terminal_id):
    return list(TERMINALS.get(terminal_id, {}).get("gradeScopes") or [])


def fake_resolve_window(_db, _tid, terminal_id):
    t = TERMINALS.get(terminal_id, {})
    if t.get("windowOpen") and t.get("windowClose"):
        return {"open": t["windowOpen"], "close": t["windowClose"]}
    return None


def fake_release_group(_db, _tid, terminal_id):
    return TERMINALS.get(terminal_id, {}).get("releaseGroupId")


def fake_settings(_tid):
    return dict(SETTINGS)


def fake_terminal_doc_cached(_db, _tid, terminal_id):
    return dict(TERMINALS.get(terminal_id, {}))


# ── Scan generator: realistic chaperone arrival pattern ─────────────────────
def generate_scans(n: int):
    """Yield (sim_clock, employee_no, terminal_id) tuples.

    Mix:
      60% normal known chaperones at the right terminal
      10% same-parent re-tap within cooldown   (will be SKIPPED)
      10% back-to-back different parents       (50% will be THROTTLED)
       8% unknown card (9999)                  → silent
       7% wrong terminal (grade mismatch)      → silent
       3% suspended / re-enroll due            → red / yellow
       2% chaperone with no students           → yellow

    Time advances by U(0.5, 5.0) seconds between scans (real pickup rush).
    """
    sim_t = datetime(2026, 5, 25, 14, 35, 0, tzinfo=WIB)  # 14:35 = in-window
    known = ["9001", "9002", "9003"]
    last_scan = None
    for _ in range(n):
        roll = rng.random()
        if roll < 0.60:
            emp = rng.choice(known)
            term = {"9001": "term-gate-4", "9002": "term-gate-5",
                    "9003": "term-myp-tower"}[emp]
            gap = rng.uniform(0.5, 5.0)
        elif roll < 0.70 and last_scan:
            # Re-tap same parent same terminal — should be dropped by cooldown
            emp, term = last_scan
            gap = rng.uniform(0.5, 4.0)
        elif roll < 0.80:
            # Back-to-back different parents at same terminal
            emp = rng.choice(known)
            term = {"9001": "term-gate-4", "9002": "term-gate-5",
                    "9003": "term-myp-tower"}[emp]
            gap = rng.uniform(0.1, 1.5)   # <2s → throttled
        elif roll < 0.88:
            emp, term = "9999", rng.choice(list(TERMINALS.keys()))
            gap = rng.uniform(1.0, 5.0)
        elif roll < 0.95:
            # Wrong-terminal: PYP parent (kid in 4C / 5C) at MYP tower
            emp = rng.choice(["9001", "9002"])
            term = "term-myp-tower"
            gap = rng.uniform(1.0, 5.0)
        elif roll < 0.98:
            emp = rng.choice(["9004", "9005"])
            term = "term-gate-5"
            gap = rng.uniform(1.0, 5.0)
        else:
            emp, term = "9006", "term-gate-4"
            gap = rng.uniform(1.0, 5.0)

        sim_t = sim_t + timedelta(seconds=gap)
        last_scan = (emp, term)
        yield sim_t, emp, term


# ── Runner ──────────────────────────────────────────────────────────────────
def run():
    # Clear caches
    for c in (pew._chaperone_cache, pew._student_cache, pew._terminal_doc_cache,
              pew._settings_cache, pew._terminal_group_cache,
              pew._recent_scan_cache, pew._recent_terminal_scan):
        c.clear()

    db = FakeDB()
    decisions: dict[str, int] = {}
    skipped = 0
    latencies_ms: list[float] = []

    # Capture per-scan "now" so the writer sees the simulated clock.
    sim_now = {"t": None}
    pew._now = lambda: sim_now["t"]  # type: ignore

    with mock.patch.object(pew, "_get_db", return_value=db), \
         mock.patch.object(pew, "_resolve_chaperone", side_effect=fake_resolve_chaperone), \
         mock.patch.object(pew, "_resolve_students", side_effect=fake_resolve_students), \
         mock.patch.object(pew, "_get_pickup_settings", side_effect=fake_settings), \
         mock.patch.object(pew, "_terminal_grade_scopes", side_effect=fake_terminal_grade_scopes), \
         mock.patch.object(pew, "_resolve_window", side_effect=fake_resolve_window), \
         mock.patch.object(pew, "_resolve_release_group_id", side_effect=fake_release_group), \
         mock.patch.object(pew, "_get_terminal_doc_cached", side_effect=fake_terminal_doc_cached), \
         mock.patch.object(pew._post_write_pool, "submit",
                           side_effect=lambda fn, *a, **kw: types.SimpleNamespace(
                               result=lambda: None, done=lambda: True)), \
         mock.patch.object(pew.tenancy, "get_tenant_id", return_value=TENANT):

        print(f"\n▶ Simulating {SIM_SCANS} chaperone scans (seed={SEED})")
        print(f"  fake-Firestore, 14:35–pickup-rush window, {len(CHAPERONES)} chaperones,"
              f" {len(TERMINALS)} terminals\n")

        wall_t0 = time.perf_counter()
        for i, (clock, emp, term) in enumerate(generate_scans(SIM_SCANS), 1):
            sim_now["t"] = clock
            t0 = time.perf_counter()
            res = pew.record_pickup_event(
                tenant_id=TENANT,
                employee_no=emp,
                chaperone_name_hint=None,
                scanned_at=clock,
                device_name=term.replace("term-", "Term "),
                gate=term.replace("term-", ""),
                terminal_id=term,
                jpeg_bytes=None,
                # Sprinkle a few FR signals so the fr.* block exercises.
                fr_confidence=rng.uniform(0.82, 0.99),
                liveness_score=rng.uniform(0.80, 0.99),
                liveness_passed=True,
                fr_engine="hikvision",
            )
            latencies_ms.append((time.perf_counter() - t0) * 1000)
            if res is None:
                skipped += 1
                continue
            d = res.get("decision", "?")
            decisions[d] = decisions.get(d, 0) + 1
        wall_ms = (time.perf_counter() - wall_t0) * 1000

    # ── Report ─────────────────────────────────────────────────────────────
    written = sum(decisions.values())
    sorted_lat = sorted(latencies_ms)
    def pct(q): return sorted_lat[min(len(sorted_lat) - 1, int(q * len(sorted_lat)))]
    mean = sum(latencies_ms) / len(latencies_ms)

    print(f"\n──────────── Simulation Report ────────────")
    print(f"  scans submitted   : {SIM_SCANS}")
    print(f"  events written    : {written}")
    print(f"  skipped (throttle): {skipped}")
    print(f"  wall time         : {wall_ms:.1f}ms")
    print(f"  effective rate    : {SIM_SCANS / (wall_ms/1000):.0f} scans/sec")
    print(f"\n  hot-path latency (record_pickup_event):")
    print(f"    p50={pct(0.50):.2f}ms  p95={pct(0.95):.2f}ms  "
          f"p99={pct(0.99):.2f}ms  max={sorted_lat[-1]:.2f}ms  mean={mean:.2f}ms")
    print(f"\n  decision breakdown:")
    iPad_visible = {"ok", "reenroll_overdue", "suspended", "outside_window"}
    for d, n in sorted(decisions.items(), key=lambda kv: -kv[1]):
        visible = "✓ shown on iPad" if d in iPad_visible else "✗ silent (audit only)"
        print(f"    {d:<22} {n:>4}   {visible}")
    print()

    # Sanity assertions — fail loudly if behavior drifts.
    assert written + skipped == SIM_SCANS, "scan accounting mismatch"
    assert decisions.get("ok", 0) > 0, "no successful pickups recorded!"
    assert skipped > 0, "throttle never fired — expected some duplicate skips"
    print("  ✓ all sanity assertions passed\n")


if __name__ == "__main__":
    run()
