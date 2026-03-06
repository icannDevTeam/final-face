#!/usr/bin/env python3
"""
BINUS Attendance – Stress Test
Simulates a morning rush of students hitting the system concurrently.

Tests:
  1. Student lookup (BINUS API via local Next.js)
  2. Student lookup (BINUS API via Vercel)
  3. Firebase attendance write (simulating mobile check-in)
  4. Dashboard read (attendance/today)
  5. Concurrent burst (all students at once)
"""

import time
import json
import random
import threading
import statistics
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# ─── Configuration ──────────────────────────────────────────────────
LOCAL_BASE = "http://localhost:3000"
VERCEL_BASE = "https://dataset-sigma.vercel.app"

STUDENT_IDS = [
    "1870001744",  # Connor Henry Owen
    "1870002777",  # Ayla Madina Zulkarnain
    "2070003324",  # Cedric Carrington Cahaya
    "2170003338",  # Anderson Ian Roesmin
    "2370007317",  # Benjamin Arandra Siregar
    "2570010026",  # Akshay Azahran Jetty
]

# Simulated concurrent students (repeat IDs to simulate 30 students)
SIMULATED_STUDENTS = STUDENT_IDS * 5  # 30 concurrent requests

# Firebase config
import firebase_admin
from firebase_admin import credentials, firestore

if not firebase_admin._apps:
    cred = credentials.Certificate("facial-attendance-binus-firebase-adminsdk.json")
    firebase_admin.initialize_app(cred)
db = firestore.client()

# ─── Helpers ────────────────────────────────────────────────────────
class Stats:
    def __init__(self, name):
        self.name = name
        self.times = []
        self.successes = 0
        self.failures = 0
        self.errors = []
        self.lock = threading.Lock()

    def record(self, elapsed, success, error=None):
        with self.lock:
            self.times.append(elapsed)
            if success:
                self.successes += 1
            else:
                self.failures += 1
                if error:
                    self.errors.append(error)

    def report(self):
        if not self.times:
            return f"  {self.name}: NO DATA"
        avg = statistics.mean(self.times)
        med = statistics.median(self.times)
        p95 = sorted(self.times)[int(len(self.times) * 0.95)] if len(self.times) >= 2 else max(self.times)
        mx = max(self.times)
        mn = min(self.times)
        total = self.successes + self.failures
        return (
            f"  {self.name}:\n"
            f"    Requests:  {total} ({self.successes} ok, {self.failures} fail)\n"
            f"    Avg:       {avg*1000:.0f}ms\n"
            f"    Median:    {med*1000:.0f}ms\n"
            f"    P95:       {p95*1000:.0f}ms\n"
            f"    Min/Max:   {mn*1000:.0f}ms / {mx*1000:.0f}ms"
        )


def timed_request(method, url, **kwargs):
    """Make a request and return (elapsed_seconds, response, error)."""
    kwargs.setdefault("timeout", 60)
    start = time.perf_counter()
    try:
        resp = method(url, **kwargs)
        elapsed = time.perf_counter() - start
        return elapsed, resp, None
    except Exception as e:
        elapsed = time.perf_counter() - start
        return elapsed, None, str(e)


# ─── Test Functions ─────────────────────────────────────────────────

def test_student_lookup(base_url, student_id, stats):
    """POST /api/student/lookup"""
    elapsed, resp, err = timed_request(
        requests.post,
        f"{base_url}/api/student/lookup",
        json={"studentId": student_id},
        headers={"Content-Type": "application/json"},
    )
    if err:
        stats.record(elapsed, False, err)
        return
    success = resp.status_code == 200 and resp.json().get("success")
    stats.record(elapsed, success, None if success else f"HTTP {resp.status_code}: {resp.text[:100]}")


def test_firebase_write(student_id, name, stats):
    """Simulate a mobile attendance write to Firebase."""
    start = time.perf_counter()
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        doc_id = f"STRESS-{student_id}"
        db.collection("attendance").document(today).collection("records").document(doc_id).set({
            "employeeNo": doc_id,
            "name": f"[STRESS] {name}",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "stress_test",
            "homeroom": "TEST",
            "grade": "TEST",
            "status": "Test",
            "late": False,
        })
        elapsed = time.perf_counter() - start
        stats.record(elapsed, True)
    except Exception as e:
        elapsed = time.perf_counter() - start
        stats.record(elapsed, False, str(e))


def test_firebase_read(stats):
    """Read today's attendance (dashboard load)."""
    start = time.perf_counter()
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        recs = list(db.collection("attendance").document(today).collection("records").stream())
        elapsed = time.perf_counter() - start
        stats.record(elapsed, True)
    except Exception as e:
        elapsed = time.perf_counter() - start
        stats.record(elapsed, False, str(e))


def test_health(base_url, stats):
    """GET /api/health"""
    elapsed, resp, err = timed_request(requests.get, f"{base_url}/api/health")
    if err:
        stats.record(elapsed, False, err)
    else:
        stats.record(elapsed, resp.status_code == 200, None if resp.status_code == 200 else f"HTTP {resp.status_code}")


def cleanup_stress_records():
    """Remove stress test records from Firebase."""
    today = datetime.now().strftime("%Y-%m-%d")
    recs = db.collection("attendance").document(today).collection("records").stream()
    deleted = 0
    for r in recs:
        if r.id.startswith("STRESS-"):
            r.reference.delete()
            deleted += 1
    return deleted


# ─── Main ───────────────────────────────────────────────────────────

def run_stress_test():
    print("=" * 60)
    print("  BINUS Attendance — Stress Test")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} WIB")
    print("=" * 60)

    names = {
        "1870001744": "Connor Henry Owen",
        "1870002777": "Ayla Madina Zulkarnain",
        "2070003324": "Cedric Carrington Cahaya",
        "2170003338": "Anderson Ian Roesmin",
        "2370007317": "Benjamin Arandra Siregar",
        "2570010026": "Akshay Azahran Jetty",
    }

    # ── Phase 1: Sequential warm-up (1 request each) ──
    print("\n▸ Phase 1: Warm-up (sequential, 1 per endpoint)")
    s1 = Stats("Local health")
    test_health(LOCAL_BASE, s1)
    print(f"  Local health: {'OK' if s1.successes else 'FAIL'} ({s1.times[0]*1000:.0f}ms)")

    s2 = Stats("Vercel health")
    test_health(VERCEL_BASE, s2)
    print(f"  Vercel health: {'OK' if s2.successes else 'FAIL'} ({s2.times[0]*1000:.0f}ms)")

    s3 = Stats("Local lookup warmup")
    test_student_lookup(LOCAL_BASE, "2570010026", s3)
    print(f"  Local lookup: {'OK' if s3.successes else 'FAIL'} ({s3.times[0]*1000:.0f}ms)")

    # ── Phase 2: Concurrent student lookups (local) ──
    print(f"\n▸ Phase 2: Concurrent student lookups — LOCAL ({len(SIMULATED_STUDENTS)} requests)")
    local_lookup = Stats("Local /api/student/lookup")
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=15) as pool:
        futures = [
            pool.submit(test_student_lookup, LOCAL_BASE, sid, local_lookup)
            for sid in SIMULATED_STUDENTS
        ]
        for f in as_completed(futures):
            f.result()
    phase2_time = time.perf_counter() - start
    print(local_lookup.report())
    print(f"    Throughput: {len(SIMULATED_STUDENTS)/phase2_time:.1f} req/s (wall: {phase2_time:.1f}s)")

    # ── Phase 3: Concurrent student lookups (Vercel) ──
    print(f"\n▸ Phase 3: Concurrent student lookups — VERCEL ({len(STUDENT_IDS)} requests)")
    vercel_lookup = Stats("Vercel /api/student/lookup")
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [
            pool.submit(test_student_lookup, VERCEL_BASE, sid, vercel_lookup)
            for sid in STUDENT_IDS
        ]
        for f in as_completed(futures):
            f.result()
    phase3_time = time.perf_counter() - start
    print(vercel_lookup.report())
    print(f"    Throughput: {len(STUDENT_IDS)/phase3_time:.1f} req/s (wall: {phase3_time:.1f}s)")

    # ── Phase 4: Concurrent Firebase writes (simulating 30 check-ins) ──
    print(f"\n▸ Phase 4: Concurrent Firebase writes ({len(SIMULATED_STUDENTS)} check-ins)")
    fb_write = Stats("Firebase attendance write")
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=15) as pool:
        futures = [
            pool.submit(test_firebase_write, sid, names.get(sid, "Unknown"), fb_write)
            for sid in SIMULATED_STUDENTS
        ]
        for f in as_completed(futures):
            f.result()
    phase4_time = time.perf_counter() - start
    print(fb_write.report())
    print(f"    Throughput: {len(SIMULATED_STUDENTS)/phase4_time:.1f} writes/s (wall: {phase4_time:.1f}s)")

    # ── Phase 5: Concurrent Firebase reads (dashboard load) ──
    print(f"\n▸ Phase 5: Concurrent dashboard reads (10 simultaneous)")
    fb_read = Stats("Firebase attendance read")
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(test_firebase_read, fb_read) for _ in range(10)]
        for f in as_completed(futures):
            f.result()
    phase5_time = time.perf_counter() - start
    print(fb_read.report())
    print(f"    Throughput: {10/phase5_time:.1f} reads/s (wall: {phase5_time:.1f}s)")

    # ── Phase 6: Full morning rush simulation ──
    print(f"\n▸ Phase 6: Morning rush — mixed workload (lookup + write + read)")
    rush_lookup = Stats("Rush: student lookup")
    rush_write = Stats("Rush: Firebase write")
    rush_read = Stats("Rush: dashboard read")

    def rush_student(sid):
        """Simulate one student's full check-in flow."""
        # Random delay 0-2s to simulate staggered arrivals
        time.sleep(random.uniform(0, 2))
        # Step 1: Lookup student
        test_student_lookup(LOCAL_BASE, sid, rush_lookup)
        # Step 2: Write attendance
        test_firebase_write(sid, names.get(sid, "Unknown"), rush_write)

    def rush_dashboard():
        """Simulate teacher checking dashboard."""
        time.sleep(random.uniform(0.5, 3))
        test_firebase_read(rush_read)

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = []
        # 30 students checking in
        for sid in SIMULATED_STUDENTS:
            futures.append(pool.submit(rush_student, sid))
        # 5 dashboard reads from teachers
        for _ in range(5):
            futures.append(pool.submit(rush_dashboard))
        for f in as_completed(futures):
            f.result()
    phase6_time = time.perf_counter() - start
    print(rush_lookup.report())
    print(rush_write.report())
    print(rush_read.report())
    print(f"    Total wall time: {phase6_time:.1f}s for 35 operations")

    # ── Cleanup ──
    print("\n▸ Cleanup: removing stress test records...")
    deleted = cleanup_stress_records()
    print(f"  Deleted {deleted} stress test records")

    # ── Summary ──
    all_stats = [local_lookup, vercel_lookup, fb_write, fb_read, rush_lookup, rush_write, rush_read]
    total_req = sum(s.successes + s.failures for s in all_stats)
    total_fail = sum(s.failures for s in all_stats)
    print("\n" + "=" * 60)
    print(f"  SUMMARY: {total_req} total requests, {total_fail} failures")
    if total_fail == 0:
        print("  ✅ ALL PASSED — System handles concurrent load well")
    else:
        print(f"  ⚠️  {total_fail} FAILURES detected:")
        for s in all_stats:
            if s.failures > 0:
                print(f"    {s.name}: {s.failures} failures")
                for e in s.errors[:3]:
                    print(f"      → {e}")
    print("=" * 60)


if __name__ == "__main__":
    run_stress_test()
