"""
Unit tests for pickup_event_writer.record_pickup_event — the chaperone-
at-terminal hot path.

Covers the decision matrix that drives what the iPad teacher PWA actually
shows when a parent taps the Hikvision face terminal at pickup time:

  decision              card_state   iPad shows?   note
  ──────────────────    ──────────   ───────────   ──────────────────────────
  ok                    green         ✓            normal release
  reenroll_overdue      yellow        ✓            still releases, banner
  suspended             red           ✓            block, officer must approve
  unknown_chaperone     silent        ✗            audit only
  wrong_terminal        silent        ✗            grade-scoped terminal
  outside_window        info          ✓            outside pickupWindow

Plus throttle/cooldown short-circuits that return None (no Firestore write).

We do NOT touch real Firestore. The module under test is patched so:
  * _get_db() returns a fake DB
  * Firestore reads (_resolve_chaperone, _resolve_students, _resolve_window,
    _terminal_grade_scopes, _get_pickup_settings, _resolve_release_group_id,
    _get_terminal_doc_cached) return injectable fixtures
  * _post_write_pool.submit() runs callables inline (synchronous) so side
    effects can be asserted — but we mostly assert on the doc payload.
"""
from __future__ import annotations

import importlib
import sys
import types
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

# Ensure backend/ is on path
BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

# Stub firebase_admin BEFORE importing the writer (it imports at module level).
if "firebase_admin" not in sys.modules:
    fake_fa = types.ModuleType("firebase_admin")
    fake_fa._apps = {}            # truthy/falsy probe in _get_db
    sys.modules["firebase_admin"] = fake_fa
    fake_fs = types.ModuleType("firebase_admin.firestore")
    fake_fs.client = lambda: None
    sys.modules["firebase_admin.firestore"] = fake_fs

import pickup_event_writer as pew  # noqa: E402

WIB = timezone(timedelta(hours=7))


class FakeDocRef:
    def __init__(self, store: dict, path: str):
        self._store = store
        self._path = path
        self.id = path.rsplit("/", 1)[-1]

    def set(self, data, merge=False):
        if merge and self._path in self._store:
            self._store[self._path] = {**self._store[self._path], **data}
        else:
            self._store[self._path] = dict(data)
        return None


class FakeDB:
    """Captures writes; the test asserts on what landed."""
    def __init__(self):
        self.writes: dict[str, dict] = {}

    def document(self, path: str):
        return FakeDocRef(self.writes, path)

    def collection(self, _path: str):
        # not exercised in this test set (no cooldown Firestore lookups
        # because the in-process cache short-circuits first, and we keep
        # cooldownSeconds=0 in default settings).
        class _Empty:
            def where(self, *a, **kw): return self
            def order_by(self, *a, **kw): return self
            def limit(self, *a, **kw): return self
            def stream(self): return iter([])
        return _Empty()


class _Snap:
    def __init__(self, doc_id, data=None, exists=True):
        self.id = doc_id
        self._data = data or {}
        self.exists = exists

    def to_dict(self):
        return dict(self._data)


class _Query:
    def __init__(self, docs):
        self._docs = docs

    def where(self, *args, **kwargs):
        return self

    def limit(self, *_args, **_kwargs):
        return self

    def stream(self):
        return iter(self._docs)


class PairingDB:
    def __init__(self, release_group_docs=None):
        self.terminal_snap = _Snap("term-g5", {}, exists=True)
        self.release_group_docs = [
            _Snap("rg-grade-5", {"terminalIds": ["term-g5"], "tabletDeviceId": "ipad-grade-5"}),
        ] if release_group_docs is None else release_group_docs

    def document(self, _path: str):
        class _Doc:
            def __init__(self, snap):
                self._snap = snap

            def get(self):
                return self._snap
        return _Doc(self.terminal_snap)

    def collection(self, _path: str):
        return _Query(self.release_group_docs)


class PickupEventWriterTests(unittest.TestCase):
    """Each test gets a clean cache + fake DB + patched lookups."""

    def setUp(self):
        # Clear ALL hot-path caches between tests so iterations don't leak.
        pew._chaperone_cache.clear()
        pew._student_cache.clear()
        pew._terminal_doc_cache.clear()
        pew._settings_cache.clear()
        pew._terminal_group_cache.clear()
        pew._recent_scan_cache.clear()
        pew._recent_terminal_scan.clear()

        self.db = FakeDB()
        self.scanned_at = datetime(2026, 5, 25, 14, 30, 0, tzinfo=WIB)  # 14:30 WIB
        # Freeze "now" inside the writer to a fixed within-window time.
        self._now_patch = mock.patch.object(pew, "_now", return_value=self.scanned_at)
        self._now_patch.start()
        # _get_db must return our fake.
        self._db_patch = mock.patch.object(pew, "_get_db", return_value=self.db)
        self._db_patch.start()
        # Run post-write work inline so side effects (if any) don't leak past test.
        # We deliberately wrap submit to capture but NOT execute by default —
        # the doc that lands in self.db.writes is what we care about.
        self._pool_patch = mock.patch.object(
            pew._post_write_pool, "submit",
            side_effect=lambda fn, *a, **kw: types.SimpleNamespace(
                result=lambda: None, done=lambda: True))
        self._pool_patch.start()
        # Default tenancy resolution
        self._tid_patch = mock.patch.object(pew.tenancy, "get_tenant_id",
                                            return_value="binus-simprug")
        self._tid_patch.start()

    def tearDown(self):
        mock.patch.stopall()

    # ── helpers ─────────────────────────────────────────────────────────
    def _stub_lookups(self, *, chaperone=None, students=None, settings=None,
                      grade_scopes=None, window=None, release_group=None,
                      terminal_doc=None):
        """Install per-test stubs for every Firestore-touching helper."""
        self.enterContext(mock.patch.object(
            pew, "_resolve_chaperone", return_value=chaperone))
        self.enterContext(mock.patch.object(
            pew, "_resolve_students", return_value=list(students or [])))
        self.enterContext(mock.patch.object(
            pew, "_get_pickup_settings", return_value=dict(settings or {})))
        self.enterContext(mock.patch.object(
            pew, "_terminal_grade_scopes", return_value=list(grade_scopes or [])))
        self.enterContext(mock.patch.object(
            pew, "_resolve_window", return_value=window))
        self.enterContext(mock.patch.object(
            pew, "_resolve_release_group_id", return_value=release_group))
        self.enterContext(mock.patch.object(
            pew, "_get_terminal_doc_cached", return_value=dict(terminal_doc or {})))

    def enterContext(self, cm):
        # Backport for py<3.11
        if hasattr(super(), "enterContext"):
            return super().enterContext(cm)
        self.addCleanup(cm.__exit__, None, None, None)
        return cm.__enter__()

    def _record(self, **overrides):
        kwargs = dict(
            tenant_id="binus-simprug",
            employee_no="9001",
            chaperone_name_hint="Mr. Test",
            scanned_at=self.scanned_at,
            device_name="Basement 1",
            gate="basement",
            terminal_id="term-bsm-1",
            jpeg_bytes=None,
        )
        kwargs.update(overrides)
        return pew.record_pickup_event(**kwargs)

    def _written_event(self):
        """Return the single pickup_events doc that hit the fake DB."""
        events = [v for k, v in self.db.writes.items()
                  if "pickup_events/" in k]
        self.assertEqual(len(events), 1, f"expected 1 event, got {len(events)}")
        return events[0]

    # ── 1. Happy path: known chaperone + 1 student + within window ──────
    def test_known_chaperone_in_window_emits_green_ok(self):
        self._stub_lookups(
            chaperone={
                "_id": "chap-9001",
                "name": "Mr. Test",
                "relation": "father",
                "facePaths": ["tenants/binus-simprug/chaperones/9001/face.jpg"],
                "phone": "+62-xxx",
                "authorizedStudentIds": ["std-001"],
            },
            students=[{"id": "std-001", "name": "Alice", "homeroom": "4A"}],
            settings={"cooldownSeconds": 0, "holdSeconds": 45},
            window={"open": "14:00", "close": "16:00"},
        )
        out = self._record()
        self.assertIsNotNone(out)
        doc = self._written_event()
        self.assertEqual(doc["decision"], "ok")
        self.assertEqual(doc["cardState"], "green")
        self.assertEqual(doc["status"], "pending")
        self.assertEqual(doc["holdSeconds"], 45)
        self.assertEqual(doc["chaperone"]["id"], "chap-9001")
        self.assertEqual(len(doc["students"]), 1)
        self.assertEqual(doc["students"][0]["name"], "Alice")
        self.assertIsNone(doc["overrideCode"], "ok events should not have override code")
        self.assertEqual(doc["terminalId"], "term-bsm-1")

    # ── 2. Unknown chaperone → silent audit event ───────────────────────
    def test_unknown_chaperone_is_silent_audit(self):
        self._stub_lookups(chaperone=None, students=[], settings={"cooldownSeconds": 0})
        # No name hint → writer must synthesize "Unknown (9999)".
        out = self._record(employee_no="9999", chaperone_name_hint=None)
        self.assertIsNotNone(out)
        doc = self._written_event()
        self.assertEqual(doc["decision"], "unknown_chaperone")
        self.assertEqual(doc["cardState"], "silent")
        self.assertEqual(doc["status"], "hidden")
        self.assertEqual(doc["chaperone"]["id"], None)
        self.assertIn("Unknown", doc["chaperone"]["name"])
        self.assertIn("9999", doc["chaperone"]["name"])
        self.assertIsNotNone(doc["overrideCode"])
        self.assertRegex(doc["overrideCode"], r"^\d{6}$")

    # ── 3. Suspended chaperone → red card, bypasses cooldown ────────────
    def test_suspended_chaperone_emits_red_card(self):
        self._stub_lookups(
            chaperone={
                "_id": "chap-9002", "name": "Susp Ended",
                "suspendedAt": "2026-05-01T00:00:00Z",
                "facePaths": [], "authorizedStudentIds": ["s1"],
            },
            students=[{"id": "s1", "name": "Bobby", "homeroom": "5B"}],
            settings={"cooldownSeconds": 0},
        )
        out = self._record(employee_no="9002")
        doc = self._written_event()
        self.assertEqual(doc["decision"], "suspended")
        self.assertEqual(doc["cardState"], "red")
        self.assertEqual(doc["status"], "pending")
        self.assertTrue(doc["chaperone"]["suspended"])
        self.assertIsNotNone(doc["overrideCode"])

    # ── 4. Re-enroll overdue → yellow ───────────────────────────────────
    def test_reenroll_overdue_emits_yellow(self):
        past = (self.scanned_at - timedelta(days=30)).isoformat()
        self._stub_lookups(
            chaperone={
                "_id": "chap-9003", "name": "Past Due",
                "reEnrollDueAt": past,
                "facePaths": [], "authorizedStudentIds": ["s1"],
            },
            students=[{"id": "s1", "name": "Cody", "homeroom": "6A"}],
            settings={"cooldownSeconds": 0},
        )
        out = self._record(employee_no="9003")
        doc = self._written_event()
        self.assertEqual(doc["decision"], "reenroll_overdue")
        self.assertEqual(doc["cardState"], "yellow")
        self.assertTrue(doc["chaperone"]["reEnrollOverdue"])

    # ── 5. Wrong terminal (Grade 2 parent at Grade 5 gate) → silent ─────
    def test_wrong_terminal_when_grade_scope_mismatch(self):
        self._stub_lookups(
            chaperone={
                "_id": "chap-9004", "name": "Wrong Gate",
                "facePaths": [], "authorizedStudentIds": ["s1"],
            },
            students=[{"id": "s1", "name": "Kid2", "homeroom": "2A"}],
            grade_scopes=["5", "6"],   # terminal restricted to grades 5–6
            settings={"cooldownSeconds": 0},
        )
        out = self._record(employee_no="9004")
        doc = self._written_event()
        self.assertEqual(doc["decision"], "wrong_terminal")
        self.assertEqual(doc["cardState"], "silent")
        self.assertEqual(doc["status"], "hidden")

    # ── 5b. Terminal with matching grade scope still goes green ─────────
    def test_grade_scope_match_passes_through(self):
        self._stub_lookups(
            chaperone={
                "_id": "chap-9005", "name": "Right Gate",
                "facePaths": [], "authorizedStudentIds": ["s1"],
            },
            students=[{"id": "s1", "name": "Kid5", "homeroom": "5C"}],
            grade_scopes=["5", "6"],
            settings={"cooldownSeconds": 0},
        )
        doc = (self._record(employee_no="9005"), self._written_event())[1]
        self.assertEqual(doc["decision"], "ok")
        self.assertEqual(doc["cardState"], "green")

    def test_ey_scope_matches_ey_homeroom(self):
        self._stub_lookups(
            chaperone={
                "_id": "chap-9010", "name": "EY Parent",
                "facePaths": [], "authorizedStudentIds": ["s1"],
            },
            students=[{"id": "s1", "name": "EY Kid", "homeroom": "EY1"}],
            grade_scopes=["EY"],
            settings={"cooldownSeconds": 0},
        )
        doc = (self._record(employee_no="9010"), self._written_event())[1]
        self.assertEqual(doc["decision"], "ok")
        self.assertEqual(doc["cardState"], "green")

    def test_ey_scope_matches_exact_ey1_scope(self):
        self._stub_lookups(
            chaperone={
                "_id": "chap-9011", "name": "EY Parent Exact",
                "facePaths": [], "authorizedStudentIds": ["s1"],
            },
            students=[{"id": "s1", "name": "EY Kid", "homeroom": "EY1"}],
            grade_scopes=["EY1"],
            settings={"cooldownSeconds": 0},
        )
        doc = (self._record(employee_no="9011"), self._written_event())[1]
        self.assertEqual(doc["decision"], "ok")
        self.assertEqual(doc["cardState"], "green")

    def test_release_group_resolves_from_release_group_terminal_ids(self):
        pew._terminal_group_cache.clear()
        self.assertEqual(
            pew._resolve_release_group_id(PairingDB(), "binus-simprug", "term-g5"),
            "rg-grade-5",
        )

    def test_missing_release_group_is_not_cached(self):
        pew._terminal_group_cache.clear()
        self.assertIsNone(
            pew._resolve_release_group_id(PairingDB(release_group_docs=[]), "binus-simprug", "term-g5")
        )
        self.assertNotIn("binus-simprug|term-g5", pew._terminal_group_cache)

    def test_release_group_cache_ttl_is_short_for_admin_pairing_changes(self):
        self.assertLessEqual(pew._TERMINAL_CACHE_TTL, 5.0)

    # ── 6. Outside-window → info card ───────────────────────────────────
    def test_outside_window_emits_info_card(self):
        # Set "now" to 09:00 WIB — well before any pickupWindow open.
        early = datetime(2026, 5, 25, 9, 0, 0, tzinfo=WIB)
        mock.patch.object(pew, "_now", return_value=early).start()
        self._stub_lookups(
            chaperone={
                "_id": "chap-9006", "name": "Too Early",
                "facePaths": [], "authorizedStudentIds": ["s1"],
            },
            students=[{"id": "s1", "name": "Earl", "homeroom": "4A"}],
            settings={"cooldownSeconds": 0, "warmupMinutes": 0, "enforceWindow": True},
            window={"open": "14:00", "close": "16:00"},
        )
        out = self._record(employee_no="9006")
        doc = self._written_event()
        self.assertEqual(doc["decision"], "outside_window")
        self.assertEqual(doc["cardState"], "info")

    # ── 7. Chaperone with no authorized students → yellow ───────────────
    def test_no_students_emits_yellow(self):
        self._stub_lookups(
            chaperone={
                "_id": "chap-9007", "name": "Orphaned Parent",
                "facePaths": [], "authorizedStudentIds": [],
            },
            students=[],
            settings={"cooldownSeconds": 0},
        )
        out = self._record(employee_no="9007")
        doc = self._written_event()
        self.assertEqual(doc["decision"], "ok")
        self.assertEqual(doc["cardState"], "yellow",
                         "chaperone with zero students must show yellow")

    # ── 8. Inter-parent throttle (2 scans <2s apart) drops the second ───
    def test_inter_parent_throttle_drops_back_to_back(self):
        self._stub_lookups(
            chaperone={
                "_id": "chap-9008", "name": "Fast Parent",
                "facePaths": [], "authorizedStudentIds": ["s1"],
            },
            students=[{"id": "s1", "name": "Kid", "homeroom": "4A"}],
            settings={"cooldownSeconds": 0, "interParentSeconds": 2},
        )
        first = self._record(employee_no="9008")
        self.assertIsNotNone(first)
        # Second scan at "same instant" → throttle blocks it.
        second = self._record(employee_no="9009")  # different parent, same terminal
        self.assertIsNone(second, "back-to-back terminal scan must be throttled")

    # ── 9. Cooldown blocks same-parent re-tap on same terminal ──────────
    def test_same_parent_cooldown_blocks_retap(self):
        self._stub_lookups(
            chaperone={
                "_id": "chap-9010", "name": "Repeat Parent",
                "facePaths": [], "authorizedStudentIds": ["s1"],
            },
            students=[{"id": "s1", "name": "Kid", "homeroom": "4A"}],
            settings={"cooldownSeconds": 300, "interParentSeconds": 0},
        )
        a = self._record(employee_no="9010")
        b = self._record(employee_no="9010")
        self.assertIsNotNone(a)
        self.assertIsNone(b, "same chap+terminal within cooldown must be skipped")

    # ── 10. FR signals flow through to the doc ──────────────────────────
    def test_fr_signals_persisted_when_provided(self):
        self._stub_lookups(
            chaperone={
                "_id": "chap-9011", "name": "FR Parent",
                "facePaths": ["tenants/binus-simprug/chaperones/9011/face.jpg"],
                "authorizedStudentIds": ["s1"],
            },
            students=[{"id": "s1", "name": "Kid", "homeroom": "4A"}],
            settings={"cooldownSeconds": 0},
        )
        self._record(
            employee_no="9011",
            fr_confidence=0.93, fr_distance=0.31,
            liveness_score=0.88, liveness_passed=True, spoof_flag=False,
            fr_retries=1, fr_engine="hikvision",
        )
        doc = self._written_event()
        fr = doc["fr"]
        self.assertAlmostEqual(fr["confidence"], 0.93)
        self.assertAlmostEqual(fr["distance"], 0.31)
        self.assertAlmostEqual(fr["liveness"], 0.88)
        self.assertTrue(fr["livenessPassed"])
        self.assertFalse(fr["spoof"])
        self.assertEqual(fr["retries"], 1)
        self.assertEqual(fr["engine"], "hikvision")
        self.assertEqual(fr["enrolledPhotoPath"],
                         "tenants/binus-simprug/chaperones/9011/face.jpg")

    # ── 11. JPEG capture path is baked even when upload is deferred ─────
    def test_jpeg_capture_path_set_when_bytes_provided(self):
        self._stub_lookups(
            chaperone={
                "_id": "chap-9012", "name": "JPEG Parent",
                "facePaths": [], "authorizedStudentIds": ["s1"],
            },
            students=[{"id": "s1", "name": "Kid", "homeroom": "4A"}],
            settings={"cooldownSeconds": 0},
        )
        self._record(employee_no="9012", jpeg_bytes=b"\xff\xd8\xff\xe0FAKE")
        doc = self._written_event()
        self.assertIsNotNone(doc["capturePath"])
        self.assertIn(doc["eventId"], doc["capturePath"])


if __name__ == "__main__":
    unittest.main()
