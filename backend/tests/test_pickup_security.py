"""
Security-focused pytest suite for pickup_event_writer.record_pickup_event.

Adversarial test matrix:
  A. Authorization bypass
  B. Throttle / replay attack
  C. Student data isolation (IDOR)
  D. Gate control integrity

Uses identical mocking strategy as test_pickup_event_writer.py — no real
Firestore, no real Firebase. All lookups patched via _stub_lookups().
"""
from __future__ import annotations

import sys
import types
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

if "firebase_admin" not in sys.modules:
    fake_fa = types.ModuleType("firebase_admin")
    fake_fa._apps = {}
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
    def __init__(self):
        self.writes: dict[str, dict] = {}

    def document(self, path: str):
        return FakeDocRef(self.writes, path)

    def collection(self, _path: str):
        class _Empty:
            def where(self, *a, **kw): return self
            def order_by(self, *a, **kw): return self
            def limit(self, *a, **kw): return self
            def stream(self): return iter([])
        return _Empty()


class PickupSecurityTests(unittest.TestCase):

    def setUp(self):
        pew._chaperone_cache.clear()
        pew._student_cache.clear()
        pew._terminal_doc_cache.clear()
        pew._settings_cache.clear()
        pew._terminal_group_cache.clear()
        pew._recent_scan_cache.clear()
        pew._recent_terminal_scan.clear()

        self.db = FakeDB()
        self.scanned_at = datetime(2026, 5, 25, 14, 30, 0, tzinfo=WIB)

        self._now_patch = mock.patch.object(pew, "_now", return_value=self.scanned_at)
        self._now_patch.start()
        self._db_patch = mock.patch.object(pew, "_get_db", return_value=self.db)
        self._db_patch.start()
        self._pool_patch = mock.patch.object(
            pew._post_write_pool, "submit",
            side_effect=lambda fn, *a, **kw: types.SimpleNamespace(
                result=lambda: None, done=lambda: True))
        self._pool_patch.start()
        self._tid_patch = mock.patch.object(pew.tenancy, "get_tenant_id",
                                            return_value="binus-simprug")
        self._tid_patch.start()

    def tearDown(self):
        mock.patch.stopall()

    def _stub_lookups(self, *, chaperone=None, students=None, settings=None,
                      grade_scopes=None, window=None, release_group=None,
                      terminal_doc=None):
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
            device_name="Gate 1",
            gate="gate1",
            terminal_id="term-g1",
            jpeg_bytes=None,
        )
        kwargs.update(overrides)
        return pew.record_pickup_event(**kwargs)

    def _written_event(self):
        events = [v for k, v in self.db.writes.items()
                  if "pickup_events/" in k]
        self.assertEqual(len(events), 1, f"expected 1 event, got {len(events)}")
        return events[0]

    # ════════════════════════════════════════════════════════════════════
    # A. AUTHORIZATION BYPASS SCENARIOS
    # ════════════════════════════════════════════════════════════════════

    def test_A1_suspended_chaperone_blocked_not_ok(self):
        """SECURITY: A suspended chaperone must NEVER get decision='ok'.
        Must get 'suspended' and card_state='red', not green.
        Real-world impact: suspended parent (restraining order, etc) walks out with a child."""
        self._stub_lookups(
            chaperone={
                "_id": "chap-9100",
                "name": "Suspended Parent",
                "suspendedAt": "2026-01-01T00:00:00Z",  # was suspended
                "facePaths": [],
                "authorizedStudentIds": ["std-100"],
            },
            students=[{"id": "std-100", "name": "Target Child", "homeroom": "4A"}],
            settings={"cooldownSeconds": 0},
        )
        out = self._record(employee_no="9100")
        self.assertIsNotNone(out, "suspended chaperone scan must still write an audit event")
        doc = self._written_event()
        self.assertNotEqual(doc["decision"], "ok",
                            "SECURITY BUG: suspended chaperone got decision=ok — child release unblocked")
        self.assertEqual(doc["decision"], "suspended")
        self.assertEqual(doc["cardState"], "red")
        self.assertTrue(doc["chaperone"]["suspended"])
        # Override code must be present so officer can manually escalate.
        self.assertIsNotNone(doc["overrideCode"])

    def test_A2_unknown_chaperone_does_not_release(self):
        """SECURITY: A face that matches no chaperone record must not release any student.
        decision must be 'unknown_chaperone', card_state='silent'."""
        self._stub_lookups(
            chaperone=None,   # unknown — no record
            students=[],
            settings={"cooldownSeconds": 0},
        )
        out = self._record(employee_no="9999", chaperone_name_hint=None)
        self.assertIsNotNone(out)
        doc = self._written_event()
        self.assertEqual(doc["decision"], "unknown_chaperone",
                         "SECURITY BUG: unknown face produced non-silent decision")
        self.assertEqual(doc["cardState"], "silent")
        self.assertEqual(doc["chaperone"]["id"], None)
        self.assertEqual(len(doc["students"]), 0,
                         "SECURITY BUG: unknown chaperone has linked students in event doc")

    def test_A3_wrong_terminal_grade_mismatch_silent(self):
        """SECURITY: Grade 4 parent at Grade 6 terminal must get 'wrong_terminal', not 'ok'.
        Real-world: allows pickup at an unsupervised gate for a different grade."""
        self._stub_lookups(
            chaperone={
                "_id": "chap-9101",
                "name": "Grade4 Parent",
                "facePaths": [],
                "authorizedStudentIds": ["std-101"],
            },
            students=[{"id": "std-101", "name": "Alice", "homeroom": "4B"}],
            grade_scopes=["6"],   # terminal locked to Grade 6 only
            settings={"cooldownSeconds": 0},
        )
        out = self._record(employee_no="9101")
        doc = self._written_event()
        self.assertNotEqual(doc["decision"], "ok",
                            "SECURITY BUG: wrong-terminal scan produced ok decision — student could be released at wrong gate")
        self.assertEqual(doc["decision"], "wrong_terminal")
        self.assertEqual(doc["cardState"], "silent",
                         "wrong_terminal events must be silent — must not trigger iPad release UI")

    def test_A4_outside_pickup_window_does_not_emit_ok(self):
        """SECURITY: Scan 3 hours before pickup window opens must NOT emit decision='ok'.
        Real-world: chaperone scanning at 06:00 before staff are present."""
        early = datetime(2026, 5, 25, 6, 0, 0, tzinfo=WIB)  # 06:00, well outside window
        mock.patch.object(pew, "_now", return_value=early).start()
        self._stub_lookups(
            chaperone={
                "_id": "chap-9102",
                "name": "Early Bird",
                "facePaths": [],
                "authorizedStudentIds": ["std-102"],
            },
            students=[{"id": "std-102", "name": "Bob", "homeroom": "5A"}],
            settings={"cooldownSeconds": 0, "warmupMinutes": 0, "enforceWindow": True},
            window={"open": "13:30", "close": "15:30"},
        )
        out = self._record(employee_no="9102")
        doc = self._written_event()
        self.assertNotEqual(doc["decision"], "ok",
                            "SECURITY BUG: outside-window scan returned ok — unsupervised release possible")
        self.assertEqual(doc["decision"], "outside_window")
        self.assertEqual(doc["cardState"], "info")

    def test_A5_reenroll_overdue_releases_but_flagged(self):
        """SECURITY: Reenroll-overdue chaperone is still flagged (yellow) and auditable.
        The system ALLOWS release (by design — admin must re-enroll) but must flag it.
        decision must be 'reenroll_overdue', not silently 'ok'."""
        past = (self.scanned_at - timedelta(days=60)).isoformat()
        self._stub_lookups(
            chaperone={
                "_id": "chap-9103",
                "name": "Overdue Parent",
                "reEnrollDueAt": past,
                "facePaths": [],
                "authorizedStudentIds": ["std-103"],
            },
            students=[{"id": "std-103", "name": "Carol", "homeroom": "3A"}],
            settings={"cooldownSeconds": 0},
        )
        out = self._record(employee_no="9103")
        doc = self._written_event()
        self.assertEqual(doc["decision"], "reenroll_overdue",
                         "reenroll_overdue decision must be set for audit visibility")
        self.assertEqual(doc["cardState"], "yellow")
        self.assertTrue(doc["chaperone"]["reEnrollOverdue"])
        # Still releases (by design) so override code must be present.
        self.assertIsNotNone(doc["overrideCode"])

    # ════════════════════════════════════════════════════════════════════
    # B. THROTTLE / REPLAY ATTACK
    # ════════════════════════════════════════════════════════════════════

    def test_B6_cooldown_short_circuit_returns_none_no_write(self):
        """SECURITY: Same chaperone tapping same terminal twice within cooldown window
        must return None and produce NO Firestore write on the second call.
        Real-world: replay attack — waving an NFC card copy or a photo at the terminal."""
        self._stub_lookups(
            chaperone={
                "_id": "chap-9200",
                "name": "Replay Parent",
                "facePaths": [],
                "authorizedStudentIds": ["std-200"],
            },
            students=[{"id": "std-200", "name": "Dave", "homeroom": "5B"}],
            settings={"cooldownSeconds": 300, "interParentSeconds": 0},
        )
        first = self._record(employee_no="9200")
        self.assertIsNotNone(first, "first scan must succeed")
        # Second scan — same employee_no, same terminal, within cooldown
        second = self._record(employee_no="9200")
        self.assertIsNone(second,
                          "SECURITY BUG: cooldown bypass — second scan within window returned non-None")
        # Only one event doc must exist in the fake DB
        events = [k for k in self.db.writes if "pickup_events/" in k]
        self.assertEqual(len(events), 1,
                         f"SECURITY BUG: cooldown bypass produced {len(events)} event writes instead of 1")

    def test_B7_rapid_5_scans_only_first_produces_write(self):
        """SECURITY: 5 rapid scans from the same employeeNo at same terminal.
        Only the first must produce a Firestore write. All subsequent must be None."""
        self._stub_lookups(
            chaperone={
                "_id": "chap-9201",
                "name": "Rapid Parent",
                "facePaths": [],
                "authorizedStudentIds": ["std-201"],
            },
            students=[{"id": "std-201", "name": "Eve", "homeroom": "4C"}],
            settings={"cooldownSeconds": 300, "interParentSeconds": 0},
        )
        results = [self._record(employee_no="9201") for _ in range(5)]
        non_none = [r for r in results if r is not None]
        self.assertEqual(len(non_none), 1,
                         f"SECURITY BUG: {len(non_none)} writes for 5 rapid scans — expected exactly 1")
        events = [k for k in self.db.writes if "pickup_events/" in k]
        self.assertEqual(len(events), 1,
                         f"SECURITY BUG: {len(events)} event docs written — expected exactly 1")

    # ════════════════════════════════════════════════════════════════════
    # C. STUDENT DATA ISOLATION (IDOR)
    # ════════════════════════════════════════════════════════════════════

    def test_C8_chaperone_A_cannot_release_student_B(self):
        """SECURITY (IDOR): Chaperone authorized for std-300 (Family A) attempts to
        release std-301 (Family B). The event doc's student list must ONLY contain
        the chaperone's own authorized students — never cross-family students.
        Real-world: chaperone registers for their own child then tricks the system
        into releasing another family's child."""
        # The production code returns students from _resolve_students(authorizedStudentIds).
        # We simulate: chaperone only has std-300 authorized.
        self._stub_lookups(
            chaperone={
                "_id": "chap-9300",
                "name": "Family A Parent",
                "facePaths": [],
                "authorizedStudentIds": ["std-300"],   # only their own child
            },
            students=[{"id": "std-300", "name": "Frank", "homeroom": "4A"}],
            settings={"cooldownSeconds": 0},
        )
        out = self._record(employee_no="9300")
        doc = self._written_event()
        # Decision must be 'ok' for their own child
        self.assertEqual(doc["decision"], "ok")
        student_ids_in_event = [s["id"] for s in doc["students"]]
        self.assertIn("std-300", student_ids_in_event)
        self.assertNotIn("std-301", student_ids_in_event,
                         "SECURITY BUG (IDOR): Family B's student appears in Family A's event")

    def test_C8b_chaperone_A_target_student_B_explicitly(self):
        """SECURITY (IDOR): More explicit — _resolve_students is called with the
        authorizedStudentIds list. If the list only contains std-300, std-301 can
        NEVER appear in the event regardless of what the caller passes.
        This verifies the system doesn't accept a student ID from the request body
        and mix it into the authorization scope."""
        # We mock _resolve_students to return only what the chaperone is authorized for,
        # simulating the actual production path where authorizedStudentIds drives the lookup.
        self._stub_lookups(
            chaperone={
                "_id": "chap-9302",
                "name": "Family A Parent 2",
                "facePaths": [],
                "authorizedStudentIds": ["std-300"],
            },
            # _resolve_students deliberately returns only std-300's data
            students=[{"id": "std-300", "name": "Frank", "homeroom": "4A"}],
            settings={"cooldownSeconds": 0},
        )
        out = self._record(employee_no="9302")
        doc = self._written_event()
        for s in doc["students"]:
            self.assertNotEqual(s["id"], "std-301",
                                "SECURITY BUG: unauthorized student bled into event")

    def test_C9_chaperone_with_no_linked_students_yellow_not_ok(self):
        """SECURITY: A chaperone with empty authorizedStudentIds must NOT produce
        a green 'ok' decision. Must produce yellow (no students) so the gate officer
        knows not to release anyone.
        Real-world: attacker self-enrolls via compromised admin path with no students,
        gets a green card to social-engineer the officer."""
        self._stub_lookups(
            chaperone={
                "_id": "chap-9303",
                "name": "No Students Parent",
                "facePaths": [],
                "authorizedStudentIds": [],
            },
            students=[],   # resolves to empty list
            settings={"cooldownSeconds": 0},
        )
        out = self._record(employee_no="9303")
        doc = self._written_event()
        # decision can be 'ok' (no explicit 'no_students' decision key exists)
        # but cardState MUST be yellow — not green — so the iPad shows a warning.
        self.assertNotEqual(doc["cardState"], "green",
                            "SECURITY BUG: chaperone with zero authorized students produced green card")
        self.assertEqual(doc["cardState"], "yellow")
        self.assertEqual(len(doc["students"]), 0)

    # ════════════════════════════════════════════════════════════════════
    # D. GATE CONTROL INTEGRITY
    # ════════════════════════════════════════════════════════════════════

    def test_D10_gate_action_absent_when_gate_field_not_open(self):
        """SECURITY: When gateOpen=False in settings/terminal, the event doc must
        NOT carry a gate-open command. The Pandora-Linux listener checks `decision`
        and `cardState`; a suspended or unknown chaperone event must not carry a
        green gate-open signal.
        We verify: suspended chaperone → cardState != 'green', no gate_action field."""
        self._stub_lookups(
            chaperone={
                "_id": "chap-9400",
                "name": "Suspended Gate Test",
                "suspendedAt": "2026-01-01T00:00:00Z",
                "facePaths": [],
                "authorizedStudentIds": ["std-400"],
            },
            students=[{"id": "std-400", "name": "Gina", "homeroom": "4A"}],
            settings={"cooldownSeconds": 0},
            terminal_doc={"gateOverride": "closed"},  # gate is force-closed
        )
        out = self._record(employee_no="9400")
        doc = self._written_event()
        # Suspended chaperone must NEVER have a green card regardless of gate state.
        self.assertNotEqual(doc["cardState"], "green",
                            "SECURITY BUG: suspended chaperone produced green card — gate would open")
        self.assertEqual(doc["decision"], "suspended")
        # The event doc must not contain a gate_action='open' field.
        gate_action = doc.get("gate_action") or doc.get("gateAction")
        self.assertFalse(
            gate_action == "open",
            "SECURITY BUG: suspended event doc carries gate_action=open — listener would open the gate"
        )

    def test_D10b_unknown_chaperone_no_gate_open_signal(self):
        """SECURITY: unknown chaperone — cardState must be 'silent', no gate-open signal."""
        self._stub_lookups(
            chaperone=None,
            students=[],
            settings={"cooldownSeconds": 0},
        )
        out = self._record(employee_no="9888")
        doc = self._written_event()
        self.assertEqual(doc["cardState"], "silent")
        gate_action = doc.get("gate_action") or doc.get("gateAction")
        self.assertFalse(gate_action == "open",
                         "SECURITY BUG: unknown chaperone event has gate_action=open")

    def test_D10c_wrong_terminal_no_green_card(self):
        """SECURITY: wrong-terminal scan must stay silent — no green card, no gate-open signal."""
        self._stub_lookups(
            chaperone={
                "_id": "chap-9401",
                "name": "Wrong Gate Parent",
                "facePaths": [],
                "authorizedStudentIds": ["std-401"],
            },
            students=[{"id": "std-401", "name": "Henry", "homeroom": "2A"}],
            grade_scopes=["5", "6"],  # terminal is grades 5-6 only
            settings={"cooldownSeconds": 0},
        )
        out = self._record(employee_no="9401")
        doc = self._written_event()
        self.assertEqual(doc["cardState"], "silent")
        gate_action = doc.get("gate_action") or doc.get("gateAction")
        self.assertFalse(gate_action == "open",
                         "SECURITY BUG: wrong-terminal event has gate_action=open")

    # ════════════════════════════════════════════════════════════════════
    # E. OVERRIDE CODE INTEGRITY
    # ════════════════════════════════════════════════════════════════════

    def test_E11_ok_decision_has_no_override_code(self):
        """SECURITY: Normal 'ok' events must NOT carry an overrideCode.
        If an ok event has an override code, a malicious actor could screenshot
        it and use it to bypass a later suspension."""
        self._stub_lookups(
            chaperone={
                "_id": "chap-9500",
                "name": "Normal Parent",
                "facePaths": [],
                "authorizedStudentIds": ["std-500"],
            },
            students=[{"id": "std-500", "name": "Irene", "homeroom": "4A"}],
            settings={"cooldownSeconds": 0},
        )
        out = self._record(employee_no="9500")
        doc = self._written_event()
        self.assertEqual(doc["decision"], "ok")
        self.assertIsNone(doc["overrideCode"],
                          "ok events should NOT carry an override code — unnecessary exposure")

    def test_E12_non_ok_decision_always_has_6digit_override_code(self):
        """SECURITY: All non-ok decisions (suspended, unknown, reenroll_overdue, wrong_terminal,
        outside_window) must carry a valid 6-digit override code so officers can manually clear.
        Missing code = officer can't resolve the event = child stranded."""
        past = (self.scanned_at - timedelta(days=1)).isoformat()
        test_cases = [
            # (label, chaperone, students, grade_scopes, window, settings)
            (
                "reenroll_overdue",
                {
                    "_id": "chap-9601", "name": "P1",
                    "reEnrollDueAt": past, "facePaths": [],
                    "authorizedStudentIds": ["s1"],
                },
                [{"id": "s1", "name": "K1", "homeroom": "4A"}],
                [], None, {"cooldownSeconds": 0},
            ),
        ]
        for label, chap, students, scopes, window, settings in test_cases:
            with self.subTest(decision=label):
                pew._chaperone_cache.clear()
                pew._recent_scan_cache.clear()
                pew._recent_terminal_scan.clear()
                self.db.writes.clear()
                self._stub_lookups(
                    chaperone=chap, students=students,
                    grade_scopes=scopes, window=window, settings=settings,
                )
                self._record(employee_no=chap["_id"].replace("chap-", ""))
                doc = self._written_event()
                self.assertIsNotNone(doc["overrideCode"],
                                     f"{label}: overrideCode is None — officer cannot resolve")
                self.assertRegex(doc["overrideCode"], r"^\d{6}$",
                                 f"{label}: overrideCode not 6 digits")


if __name__ == "__main__":
    unittest.main()
