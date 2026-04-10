#!/usr/bin/env python3
"""
fetch_all_students.py — Fetch all students from BINUS API (Grade 1–6)
=====================================================================
Retrieves student photos and enrollment data for Grades 1–6,
identifies students with valid IDs + photos, and preps for device enrollment.

Usage:
    python3 fetch_all_students.py              # Fetch & report
    python3 fetch_all_students.py --enroll     # Fetch, report, then enroll on devices
    python3 fetch_all_students.py --enroll --dry-run  # Show what would be enrolled
"""

import sys
import os
import json
import time
import requests
from pathlib import Path
from datetime import datetime, timezone, timedelta

# Add parent dir for imports
sys.path.insert(0, os.path.dirname(__file__))

import api_integrate
import student_metadata

WIB = timezone(timedelta(hours=7))
DATA_DIR = Path(__file__).parent / "data"
REPORT_FILE = DATA_DIR / "binus_students_all.json"

# Grades and known homeroom sections
GRADES = {
    "1": ["1A", "1B", "1C"],
    "2": ["2A", "2B", "2C"],
    "3": ["3A", "3B", "3C"],
    "4": ["4A", "4B", "4C"],
    "5": ["5A", "5B", "5C"],
    "6": ["6A", "6B", "6C"],
}


def fetch_all_students():
    """Fetch all students from Grade 1–6 via BINUS API."""
    token = api_integrate.get_auth_token()
    if not token:
        print("❌ Failed to get auth token")
        return None

    all_students = []
    stats = {"total": 0, "with_photo": 0, "with_ids": 0, "enrollable": 0}

    for grade, homerooms in GRADES.items():
        # First try grade-wide (no homeroom filter)
        print(f"\n📚 Grade {grade}:")
        students = api_integrate.get_student_photos(grade=grade, homeroom=None, token=token)

        # Token expires after 60 min — refresh if needed
        if students is None:
            token = api_integrate.get_auth_token()
            students = api_integrate.get_student_photos(grade=grade, homeroom=None, token=token)

        if not students:
            # Fall back to per-homeroom
            print(f"  ⚠ Grade-wide query returned nothing, trying per-homeroom...")
            students = []
            for hr in homerooms:
                hr_students = api_integrate.get_student_photos(grade=grade, homeroom=hr, token=token)
                if hr_students is None:
                    token = api_integrate.get_auth_token()
                    hr_students = api_integrate.get_student_photos(grade=grade, homeroom=hr, token=token)
                if hr_students:
                    students.extend(hr_students)
                    print(f"    {hr}: {len(hr_students)} students")
                else:
                    print(f"    {hr}: 0 students")
                time.sleep(0.3)  # rate limit

        if not students:
            print(f"  No students found for Grade {grade}")
            continue

        # Enrich with enrollment data (to get name + class)
        seen_ids = set()
        for s in students:
            sid = s.get("idStudent", "")
            if sid in seen_ids:
                continue
            seen_ids.add(sid)

            bid = s.get("idBinusian", "")
            file_name = s.get("fileName", "") or ""
            file_path = s.get("filePath", "") or ""
            has_photo = bool(file_name.strip() and file_path.strip())
            has_ids = bool(sid and bid and bid != "-")

            # Get enrollment data for name + class
            enrollment = api_integrate.get_student_by_id_c2(sid, token=token)
            if enrollment is None:
                # Token might have expired
                token = api_integrate.get_auth_token()
                enrollment = api_integrate.get_student_by_id_c2(sid, token=token)

            name = ""
            homeroom = ""
            grade_code = grade
            if enrollment:
                name = enrollment.get("studentName", "")
                homeroom = enrollment.get("class", "")
                grade_code = enrollment.get("gradeCode", grade)

            record = {
                "idStudent": sid,
                "idBinusian": bid,
                "name": name,
                "grade": grade_code,
                "homeroom": homeroom,
                "fileName": file_name,
                "filePath": file_path,
                "hasPhoto": has_photo,
                "hasIds": has_ids,
                "enrollable": has_photo and has_ids,
            }
            all_students.append(record)

            stats["total"] += 1
            if has_photo:
                stats["with_photo"] += 1
            if has_ids:
                stats["with_ids"] += 1
            if has_photo and has_ids:
                stats["enrollable"] += 1

        enrolled_count = sum(1 for s in students if s.get("idStudent", "") in seen_ids)
        photo_count = sum(
            1 for s in all_students
            if s.get("grade") == grade and s.get("hasPhoto")
        )
        print(f"  Found {len(seen_ids)} students, {photo_count} with photos")
        time.sleep(0.5)  # rate limit between grades

    return all_students, stats


def print_report(all_students, stats):
    """Print summary report."""
    print("\n" + "=" * 70)
    print("  BINUS STUDENT REPORT — Grades 1–6")
    print("=" * 70)
    print(f"  Total students:      {stats['total']}")
    print(f"  With valid IDs:      {stats['with_ids']}")
    print(f"  With photos:         {stats['with_photo']}")
    print(f"  Enrollable (ID+photo): {stats['enrollable']}")
    print()

    # Per-grade breakdown
    by_grade = {}
    for s in all_students:
        g = s["grade"]
        if g not in by_grade:
            by_grade[g] = {"total": 0, "photo": 0, "enrollable": 0, "no_photo": [], "no_ids": []}
        by_grade[g]["total"] += 1
        if s["hasPhoto"]:
            by_grade[g]["photo"] += 1
        if s["enrollable"]:
            by_grade[g]["enrollable"] += 1
        if not s["hasPhoto"]:
            by_grade[g]["no_photo"].append(s["name"] or s["idStudent"])
        if not s["hasIds"]:
            by_grade[g]["no_ids"].append(s["name"] or s["idStudent"])

    print(f"  {'Grade':<8} {'Total':>6} {'Photos':>7} {'Enrollable':>11} {'Missing Photo':>14} {'Missing IDs':>12}")
    print(f"  {'-'*8} {'-'*6} {'-'*7} {'-'*11} {'-'*14} {'-'*12}")
    for g in sorted(by_grade.keys(), key=lambda x: (len(x), x)):
        d = by_grade[g]
        no_p = len(d["no_photo"])
        no_id = len(d["no_ids"])
        print(f"  {g:<8} {d['total']:>6} {d['photo']:>7} {d['enrollable']:>11} {no_p:>14} {no_id:>12}")

    print()

    # List students missing photos
    no_photo = [s for s in all_students if not s["hasPhoto"] and s["hasIds"]]
    if no_photo:
        print(f"  ⚠ Students with IDs but NO photo ({len(no_photo)}):")
        for s in no_photo:
            print(f"    - {s['name'] or '?':35s} Grade {s['grade']} {s['homeroom']:4s} sid={s['idStudent']}")

    # List students missing IDs
    no_ids = [s for s in all_students if not s["hasIds"]]
    if no_ids:
        print(f"\n  ⚠ Students with missing idBinusian ({len(no_ids)}):")
        for s in no_ids:
            print(f"    - {s['name'] or '?':35s} Grade {s['grade']} {s['homeroom']:4s} sid={s['idStudent']} bid={s['idBinusian']}")


def check_already_enrolled(all_students):
    """Check which enrollable students are already on our devices."""
    meta = student_metadata._load_local()

    already = []
    new = []
    for s in all_students:
        if not s["enrollable"]:
            continue
        # Check if student is already in metadata (by idStudent or name)
        found = False
        for emp, m in meta.items():
            if m.get("idStudent") == s["idStudent"]:
                found = True
                break
            if m.get("name", "").strip().lower() == (s.get("name") or "").strip().lower() and s.get("name"):
                found = True
                break
        if found:
            already.append(s)
        else:
            new.append(s)

    print(f"\n  📋 Enrollment status:")
    print(f"    Already enrolled: {len(already)}")
    print(f"    NEW (not on device): {len(new)}")

    if new:
        print(f"\n  🆕 Students to enroll ({len(new)}):")
        for s in new:
            print(f"    + {s['name'] or '?':35s} Grade {s['grade']} {s['homeroom']:4s} sid={s['idStudent']}")

    return new, already


def save_report(all_students, stats):
    """Save full report to JSON."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "fetchedAt": datetime.now(WIB).isoformat(),
        "stats": stats,
        "students": all_students,
    }
    REPORT_FILE.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n  💾 Full report saved to {REPORT_FILE}")


if __name__ == "__main__":
    do_enroll = "--enroll" in sys.argv
    dry_run = "--dry-run" in sys.argv

    print("🔍 Fetching all students from BINUS API (Grades 1–6)...")
    result = fetch_all_students()
    if not result:
        print("❌ Failed to fetch students")
        sys.exit(1)

    all_students, stats = result
    print_report(all_students, stats)
    save_report(all_students, stats)
    new_students, already_enrolled = check_already_enrolled(all_students)

    if do_enroll and new_students:
        print(f"\n{'🏷️  DRY RUN — ' if dry_run else ''}Enrolling {len(new_students)} new students on devices...")
        # TODO: Hook into hikvision_attendance.py enrollment
        if dry_run:
            print("  (dry run — no changes made)")
        else:
            print("  ⚠ Enrollment not yet implemented in this script.")
            print("  Use the dashboard enrollment page or hikvision_attendance.py")
    elif do_enroll:
        print("\n  ✅ All enrollable students are already on devices!")
