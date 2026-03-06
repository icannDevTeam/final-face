#!/usr/bin/env python3
"""
test_attendance_insert.py — B.2 Attendance Insert Simulation
=============================================================
Sends a test attendance record to the Binus School e-Desk API
endpoint (bss-add-simprug-attendance-fr) to confirm the integration
is working end-to-end.

Usage:
  python test_attendance_insert.py                          # Use defaults
  python test_attendance_insert.py --id-student 1111111 --id-binusian 22222222
  python test_attendance_insert.py --dry-run                # Show payload only, don't send

Requires:
  - .env with API_KEY set
  - Network access to binusian.ws
"""

import argparse
import json
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from api_integrate import get_auth_token, insert_student_attendance

# ─── Test student data (use real IDs from e-Desk if available) ───────────────
TEST_STUDENTS = [
    {
        "name": "Test Student A",
        "IdStudent": "1111111",
        "IdBinusian": "22222222",
        "ImageDesc": "-",
        "UserAction": "TEACHER7",
    },
]


def main():
    parser = argparse.ArgumentParser(
        description="B.2 Attendance Insert — Simulation Test for e-Desk Team"
    )
    parser.add_argument("--id-student", type=str, default=None, help="IdStudent to test with")
    parser.add_argument("--id-binusian", type=str, default=None, help="IdBinusian to test with")
    parser.add_argument("--image-desc", type=str, default="-", help="ImageDesc value (default: '-')")
    parser.add_argument("--user-action", type=str, default="TEACHER7", help="UserAction value (default: 'TEACHER7')")
    parser.add_argument("--dry-run", action="store_true", help="Show payload without sending")
    args = parser.parse_args()

    print("=" * 60)
    print("  B.2 Attendance Insert — Simulation Test")
    print("=" * 60)
    print(f"  Endpoint: http://binusian.ws/binusschool/bss-add-simprug-attendance-fr")
    print(f"  Time:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # ── Step 1: Get auth token ──────────────────────────────────────────
    print("Step 1: Requesting auth token...")
    token = get_auth_token()
    if token:
        print(f"  ✅ Token acquired: {token[:20]}...")
    else:
        print("  ❌ Failed to get token. Check API_KEY in .env")
        sys.exit(1)
    print()

    # ── Step 2: Build test payload ──────────────────────────────────────
    if args.id_student or args.id_binusian:
        # Custom IDs from CLI
        test_data = {
            "IdStudent": args.id_student or "1111111",
            "IdBinusian": args.id_binusian or "22222222",
            "ImageDesc": args.image_desc,
            "UserAction": args.user_action,
        }
        test_list = [test_data]
    else:
        test_list = TEST_STUDENTS

    for i, student in enumerate(test_list, 1):
        payload = {
            "IdStudent": str(student["IdStudent"]),
            "IdBinusian": str(student["IdBinusian"]),
            "ImageDesc": str(student.get("ImageDesc", "-")),
            "UserAction": str(student.get("UserAction", "TEACHER7")),
        }

        print(f"Step 2: Test #{i} — Payload:")
        print(f"  {json.dumps(payload, indent=2)}")
        print()

        if args.dry_run:
            print("  ⏭️  Dry run — skipping actual API call")
            print()
            continue

        # ── Step 3: Send request ────────────────────────────────────────
        print(f"Step 3: Sending POST request...")
        success = insert_student_attendance(payload, token=token)

        if success:
            print(f"  ✅ SUCCESS — Attendance record inserted for IdStudent={payload['IdStudent']}")
        else:
            print(f"  ❌ FAILED — Check api_testing/org_api_test.log for details")
        print()

    # ── Summary ─────────────────────────────────────────────────────────
    print("=" * 60)
    if args.dry_run:
        print("  Dry run complete. No data was sent.")
    else:
        print("  Test complete. Share this output with the e-Desk team.")
    print("=" * 60)


if __name__ == "__main__":
    main()
