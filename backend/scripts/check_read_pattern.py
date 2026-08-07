#!/usr/bin/env python3
"""
check_read_pattern.py — Daily Firestore read-pattern sanity check.

Pulls hourly document read counts from Cloud Monitoring for the last 36h,
prints a WIB-hour histogram, and flags any hour OUTSIDE the dismissal window
(Mon-Thu 12:00-15:30 WIB, Fri 10:40-12:30 WIB) that exceeds the leak threshold.
Exits non-zero when a
leak is detected so it can run under cron/CI.

Auth: uses `gcloud auth print-access-token` (same pattern verified 2026-08-05).

Usage:
    python3 backend/scripts/check_read_pattern.py [--hours 36] [--threshold 50000]

Background: on 2026-08-04→05 an overnight leak burned ~770k reads/hr all
night (≈$114/mo). Root cause was un-indexed release-group query fallbacks;
fixed by deploying firestore.indexes.json. This script verifies it stays fixed.
"""
import argparse
import datetime as dt
import json
import subprocess
import sys
import urllib.parse
import urllib.request

PROJECT = "facial-attendance-binus"
METRIC = "firestore.googleapis.com/document/read_count"
WIB = dt.timezone(dt.timedelta(hours=7))
DEFAULT_WINDOW = (12 * 60, 15 * 60 + 30)
WINDOW_BY_WEEKDAY = {
    "fri": (10 * 60 + 40, 12 * 60 + 30),
}
WEEKDAY_KEYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
READ_PRICE_PER_DOC = 0.00000038  # asia-southeast2 (Jakarta), 2026-08 catalog


def window_for_wib_time(d: dt.datetime) -> tuple[int, int]:
    """Return (open_minute, close_minute) for the WIB weekday of datetime d."""
    key = WEEKDAY_KEYS[d.weekday()]
    return WINDOW_BY_WEEKDAY.get(key, DEFAULT_WINDOW)


def bucket_overlaps_window(start_minute: int, end_minute: int,
                           open_minute: int, close_minute: int) -> bool:
    """True if a [start,end) hourly bucket overlaps today's dismissal window."""
    return max(start_minute, open_minute) < min(end_minute, close_minute)


def access_token() -> str:
    return subprocess.check_output(
        ["gcloud", "auth", "print-access-token"], text=True
    ).strip()


def fetch_hourly_reads(hours: int) -> dict:
    """Return {aligned_end_utc_datetime: read_count} summed across all labels."""
    now = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = now - dt.timedelta(hours=hours)
    params = {
        "filter": f'metric.type="{METRIC}"',
        "interval.startTime": start.isoformat().replace("+00:00", "Z"),
        "interval.endTime": now.isoformat().replace("+00:00", "Z"),
        "aggregation.alignmentPeriod": "3600s",
        "aggregation.perSeriesAligner": "ALIGN_SUM",
        "aggregation.crossSeriesReducer": "REDUCE_SUM",
    }
    url = (
        f"https://monitoring.googleapis.com/v3/projects/{PROJECT}/timeSeries?"
        + urllib.parse.urlencode(params)
    )
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {access_token()}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        body = json.load(r)

    buckets = {}
    for series in body.get("timeSeries", []):
        for pt in series.get("points", []):
            end = dt.datetime.fromisoformat(pt["interval"]["endTime"].replace("Z", "+00:00"))
            val = int(pt["value"].get("int64Value") or pt["value"].get("doubleValue") or 0)
            buckets[end] = buckets.get(end, 0) + val
    return buckets


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=36, help="lookback window (default 36)")
    ap.add_argument("--threshold", type=int, default=50_000,
                    help="reads/hr outside dismissal window that count as a leak")
    args = ap.parse_args()

    buckets = fetch_hourly_reads(args.hours)
    if not buckets:
        print("No datapoints returned — check auth/project.")
        return 2

    total = 0
    leaks = []
    print(f"Hourly Firestore reads, last {args.hours}h (WIB) — project {PROJECT}")
    print(f"{'hour ending (WIB)':>20}  {'reads':>12}  bar")
    peak = max(buckets.values()) or 1
    for end_utc in sorted(buckets):
        reads = buckets[end_utc]
        total += reads
        end_wib = end_utc.astimezone(WIB)
        # The bucket covers [end-1h, end); treat it as in-window if any overlap
        # exists with the configured 12:00-15:30 WIB dismissal window.
        bucket_start = end_wib - dt.timedelta(hours=1)
        start_minute = bucket_start.hour * 60 + bucket_start.minute
        end_minute = start_minute + 60
        open_minute, close_minute = window_for_wib_time(bucket_start)
        in_window = bucket_overlaps_window(start_minute, end_minute, open_minute, close_minute)
        bar = "#" * max(1, round(40 * reads / peak)) if reads else ""
        flag = ""
        if not in_window and reads > args.threshold:
            leaks.append((end_wib, reads))
            flag = "  << LEAK?"
        print(f"{end_wib.strftime('%a %d %H:%M'):>20}  {reads:>12,}  {bar}{flag}")

    daily = total * 24 / max(args.hours, 1)
    print(f"\nTotal reads in window: {total:,}")
    print(f"Extrapolated: {daily:,.0f}/day ≈ ${daily * READ_PRICE_PER_DOC * 30:,.2f}/mo (30d)")

    if leaks:
        print(f"\nLEAK DETECTED — {len(leaks)} off-window hour(s) above {args.threshold:,} reads:")
        for when, reads in leaks:
            print(f"  {when.strftime('%a %d %H:%M')} WIB — {reads:,} reads")
        print("Investigate: tablet pollers outside dismissal window, un-indexed "
              "query fallbacks (firestore.indexes.json), rogue scripts.")
        return 1

    print("\nOK — no off-window hour exceeded the threshold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
