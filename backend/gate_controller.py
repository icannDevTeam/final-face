"""
gate_controller.py — per-terminal pickup-gate state enforcer.

Watches Firestore `tenants/{tid}/terminals/{terminalId}` for this device's
schedule + manual override and, only on **state transitions**, issues a single
Hikvision RemoteControl command to the door relay:

    desired = "alwaysOpen"   →  PUT /ISAPI/AccessControl/RemoteControl/door/1
                                  body: {"RemoteControlDoor":{"cmd":"alwaysOpen"}}
    desired = "alwaysClose"  →  body: {"RemoteControlDoor":{"cmd":"alwaysClose"}}
    desired = "resume"       →  body: {"RemoteControlDoor":{"cmd":"resume"}}

Precedence (highest first):
    1. gateOverride == "closed"                  → alwaysClose   (admin force-close)
    2. gateOverride == "open"                    → alwaysOpen    (admin force-open)
    3. windowOpen + windowClose set, in window   → alwaysOpen
    4. windowOpen + windowClose set, out of win  → alwaysClose
    5. no schedule + no override                 → resume        (normal face-match unlock)

This intentionally avoids "constant firing": the relay command is only sent
when `desired` changes. A heartbeat re-issue every HEARTBEAT_SECS confirms
the device still holds the chosen mode (some terminals revert after reboot).

Runs in a daemon thread started by attendance_listener.run_listener().
"""
from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone, timedelta

import requests

WIB = timezone(timedelta(hours=7))
POLL_SECS = int(os.getenv("GATE_POLL_SECS", "5"))           # how often we read Firestore
HEARTBEAT_SECS = int(os.getenv("GATE_HEARTBEAT_SECS", "120"))  # re-issue same cmd every 2min


def _hhmm_to_minutes(s):
    if not s or not isinstance(s, str) or len(s) != 5 or s[2] != ":":
        return None
    try:
        h = int(s[:2]); m = int(s[3:])
        if 0 <= h < 24 and 0 <= m < 60:
            return h * 60 + m
    except ValueError:
        pass
    return None


def _wib_minutes(now=None):
    now = now or datetime.now(WIB)
    return now.hour * 60 + now.minute


def compute_desired(terminal_doc, now=None):
    """Return ('alwaysOpen'|'alwaysClose'|'resume', reason)."""
    override = terminal_doc.get("gateOverride") if terminal_doc else None
    if override == "closed":
        return "alwaysClose", "manual-closed"
    if override == "open":
        return "alwaysOpen", "manual-open"

    open_min = _hhmm_to_minutes(terminal_doc.get("windowOpen") if terminal_doc else None)
    close_min = _hhmm_to_minutes(terminal_doc.get("windowClose") if terminal_doc else None)
    if open_min is not None and close_min is not None:
        cur = _wib_minutes(now)
        if open_min <= close_min:
            in_window = open_min <= cur <= close_min
        else:
            in_window = cur >= open_min or cur <= close_min   # crosses midnight
        return ("alwaysOpen", "in-window") if in_window else ("alwaysClose", "out-of-window")

    return "resume", "always-open"


class GateEnforcer(threading.Thread):
    """Background thread that pushes door state to a single Hikvision terminal."""

    def __init__(self, terminal_id, hik_ip, send_cmd, get_terminal_doc, log=print):
        """
        Parameters
        ----------
        terminal_id : str
            Firestore terminal id (sha1 of device name).
        hik_ip : str
            For logging only.
        send_cmd : callable(cmd: str) -> bool
            Function that PUTs the RemoteControl command to the device.
            Should return True on HTTP 2xx.
        get_terminal_doc : callable() -> dict | None
            Reads the latest terminal doc from Firestore (or returns None).
        """
        super().__init__(daemon=True, name=f"gate-enforcer-{terminal_id[:6]}")
        self.terminal_id = terminal_id
        self.hik_ip = hik_ip
        self.send_cmd = send_cmd
        self.get_terminal_doc = get_terminal_doc
        self.log = log
        self._stop = threading.Event()
        self._last_cmd = None
        self._last_send_ts = 0
        self._last_reason = None
        # Track failed cmd so we don't log the same HTTP 400 every 5 seconds
        self._last_failed_cmd = None
        self._last_failed_log_ts = 0

    def stop(self):
        self._stop.set()

    def run(self):
        self.log(f"  🚪 Gate enforcer started (poll={POLL_SECS}s heartbeat={HEARTBEAT_SECS}s)")
        # Try to push initial state quickly so the device is in a known mode.
        time.sleep(2)
        while not self._stop.is_set():
            try:
                doc = self.get_terminal_doc()
                if doc is None:
                    # No registry entry — leave device in default behaviour.
                    self._stop.wait(POLL_SECS)
                    continue
                desired, reason = compute_desired(doc)
                now_ts = time.time()
                changed = desired != self._last_cmd
                stale = (now_ts - self._last_send_ts) >= HEARTBEAT_SECS
                if changed or stale:
                    ok = False
                    try:
                        ok = self.send_cmd(desired)
                    except Exception as e:
                        self.log(f"  ⚠ Gate cmd '{desired}' failed: {e}")
                    if ok:
                        if changed:
                            self.log(f"  🚪 Gate → {desired.upper()} ({reason})")
                        self._last_cmd = desired
                        self._last_send_ts = now_ts
                        self._last_reason = reason
                        self._last_failed_cmd = None
                    else:
                        # Throttle repeat failures: log once, then once every 5 min.
                        if (
                            self._last_failed_cmd != desired
                            or (now_ts - self._last_failed_log_ts) >= 300
                        ):
                            self.log(
                                f"  ⚠ Gate cmd '{desired}' rejected by device "
                                f"(suppressing repeats for 5 min)"
                            )
                            self._last_failed_cmd = desired
                            self._last_failed_log_ts = now_ts
                        # Treat as "sent" so we don't retry every 5s in tight loop;
                        # heartbeat will retry naturally after HEARTBEAT_SECS.
                        self._last_cmd = desired
                        self._last_send_ts = now_ts
            except Exception as e:
                self.log(f"  ⚠ Gate enforcer loop error: {e}")
            self._stop.wait(POLL_SECS)


def hik_remote_control_door(hik_ip, build_auth, cmd, door_no=1, timeout=8, log=print):
    """Send RemoteControl door command to a Hikvision terminal.

    cmd in {'open','close','alwaysOpen','alwaysClose','resume'}.
    `build_auth` is a callable(method, uri) -> Authorization header value
    (so we reuse the listener's existing digest implementation).
    """
    uri = f"/ISAPI/AccessControl/RemoteControl/door/{door_no}"
    body = {"RemoteControlDoor": {"cmd": cmd}}
    try:
        auth = build_auth("PUT", uri)
        r = requests.put(
            f"http://{hik_ip}{uri}",
            json=body,
            headers={"Authorization": auth, "Content-Type": "application/json"},
            timeout=timeout,
        )
        if r.status_code == 401:
            # Stale digest — the listener will refresh on next call.
            log(f"  ⚠ Gate cmd '{cmd}' got 401 (stale digest); will retry")
            return False
        if 200 <= r.status_code < 300:
            return True
        log(f"  ⚠ Gate cmd '{cmd}' HTTP {r.status_code}: {r.text[:120]}")
        return False
    except requests.RequestException as e:
        log(f"  ⚠ Gate cmd '{cmd}' transport error: {e}")
        return False
