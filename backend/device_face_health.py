#!/usr/bin/env python3
"""
device_face_health.py — Continuous health monitor for all Hikvision face devices.

Detects the failure mode that caused today's PYP/Basement incident:
device's face library content drops while user accounts stay intact
(faces/users ratio collapses), making everyone show "not registered".

Modes:
    --once             Run a single probe sweep and exit (good for cron).
    --watch [SECONDS]  Loop forever (default 300s) and re-probe.
    --auto-rebind      When unhealthy, automatically run _rebind_pyp_faces.py
                       for the affected device. Otherwise just log + alert.
    --threshold 0.9    Trigger if face_count < user_count * threshold.

Outputs:
    data/device_health.log     Append-only timeline of every probe.
    data/device_health.json    Latest snapshot (machine-readable).
    data/ALERT_<device>.txt    Marker file created when a device is unhealthy
                               (delete to silence; auto-cleared when healthy).

Cron suggestion (every 5 min, auto-heal):
    */5 * * * * cd /home/pandora/Downloads/final-face/backend && \
        ./_venv_run.sh device_face_health.py --once --auto-rebind \
        >> data/device_health.cron.log 2>&1
"""
import json
import os
import sys
import time
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from batch_enroll import DeviceClient

HERE = Path(__file__).parent
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)
LOG = DATA / "device_health.log"
SNAPSHOT = DATA / "device_health.json"
WIB = timezone(timedelta(hours=7))

# Pinned matcher mode: changing this on V4.38 firmware WIPES the FDLib face index
# (root cause of 2026-05-19 incident). Health monitor alerts if any device drifts
# off this value. To intentionally roll a new mode, update this constant AND
# expect to re-bind every device afterwards.
EXPECTED_FACE_MODE = "normalMode"

# Skip-list of known non-student employeeNos (always missing real photos).
# Used only for "expected faces" expectations so test users don't inflate alerts.
NON_STUDENT_PREFIXES = ("9000000",)


def _now():
    return datetime.now(WIB).isoformat(timespec="seconds")


def _log(msg):
    line = f"[{_now()}] {msg}"
    print(line)
    with LOG.open("a") as f:
        f.write(line + "\n")


def load_devices():
    devs_cfg = json.loads((HERE / "devices.json").read_text())
    devs_loc = {d["name"]: d["password"]
                for d in json.loads((HERE / "devices.local.json").read_text())}
    out = []
    for d in devs_cfg:
        if not d.get("enabled"):
            continue
        pw = devs_loc.get(d["name"])
        if not pw:
            _log(f"⚠️  No password for {d['name']} — skipping")
            continue
        out.append({"name": d["name"], "ip": d["ip"], "password": pw})
    return out


def probe(dev):
    """Return dict with users, faces, reachable, error, mode."""
    res = {"name": dev["name"], "ip": dev["ip"],
           "users": None, "faces": None, "mode": None,
           "reachable": False, "error": None}
    try:
        c = DeviceClient(dev["ip"], "admin", dev["password"])
        s, uc = c.api_json("get", "/ISAPI/AccessControl/UserInfo/Count?format=json")
        if s != 200:
            res["error"] = f"UserInfo/Count HTTP {s}"
            return res
        res["users"] = uc.get("UserInfoCount", {}).get("userNumber", 0)
        # FDSearch totalMatches is the authoritative face count on these firmwares.
        body = {"searchResultPosition": 0, "maxResults": 1,
                "faceLibType": "blackFD", "FDID": "1"}
        s, fd = c.api_json("post", "/ISAPI/Intelligent/FDLib/FDSearch?format=json",
                           json=body, timeout=15)
        if s != 200:
            res["error"] = f"FDSearch HTTP {s}"
            return res
        res["faces"] = fd.get("totalMatches", 0)
        # Matcher mode — drift wipes the FDLib on this firmware. Read it every probe.
        try:
            s, fm = c.api_json("get", "/ISAPI/AccessControl/FaceRecognizeMode?format=json")
            if s == 200:
                res["mode"] = fm.get("FaceRecognizeMode", {}).get("mode")
        except Exception:
            pass
        res["reachable"] = True
    except Exception as e:
        res["error"] = str(e)[:200]
    return res


def alert_marker(dev_name, msg):
    safe = dev_name.replace(" ", "_").replace("/", "_")
    p = DATA / f"ALERT_{safe}.txt"
    p.write_text(f"[{_now()}] {msg}\n")
    return p


def clear_marker(dev_name):
    safe = dev_name.replace(" ", "_").replace("/", "_")
    p = DATA / f"ALERT_{safe}.txt"
    if p.exists():
        p.unlink()


def auto_rebind(dev_name):
    _log(f"🔧 Auto-rebind triggered for {dev_name}")
    cmd = [sys.executable, str(HERE / "_rebind_pyp_faces.py"),
           "--device", dev_name]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        tail = "\n".join(proc.stdout.splitlines()[-5:])
        _log(f"   rebind exit={proc.returncode}; tail:\n{tail}")
        return proc.returncode == 0
    except Exception as e:
        _log(f"   ❌ rebind subprocess failed: {e}")
        return False


def run_once(threshold=0.9, do_auto_rebind=False):
    devices = load_devices()
    snap = {"ts": _now(), "threshold": threshold, "devices": []}
    for d in devices:
        r = probe(d)
        snap["devices"].append(r)
        if not r["reachable"]:
            _log(f"❌ {d['name']} ({d['ip']}) UNREACHABLE: {r['error']}")
            alert_marker(d['name'], f"unreachable: {r['error']}")
            continue
        users, faces = r["users"], r["faces"]
        ratio = (faces / users) if users else 0.0
        status = "✅" if ratio >= threshold else "🚨"
        mode_tag = f" mode={r['mode']}" if r.get("mode") else ""
        _log(f"{status} {d['name']:42s} users={users:3d} faces={faces:3d}  ratio={ratio:.2f}{mode_tag}")
        mode_drift = bool(r.get("mode") and r["mode"] != EXPECTED_FACE_MODE)
        if mode_drift:
            _log(f"   ⚠️  MODE DRIFT: expected {EXPECTED_FACE_MODE!r}, got {r['mode']!r} — "
                 f"changing this WIPES the face library. Investigate immediately.")
            alert_marker(d['name'],
                f"MODE DRIFT: expected {EXPECTED_FACE_MODE!r}, got {r['mode']!r}")
        if ratio < threshold:
            alert_marker(d['name'],
                f"DEGRADED: {faces}/{users} faces ({ratio:.0%}) below {threshold:.0%}")
            if do_auto_rebind:
                ok = auto_rebind(d['name'])
                if ok:
                    # Re-probe to confirm
                    r2 = probe(d)
                    if r2["faces"] and r2["users"]:
                        new_ratio = r2["faces"] / r2["users"]
                        _log(f"   post-rebind: {r2['faces']}/{r2['users']} ratio={new_ratio:.2f}")
                        if new_ratio >= threshold and not mode_drift:
                            clear_marker(d['name'])
        else:
            if not mode_drift:
                clear_marker(d['name'])
    SNAPSHOT.write_text(json.dumps(snap, indent=2))
    return snap


def main():
    args = sys.argv[1:]
    threshold = 0.9
    do_auto = "--auto-rebind" in args
    if "--threshold" in args:
        i = args.index("--threshold")
        threshold = float(args[i + 1])

    if "--watch" in args:
        idx = args.index("--watch")
        period = 300
        if idx + 1 < len(args) and args[idx + 1].isdigit():
            period = int(args[idx + 1])
        _log(f"👁  watch mode every {period}s; threshold={threshold} auto-rebind={do_auto}")
        while True:
            try:
                run_once(threshold, do_auto)
            except KeyboardInterrupt:
                _log("interrupted; bye"); return
            except Exception as e:
                _log(f"❌ probe loop error: {e}")
            time.sleep(period)
    else:
        run_once(threshold, do_auto)


if __name__ == "__main__":
    main()
