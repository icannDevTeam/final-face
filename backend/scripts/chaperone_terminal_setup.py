#!/usr/bin/env python3
"""
chaperone_terminal_setup.py
===========================
Safely prepare Hikvision terminals for the pickup chaperone system.

What this script does:
1. Loads terminal registry from Firestore (tenants/{tid}/terminals).
2. Validates target devices are reachable and match baseline mode from a template IP.
3. Backs up current users on selected wipe devices.
4. Wipes selected devices (all users + face mappings).
5. Re-enrolls chaperones (create user + upload face) based on class/grade scope mapping.

Safety:
- Does NOT mutate terminal class/grade assignments in Firestore.
- Does NOT write FaceRecognizeMode (only checks parity).
- Creates backup JSON before any wipe operation.
"""

import argparse
import json
import os
import random
import re
import string
import time
from datetime import datetime
from pathlib import Path

import requests
from requests.auth import HTTPDigestAuth

try:
    import firebase_admin
    from firebase_admin import credentials, firestore, storage
except Exception as e:
    raise SystemExit(f"firebase_admin import failed: {e}")

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "backend" / "data"
BACKUP_DIR = DATA_DIR / "backups"
BACKUP_DIR.mkdir(parents=True, exist_ok=True)


def normalize_token(token: str) -> list[str]:
    t = str(token or "").strip().upper().replace(" ", "")
    if not t:
        return []
    out = [t]
    m = re.match(r"^(\d+)([A-Z])$", t)
    if m:
        out.append(m.group(1))
        out.append(m.group(2))
    return out


def normalize_scopes(scopes) -> set[str]:
    out = set()
    for raw in scopes or []:
        for x in normalize_token(raw):
            out.add(x)
    return out


def parse_grade_label(label: str) -> list[str]:
    if not label:
        return []
    s = str(label).strip().upper()
    if s in ("", "ALL", "ANY", "SHARED"):
        return []
    out = set()
    rm = re.search(r"(\d+)\s*[-–—]\s*(\d+)", s)
    if rm:
        a, b = int(rm.group(1)), int(rm.group(2))
        if a > b:
            a, b = b, a
        for g in range(a, b + 1):
            out.add(str(g))
    for m in re.finditer(r"\b\d+[A-Z]?\b|\b[A-Z]\b", s):
        out.add(m.group(0))
    return sorted(out)


class DeviceClient:
    def __init__(self, ip: str, user: str, password: str, timeout: int = 20):
        self.ip = ip
        self.base = f"http://{ip}"
        self.timeout = timeout
        self.user = user
        self.password = password

    def _json(self, method: str, path: str, payload=None, headers=None, raw=False):
        url = f"{self.base}{path}"
        kwargs = {"timeout": self.timeout}
        if headers:
            kwargs["headers"] = headers
        if payload is not None:
            if raw:
                kwargs["data"] = payload
            else:
                kwargs["json"] = payload
        # Fresh digest handshake per request — the device invalidates cached
        # digest state after a completed search session, which made reused
        # sessions silently return 401/empty on subsequent calls.
        try:
            r = requests.request(method, url, auth=HTTPDigestAuth(self.user, self.password), **kwargs)
            if r.status_code == 401:
                time.sleep(0.3)
                r = requests.request(method, url, auth=HTTPDigestAuth(self.user, self.password), **kwargs)
        except requests.RequestException as e:
            return 0, {"error": str(e)[:300]}
        try:
            body = r.json()
        except Exception:
            body = {"raw": (r.text or "")[:500]}
        return r.status_code, body

    def reachable(self):
        c, _ = self._json("get", "/ISAPI/System/deviceInfo?format=json")
        return c == 200

    def face_mode(self):
        c, body = self._json("get", "/ISAPI/AccessControl/FaceRecognizeMode?format=json")
        if c != 200:
            return None
        return ((body or {}).get("FaceRecognizeMode") or {}).get("mode")

    def list_users(self):
        out = []
        pos = 0
        search_id = "prep-" + "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
        while True:
            body = {
                "UserInfoSearchCond": {
                    "searchID": search_id,
                    "searchResultPosition": pos,
                    "maxResults": 50,
                }
            }
            c, data = self._json("post", "/ISAPI/AccessControl/UserInfo/Search?format=json", body)
            if c != 200:
                break
            ui = (data or {}).get("UserInfoSearch") or {}
            users = ui.get("UserInfo") or []
            if isinstance(users, dict):
                users = [users]
            out.extend(users)
            total = int(ui.get("totalMatches") or 0)
            pos += len(users)
            if not users or pos >= total:
                break
        return out

    def delete_face(self, employee_no: str):
        body = {
            "FPID": [{"value": str(employee_no)}],
            "faceLibType": "blackFD",
            "FDID": "1",
        }
        c, data = self._json("put", "/ISAPI/Intelligent/FDLib/FDSearch/Delete?format=json", body)
        return c, data

    def delete_user(self, employee_no: str):
        body = {"UserInfoDelCond": {"EmployeeNoList": [{"employeeNo": str(employee_no)}]}}
        c, data = self._json("put", "/ISAPI/AccessControl/UserInfo/Delete?format=json", body)
        return c, data

    def user_count(self):
        for _ in range(3):
            c, data = self._json("get", "/ISAPI/AccessControl/UserInfo/Count?format=json")
            if c == 200:
                return int(((data or {}).get("UserInfoCount") or {}).get("userNumber") or 0)
            time.sleep(1.0)
        return None

    def wipe_all_users(self, max_rounds: int = 5):
        """Delete every user + face. Verifies with count and retries until empty."""
        deleted_total = 0
        for _ in range(max_rounds):
            users = self.list_users()
            if not users:
                break
            for u in users:
                eno = str(u.get("employeeNo") or "").strip()
                if not eno:
                    continue
                self.delete_face(eno)
                c, data = self.delete_user(eno)
                if c == 200:
                    deleted_total += 1
                else:
                    print(f"    ! delete {eno} on {self.ip} -> {c} {json.dumps(data)[:150]}")
                time.sleep(0.05)
            time.sleep(0.5)
        remaining = self.user_count()
        return deleted_total, remaining

    def create_user(self, employee_no: str, name: str):
        body = {
            "UserInfo": {
                "employeeNo": str(employee_no),
                "name": str(name or employee_no),
                "userType": "normal",
                "gender": "unknown",
                "Valid": {
                    "enable": True,
                    "beginTime": "2024-01-01T00:00:00",
                    "endTime": "2037-12-31T23:59:59",
                    "timeType": "local",
                },
                "doorRight": "1",
                "RightPlan": [{"doorNo": 1, "planTemplateNo": "1"}],
            }
        }
        c, data = self._json("post", "/ISAPI/AccessControl/UserInfo/Record?format=json", body)
        if c == 200:
            return True
        txt = json.dumps(data)
        return "employeeNoAlreadyExist" in txt or "deviceUserAlreadyExist" in txt

    def upload_face_multipart(self, employee_no: str, name: str, img_bytes: bytes):
        boundary = "----WebKitFormBoundary" + "".join(random.choices(string.hexdigits.lower(), k=16))
        meta = json.dumps({
            "faceLibType": "blackFD",
            "FDID": "1",
            "FPID": str(employee_no),
            "name": str(name or employee_no),
        })
        head = (
            f"--{boundary}\r\n"
            "Content-Disposition: form-data; name=\"FaceDataRecord\"\r\n"
            "Content-Type: application/json\r\n\r\n"
            f"{meta}\r\n"
            f"--{boundary}\r\n"
            "Content-Disposition: form-data; name=\"FaceImage\"; filename=\"face.jpg\"\r\n"
            "Content-Type: image/jpeg\r\n\r\n"
        ).encode("utf-8")
        foot = f"\r\n--{boundary}--\r\n".encode("utf-8")
        body = head + img_bytes + foot
        c, data = self._json(
            "put",
            "/ISAPI/Intelligent/FDLib/FDSetUp?format=json",
            payload=body,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(body)),
            },
            raw=True,
        )
        if c == 200:
            return True, data
        return False, data


def init_firebase():
    if firebase_admin._apps:
        return
    cred_path = os.getenv("FIREBASE_CREDENTIALS", str(ROOT / "facial-attendance-binus-firebase-adminsdk.json"))
    bucket = os.getenv("FIREBASE_STORAGE_BUCKET", "facial-attendance-binus.firebasestorage.app")
    firebase_admin.initialize_app(credentials.Certificate(cred_path), {"storageBucket": bucket})


def load_terminals(db, tenant_id: str):
    docs = db.collection(f"tenants/{tenant_id}/terminals").get()
    out = []
    for d in docs:
        t = d.to_dict() or {}
        if t.get("enabled") is False:
            continue
        ip = t.get("ip")
        if not ip:
            continue
        scopes = t.get("gradeScopes")
        if not isinstance(scopes, list):
            scopes = parse_grade_label(t.get("gradeLabel"))
        out.append({
            "id": d.id,
            "name": t.get("name") or d.id,
            "ip": ip,
            "gradeScopes": [str(x).strip() for x in (scopes or []) if str(x).strip()],
        })
    return out


def load_chaperones(db, tenant_id: str):
    docs = db.collection(f"tenants/{tenant_id}/chaperones").get()
    out = []
    for d in docs:
        c = d.to_dict() or {}
        emp = str(c.get("employeeNo") or "").strip()
        if not emp:
            continue
        if c.get("active") is False:
            continue
        face_paths = c.get("facePaths") or []
        if not isinstance(face_paths, list) or len(face_paths) == 0:
            continue
        scopes = []
        sg = c.get("studentGrades") or []
        sc = c.get("studentClasses") or []
        if isinstance(sg, list):
            scopes.extend([str(x) for x in sg])
        if isinstance(sc, list):
            scopes.extend([str(x) for x in sc])
        out.append({
            "id": d.id,
            "employeeNo": emp,
            "name": c.get("name") or emp,
            "facePath": face_paths[0],
            "scopes": scopes,
        })
    return out


def pick_devices_for_chaperone(chap_scopes, terminals, allowed_ips=None):
    allowed = set(allowed_ips or [])
    scopes = normalize_scopes(chap_scopes)
    result = []
    for t in terminals:
        if allowed and t["ip"] not in allowed:
            continue
        ds = normalize_scopes(t.get("gradeScopes") or [])
        if not ds:
            result.append(t)
            continue
        if not scopes:
            result.append(t)
            continue
        if ds.intersection(scopes):
            result.append(t)
    return result


def main():
    ap = argparse.ArgumentParser(description="Prepare terminals and enroll chaperones")
    ap.add_argument("--tenant", default="binus-simprug")
    ap.add_argument("--hik-user", default=os.getenv("HIKVISION_USER", "admin"))
    ap.add_argument("--hik-pass", default=os.getenv("HIKVISION_PASS", "aiclub@406013"))
    ap.add_argument("--baseline-ip", default="10.26.30.201")
    ap.add_argument("--wipe-ips", default="10.26.30.41,10.26.80.20,10.26.30.201")
    ap.add_argument("--target-ips", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--wipe-only", action="store_true",
                    help="Backup + wipe devices, verify parity, but do NOT enroll chaperones")
    args = ap.parse_args()

    wipe_ips = [x.strip() for x in args.wipe_ips.split(",") if x.strip()]
    target_ips = [x.strip() for x in args.target_ips.split(",") if x.strip()]

    init_firebase()
    db = firestore.client()
    bucket = storage.bucket()

    terminals = load_terminals(db, args.tenant)
    if not terminals:
        raise SystemExit("No enabled terminals in Firestore")

    clients = {}
    reachable_ips = []
    mode_errors = []

    base = DeviceClient(args.baseline_ip, args.hik_user, args.hik_pass)
    base_mode = base.face_mode()
    if not base_mode:
        raise SystemExit(f"Baseline IP not reachable or no mode: {args.baseline_ip}")

    ips_to_check = sorted({t["ip"] for t in terminals} | set(wipe_ips) | set(target_ips))
    for ip in ips_to_check:
        cli = DeviceClient(ip, args.hik_user, args.hik_pass)
        clients[ip] = cli
        if cli.reachable():
            reachable_ips.append(ip)
            mode = cli.face_mode()
            if mode != base_mode:
                mode_errors.append({"ip": ip, "mode": mode, "baseline": base_mode})

    print(f"Baseline {args.baseline_ip} mode = {base_mode}")
    print(f"Reachable devices: {len(reachable_ips)} / {len(ips_to_check)}")
    if mode_errors:
        print("Mode mismatch detected:")
        for m in mode_errors:
            print(f"  - {m['ip']}: {m['mode']} (expected {m['baseline']})")

    if args.dry_run:
        print("Dry run: no wipe and no enrollment executed.")
        return

    # Backup + wipe requested devices (only if reachable)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = {"createdAt": datetime.now().isoformat(), "tenant": args.tenant, "devices": {}}
    for ip in wipe_ips:
        cli = clients.get(ip)
        if not cli or ip not in reachable_ips:
            backup["devices"][ip] = {"reachable": False, "users": []}
            continue
        users = cli.list_users()
        backup["devices"][ip] = {"reachable": True, "users": users}

    backup_file = BACKUP_DIR / f"chaperone_prep_backup_{ts}.json"
    backup_file.write_text(json.dumps(backup, indent=2))
    print(f"Backup saved: {backup_file}")

    wipe_stats = {}
    for ip in wipe_ips:
        cli = clients.get(ip)
        if not cli or ip not in reachable_ips:
            wipe_stats[ip] = {"ok": False, "reason": "unreachable"}
            continue
        deleted, remaining = cli.wipe_all_users()
        wipe_stats[ip] = {"ok": remaining == 0, "deletedUsers": deleted, "remaining": remaining}
        print(f"Wiped {ip}: {deleted} deleted, {remaining} remaining")

    if args.wipe_only:
        print("\n=== Wipe-Only Summary ===")
        for ip in wipe_ips:
            st = wipe_stats.get(ip, {})
            print(f"  {ip}: ok={st.get('ok')} deleted={st.get('deletedUsers', 0)} remaining={st.get('remaining', 'n/a')}")
        print("Enrollment skipped (--wipe-only).")
        return

    # Enrollment targets = reachable terminals, optionally constrained to target-ips
    allowed = set(target_ips) if target_ips else set(reachable_ips)
    active_terminals = [t for t in terminals if t["ip"] in allowed]

    chaperones = load_chaperones(db, args.tenant)
    print(f"Chaperones with face photo: {len(chaperones)}")

    enroll_ok = 0
    enroll_fail = 0
    device_results = {}

    for c in chaperones:
        face_blob = bucket.blob(c["facePath"])
        if not face_blob.exists():
            enroll_fail += 1
            continue
        img = face_blob.download_as_bytes()
        if not img:
            enroll_fail += 1
            continue

        devices = pick_devices_for_chaperone(c["scopes"], active_terminals)
        if not devices:
            continue

        for d in devices:
            ip = d["ip"]
            cli = clients.get(ip)
            if not cli or ip not in reachable_ips:
                device_results.setdefault(ip, {"ok": 0, "fail": 0})
                device_results[ip]["fail"] += 1
                continue

            created = cli.create_user(c["employeeNo"], c["name"])
            if not created:
                device_results.setdefault(ip, {"ok": 0, "fail": 0})
                device_results[ip]["fail"] += 1
                enroll_fail += 1
                continue

            ok = False
            for _ in range(2):
                ok, err = cli.upload_face_multipart(c["employeeNo"], c["name"], img)
                if ok:
                    break
                txt = json.dumps(err)
                if re.search(r"alreadyExist|FPIDAlreadyExist|writeDatabaseError|saveFacePic", txt, re.I):
                    cli.delete_face(c["employeeNo"])
                    time.sleep(0.3)
            device_results.setdefault(ip, {"ok": 0, "fail": 0})
            if ok:
                device_results[ip]["ok"] += 1
                enroll_ok += 1
            else:
                device_results[ip]["fail"] += 1
                enroll_fail += 1

    print("\n=== Setup Summary ===")
    print(f"Tenant: {args.tenant}")
    print(f"Baseline mode: {base_mode}")
    print(f"Wipe targets: {', '.join(wipe_ips)}")
    print(f"Enrollment writes OK: {enroll_ok}")
    print(f"Enrollment writes FAIL: {enroll_fail}")
    print("Per-device results:")
    for ip in sorted(device_results.keys()):
        r = device_results[ip]
        print(f"  - {ip}: ok={r['ok']} fail={r['fail']}")


if __name__ == "__main__":
    main()
