#!/usr/bin/env python3
"""
quick_terminal_enroll.py

Fast one-off enrollment to a Hikvision terminal:
1) create/update a user
2) upload one face photo to FDLib
3) verify the user exists

Example:
  /home/pandora/Downloads/final-face/.venv/bin/python backend/scripts/quick_terminal_enroll.py \
    --ip 10.26.30.201 --password 'aiclub@406013' \
    --employee-no 9000000999 --name 'Quick Trial' \
    --image /path/to/photo.jpg --delete-first
"""

from __future__ import annotations

import argparse
import json
import random
import re
import string
import sys
from pathlib import Path

import requests
from requests.auth import HTTPDigestAuth


class HikClient:
    def __init__(self, ip: str, username: str, password: str, timeout: int = 25):
        self.ip = ip
        self.base = f"http://{ip}"
        self.username = username
        self.password = password
        self.timeout = timeout

    def req(self, method: str, path: str, payload=None, headers=None, raw=False):
        url = f"{self.base}{path}"
        kwargs = {"timeout": self.timeout}
        if headers:
            kwargs["headers"] = headers
        if payload is not None:
            if raw:
                kwargs["data"] = payload
            else:
                kwargs["json"] = payload

        r = requests.request(method, url, auth=HTTPDigestAuth(self.username, self.password), **kwargs)
        if r.status_code == 401:
            r = requests.request(method, url, auth=HTTPDigestAuth(self.username, self.password), **kwargs)

        try:
            body = r.json()
        except Exception:
            body = {"raw": (r.text or "")[:800]}
        return r.status_code, body

    def reachable(self) -> bool:
        code, _ = self.req("get", "/ISAPI/System/deviceInfo?format=json")
        return code == 200

    def delete_face(self, employee_no: str):
        body = {
            "FPID": [{"value": str(employee_no)}],
            "faceLibType": "blackFD",
            "FDID": "1",
        }
        return self.req("put", "/ISAPI/Intelligent/FDLib/FDSearch/Delete?format=json", body)

    def delete_user(self, employee_no: str):
        body = {"UserInfoDelCond": {"EmployeeNoList": [{"employeeNo": str(employee_no)}]}}
        return self.req("put", "/ISAPI/AccessControl/UserInfo/Delete?format=json", body)

    def create_user(self, employee_no: str, name: str) -> bool:
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
        code, data = self.req("post", "/ISAPI/AccessControl/UserInfo/Record?format=json", body)
        if code == 200:
            return True
        txt = json.dumps(data)
        return "employeeNoAlreadyExist" in txt or "deviceUserAlreadyExist" in txt

    def upload_face(self, employee_no: str, name: str, img_bytes: bytes):
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

        code, data = self.req(
            "put",
            "/ISAPI/Intelligent/FDLib/FDSetUp?format=json",
            payload=body,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(body)),
            },
            raw=True,
        )
        return code == 200, code, data

    def user_exists(self, employee_no: str) -> bool:
        body = {
            "UserInfoSearchCond": {
                "searchID": "quick-enroll-check",
                "searchResultPosition": 0,
                "maxResults": 1,
                "EmployeeNoList": [{"employeeNo": str(employee_no)}],
            }
        }
        code, data = self.req("post", "/ISAPI/AccessControl/UserInfo/Search?format=json", body)
        if code != 200:
            return False
        section = (data or {}).get("UserInfoSearch") or {}
        users = section.get("UserInfo") or []
        if isinstance(users, dict):
            users = [users]
        return any(str(u.get("employeeNo") or "") == str(employee_no) for u in users)


def main() -> int:
    ap = argparse.ArgumentParser(description="Quick enroll one face to one Hikvision terminal")
    ap.add_argument("--ip", required=True, help="Terminal IP")
    ap.add_argument("--username", default="admin")
    ap.add_argument("--password", required=True)
    ap.add_argument("--employee-no", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--image", required=True, help="Path to JPG/JPEG/PNG image")
    ap.add_argument("--delete-first", action="store_true", help="Delete existing user+face before re-enroll")
    args = ap.parse_args()

    img_path = Path(args.image)
    if not img_path.exists() or not img_path.is_file():
        print(f"ERROR: image file not found: {img_path}")
        return 2
    if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        print("ERROR: image must be .jpg/.jpeg/.png")
        return 2

    img_bytes = img_path.read_bytes()
    if len(img_bytes) < 5_000:
        print(f"ERROR: image file is too small ({len(img_bytes)} bytes); likely invalid for FDSetUp")
        return 2

    cli = HikClient(args.ip, args.username, args.password)
    print(f"Checking device {args.ip}...")
    if not cli.reachable():
        print("ERROR: terminal not reachable or auth failed")
        return 3

    employee_no = str(args.employee_no).strip()
    if not re.fullmatch(r"\d+", employee_no):
        print("ERROR: --employee-no must be numeric")
        return 2

    if args.delete_first:
        cli.delete_face(employee_no)
        cli.delete_user(employee_no)

    print(f"Creating user {employee_no} ({args.name})...")
    if not cli.create_user(employee_no, args.name):
        print("ERROR: create user failed")
        return 4

    print("Uploading face image...")
    ok, code, body = cli.upload_face(employee_no, args.name, img_bytes)
    if not ok:
        print(f"ERROR: upload face failed (HTTP {code})")
        print(json.dumps(body)[:800])
        return 5

    print("Verifying user search...")
    if not cli.user_exists(employee_no):
        print("WARN: user search did not find employeeNo after upload")
        return 6

    print("SUCCESS: user + face enrolled")
    print(json.dumps({
        "ip": args.ip,
        "employeeNo": employee_no,
        "name": args.name,
        "image": str(img_path),
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
