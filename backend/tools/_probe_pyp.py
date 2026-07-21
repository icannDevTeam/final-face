#!/usr/bin/env python3
"""Quick PYP Lobby health probe. Uses devices.local.json for the password."""
import json, requests
from pathlib import Path
from requests.auth import HTTPDigestAuth

HERE = Path(__file__).parent
local = {d["name"]: d for d in json.loads((HERE / "devices.local.json").read_text())}
pw = local["PYP Lobby Entrance (DS-K1T342MFX)"]["password"]
ip = "10.26.30.59"
auth = HTTPDigestAuth("admin", pw)

def hit(method, path, **kw):
    url = f"http://{ip}{path}"
    try:
        r = requests.request(method, url, auth=auth, timeout=10, **kw)
        return r.status_code, r.text[:400]
    except Exception as e:
        return "ERR", str(e)[:200]

print("--- userCheck (lockout status) ---")
print(*hit("get", "/ISAPI/Security/userCheck"))
print("--- UserInfo/Count ---")
print(*hit("get", "/ISAPI/AccessControl/UserInfo/Count?format=json"))
print("--- FDLib/FDSearch/Count (face DB) ---")
print(*hit("post",
          "/ISAPI/Intelligent/FDLib/FDSearch/Count?format=json",
          json={"FDID": "1", "faceLibType": "blackFD"}))
print("--- FaceRecognizeMode ---")
print(*hit("get", "/ISAPI/AccessControl/FaceRecognizeMode?format=json"))
print("--- DeviceInfo ---")
print(*hit("get", "/ISAPI/System/deviceInfo?format=json"))
