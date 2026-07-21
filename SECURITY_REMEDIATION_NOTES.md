# Security Remediation Notes

This file documents vulnerabilities that require architectural changes, external action,
or a dedicated remediation pass — and therefore were NOT auto-applied by the surgical
fix pass. Each entry describes the threat, what must be done, and what depends on it.

## Status Update — 2026-06-12

| Change | Impact |
|--------|--------|
| **Mobile attendance app taken down** | VULN-002, VULN-007, VULN-012, VULN-013, VULN-016 are **no longer active** — the attack surface they described (client-side liveness, GPS spoofing, localStorage PII, client clock, SW cache poisoning) is gone while the mobile app is offline. Revisit if/when the app is re-deployed. |
| **Hikvision device passwords rotated** | VULN-001 Hikvision credential exposure is **mitigated**. Old password `password.123` in git history can no longer be used to authenticate to any device. Git history still contains the old value — run `git filter-repo` cleanup when convenient but it is no longer an active credential. |
| **BINUS API was sandbox environment** | VULN-001 BINUS API key and VULN-005 plaintext HTTP are **lower risk** — sandbox data, not production student PII. Rotate the sandbox key anyway (good hygiene) and enforce HTTPS before connecting to any production BINUS API endpoint. |

---

## VULN-001 (CRITICAL) — Git History Credential Leak

**Threat**: A BINUS API key (`OUQyQjdEN0EtREFDQy00QkEyLTg3QTAtNUFGNDVDOUZCRTgy`) and
a Hikvision device password were committed to git history (in `DOCS.md` or similar).
Both credentials must be treated as **permanently compromised** even after the file is
removed from the working tree.

**Required actions (external)**:
1. Rotate the BINUS School API key with the BINUS IT team immediately.
2. Rotate the Hikvision device password via the device admin panel.
3. Remove the file from all branches and tags:
   ```
   git filter-repo --path DOCS.md --invert-paths
   ```
4. Force-push all branches: `git push --force --all`
5. Ask all collaborators to re-clone — cached refs still carry the old objects.
6. Revoke any GitHub/GitLab deploy keys or CI tokens that had access to the old history.

**Cannot be automated**: requires external credential rotation and destructive git history rewrite.

---

## VULN-002 (CRITICAL) — Client-Controlled Liveness & Confidence Score

**Threat**: The mobile PWA (`mobile-attendance/src/lib/api.js`) sends `livenessVerified: true`
and `confidence` as client-supplied values in the Firestore write. An attacker who can
intercept or replay the Firestore write can forge attendance with arbitrary confidence.

**Required architecture**:
- Add a server-side face validation API route (e.g. `/api/verify-face`) that:
  1. Accepts the raw face image + studentId
  2. Runs liveness + matching server-side (Python backend or serverless)
  3. Issues a short-lived signed token (JWT or HMAC, 60 s TTL) containing `{studentId, livenessVerified: true, confidence}`
- Modify `checkIn()` in `api.js` to attach the server-issued token to the Firestore write
- Add a Firestore rule condition that verifies the token signature before accepting `create`

**Cannot be done** without changing the mobile app architecture and adding a new backend endpoint.

---

## VULN-007 (HIGH) — GPS Anti-Spoofing Is Client-Side Only

**Threat**: The spoof-score computation in `mobile-attendance/src/lib/geolocation.js` runs
entirely in the browser. A motivated attacker can patch the JS bundle or intercept the
Firestore write and set `gpsSpoofScore: 0` with a fake location.

**Required architecture**:
- Add a server-side location verification endpoint (e.g. `/api/verify-location`) that:
  1. Accepts `{lat, lng, accuracy, userAgent, timestamp}`
  2. Verifies against IP geolocation as a secondary signal
  3. Issues a short-lived signed location token
- Modify `checkIn()` to include the server-issued location token
- Reject Firestore writes missing a valid location token

**Cannot be done** without a new backend endpoint and mobile app changes.

---

## VULN-005 / VULN-006 (HIGH) — Plaintext HTTP to BINUS API and Hikvision

**Threat**:
- BINUS School API calls go to `http://binusian.ws` (port 80). Bearer tokens and student
  PII are transmitted in cleartext.
- Hikvision ISAPI calls go to `http://<device-ip>` (port 80). Device credentials and face
  images are transmitted in cleartext on the school LAN.

**Required actions**:
- **BINUS API**: Contact BINUS IT team to enable HTTPS on port 443. Once confirmed, change
  `BINUS_BASE` in `web-dataset-collector/pages/api/student/lookup.js` from `http://` to `https://`.
- **Hikvision**: Enable HTTPS on the device via `/ISAPI/Security/TLS` (device admin panel or
  Hikvision iVMS-4200). Then change `HIK_BASE` in `enroll.js` and equivalent Python scripts
  from `http://` to `https://`.

**Cannot be automated**: requires BINUS IT and physical/admin access to Hikvision device.

---

## VULN-013 (MEDIUM) — Client Clock Manipulation for Late/Present Status

**Threat**: `getWIBNow()` in `mobile-attendance/src/lib/api.js` computes WIB time as
`Date.now() + 7*3600*1000`. A student can manipulate the device clock to clock in "on time"
while physically arriving late.

**Required fix**:
- Replace client-side `Date.now()` with Firebase `serverTimestamp()` in the Firestore write
- Recompute `isLate` server-side (in a Cloud Function or the Python listener) based on the
  server-recorded timestamp, not the client-supplied `timestamp` field
- The `status` field should be set by a trusted process, not the client

**Cannot be done** without changing the attendance write path and adding server-side recomputation.

---

## VULN-014 (MEDIUM) — MD5 Digest Authentication to Hikvision

**Threat**: The Python (`attendance_listener.py`) and Node.js (`lib/hikvision.js`) Hikvision
clients use HTTP Digest auth with MD5, which is susceptible to offline dictionary attacks
if traffic is captured on the LAN.

**Required actions**:
- Configure the Hikvision device to require SHA-256 Digest via:
  `PUT /ISAPI/Security/UserCheck` with `<digestAlgorithm>SHA256</digestAlgorithm>`
- Update both the Python `requests` digest auth (use `requests.auth.HTTPDigestAuth` — it
  auto-negotiates, but verify the device is actually offering SHA-256 in its `WWW-Authenticate`)
- Update the Node.js digest implementation in `lib/hikvision.js` to use SHA-256

**Cannot be automated**: requires device configuration + code changes to both client implementations.

---

## VULN-009 (HIGH) — npm Dependency CVEs

**Threat**: `npm audit` likely surfaces known CVEs in `mobile-attendance/` and
`web-dataset-collector/` dependencies.

**Required action** (dedicated pass):
```bash
cd mobile-attendance && npm audit fix
cd web-dataset-collector && npm audit fix
```
Verify after each run that no breaking changes were introduced (run dev build + smoke test).
Do NOT run `npm audit fix --force` blindly — it may introduce breaking major-version upgrades.

**Not done here**: requires a dedicated testing pass to verify no regressions.

---

## VULN-020 (LOW) — Grafana Admin Password in Plaintext docker-compose

**Threat**: `monitoring/docker-compose.yml` contains `GF_SECURITY_ADMIN_PASSWORD=binus2026`
in plaintext. If this file is in git history, the password is exposed.

**Required action**:
1. Rotate the Grafana admin password in the running stack:
   ```bash
   docker exec -it grafana grafana-cli admin reset-admin-password <new-password>
   ```
2. Move the password to `monitoring/.env` (already gitignored):
   ```
   GF_SECURITY_ADMIN_PASSWORD=<new-strong-password>
   ```
3. Reference it in `docker-compose.yml`:
   ```yaml
   environment:
     - GF_SECURITY_ADMIN_PASSWORD=${GF_SECURITY_ADMIN_PASSWORD}
   ```
4. Confirm `monitoring/.env` is in `.gitignore` (it is — already listed).

**Cannot be automated**: requires running stack access and secret rotation.

---

## VULN-004 (HIGH) — PII in Log Files on Jetson

**Threat**: `backend/attendance_listener.py` may log full student name/ID lists to
`backend/listeners.log.*` during enrollment dumps. These log files on the Jetson Nano
are not encrypted and may be accessible to anyone with device access.

**Required fix**:
- In `attendance_listener.py`, replace any enrollment dump that logs individual student
  names or IDs with a count-only summary:
  ```python
  # Instead of: logging.info(f"Enrolled students: {student_list}")
  logging.info("Enrolled students: %d total", len(student_list))
  ```
- Implement log rotation with retention limits (e.g. `logging.handlers.RotatingFileHandler`
  with `maxBytes=5MB, backupCount=3`)
- Consider encrypting the log directory or restricting permissions: `chmod 600 backend/*.log`

**Not done here**: requires careful audit of all log call sites in `attendance_listener.py`
to avoid breaking the monitoring/debugging utility of logs.

---

## Implementation Notes

- FIX-6 (VULN-017): The file `mobile-attendance/public/logo.jpe` is a valid JPEG with an
  unusual extension. It was copied to `logo.png` and the reference in `HomePage.jsx` was
  updated. The original `logo.jpe` was left in place to avoid breaking any CDN caches or
  direct links that may reference it. It can be removed once confirmed no external references
  exist to the old path.

- FIX-3 (VULN-015): `verifyCookie()` in `lib/session-cookie.js` returns `{ email, timestamp }`
  or `null`. The `enroll.js` fix extracts `.email` from the result. If `SESSION_SECRET` (or
  `DASHBOARD_API_KEY`) is not configured, `verifyCookie` returns `null` and `enrolledBy`
  remains `'unknown'` — this is fail-closed and safe.
