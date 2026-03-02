# Project: BINUS Facial Attendance System

> Project-level Claude memory. Loaded at session start for this workspace.

## Overview

Web-based facial recognition attendance system with Hikvision device integration, real-time dashboard, and mobile PWA for face-scan check-in.

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Hikvision       │    │  Jetson Nano     │    │  Firebase       │
│  DS-K1T341AMF   │───▶│  (Python listener)│───▶│  Firestore +    │
│  Face Terminal   │    │  attendance_      │    │  Storage        │
└─────────────────┘    │  listener.py      │    └────────┬────────┘
                       └──────────────────┘             │
                                                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  mobile-         │    │  web-dataset-    │    │  Vercel         │
│  attendance/     │    │  collector/      │    │  (deployment)   │
│  (Vite PWA)     │    │  (Next.js 14)    │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Key Components

| Component                | Path                        | Purpose                              |
|-------------------------|-----------------------------|--------------------------------------|
| Face recognition engine | `main.py`                   | OpenCV + dlib face processing (2578 lines) |
| Attendance listener     | `attendance_listener.py`    | Live Hikvision event stream monitor  |
| Dataset collector       | `web-dataset-collector/`    | Next.js app for photo capture/upload |
| Mobile attendance       | `mobile-attendance/`        | Vite PWA with face-api.js            |
| API integration         | `api_integrate.py`          | BINUS School API bridge              |
| Student metadata        | `student_metadata.py`       | employeeNo → BINUS ID mapping        |
| Monitoring stack        | `monitoring/`               | Prometheus + Grafana                 |
| Stress testing          | `stress_test.py`            | Load/performance testing             |

## Data Flow

1. **Enrollment**: Photo captured via web UI → Firebase Storage → Hikvision device
2. **Attendance**: Face scanned at terminal → event stream → `attendance_listener.py` → Firestore + BINUS API
3. **Mobile**: PWA scans face via camera → face-api.js matches → geolocation verified → attendance logged
4. **Dashboard**: Real-time Firestore listener → auto-refresh every 10s

## Important Files

- `data/student_metadata.json` — Student records (employeeNo, binusId, name)
- `data/attendance/*.json` — Daily attendance logs (by date)
- `facial-attendance-binus-firebase-adminsdk.json` — Firebase service account (DO NOT COMMIT)
- `.env` — Environment secrets

## Conventions

- Python scripts are standalone modules (one file = one concern)
- Next.js pages in `pages/`, API routes in `pages/api/`
- Local JSON persistence in `data/` directory
- Firebase for cloud state, local JSON for edge/backup
- Hikvision integration uses custom HTTP Digest Auth (`lib/hikvision.js`)

## Known Gotchas

- Hikvision device event *search* endpoint is not supported — must use live event stream
- `main.py` is large (2578 lines) — the core face recognition engine, handles multiple modes
- Firebase SDK needs explicit initialization with service account JSON
- Mobile PWA uses client-side face-api.js models served from `public/models/`

## Active TODOs

| # | Task | Priority | Status |
|---|------|----------|--------|
| 1 | Implement Firebase Auth in mobile PWA (Firestore rules require it) | **Critical** | Not started |
| 2 | Add auth/middleware to Next.js API routes (all unprotected) | **High** | Not started |
| 3 | Fix `ScanPage.jsx` referencing `/logo.jpe` — likely should be `.jpg`/`.jpeg` | Medium | Not started |
| 4 | Add missing PWA icons (`icon-192.png`, `icon-512.png`, `apple-touch-icon.png`) to `mobile-attendance/public/` | Medium | Not started |
| 5 | Implement `HomePage.jsx` (currently empty file) | Medium | **Done** |
| 6 | Implement `InstallPrompt.jsx` (currently empty file) | Low | **Done** |
| 7 | Replace optical flow placeholder in `main.py:645` (always returns 0.5) | Low | Not started |
| 8 | Replace health status placeholder in `main.py:2056` (hardcoded 'OK') | Low | Not started |
| 9 | Add 404 catch-all route to `mobile-attendance/src/App.jsx` | Low | Not started |
| 10 | Add React error boundaries to both web apps | Low | Not started |
| 11 | Add offline attendance queueing in service worker | Low | Not started |
| 12 | Add "my attendance history" page to mobile app | Low | Not started |
| 13 | Extend Prometheus monitoring to cover `attendance_listener.py` + mobile | Low | Not started |

## Known Bugs

| Bug | File(s) | Details |
|-----|---------|---------|
| **Firestore writes will be denied** | `mobile-attendance/src/lib/firebase.js`, `firestore.rules` | Firestore rules require `request.auth != null` but mobile app never initializes Firebase Auth — all writes silently fail |
| **Missing PWA icons** | `mobile-attendance/public/sw.js:8-10` | Service worker precaches `icon-192.png`, `icon-512.png`, `apple-touch-icon.png` — none exist in `public/` |
| **Logo 404** | `mobile-attendance/src/pages/ScanPage.jsx:216` | References `/logo.jpe` — unusual extension, likely a typo |
| **Late cutoff duplicated & divergent** | `attendance_listener.py:55`, `mobile-attendance/src/lib/api.js:97` | Hardcoded `08:15` in two places (Python + JS) with no shared constant — easy to get out of sync |
| **WIB time uses naive UTC+7 offset** | `api.js:22`, `attendance-monitor.js:24` | `Date.now() + 7*3600*1000` — no DST/clock-drift handling |

## Technical Debt

| Item | Details |
|------|---------|
| **`main.py` is 2,578 lines** | Monolithic god-file. Needs decomposition into modules (detection, recognition, tracking, liveness, utils). |
| **Zero test coverage** | No pytest/jest config. `test_attendance_insert.py` is an API sim, `stress_test.py` is load testing. No unit/integration tests. |
| **Duplicate Hikvision digest auth** | Python impl in `attendance_listener.py` + Node.js impl in `lib/hikvision.js`. Two codebases to maintain. |
| **Hardcoded Hikvision password fallback** | `password.123` as env fallback in `attendance_listener.py:51`, `hikvision_attendance.py:57`, `capture_and_upload.py:45` |
| **Grafana password in plaintext** | `binus2026` in `monitoring/docker-compose.yml:46` |
| **Wrong deps in mobile-attendance** | `firebase-admin` (server-only, 1.6MB) and `canvas` (native) are in `devDependencies` — shouldn't be in a browser-only Vite app |
| **No input validation on API routes** | Most Next.js API routes lack request body validation/sanitization |
| **No rate limiting** | API routes vulnerable to abuse — no rate limiting on BINUS API proxy or Hikvision operations |
| **Student IDs in stress test** | Real student IDs hardcoded in `stress_test.py:28-34` — data privacy concern in source control |

## Security Concerns

| Concern | Severity |
|---------|----------|
| Firebase Auth not implemented in mobile app — rules either wide open or app is broken | **Critical** |
| All Next.js API routes have zero authentication — device creds flow through unprotected endpoints | **High** |
| Hardcoded Hikvision password fallback `password.123` if `.env` is missing | **Medium** |
| Client-side Firestore writes — possible to forge attendance if auth bypassed | **Medium** |
| Grafana admin password in plaintext docker-compose | **Medium** |
| Firebase service account JSON on disk (but correctly `.gitignore`d — verified safe) | **Low** |

## Decisions Log

| Date       | Decision                                  | Reason                                    |
|-----------|-------------------------------------------|-------------------------------------------|
| 2026-03-02 | Created CLAUDE.md for session persistence | Track project context across Claude sessions |
| 2026-03-02 | Set Vite root explicitly in vite.config.js | Vite was resolving wrong root when run from workspace directory |
| 2026-03-02 | CSS Modules for HomePage | Consistent with ScanPage pattern, no Tailwind dependency |

## Session History

| Date       | What was done                              |
|-----------|---------------------------------------------|
| 2026-03-02 | Initialized project + system CLAUDE.md     |
| 2026-03-02 | Full project audit — 13 TODOs, 5 bugs, 9 debt items, 6 security concerns |
| 2026-03-02 | Built HomePage.jsx — real GPS, BINUS logo, fingerprint icon, map preview, nav to /scan |
| 2026-03-02 | Built InstallPrompt.jsx — PWA install banner with beforeinstallprompt |
| 2026-03-02 | Fixed broken SVG path in fingerprint icon, fixed Vite root config |

---
*Update after each session with decisions, changes, and learnings.*
