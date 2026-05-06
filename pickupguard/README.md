# PickupGuard

Workspace overview of the PickupGuard / chaperone pickup feature. Every file
in this folder is a **symlink** to the real source location — edits in either
place go to the same file.

We use symlinks instead of physical moves because:

- Next.js requires `pages/*` and `pages/api/*` to live inside `web-dataset-collector/`.
- The Python helpers (`pickup_event_writer.py`, `chaperone_allocator.py`) are
  imported by `backend/attendance_listener.py` from its own directory.

Moving any of these would break URLs, API routes, or the live attendance
listener. Symlinks give us a clean per-feature view with zero runtime risk.

## Layout

```
pickupguard/
├── parent-onboarding/        Public form a parent uses to register chaperones
│   ├── page/                 /pickup/onboarding/[token]
│   └── api/                  lookup, submit, face upload
│
├── tv-display/               Always-on FireTV/kiosk display at each gate
│   ├── page/                 /pickup/tv
│   └── api/                  feed, pairing flow (start/poll/claim/whoami)
│
├── admin/                    School-staff review queue + kiosk management
│   ├── page/                 /v2/pickup-admin (+ legacy redirect)
│   ├── components/           KioskManager (TV profile + device admin UI)
│   └── api/                  approve, reject, bulk-action, list,
│                             reenroll, tv-devices, kiosk-profiles, kiosk-status
│
├── shared-lib/               Code reused across the three sub-apps
│   ├── pickup-token.js       HMAC-signed onboarding token
│   ├── tv-devices.js         TV pairing / device-token store
│   ├── kiosk-profiles.js     Per-gate display profile (gate hours, etc.)
│   ├── rate-limit.js         Per-IP limiter for public TV/onboarding APIs
│   └── chaperone-enroll.js   Push approved chaperones to all Hikvision devices
│
└── backend/                  Python pieces (called from attendance_listener)
    ├── pickup_event_writer.py        Writes pickup_events from face scans
    ├── chaperone_allocator.py        Atomic 9XXX employeeNo allocator
    ├── init_pickup.py                One-shot bootstrap of pickup_settings
    ├── issue_pickup_onboarding_token.py   CLI for sending parents an onboarding link
    └── seed_pickup_demo.py           Demo data for local dev
```

## Live URLs (when `npm run dev` is up in `web-dataset-collector/`)

| URL | Purpose |
|---|---|
| `/pickup/onboarding/<token>` | Parent self-service chaperone registration |
| `/pickup/tv` | TV / kiosk display for each gate |
| `/v2/pickup-admin` | Staff review queue (Pending / Approved / Rejected) |
| `/v2/pickup-admin?view=kiosks` | TV kiosk profile + device pairing UI |

## Data model (Firestore, under `tenants/{tid}/`)

| Collection | Written by | Read by |
|---|---|---|
| `pickup_onboarding/{id}` | parent submit | admin queue |
| `chaperones/{chap-9XXX}` | admin approve | listener, TV feed |
| `pickup_events/{id}` | listener (Python) | TV feed |
| `tv_devices/{id}` | TV pair flow | TV feed, admin kiosks |
| `kiosk_profiles/{id}` | admin kiosks | TV feed |
| `security_incidents/{id}` | listener (Python) | admin (read-only) |
| `settings/pickup` | admin | listener |
| `id_allocations/chaperone-counter` | allocator (atomic txn) | — |

## Auth & rate limits

- Parent onboarding token: HMAC-SHA256, signed with `CONSENT_SIGNING_SECRET`,
  TTL 30 days, purpose-bound (`p:'pickup-onboarding'`).
- TV pairing: device gets a long-lived token stored in `localStorage`;
  admin can revoke server-side.
- Public APIs (`onboarding/*`, `tv/*` claim/poll/start) are rate-limited
  per-IP via `shared-lib/rate-limit.js`.
- Admin APIs require Firebase ID token (verified via the dashboard middleware).

## Running locally

```bash
# Dashboard + TV + parent onboarding (single Next.js app)
cd web-dataset-collector && npm run dev          # → http://localhost:3000

# Live attendance listeners (which write pickup_events when a chaperone scans)
cd backend && python3 run_listeners.py
```
