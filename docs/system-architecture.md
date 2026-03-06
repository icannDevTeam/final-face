# BINUS Facial Attendance — System Architecture

## 1. High-Level Flow

```
[Hikvision Terminals]
        │  (ISAPI alertStream)
        ▼
[Edge Gateway / attendance_listener.py]
        │  (REST + Firestore SDK)
        ├────────────► [Firebase Firestore]
        │                └─ attendance/{date}/records
        │
        └────────────► [BINUS School API]  (best-effort sync)

[Mobile PWA (Vite/React)] ───► Firestore + BINUS API proxy (Vercel)
           │
           └─ Face recognition via face-api.js models (public/models)

[Web Dataset Collector / Dashboard (Next.js)]
           └─ Device enrollment, monitoring, manual lookup
```

## 2. System Layers

### Edge Capture Layer
- **Devices**: Hikvision DS-K1T341AMF terminals running face access control firmware.
- **Protocols**: HTTP Digest Auth against `/ISAPI/AccessControl/**` endpoints and persistent `/ISAPI/Event/notification/alertStream` for live events.
- **Gateway**: Mini PC / Jetson running [`attendance_listener.py`](../attendance_listener.py) for each campus; responsibilities:
  - Maintain digest auth session and reconnect with backoff.
  - Enrich events with local metadata (`student_metadata.py`).
  - Deduplicate within an 8-hour window and store JSON backups under `data/attendance/`.
  - Push records to Firestore and trigger BINUS API uploads via [`api_integrate.py`](../api_integrate.py).

### Cloud Data Plane
- **Firestore collections**:
  - `attendance/{date}/records/{employeeNo}` — canonical record shared by all channels.
  - `students`, `student_metadata` — lookup tables used by both edge listener and PWA.
  - `face_descriptors` — embeddings referenced by the PWA face matcher.
- **Security**: Firebase Admin SDK on edge/server components; Firebase Web SDK (anonymous auth) for the PWA.
- **Caching**: [`mobile-attendance/src/lib/cache.js`](../mobile-attendance/src/lib/cache.js) memoizes descriptors, student profiles, and check-in state to minimize reads.

### Mobile PWA (Vite + React)
- Codebase: [`mobile-attendance/`](../mobile-attendance/).
- Features:
  - GPS-based geofence with Leaflet map ([`LiveMap.jsx`](../mobile-attendance/src/components/LiveMap.jsx)).
  - Face recognition pipeline using `face-api.js` models stored in `/public/models`.
  - Offline-ready via [`public/sw.js`](../mobile-attendance/public/sw.js) — cache v5 precaches manifest, BINUS icons, and ML models for instant loads.
  - BINUS-branded install surface defined in [`public/manifest.json`](../mobile-attendance/public/manifest.json) and regenerated icons (logo-based PNGs).

### Web Control Room (Next.js 14)
- Codebase: [`web-dataset-collector/`](../web-dataset-collector/).
- Responsibilities:
  - Device enrollment (photo capture, metadata push) via `/pages/api/hikvision/*`.
  - Dashboards for live attendance, metrics, and manual lookup.
  - BINUS API proxy deployed on Vercel for secure server-side credential handling (`/api/binus-attendance`).

### External Integrations
- **BINUS School API**: HTTP-only endpoint requiring API key; all calls funnel through `api_integrate.py` (edge) or Vercel proxy (web/mobile) to avoid exposing credentials.
- **face-api.js Models**: Bundled assets loaded by both PWA and seed scripts (`scripts/seed-descriptors.cjs`).
- **Monitoring**: Prometheus + Grafana docker stack (`monitoring/`) collects listener/device metrics.

## 3. Data Flow Summary

1. **Enrollment**
   - Admin captures photos via Next.js collector → uploads to Firebase Storage → descriptors stored in Firestore `face_descriptors`.
   - Student metadata (BINUS IDs, homeroom) synced to `student_metadata` collection and local JSON backup.

2. **Attendance via Device**
   - Terminal emits AccessControllerEvent.
   - Edge listener validates, dedups, saves backup JSON, writes Firestore, then calls BINUS API.

3. **Attendance via Mobile PWA**
   - User passes GPS geofence, loads face models, matches descriptors locally.
   - App writes to Firestore via `checkIn()`; BINUS API sync is queued through `/api/binus-attendance`.
   - Service worker ensures icons/manifest are offline-ready for install.

4. **Dashboard & Lookup**
   - Next.js app reads Firestore directly for real-time stats.
   - API routes manage Hikvision enrollment, capturing device snapshots, and surfacing metrics.

## 4. Scaling & Deployment
- **Deployment Pipeline**: GitHub → Vercel (mobile + dashboard). Edge listener deployed manually on campus mini PCs.
- **Per-Campus Scale**: One listener handles ~6 terminals / ~5k daily scans. Current Firebase/BINUS throughput supports ~3 campuses (~15k/day) without re-architecture.
- **Future Scaling**: Introduce regional queueing (Pub/Sub) between listeners and Firestore, add automated device management, and shard attendance collections when expanding to 10+ campuses.

---
Use this document as the single source when generating infographics or executive summaries.
