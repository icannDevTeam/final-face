# BINUS Facial Attendance System

Web-based facial recognition attendance with Hikvision device integration, real-time dashboard, and mobile PWA.

## Repository Structure

```
final-face/
├── backend/                    # Python backend (face recognition, listeners, APIs)
│   ├── main.py                 # Core face recognition engine (OpenCV + dlib)
│   ├── attendance_listener.py  # Live Hikvision event stream monitor
│   ├── hikvision_attendance.py # Hikvision device management
│   ├── api_integrate.py        # BINUS School API bridge
│   ├── student_metadata.py     # employeeNo → BINUS ID mapping
│   ├── enroll.py               # Face enrollment pipeline
│   ├── capture_and_upload.py   # Camera capture → Firebase upload
│   ├── firebase_dataset_sync.py# Firebase ↔ local dataset sync
│   ├── collect_metrics.py      # Recognition metrics collection
│   ├── stress_test.py          # Load/performance testing
│   ├── test_attendance_insert.py # API simulation tests
│   ├── requirements.txt        # Python dependencies
│   └── data/                   # Runtime data (attendance logs, metadata)
├── mobile-attendance/          # Vite + React PWA (face scan check-in)
├── web-dataset-collector/      # Next.js 14 (photo capture, enrollment, dashboard) [submodule]
├── monitoring/                 # Prometheus + Grafana stack
├── docs/                       # Architecture documentation
├── firebase.json               # Firebase project config
├── firestore.rules             # Firestore security rules
└── .env                        # Shared environment variables
```

## Quick Start

### Backend (Python)
```bash
cd backend
pip install -r requirements.txt
python3 attendance_listener.py    # Start Hikvision event listener
```

### Web Dashboard
```bash
cd web-dataset-collector
npm install
npm run dev                       # http://localhost:3000
```
- `/` — Photo capture & upload to Firebase
- `/hikvision` — Batch enroll students to device
- `/dashboard` — Real-time attendance monitoring
- `/device-manager` — Hikvision device management

### Mobile PWA
```bash
cd mobile-attendance
npm install
npm run dev                       # http://localhost:5173
```

**Production:** https://final-face-ten.vercel.app/

## Architecture

| Component | Tech | Location |
|-----------|------|----------|
| Face Recognition Engine | OpenCV + dlib + Python | `backend/` |
| Attendance Listener | Python (Jetson Nano) | `backend/attendance_listener.py` |
| Web Dashboard | Next.js 14 + Vercel | `web-dataset-collector/` |
| Mobile Check-in | Vite + React + face-api.js | `mobile-attendance/` |
| Face Terminal | Hikvision DS-K1T341AMF | Hardware (ISAPI) |
| Database | Firebase Firestore + Storage | Cloud |
| Monitoring | Prometheus + Grafana | `monitoring/` |

## Notes

- Device event search not supported — using live stream instead
- Dashboard auto-refreshes every 10s
- Listener logs to local JSON + Firestore
- `web-dataset-collector/` is a git submodule (`icannDevTeam/dataset`)
