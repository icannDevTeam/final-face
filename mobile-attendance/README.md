# BINUS Mobile Attendance — Face Recognition Clock-In

A Workday-style Progressive Web App (PWA) where students tap "Clock In", the camera opens, their face is identified using face-api.js, and attendance is recorded automatically.

## Tech Stack

| Layer          | Technology                           |
|---------------|--------------------------------------|
| Framework     | Vite 7 + React 19                    |
| Face Detection| face-api.js (SSD MobileNet v1)       |
| Face Recog.   | face-api.js (128-d ResNet descriptors)|
| Styling       | CSS Modules (dark theme)             |
| Routing       | React Router v7                      |
| Backend       | Firebase Firestore (client SDK)      |
| PWA           | Service Worker + Web Manifest        |

## How It Works

```
┌──────────────┐
│   HomePage   │  Big clock + "Clock In" button
│  (Workday)   │  On mount: loads face-api models + face descriptors from Firestore
└──────┬───────┘
       │ tap Clock In
       ▼
┌──────────────┐
│   ScanPage   │  Opens front camera (selfie mode)
│  (Camera)    │  Runs SSD face detection → bounding box overlay
│              │  Every 600ms: detectAndMatch() → compares against stored descriptors
│              │  3 consecutive matches = confirmed identity
└──────┬───────┘
       │ identity confirmed
       ▼
  checkIn() → Firestore attendance/{date}/records/{studentId}
       │
       ▼
  ✅ "Clocked In!" overlay → auto-return to home
```

## Face Descriptor Pipeline

1. **Enrollment** — Student photos are captured & stored in Firebase Storage (`face_dataset/{class}/{name}/`)
2. **Seeding** — `scripts/seed-descriptors.cjs` downloads photos, computes 128-d face descriptors using face-api.js, stores them in Firestore `face_descriptors/{studentId}`
3. **Runtime** — The mobile app loads descriptors from Firestore on boot, then matches camera frames against them using Euclidean distance (threshold: 0.5)

## Quick Start

```bash
cd mobile-attendance
npm install

# Seed face descriptors (requires student photos in Firebase Storage)
node scripts/seed-descriptors.cjs

# Dev server
npm run dev
```

## Project Structure

```
mobile-attendance/
├── public/
│   ├── manifest.json          # PWA manifest
│   ├── sw.js                  # Service worker (network-first)
│   ├── favicon.svg            # App icon
│   └── models/                # face-api.js ML models (~12MB)
│       ├── ssd_mobilenetv1_*  # Face detection
│       ├── face_landmark_68_* # Landmark detection
│       └── face_recognition_* # 128-d descriptor extraction
├── scripts/
│   └── seed-descriptors.cjs   # Compute & store face descriptors
├── src/
│   ├── main.jsx               # Entry + SW registration
│   ├── App.jsx                # Router (/ → Home, /scan → Scan)
│   ├── lib/
│   │   ├── firebase.js        # Firebase Firestore init
│   │   ├── faceRecognition.js # face-api.js wrapper (load, detect, match)
│   │   └── api.js             # Attendance CRUD (checkIn, lookup, summary)
│   ├── pages/
│   │   ├── HomePage.jsx       # Workday-style clock + Clock In button
│   │   └── ScanPage.jsx       # Camera view + face detection overlay
│   └── styles/
│       ├── Home.module.css    # Clock UI, big button
│       └── Scan.module.css    # Camera viewport, overlays
├── .env.example
└── index.html
```

## Attendance Record Schema

```json
{
  "name": "Anderson Ian Roesmin",
  "employeeNo": "2170003338",
  "timestamp": "2026-02-18 07:45:30",
  "status": "Present",
  "late": false,
  "homeroom": "6B",
  "grade": "6",
  "source": "mobile_face"
}
```

The `source: "mobile_face"` field distinguishes these records from Hikvision (`hikvision`) and proximity (`mobile_proximity`) check-ins.
