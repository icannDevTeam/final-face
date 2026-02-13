# BINUS Facial Attendance System

Web-based facial recognition attendance with Hikvision device integration and real-time dashboard.

## Quick Start

**Web App:** https://dataset-sigma.vercel.app/
- `/` — Photo capture & upload to Firebase
- `/hikvision` — Batch enroll students to device
- `/dashboard` — Real-time attendance monitoring

**Start Listener (Jetson):**
\`\`\`bash
python3 attendance_listener.py
\`\`\`

## Architecture

- Next.js 14 web app on Vercel
- Python listener on Jetson (monitors Hikvision event stream)
- Hikvision DS-K1T341AMF face terminal
- Firebase Storage + Firestore

## Key Files

- \`attendance_listener.py\` — Live event stream monitor (production)
- \`pages/dashboard.js\` — Dashboard UI
- \`pages/api/attendance/\` — API routes
- \`lib/hikvision.js\` — Custom HTTP Digest Auth

## Notes

- Device event search not supported — using live stream instead
- Dashboard auto-refreshes every 10s
- Listener logs to local JSON + Firestore
