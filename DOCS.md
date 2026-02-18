# Facial Attendance System — Complete Documentation

> **BINUS School Simprug — Hikvision DS-K1T341AMF Face Recognition Attendance System**
>
> Last updated: 2026-02-18

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [System Workflows](#3-system-workflows)
   - 3.1 [Photo Capture & Dataset Collection](#31-photo-capture--dataset-collection)
   - 3.2 [Student Enrollment to Device](#32-student-enrollment-to-device)
   - 3.3 [Live Attendance Recording](#33-live-attendance-recording)
   - 3.4 [OpenCV Local Recognition (Alternative)](#34-opencv-local-recognition-alternative)
4. [External APIs](#4-external-apis)
   - 4.1 [BINUS School API (binusian.ws)](#41-binus-school-api-binusianws)
   - 4.2 [Hikvision ISAPI (Device)](#42-hikvision-isapi-device)
   - 4.3 [Firebase Services](#43-firebase-services)
5. [Internal APIs (Next.js Dashboard)](#5-internal-apis-nextjs-dashboard)
6. [Data Schemas](#6-data-schemas)
7. [File Reference](#7-file-reference)
8. [Environment Variables](#8-environment-variables)
9. [Quick Start Commands](#9-quick-start-commands)

---

## 1. System Overview

This system provides automated facial-recognition-based attendance for BINUS School Simprug. It consists of:

| Component | Technology | Purpose |
|---|---|---|
| **Hikvision DS-K1T341AMF** | Face recognition terminal | Hardware device that detects faces and fires events |
| **Python Backend** | Python 3.11+ | Enrollment, event listening, BINUS API integration |
| **Next.js Web App** | Next.js + React | Photo capture UI, student lookup, attendance dashboard |
| **Firebase** | Firestore + Storage | Cloud database for attendance records + face image storage |
| **BINUS School API** | REST (binusian.ws) | Student lookup, photo data, attendance insertion to e-Desk |

### Key Design Decision

The **BINUS Student ID** (e.g. `2370007317`) is used as the device's `employeeNo`. This means when the device fires a face recognition event, the `employeeNo` field directly contains the student's BINUS ID — enabling seamless attendance insertion to the BINUS e-Desk system without any intermediate mapping.

---

## 2. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        SYSTEM ARCHITECTURE                             │
└─────────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐       ┌──────────────────┐       ┌─────────────────┐
  │   Student    │       │  Next.js Web App │       │  BINUS School   │
  │  (Browser)   │──────▶│  (port 3000)     │──────▶│  API            │
  │              │       │                  │       │  (binusian.ws)  │
  └──────────────┘       └────────┬─────────┘       └────────┬────────┘
                                  │                          │
                           Upload photos             C.0 Auth Token
                           Student lookup            C.1 Student Photos
                                  │                  C.2 Student Enrollment
                                  ▼                  B.2 Attendance Insert
                         ┌────────────────┐                  │
                         │   Firebase     │                  │
                         │  ┌──────────┐  │                  │
                         │  │Firestore │  │                  │
                         │  │ students │  │                  │
                         │  │attendance│  │                  │
                         │  │ metadata │  │                  │
                         │  └──────────┘  │                  │
                         │  ┌──────────┐  │                  │
                         │  │ Storage  │  │                  │
                         │  │face_data │  │                  │
                         │  └──────────┘  │                  │
                         └───────┬────────┘                  │
                                 │                           │
                          Download images                    │
                          Read metadata                      │
                                 │                           │
                                 ▼                           │
                    ┌─────────────────────┐                  │
                    │   Python Backend    │                  │
                    │                     │──────────────────┘
                    │  hikvision_attend.  │     Insert attendance
                    │  attendance_listen. │     via B.2 API
                    │  student_metadata   │
                    │  api_integrate      │
                    └──────────┬──────────┘
                               │
                        ISAPI (HTTP Digest)
                        Enroll faces
                        Listen alertStream
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Hikvision Device   │
                    │  DS-K1T341AMF       │
                    │  IP: 10.26.30.200   │
                    │                     │
                    │  Face Recognition   │
                    │  LCD Display        │
                    │  DeepLearn Engine   │
                    └─────────────────────┘
```

---

## 3. System Workflows

### 3.1 Photo Capture & Dataset Collection

**Purpose:** Collect face photos of students and store them in Firebase for later enrollment to the device.

```
Student opens web app → Enters BINUS Student ID → Web app calls C.2 API
→ Student name/class returned → Student takes 3 photos via webcam
→ Photos uploaded to Firebase Storage → Metadata saved to Firestore
```

**Detailed Steps:**

| Step | Action | Endpoint/Location |
|------|--------|-------------------|
| 1 | Student enters their BINUS Student ID in the web app | `http://localhost:3000` |
| 2 | Web app calls internal API to look up student | `POST /api/student/lookup` |
| 3 | Internal API calls BINUS C.2 endpoint | `POST binusian.ws/binusschool/bss-student-enrollment` |
| 4 | Student name, grade, homeroom returned | Response parsed |
| 5 | Student takes 3 photos via browser webcam | CapturePage component |
| 6 | Each photo uploaded to internal API | `POST /api/face/upload` |
| 7 | Photo saved to Firebase Storage | `face_dataset/{className}/{studentName}/photo_{n}_{ts}.jpg` |
| 8 | Metadata saved to Firestore | `students/{studentId}` document |

**Firestore Document Created** (`students/{studentId}`):
```json
{
  "id": "2370007317",
  "name": "Benjamin Arandra Siregar",
  "homeroom": "4C",
  "gradeCode": "4",
  "gradeName": "Grade 4",
  "displayLabel": "Benjamin Arandra Siregar 4C",
  "lastCaptureAt": "2026-02-18T10:30:00.000Z",
  "totalImages": 3
}
```

---

### 3.2 Student Enrollment to Device

**Purpose:** Download face photos from Firebase, process them, and enroll students on the Hikvision device so it can recognize them.

```
Python reads Firestore `students` collection → Gets BINUS IDs
→ Batch-fetches IdBinusian from C.1 API → Downloads photos from Firebase Storage
→ dlib detects & crops faces → Temp HTTP server serves images
→ Device downloads images via ISAPI FDLib → Extracts own embeddings
→ Metadata saved locally + to Firebase
```

**Three Enrollment Methods:**

| Method | Command | Description |
|--------|---------|-------------|
| `enroll` | `python hikvision_attendance.py enroll` | Bulk enroll from Firebase Storage photos |
| `enroll-live` | `python hikvision_attendance.py enroll-live` | One-by-one, device camera captures face |
| `enroll-class` | `python hikvision_attendance.py enroll-class 4 4C` | Entire class via BINUS API + device camera |

**`enroll` Pipeline (Firebase → Device):**

| Step | Action | Detail |
|------|--------|--------|
| 1 | Load student metadata from Firestore | Collection: `students`, gets IdStudent, name, homeroom, gradeCode |
| 2 | Batch-fetch IdBinusian from BINUS C.1 API | Groups by class, calls `bss-get-simprug-studentphoto-fr` |
| 3 | Download face images from Firebase Storage | `firebase_dataset_sync.py` → local temp directory |
| 4 | Rank images by frontality | dlib 68-point landmarks → nose symmetry + eye centering score |
| 5 | Crop best face with margin | dlib face detector → 480×640 JPEG |
| 6 | Start temp HTTP server | Serves cropped images on port 8888 |
| 7 | Create user on device | `POST /ISAPI/AccessControl/UserInfo/Record` with `employeeNo = IdStudent` |
| 8 | Upload face to FDLib | `PUT /ISAPI/Intelligent/FDLib/FDSetUp` with image URL |
| 9 | Save metadata | `student_metadata.py` → local JSON + Firestore `student_metadata` collection |

**Student Metadata Saved** (`data/student_metadata.json`):
```json
{
  "2370007317": {
    "employeeNo": "2370007317",
    "name": "Benjamin Arandra Siregar",
    "idStudent": "2370007317",
    "idBinusian": "BN125377010",
    "homeroom": "4C",
    "grade": "4",
    "enrolledAt": "2026-02-18T10:00:00+07:00",
    "updatedAt": "2026-02-18T10:00:00+07:00"
  }
}
```

---

### 3.3 Live Attendance Recording

**Purpose:** When the Hikvision device recognizes a face, record attendance to local JSON, Firebase Firestore, and the BINUS e-Desk system.

```
Device recognizes face → Fires event via alertStream
→ attendance_listener.py parses MIME stream → Extracts employeeNo (= IdStudent)
→ Determines Present/Late (cutoff 08:15 WIB) → Dedup check (8hr window)
→ Save to local JSON → Save to Firebase Firestore → Call BINUS B.2 API
```

**Detailed Flow:**

| Step | Action | Detail |
|------|--------|--------|
| 1 | Listener connects to device | `GET /ISAPI/Event/notification/alertStream` (HTTP Digest Auth) |
| 2 | Device streams multipart MIME | Boundary: `--MIME_boundary`, JSON + optional JPEG |
| 3 | Parse AccessControllerEvent | `majorEventType=5`, `subEventType=75/76/104` = face verification |
| 4 | Extract identity | `employeeNoString` = BINUS Student ID, `name` = student name |
| 5 | Dedup check | Skip if already logged within 8 hours (`DUPLICATE_WINDOW = 28800s`) |
| 6 | Determine status | `Present` if before 08:15 WIB, `Late` if after |
| 7 | Save local JSON | `data/attendance/YYYY-MM-DD.json` |
| 8 | Save to Firebase | `attendance/{date}/records/{employeeNo}` |
| 9 | Look up metadata | `student_metadata.get_student(emp_no)` → get IdStudent, IdBinusian |
| 10 | Call BINUS B.2 API | `POST bss-add-simprug-attendance-fr` with `{IdStudent, IdBinusian, ImageDesc, UserAction}` |

**Event Stream Format (raw):**
```
--MIME_boundary
Content-Type: application/json; charset="UTF-8"
Content-Length: 1234

{
  "ipAddress": "10.26.30.200",
  "portNo": 80,
  "protocol": "HTTP",
  "macAddress": "c0:56:e3:xx:xx:xx",
  "channelID": 1,
  "dateTime": "2026-02-18T08:05:30+07:00",
  "activePostCount": 1,
  "eventType": "AccessControllerEvent",
  "eventState": "active",
  "eventDescription": "Access Controller Event",
  "AccessControllerEvent": {
    "deviceName": "Front Entrance",
    "majorEventType": 5,
    "subEventType": 76,
    "name": "Benjamin Arandra Siregar",
    "employeeNoString": "2370007317",
    "serialNo": 42,
    "userType": "normal",
    "currentVerifyMode": "faceOrFpOrCardOrPw"
  }
}
--MIME_boundary
Content-Type: image/jpeg
Content-Length: 45000

<binary JPEG data — captured face photo>
--MIME_boundary
```

**Local Attendance JSON** (`data/attendance/2026-02-18.json`):
```json
{
  "Benjamin Arandra Siregar": {
    "employeeNo": "2370007317",
    "timestamp": "2026-02-18 08:05:30",
    "status": "Present",
    "late": false
  },
  "Anderson Wijaya": {
    "employeeNo": "2170003338",
    "timestamp": "2026-02-18 08:20:15",
    "status": "Late",
    "late": true
  }
}
```

**Firebase Attendance Record** (`attendance/2026-02-18/records/2370007317`):
```json
{
  "name": "Benjamin Arandra Siregar",
  "employeeNo": "2370007317",
  "timestamp": "2026-02-18 08:05:30",
  "status": "Present",
  "late": false,
  "updatedAt": "2026-02-18T08:05:30+07:00"
}
```

---

### 3.4 OpenCV Local Recognition (Alternative)

**Purpose:** `main.py` provides an alternative attendance system using a local camera + OpenCV + dlib for face recognition (no Hikvision device needed).

```
Webcam/RTSP feed → OpenCV frame capture → dlib face detection
→ Face encoding comparison → Match against enrolled encodings
→ Liveness/anti-spoof checks → Record attendance
→ Firebase + BINUS API upload
```

This is a standalone mode that uses the same BINUS API integration and Firebase upload but does all face recognition locally instead of on the Hikvision device.

---

## 4. External APIs

### 4.1 BINUS School API (binusian.ws)

**Base URL:** `http://binusian.ws/binusschool`

All endpoints require authentication via the API key.

---

#### C.0 — Authentication Token

Get a Bearer token required for all other API calls.

| Field | Value |
|-------|-------|
| **URL** | `http://binusian.ws/binusschool/auth/token` |
| **Method** | `GET` |
| **Auth** | `Basic {API_KEY}` (in `Authorization` header) |

**Request:**
```http
GET /binusschool/auth/token HTTP/1.1
Host: binusian.ws
Authorization: Basic OUQyQjdEN0EtREFDQy00QkEyLTg3QTAtNUFGNDVDOUZCRTgy
```

**Response (Success):**
```json
{
  "resultCode": 200,
  "errorMessage": null,
  "data": {
    "token": "eyJhbGciOiJIUzI1NiIs...<JWT>",
    "duration": 60
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `resultCode` | int | `200` = success |
| `data.token` | string | Bearer token (JWT), valid for `duration` minutes |
| `data.duration` | int | Token validity in minutes |

**Used by:** Every other API call as `Authorization: Bearer {token}`

---

#### C.1 — Student Photos (Get IdBinusian)

Retrieve face photo metadata for an entire class. **This is the only endpoint that returns `idBinusian`**, which is required for the B.2 attendance insert.

| Field | Value |
|-------|-------|
| **URL** | `http://binusian.ws/binusschool/bss-get-simprug-studentphoto-fr` |
| **Method** | `POST` |
| **Auth** | `Bearer {token}` |
| **Content-Type** | `application/json` |

**Request:**
```http
POST /binusschool/bss-get-simprug-studentphoto-fr HTTP/1.1
Host: binusian.ws
Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
Content-Type: application/json

{
  "Grade": "4",
  "Homeroom": "4C",
  "IdStudentList": null
}
```

| Body Field | Type | Required | Description |
|------------|------|----------|-------------|
| `Grade` | string | Yes | Grade level (`"EY1"`, `"EY2"`, `"1"` - `"12"`) |
| `Homeroom` | string | Yes | Homeroom code (`"4C"`, `"1A"`, etc.) |
| `IdStudentList` | array\|null | No | Filter by specific student IDs, or `null` for all |

**Response (Success):**
```json
{
  "resultCode": 200,
  "errorMessage": null,
  "studentPhotoResponse": {
    "studentList": [
      {
        "idStudent": "2370007317",
        "idBinusian": "BN125377010",
        "fileName": "2370007317_photo.jpg",
        "filePath": "/photos/grade4/2370007317_photo.jpg"
      },
      {
        "idStudent": "2170003338",
        "idBinusian": "BN124456140",
        "fileName": null,
        "filePath": null
      }
    ]
  }
}
```

| Response Field | Type | Description |
|----------------|------|-------------|
| `studentList[].idStudent` | string | BINUS Student ID (7-10 digit number) |
| `studentList[].idBinusian` | string | BINUS internal ID (e.g. `"BN125377010"`) — **needed for B.2** |
| `studentList[].fileName` | string\|null | Photo filename if exists |
| `studentList[].filePath` | string\|null | Photo path if exists |

---

#### C.2 — Student Enrollment (Lookup by ID)

Look up a single student's information by their Student ID.

| Field | Value |
|-------|-------|
| **URL** | `http://binusian.ws/binusschool/bss-student-enrollment` |
| **Method** | `POST` |
| **Auth** | `Bearer {token}` |
| **Content-Type** | `application/json` |

**Request:**
```http
POST /binusschool/bss-student-enrollment HTTP/1.1
Host: binusian.ws
Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
Content-Type: application/json

{
  "IdStudent": "2370007317"
}
```

| Body Field | Type | Required | Description |
|------------|------|----------|-------------|
| `IdStudent` | string | Yes | BINUS Student ID |

**Response (Success):**
```json
{
  "resultCode": 200,
  "errorMessage": null,
  "studentDataResponse": {
    "studentName": "Benjamin Arandra Siregar",
    "gradeCode": "4",
    "gradeName": "Grade 4",
    "class": "4C"
  }
}
```

| Response Field | Type | Description |
|----------------|------|-------------|
| `studentDataResponse.studentName` | string | Full student name |
| `studentDataResponse.gradeCode` | string | Grade code (`"4"`, `"EY1"`, etc.) |
| `studentDataResponse.gradeName` | string | Grade display name |
| `studentDataResponse.class` | string | Homeroom / class code (`"4C"`) |

**Response (Not Found):**
```json
{
  "resultCode": 404,
  "errorMessage": "Student not found"
}
```

> **Note:** This endpoint does **NOT** return `idBinusian`. Use C.1 for that.

---

#### B.2 — Insert Attendance (Face Recognition)

Record a student's attendance in the BINUS e-Desk system.

| Field | Value |
|-------|-------|
| **URL** | `http://binusian.ws/binusschool/bss-add-simprug-attendance-fr` |
| **Method** | `POST` |
| **Auth** | `Bearer {token}` |
| **Content-Type** | `application/json` |

**Request:**
```http
POST /binusschool/bss-add-simprug-attendance-fr HTTP/1.1
Host: binusian.ws
Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
Content-Type: application/json

{
  "IdStudent": "2370007317",
  "IdBinusian": "BN125377010",
  "ImageDesc": "-",
  "UserAction": "TEACHER7"
}
```

| Body Field | Type | Required | Description |
|------------|------|----------|-------------|
| `IdStudent` | string | Yes | BINUS Student ID |
| `IdBinusian` | string | Yes | BINUS internal ID (from C.1 API) |
| `ImageDesc` | string | No | Image description, use `"-"` if no image |
| `UserAction` | string | Yes | Teacher/system identifier (e.g. `"TEACHER7"`) |

**Response (Success):**
```json
{
  "attendanceFaceRecognitionResponse": {
    "success": true,
    "msg": "OK"
  },
  "resultCode": 200,
  "errorMessage": null
}
```

**Alternative Response Format (Production):**
```json
{
  "isSuccess": true,
  "statusCode": 200,
  "message": "OK"
}
```

| Response Field | Type | Description |
|----------------|------|-------------|
| `resultCode` / `statusCode` | int | `200` = success |
| `attendanceFaceRecognitionResponse.success` / `isSuccess` | bool | `true` if recorded |

> **Important:** Both `IdStudent` and `IdBinusian` are required. The system fetches `IdBinusian` during enrollment via the C.1 API and stores it in the metadata mapping.

---

### 4.2 Hikvision ISAPI (Device)

**Base URL:** `http://10.26.30.200`
**Authentication:** HTTP Digest Auth (`admin` / `password.123`)
**Protocol:** ISAPI (Hikvision proprietary REST-like API)

All requests require HTTP Digest Authentication. The Python code uses `requests.Session()` with `HTTPDigestAuth` and auto-retries on 401 (stale nonce).

---

#### Device Info

| Field | Value |
|-------|-------|
| **URL** | `/ISAPI/System/deviceInfo` |
| **Method** | `GET` |
| **Response** | XML |

**Response:**
```xml
<?xml version="1.0" encoding="UTF-8"?>
<DeviceInfo xmlns="http://www.isapi.org/ver20/XMLSchema" version="2.0">
  <deviceName>Front Entrance</deviceName>
  <model>DS-K1T341AMF</model>
  <serialNumber>DS-K1T341AMF20211234</serialNumber>
  <macAddress>c0:56:e3:xx:xx:xx</macAddress>
  <firmwareVersion>V1.2.0</firmwareVersion>
  <firmwareReleasedDate>2024-01-15</firmwareReleasedDate>
</DeviceInfo>
```

---

#### Search Enrolled Users

| Field | Value |
|-------|-------|
| **URL** | `/ISAPI/AccessControl/UserInfo/Search?format=json` |
| **Method** | `POST` |
| **Content-Type** | `application/json` |

**Request:**
```json
{
  "UserInfoSearchCond": {
    "searchID": "enrollment_check",
    "searchResultPosition": 0,
    "maxResults": 30
  }
}
```

**Response:**
```json
{
  "UserInfoSearch": {
    "searchID": "enrollment_check",
    "responseStatusStrg": "OK",
    "numOfMatches": 6,
    "totalMatches": 6,
    "UserInfo": [
      {
        "employeeNo": "2370007317",
        "name": "Benjamin Arandra Siregar",
        "userType": "normal",
        "gender": "unknown",
        "numOfFace": 1,
        "Valid": {
          "enable": true,
          "beginTime": "2024-01-01T00:00:00",
          "endTime": "2037-12-31T23:59:59"
        }
      }
    ]
  }
}
```

---

#### Create User

| Field | Value |
|-------|-------|
| **URL** | `/ISAPI/AccessControl/UserInfo/Record?format=json` |
| **Method** | `POST` |
| **Content-Type** | `application/json` |

**Request:**
```json
{
  "UserInfo": {
    "employeeNo": "2370007317",
    "name": "Benjamin Arandra Siregar",
    "userType": "normal",
    "gender": "unknown",
    "Valid": {
      "enable": true,
      "beginTime": "2024-01-01T00:00:00",
      "endTime": "2037-12-31T23:59:59",
      "timeType": "local"
    },
    "doorRight": "1",
    "RightPlan": [
      { "doorNo": 1, "planTemplateNo": "1" }
    ]
  }
}
```

**Response (Success):**
```json
{
  "statusCode": 1,
  "statusString": "OK",
  "subStatusCode": "ok"
}
```

---

#### Upload Face to FDLib

Uploads a face image to the device's Face Database Library. The device downloads the image from the provided URL and extracts its own DeepLearn embeddings.

| Field | Value |
|-------|-------|
| **URL** | `/ISAPI/Intelligent/FDLib/FDSetUp?format=json` |
| **Method** | `PUT` |
| **Content-Type** | `application/json` |

**Request:**
```json
{
  "faceLibType": "blackFD",
  "FDID": "1",
  "FPID": "2370007317",
  "name": "Benjamin Arandra Siregar",
  "faceURL": "http://192.168.1.100:8888/2370007317_face.jpg"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `faceLibType` | string | Always `"blackFD"` for this device |
| `FDID` | string | Face Database ID, always `"1"` |
| `FPID` | string | Face Picture ID = `employeeNo` = BINUS Student ID |
| `name` | string | Display name shown on device LCD |
| `faceURL` | string | HTTP URL where device can download the face image |

> **Note:** The device must be able to reach `faceURL` over the network. A temporary HTTP server is started on the Python host (port 8888) to serve the cropped face images.

**Response (Success):**
```json
{
  "statusCode": 1,
  "statusString": "OK",
  "subStatusCode": "ok"
}
```

**Response (Failure — bad image):**
```json
{
  "statusCode": 3,
  "statusString": "Device Error",
  "subStatusCode": "facePicQualityFail"
}
```

---

#### Delete User

| Field | Value |
|-------|-------|
| **URL** | `/ISAPI/AccessControl/UserInfo/Delete?format=json` |
| **Method** | `PUT` |
| **Content-Type** | `application/json` |

**Request:**
```json
{
  "UserInfoDelCond": {
    "EmployeeNoList": [
      { "employeeNo": "2370007317" }
    ]
  }
}
```

---

#### Delete Face from FDLib

| Field | Value |
|-------|-------|
| **URL** | `/ISAPI/Intelligent/FDLib/FDDelete?format=json` |
| **Method** | `PUT` |
| **Content-Type** | `application/json` |

**Request:**
```json
{
  "faceLibType": "blackFD",
  "FDID": "1",
  "FPID": "2370007317"
}
```

---

#### Capture Face from Device Camera

Blocks until the device detects a face. Returns a multipart response with XML metadata + JPEG image.

| Field | Value |
|-------|-------|
| **URL** | `/ISAPI/AccessControl/CaptureFaceData` |
| **Method** | `POST` |
| **Content-Type** | `application/xml` |
| **Timeout** | 60 seconds (blocks until face detected) |

**Request:**
```xml
<?xml version="1.0" encoding="UTF-8"?>
<CaptureFaceDataCond xmlns="http://www.isapi.org/ver20/XMLSchema" version="2.0">
  <captureInfrared>false</captureInfrared>
  <dataType>binary</dataType>
</CaptureFaceDataCond>
```

**Response:** Multipart MIME with boundary, containing XML metadata and binary JPEG data (352×432 face image).

---

#### Event Alert Stream (Live Events)

Persistent HTTP connection that streams real-time events from the device as multipart MIME.

| Field | Value |
|-------|-------|
| **URL** | `/ISAPI/Event/notification/alertStream` |
| **Method** | `GET` |
| **Response** | `multipart/mixed; boundary=MIME_boundary` (streaming) |
| **Timeout** | Connect: 10s, Read: infinite |

**Response Stream:**
```
--MIME_boundary
Content-Type: application/json; charset="UTF-8"
Content-Length: 892

{
  "ipAddress": "10.26.30.200",
  "dateTime": "2026-02-18T08:05:30+07:00",
  "eventType": "AccessControllerEvent",
  "AccessControllerEvent": {
    "majorEventType": 5,
    "subEventType": 76,
    "employeeNoString": "2370007317",
    "name": "Benjamin Arandra Siregar",
    "serialNo": 42
  }
}
--MIME_boundary
Content-Type: image/jpeg
Content-Length: 45000

<binary JPEG>
--MIME_boundary
```

**Event Types:**

| majorEventType | subEventType | Description |
|----------------|-------------|-------------|
| 5 | 75 | Face comparison success |
| 5 | 76 | Face verification success |
| 5 | 104 | Face recognition (alternate) |

---

#### FDLib Face Record Count

| Field | Value |
|-------|-------|
| **URL** | `/ISAPI/Intelligent/FDLib/Count?format=json` |
| **Method** | `GET` |

**Response:**
```json
{
  "FDRecordDataInfo": [
    {
      "FDID": "1",
      "faceLibType": "blackFD",
      "recordDataNumber": 6
    }
  ]
}
```

---

### 4.3 Firebase Services

**Project:** `facial-attendance-binus`
**Credentials:** `facial-attendance-binus-firebase-adminsdk.json`

#### Firestore Collections

| Collection | Document ID | Purpose |
|------------|-------------|---------|
| `students/{studentId}` | BINUS Student ID | Student metadata from photo capture |
| `students/{studentId}/images/{auto}` | Auto-generated | Individual photo metadata |
| `attendance/{date}` | `YYYY-MM-DD` | Day summary with `lastUpdated` |
| `attendance/{date}/records/{employeeNo}` | BINUS Student ID | Individual attendance record |
| `student_metadata/{employeeNo}` | BINUS Student ID | Device enrollment metadata backup |

#### Firebase Storage Paths

| Path | Content |
|------|---------|
| `face_dataset/{className}/{studentName}/photo_{n}_{timestamp}.jpg` | Face photos captured via web app |

---

## 5. Internal APIs (Next.js Dashboard)

**Base URL:** `http://localhost:3000`

These are the Next.js API routes used by the web dashboard.

---

#### `POST /api/student/lookup`

Look up a student by BINUS Student ID (proxies to BINUS C.2 API).

**Request:**
```json
{ "studentId": "2370007317" }
```

**Response:**
```json
{
  "success": true,
  "id": "2370007317",
  "name": "Benjamin Arandra Siregar",
  "homeroom": "4C",
  "gradeCode": "4",
  "gradeName": "Grade 4",
  "raw": { "studentName": "...", "gradeCode": "...", "class": "..." }
}
```

---

#### `POST /api/face/upload`

Upload a face photo for a student. Accepts `multipart/form-data`.

**Request (multipart/form-data):**

| Field | Type | Description |
|-------|------|-------------|
| `studentId` | string | BINUS Student ID |
| `studentName` | string | Full name |
| `className` | string | Homeroom (e.g. `"4C"`) |
| `gradeCode` | string | Grade code |
| `gradeName` | string | Grade name |
| `photoNumber` | string | Current photo number (`"1"`, `"2"`, `"3"`) |
| `totalPhotos` | string | Total photos to capture |
| `image` | file | JPEG image file |

**Response:**
```json
{
  "success": true,
  "message": "Photo 1/3 uploaded for Benjamin Arandra Siregar 4C",
  "uploadMethod": "Firebase Storage",
  "data": {
    "studentId": "2370007317",
    "studentName": "Benjamin Arandra Siregar",
    "className": "4C",
    "photoNumber": "1",
    "totalPhotos": "3",
    "size": 125000,
    "storageUrl": "gs://facial-attendance-binus.firebasestorage.app/face_dataset/4C/Benjamin Arandra Siregar/photo_1_1739884200000.jpg"
  }
}
```

---

#### `GET /api/attendance/today`

Get today's attendance records for the dashboard.

**Query Parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `date` | string | Today (WIB) | Date in `YYYY-MM-DD` format |

**Response:**
```json
{
  "date": "2026-02-18",
  "records": [
    {
      "id": "2370007317",
      "name": "Benjamin Arandra Siregar",
      "employeeNo": "2370007317",
      "timestamp": "2026-02-18 08:05:30",
      "status": "Present",
      "late": false,
      "updatedAt": "2026-02-18T08:05:30+07:00"
    }
  ],
  "summary": {
    "total": 5,
    "present": 4,
    "late": 1,
    "lastUpdated": "2026-02-18T08:20:15+07:00"
  }
}
```

---

#### `GET /api/attendance/enrolled`

Get currently enrolled users from the Hikvision device.

**Response:**
```json
{
  "users": [
    {
      "employeeNo": "2370007317",
      "name": "Benjamin Arandra Siregar",
      "numOfFace": 1
    }
  ],
  "total": 6
}
```

---

#### `GET /api/health`

Health check endpoint.

**Response:**
```json
{ "status": "ok", "timestamp": "2026-02-18T08:00:00.000Z" }
```

---

## 6. Data Schemas

### Student Metadata (`data/student_metadata.json`)

The central mapping file that bridges device identity → BINUS identity.

```json
{
  "<employeeNo (= IdStudent)>": {
    "employeeNo": "2370007317",
    "name": "Benjamin Arandra Siregar",
    "idStudent": "2370007317",
    "idBinusian": "BN125377010",
    "homeroom": "4C",
    "grade": "4",
    "enrolledAt": "2026-02-18T10:00:00+07:00",
    "updatedAt": "2026-02-18T10:00:00+07:00"
  }
}
```

### Daily Attendance (`data/attendance/YYYY-MM-DD.json`)

```json
{
  "<Student Name>": {
    "employeeNo": "2370007317",
    "timestamp": "2026-02-18 08:05:30",
    "status": "Present",
    "late": false
  }
}
```

### Face Encodings (`encodings.pickle`)

Binary pickle file containing pre-computed face encodings for the OpenCV-based recognition mode (`main.py`). Structure:

```python
{
    "names": ["Benjamin Arandra Siregar", "Anderson Wijaya", ...],
    "encodings": [<128-dim numpy array>, ...],
    "classes": ["4C", "4C", ...],  # optional
}
```

---

## 7. File Reference

### Python Backend

| File | Purpose |
|------|---------|
| `hikvision_attendance.py` | **Main CLI** — Enrollment, device management, monitoring. Commands: `enroll`, `enroll-live`, `enroll-class`, `monitor`, `status`, `clear` |
| `attendance_listener.py` | **Live event listener** — Connects to device alertStream, records attendance to local JSON + Firebase + BINUS API |
| `api_integrate.py` | **BINUS API wrapper** — Token auth (C.0), student photos (C.1), student lookup (C.2), attendance insert (B.2) |
| `student_metadata.py` | **Metadata mapping** — Persistent `employeeNo → {IdStudent, IdBinusian}` mapping, local JSON + Firebase sync |
| `main.py` | **OpenCV recognition** — Alternative local camera face recognition with dlib/CNN, anti-spoofing, attendance logging |
| `enroll.py` | **Local enrollment** — Create `encodings.pickle` from local face images for OpenCV mode |
| `firebase_dataset_sync.py` | **Firebase Storage sync** — Download face dataset from Firebase Storage to local directory |
| `collect_metrics.py` | **Metrics collector** — Fetch student photo metrics from BINUS API, generate reports/dashboards |
| `test_attendance_insert.py` | **Test script** — Simulate attendance insertion to verify the full pipeline |

### Next.js Web App (`web-dataset-collector/`)

| File | Purpose |
|------|---------|
| `pages/index.js` | Home page — student ID entry |
| `pages/dashboard.js` | Live attendance dashboard (auto-refreshes every 10s) |
| `pages/hikvision.js` | Hikvision device management page |
| `pages/api/student/lookup.js` | API: Student lookup (C.2 proxy) |
| `pages/api/face/upload.js` | API: Photo upload to Firebase |
| `pages/api/attendance/today.js` | API: Today's attendance records |
| `pages/api/attendance/enrolled.js` | API: Enrolled users from device |
| `pages/api/attendance/metrics.js` | API: Attendance metrics/stats |
| `pages/api/health.js` | API: Health check |
| `components/CapturePage.js` | React component: Photo capture UI |
| `components/EnrollmentPage.js` | React component: Enrollment form |
| `lib/firebase-admin.js` | Firebase Admin SDK initialization |
| `lib/hikvision.js` | Hikvision ISAPI client (JS) |

### Data Files

| File/Directory | Purpose |
|----------------|---------|
| `data/student_metadata.json` | Student mapping (employeeNo → BINUS IDs) |
| `data/attendance/YYYY-MM-DD.json` | Daily attendance records |
| `.env` | Environment variables (API keys, device credentials) |
| `facial-attendance-binus-firebase-adminsdk.json` | Firebase service account credentials |
| `encodings.pickle` | Pre-computed face encodings for OpenCV mode |
| `shape_predictor_68_face_landmarks.dat` | dlib 68-point facial landmark model |
| `dlib_face_recognition_resnet_model_v1.dat` | dlib face recognition ResNet model |

---

## 8. Environment Variables

| Variable | Example | Used By | Description |
|----------|---------|---------|-------------|
| `HIKVISION_IP` | `10.26.30.200` | Python backend | Device IP address |
| `HIKVISION_USER` | `admin` | Python backend | Device admin username |
| `HIKVISION_PASS` | `password.123` | Python backend | Device admin password |
| `HIKVISION_RTSP_URL` | `rtsp://admin:password.123@10.26.30.200:554/Streaming/Channels/101` | main.py | RTSP stream URL for OpenCV mode |
| `API_KEY` | `OUQyQjdEN0EtREFDQy00QkEyLTg3QTAtNUFGNDVDOUZCRTgy` | Both | BINUS School API key (Base64 encoded) |
| `USER_ACTION` | `TEACHER7` | Python backend | Teacher identifier for B.2 API |
| `FIREBASE_STORAGE_BUCKET` | `facial-attendance-binus.firebasestorage.app` | Both | Firebase Storage bucket name |
| `FIREBASE_PROJECT_ID` | `facial-attendance-binus` | Both | Firebase project ID |
| `FIREBASE_CREDENTIALS` | `facial-attendance-binus-firebase-adminsdk.json` | Python backend | Path to Firebase credentials file |
| `FACE_SERVER_PORT` | `8888` | Python backend | Port for temp HTTP server during enrollment |
| `DUPLICATE_DETECTION_WINDOW` | `28800` | Both | Seconds before a student can log attendance again |

---

## 9. Quick Start Commands

### Start the Web Dashboard
```bash
cd web-dataset-collector
npm install
npx next dev -p 3000
```

### Enroll Students (Firebase → Device)
```bash
# Bulk enroll all students from Firebase photos
python hikvision_attendance.py enroll

# Enroll one student via device camera
python hikvision_attendance.py enroll-live

# Enroll entire class
python hikvision_attendance.py enroll-class 4 4C
```

### Start Attendance Listener
```bash
# With Firebase + BINUS API
python attendance_listener.py

# Local-only (no Firebase)
python attendance_listener.py --no-firebase
```

### Device Management
```bash
# Check enrolled users and device status
python hikvision_attendance.py status

# Clear all enrolled users
python hikvision_attendance.py clear
```

### View Student Metadata
```bash
python student_metadata.py
```

### OpenCV Local Mode (Alternative)
```bash
# Enroll faces from local images
python enroll.py

# Run local face recognition
python main.py
```

---

## API Authentication Flow Summary

```
┌──────────┐    GET /auth/token         ┌───────────────┐
│  Client   │──────────────────────────▶│  BINUS API    │
│           │  Authorization: Basic KEY  │               │
│           │◀──────────────────────────│               │
│           │  { token: "eyJ..." }       │               │
│           │                            │               │
│           │    POST (any endpoint)      │               │
│           │──────────────────────────▶│               │
│           │  Authorization: Bearer TOK │               │
│           │◀──────────────────────────│               │
│           │  { resultCode: 200, ... }  │               │
└──────────┘                            └───────────────┘
```

**Token Lifecycle:**
1. Call C.0 with `Basic {API_KEY}` → receive JWT token (valid ~60 min)
2. Use token as `Bearer {token}` in all subsequent calls
3. Token auto-refreshed by `api_integrate.py` when needed

---

*Generated from codebase analysis. For questions, check the source files referenced in each section.*
