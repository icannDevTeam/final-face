# Privacy Policy — Facial Attendance & Pickup Authorization System

**Tenant:** BINUS School Simprug
**Version:** v2
**Effective Date:** 2026-05-15
**Last Updated:** 2026-04-29
**Supersedes:** v1 (2026-05-01)

---

## What's New in v2

Version 2 adds the **Pickup System** module: a face-recognition system used at the
school gate(s) at dismissal time to verify that the adult collecting a child is
authorized by the child's parent/guardian.

This means a new category of person — the **chaperone** (parent, family driver,
nanny, grandparent, emergency contact, etc.) — now has biometric data processed
by the Service. Sections 2.1, 4, 5, 6, and 8 below have been updated. Sections
unchanged from v1 are reproduced for completeness so that consent is given
against the full policy text.

---

## 1. Who We Are

This service ("the Service") is operated on behalf of **BINUS School Simprug**
("the School"). The School is the **data controller** of all personal data
processed by the Service; the Service operator acts as a **data processor**
under a written Data Processing Agreement.

For privacy questions, contact: **dpo@binus-school.example** *(placeholder — School to confirm)*.

---

## 2. What Data We Collect

### 2.1 Biometric Data

**For students** (unchanged from v1):
- Facial photographs captured during enrollment (typically 3–10 images).
- Mathematical face descriptors derived from those photos.
- Live facial scans at attendance terminals and via the mobile app — matched
  against the descriptor and discarded; only the match result is retained.

**For chaperones** (NEW in v2):
- 1–3 facial photographs captured during the parent-driven onboarding flow.
- Mathematical face descriptors derived from those photos.
- Live face captures from the gate terminal at each pickup event, stored for
  up to 30 days as audit evidence (see §5).

### 2.2 Attendance & Pickup Data
- Date, time, and gate/terminal of each attendance event (students).
- Date, time, gate, and matched chaperone of each pickup event.
- The list of which student(s) each chaperone attempted to collect.
- Officer override decisions (manual approval at the gate) and the reason.

### 2.3 Account & Operational Data
- Student name, student ID, homeroom, grade.
- Guardian name, email, phone, relationship.
- **Chaperone name, relation to student, phone, optional email and ID/KTP number.**
- The mapping of which chaperones are authorized to pick up which students.
- Login records of staff who access the dashboard.
- Anti-spoofing telemetry when fraudulent attempts are detected.

### 2.4 What We Do **Not** Collect
- We do not record audio.
- We do not perform emotion analysis, ethnicity inference, or any classification
  beyond identity matching.
- We do not share biometric data with advertising networks or any third party
  outside the sub-processors listed in §8.
- We do not use chaperone face data outside the pickup gate context (no
  cross-matching against student attendance, no general surveillance).

---

## 3. Lawful Basis for Processing

- **Student biometric data:** consent of the parent/legal guardian.
- **Chaperone biometric data:** consent of the chaperone themselves, given on
  their behalf by the parent during onboarding (the parent confirms they have
  spoken to and obtained agreement from each person they nominate). Chaperones
  may withdraw their own consent at any time directly with the School office.
- **Legitimate interest of the School** for keeping accurate attendance records
  and for child-safety verification at dismissal.

In line with Indonesian Law No. 27 of 2022 on Personal Data Protection (UU PDP)
and, where applicable, GDPR Article 9(2)(a). Consent is freely given, specific,
informed, and unambiguous, and can be withdrawn at any time (see §6).

---

## 4. How We Use the Data

| Purpose | Data Used |
|---|---|
| Identify students at the attendance gate / on mobile | Student descriptors + live scan |
| Record attendance | Identity + timestamp + location |
| Notify parents of late arrivals | Attendance record + guardian email |
| **Identify the adult collecting a child at dismissal** | **Chaperone descriptors + live scan + authorization map** |
| **Display recent pickup events on the gate TV (rolling 5-card queue, ~60 s hold)** | **Chaperone name + photo, student name(s) + photo** |
| **Officer manual override when face match is uncertain or out-of-window** | **Chaperone record + officer reason note** |
| Detect fraudulent attendance attempts | Spoof telemetry, IP, user agent |
| Internal audit and security | Access logs, biometric access logs |

We do **not** use this data for:
- Profiling, scoring, or any automated decision with legal effects.
- Marketing.
- Surveillance outside attendance/pickup windows.

---

## 5. Retention

| Category | Retention Period |
|---|---|
| Student facial photographs | 1 year after the student departs the School |
| Student face descriptors | Same as photographs |
| Attendance records | 7 years (typical school records requirement) |
| **Chaperone facial photographs** | **Until authorization is revoked + 30 days, OR 12 months after last pickup, whichever is sooner** |
| **Chaperone face descriptors** | **Same as chaperone photographs** |
| **Pickup event captures (face JPEG at the gate)** | **30 days** |
| **Pickup event records (no JPEG, metadata only)** | **1 year** |
| Anti-spoofing logs | 90 days |
| Dashboard access logs | 1 year |
| Consent records | Lifetime of student + 7 years (legal proof of lawful processing) |

**Chaperone re-enrollment cycle:** every 12 months the system requires the
parent to reconfirm the authorization list. 1 month before expiry, an email
reminder is sent. If the parent does not reconfirm, the chaperone is
automatically suspended at the gate (manual officer override remains available).

Retention windows are configurable per tenant within these legal ceilings.
Automated purge jobs run daily.

---

## 6. Your Rights

You (or your legal guardian, if you are a minor) have the right to:

1. **Access** — request a copy of all personal data we hold about you.
2. **Rectify** — correct inaccurate or outdated information.
3. **Erase** — request deletion of biometric data ("right to be forgotten").
   Attendance and pickup records may be retained in pseudonymized form for
   legal/audit purposes.
4. **Withdraw consent** — at any time, with no penalty. Withdrawal triggers
   deletion of biometric data within 30 days.
5. **Object to processing** — and request alternative attendance / pickup
   verification methods (manual sign-in at the gate office).
6. **Lodge a complaint** with the relevant supervisory authority (in Indonesia:
   Ministry of Communications and Informatics; in the EU: your national Data
   Protection Authority).

**For chaperones specifically:** any chaperone listed in the authorization
system may directly request removal at any time, without parental approval, by
contacting the School office. Removal takes effect within 24 hours.

---

## 7. Security

- All data is transmitted over TLS 1.2+.
- Photos are encrypted at rest using Google-managed encryption keys.
- Face descriptors are stored in Google Cloud Firestore with project-level
  access controls.
- Access to biometric data is logged (who viewed which person's photo, when,
  and why).
- Hikvision device credentials are stored in encrypted environment variables.
- The pickup TV display kiosk authenticates with a short-lived signed token
  scoped to a single gate; it cannot read or write any data outside the rolling
  pickup queue.
- Annual penetration testing and quarterly access reviews are performed.

---

## 8. Sub-processors

| Sub-processor | Purpose | Region |
|---|---|---|
| Google Cloud (Firebase, Cloud Storage, Cloud Firestore) | Database, file storage, authentication | Asia (asia-southeast2) and us-central1 |
| Vercel | Hosting of the dashboard, mobile web app, parent onboarding pages | Global edge |
| Hikvision OEM | On-premise facial-recognition terminal hardware/firmware | On-premise (data does not leave the School) |
| **Local Jetson edge appliance** | **Real-time pickup event fan-out to the gate TV display** | **On-premise (no cloud egress for the SSE stream)** |

We will notify the School of any new sub-processor at least 30 days before
engagement.

---

## 9. International Transfers

Where data is stored or processed outside Indonesia (e.g., Google Cloud's
`us-central1` region used by some Firebase services), transfers are protected by
Standard Contractual Clauses and Google's data-processing terms. The pickup
event stream itself stays on-premise on the local Jetson appliance.

---

## 10. Changes to This Policy

We will notify the guardian of every consenting student via email when this
policy materially changes. Continued use of the Service after a material change
requires re-consent. The full version history of this policy is preserved and
any consent record is permanently linked to the version it was given against.

---

## 11. Contact

| Role | Contact |
|---|---|
| School Data Protection Contact | privacy@binus-school.example *(placeholder)* |
| Service Operator | ops@final-face.example *(placeholder)* |
| Postal Address | *to be confirmed by the School* |

---

*This document, together with its SHA-256 hash and version identifier, is stored
immutably in the Service's policy registry. The hash printed when seeding the
policy is the canonical proof of what was published.*
