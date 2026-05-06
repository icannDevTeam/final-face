# Privacy Policy — Facial Attendance System

**Tenant:** BINUS School Simprug
**Version:** v1
**Effective Date:** 2026-05-01
**Last Updated:** 2026-04-29

---

## 1. Who We Are

This facial attendance service ("the Service") is operated on behalf of **BINUS School Simprug** ("the School"). The School is the **data controller** of student personal data; the Service operator acts as a **data processor** under a Data Processing Agreement with the School.

For privacy questions, contact: **dpo@binus-school.example** (placeholder — School to confirm).

---

## 2. What Data We Collect

### 2.1 Biometric Data
- **Facial photographs** captured during enrollment (typically 3–10 images per student).
- **Mathematical face descriptors** (numerical vectors derived from photos) used for recognition. These cannot be reversed back into a photograph.
- **Live facial scans** at attendance terminals and via the mobile app — these are matched against the descriptor and discarded; only the match result is retained.

### 2.2 Attendance Data
- Date, time, and location (campus terminal or geofenced mobile check-in) of each attendance event.
- "On time" / "Late" status, computed against the School's published cutoff time.
- Source: which device or mobile session recorded the event.

### 2.3 Account & Operational Data
- Student name, student ID number, homeroom, grade.
- Guardian name, email address, and relationship (collected during consent).
- Login records of staff/administrators who access the dashboard.
- Anti-spoofing telemetry (device user agent, anonymized indicators) when a fraudulent attempt is detected.

### 2.4 What We Do **Not** Collect
- We do not record audio.
- We do not perform emotion analysis, ethnicity inference, or any classification beyond identity matching.
- We do not share biometric data with advertising networks or any third party outside the sub-processors listed in §8.

---

## 3. Lawful Basis for Processing

- **Consent of the parent/legal guardian** is the primary lawful basis for processing minors' biometric data, in line with Indonesian Law No. 27 of 2022 on Personal Data Protection (UU PDP) and, where applicable, GDPR Article 9(2)(a).
- **Legitimate interest of the School** for keeping accurate attendance records as required by national education regulations.

Consent is freely given, specific, informed, and unambiguous. It can be **withdrawn at any time** (see §6).

---

## 4. How We Use the Data

| Purpose | Data Used |
|---|---|
| Identify students at the gate / on mobile | Face descriptors, live scan |
| Record attendance | Identity + timestamp + location |
| Notify parents of late arrivals | Attendance record + guardian email |
| Detect fraudulent attendance attempts | Spoof telemetry, IP, user agent |
| Internal audit and security | Access logs, biometric access logs |

We do **not** use this data for:
- Profiling, scoring, or any automated decision with legal effects.
- Marketing.
- Surveillance outside attendance windows.

---

## 5. Retention

| Category | Retention Period |
|---|---|
| Facial photographs | 1 year after the student departs the School |
| Face descriptors | Same as photographs |
| Attendance records | 7 years (typical school records requirement) |
| Anti-spoofing logs | 90 days |
| Dashboard access logs | 1 year |
| Consent records | Lifetime of student + 7 years (legal proof of lawful processing) |

Retention windows are configurable per tenant within these legal ceilings. Automated purge jobs run daily (see Workstream B6 in the implementation roadmap).

---

## 6. Your Rights

You (or your legal guardian, if you are a minor) have the right to:

1. **Access** — request a copy of all personal data we hold about you.
2. **Rectify** — correct inaccurate or outdated information.
3. **Erase** — request deletion of biometric data ("right to be forgotten"). Attendance records may be retained in pseudonymized form for legal/audit purposes.
4. **Withdraw consent** — at any time, with no penalty. Withdrawal triggers deletion of biometric data within 30 days.
5. **Object to processing** — and request alternative attendance methods (manual sign-in).
6. **Lodge a complaint** with the relevant supervisory authority (in Indonesia: Ministry of Communications and Informatics; in the EU: your national Data Protection Authority).

All requests can be submitted via the consent portal or by emailing the School's privacy contact. We respond within 30 days (target: 7 days).

---

## 7. Security

- All data is transmitted over TLS 1.2+.
- Photos are encrypted at rest using Google-managed encryption keys (Google Cloud Storage).
- Face descriptors are stored in Google Cloud Firestore with project-level access controls.
- Access to biometric data is logged (who viewed which student's photo, when, and why).
- Hikvision device credentials are stored in encrypted environment variables, not in the codebase.
- Annual penetration testing and quarterly access reviews are performed.

---

## 8. Sub-processors

The Service relies on the following third parties to deliver core functionality:

| Sub-processor | Purpose | Region |
|---|---|---|
| Google Cloud (Firebase, Cloud Storage, Cloud Firestore) | Database, file storage, authentication | Asia (asia-southeast2 / us-central1 default) |
| Vercel | Hosting of the dashboard and mobile web app | Global edge |
| Hikvision OEM | On-premise facial-recognition terminal hardware/firmware | On-premise (data does not leave the School) |

We will notify the School of any new sub-processor at least 30 days before engagement.

---

## 9. International Transfers

Where data is stored or processed outside Indonesia (e.g., Google Cloud's `us-central1` region used by some Firebase services), transfers are protected by Standard Contractual Clauses and Google's data-processing terms.

---

## 10. Changes to This Policy

We will notify the guardian of every consenting student via email when this policy materially changes. Continued use of the Service after a material change requires re-consent. The full version history of this policy is preserved and any consent record is permanently linked to the version it was given against.

---

## 11. Contact

| Role | Contact |
|---|---|
| School Data Protection Contact | privacy@binus-school.example *(placeholder)* |
| Service Operator | ops@final-face.example *(placeholder)* |
| Postal Address | *to be confirmed by the School* |

---

*This document, together with its SHA-256 hash and version identifier, is stored immutably in the Service's policy registry. The hash printed when seeding the policy is the canonical proof of what was published.*
