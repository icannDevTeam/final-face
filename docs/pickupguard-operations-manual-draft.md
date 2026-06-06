# PickupGuard Operations Manual (Draft)

Version: v1.0-draft  
Date: 2026-06-05  
Audience: Academic Operations (ACOP), IT, Teachers, Admin  
Status: Content draft for screenshot and visual design handoff

---

## 1. Purpose and Scope

This manual explains how to run the Pick-Upp system safely and consistently across daily operations.

This document covers:
- Parent onboarding and approval workflow
- Chaperone lifecycle management
- Gate operations and release process
- Security incident handling and officer override flow
- Role responsibilities for ACOP, IT, Teachers, and Admin
- Compliance and data handling boundaries
- Troubleshooting and escalation paths

This document does not cover:
- UI redesign or branding decisions
- Engineering implementation details not visible to operations users
- Legal interpretation beyond approved policy language

Source references used in this manual:
- pickupguard/README.md
- docs/privacy/privacy-policy-v2.md
- web-dataset-collector/pages/v2/pickup-admin.js
- web-dataset-collector/pages/v2/chaperones.js
- web-dataset-collector/pages/v2/officer-overrides.js
- web-dataset-collector/pages/v2/security.js
- web-dataset-collector/pages/v2/terminals.js
- web-dataset-collector/pages/v2/release-groups.js

---

## 2. System Overview

PickupGuard verifies that the adult collecting a student is authorized by the parent/guardian.

Primary operating modules:
- Parent onboarding: /pickup/onboarding/[token]
- Admin queue: /v2/pickup-admin
- Chaperone management: /v2/chaperones
- Officer override desk: /v2/officer-overrides
- Security monitoring: /v2/security
- Terminal management: /v2/terminals
- Release group pairing: /v2/release-groups

Core flow summary:
1. Parent receives onboarding link and submits chaperone details.
2. Admin reviews and approves/rejects submission.
3. Approved chaperones are enrolled and available for gate scan matching.
4. Pickup event is recorded at gate.
5. If event is flagged, officer override process is used.
6. Incidents are monitored and closed through the security workflow.

---

## 3. Roles and Responsibility Matrix

### 3.1 Role definitions

- ACOP (Academic Operations): process owner for pickup safety and policy compliance during school operations.
- IT: technical owner for system availability, terminal readiness, pairing, and troubleshooting.
- Admin: operational owner for daily queue handling, approvals, re-enrollment, and records maintenance.
- Teachers: classroom release and student handoff support role, with escalation to ACOP/Admin for exceptions.

### 3.2 Responsibility matrix (RACI style)

| Activity | ACOP | IT | Admin | Teachers |
|---|---|---|---|---|
| Define and enforce pickup process rules | A | C | C | I |
| Maintain terminal and pairing readiness | I | A/R | C | I |
| Review parent onboarding submissions | C | I | A/R | I |
| Approve/reject chaperone requests | A | I | R | I |
| Run daily gate override desk | A | C | R | I |
| Handle unknown/suspended/overdue incidents | A | C | R | I |
| Support class release coordination | C | I | C | A/R |
| Investigate recurring operational failures | A | R | R | C |
| Confirm retention/compliance execution | A | C | R | I |

Legend: A = Accountable, R = Responsible, C = Consulted, I = Informed

### 3.3 Segregation of duties

- Admin should not self-approve exceptional overrides without ACOP visibility.
- IT should not change policy outcomes (approval/rejection decisions).
- Teachers should not approve overrides; they escalate to ACOP/Admin.

---

## 4. Daily Operating Model

### 4.1 Daily checkpoints

Pre-dismissal (before pickup window):
- IT confirms all active terminals are online and synchronized.
- Admin verifies pending queue has no urgent unresolved submissions.
- ACOP confirms officer desk coverage and escalation contact.
- Teachers receive release readiness notice.

During pickup window:
- Gate scans are processed automatically.
- Flagged cases are routed to officer override process.
- Teachers release students according to confirmed pickup flow.

Post-window:
- Admin reviews incident list and unresolved cases.
- ACOP signs off on high-risk or repeated incidents.
- IT checks device health and logs unresolved technical issues.

### 4.2 Standard operating windows

Default operations are governed by configured pickup windows per gate profile. Changes must be approved by ACOP and implemented by Admin/IT through terminal and kiosk profile settings.

---

## 5. End-to-End Workflow (Operational)

### 5.1 Parent onboarding to approval

Objective:
Register and validate authorized chaperones for student pickup.

Preconditions:
- Parent has a valid onboarding token.
- Admin account has access to pickup admin queue.

Steps:
1. Parent opens onboarding link and submits chaperone data.
2. Submission appears in Pending tab on /v2/pickup-admin.
3. Admin opens record and reviews identity, relation, and photo quality.
4. Admin chooses Approve or Reject (with reason for reject).
5. Approved records proceed to enrollment state and become available for gate operations.

Expected result:
- Approved chaperones are active for pickup matching.
- Rejected submissions include reason and remain auditable.

Failure modes:
- Missing data
- Invalid or low-quality photos
- Duplicate or suspicious submission

Escalation:
- Admin -> ACOP for policy exceptions
- Admin -> IT for technical upload/enrollment issues

Screenshot slots:
- SS-ONB-01: Parent onboarding form initial page
- SS-ONB-02: Parent submission confirmation state
- SS-ADM-01: Admin Pending tab with queue counts
- SS-ADM-02: Record detail drawer with approve/reject actions

### 5.2 Gate scan to release

Objective:
Allow authorized pickup quickly while blocking unsafe releases.

Preconditions:
- Terminal is enabled and in expected gate state.
- Approved chaperones are already enrolled.

Steps:
1. Chaperone scans face at gate terminal.
2. System validates match and authorization.
3. If normal, event is recorded and release proceeds.
4. If flagged, event is held for officer override decision.

Expected result:
- Safe release decision per event with audit trail.

Screenshot slots:
- SS-GATE-01: Terminal card in open state from /v2/terminals
- SS-GATE-02: Pickup event visibility in monitoring context

### 5.3 Flagged case to officer override

Objective:
Resolve flagged events with controlled manual approval.

Preconditions:
- Event status is flagged (unknown, suspended, or re-enroll overdue).
- Officer/Admin has access to /v2/officer-overrides.

Steps:
1. Officer opens Awaiting officer panel.
2. Officer verifies person and reason context.
3. Officer enters 6-digit override code in quick pad.
4. System records approver, note, and timestamp.
5. Event transitions to approved override status.

Expected result:
- Exception is resolved with auditable manual decision.

Screenshot slots:
- SS-OVR-01: Awaiting officer panel with pending flagged items
- SS-OVR-02: Quick override pad entry process
- SS-OVR-03: Approved history table entry

### 5.4 Incident review to closure

Objective:
Detect trend risk and close incidents with accountability.

Preconditions:
- Security heatmap data is available in /v2/security.

Steps:
1. Review day/hour heatmap for concentration patterns.
2. Review by-kind and by-gate breakdowns.
3. Open recent incidents and identify unresolved risk.
4. Assign action owner (Admin/IT/ACOP) and due time.
5. Mark incident as resolved after verification.

Expected result:
- Incident risk is reduced and closure evidence is recorded.

Screenshot slots:
- SS-SEC-01: Security heatmap with date range control
- SS-SEC-02: By-kind breakdown card
- SS-SEC-03: Recent incidents list with resolved state

---

## 6. Role SOP Chapters

## 6A. ACOP SOP

### ACOP-1 Daily governance check

Objective:
Ensure pickup process follows school safety policy each day.

Steps:
1. Confirm duty roster for officer override coverage.
2. Confirm Admin queue is actively managed.
3. Confirm unresolved incidents from prior day are assigned.
4. Publish same-day policy notices if special release condition applies.

Evidence:
- Daily operations log entry
- Escalation summary if exceptions occurred

### ACOP-2 Exception approval and incident closure

Objective:
Provide final operational sign-off on high-risk exceptions.

Trigger conditions:
- Repeated unknown chaperone events
- Frequent suspended/overdue overrides
- Any event escalated by Teacher/Admin/IT

Steps:
1. Review incident context in security and override history.
2. Validate that manual release was justified.
3. Approve closure or request further investigation.
4. Confirm corrective action owner and deadline.

Evidence:
- Incident closure note
- Corrective action tracker update

### ACOP-3 Weekly quality review

Objective:
Reduce repeat issues and improve process discipline.

Steps:
1. Review weekly incident totals and peak windows.
2. Compare gate-level patterns and staffing levels.
3. Approve process correction actions.
4. Sign off weekly operational report.

Screenshot slots for ACOP chapter:
- SS-ACOP-01: Security dashboard with trend view
- SS-ACOP-02: Officer override history review view

## 6B. IT SOP

### IT-1 Terminal readiness and gate-state verification

Objective:
Keep all terminals operational and correctly configured.

Steps:
1. Open /v2/terminals and check device status cards.
2. Verify enabled state and effective open/closed reason.
3. Validate release group binding per terminal.
4. Verify window open and window close values.
5. Resolve unbound, disabled, or stale devices before pickup window.

Expected result:
- No critical terminal gaps before dismissal.

### IT-2 Release group pairing management

Objective:
Ensure each gate is paired correctly with target display/control device.

Steps:
1. Open /v2/release-groups.
2. Confirm groups are paired and not expired.
3. Regenerate pairing code when needed.
4. Rebind terminals if gate assignment changed.

Expected result:
- Pairing path is valid and current for all active gates.

### IT-3 Technical incident handling

Objective:
Restore service quickly and document root cause.

Common triggers:
- Terminal not updating
- Pairing fails
- Device last-seen stale
- Unexpected out-of-window behavior

Steps:
1. Confirm issue scope (single terminal vs multiple).
2. Validate settings and gate override status.
3. Re-sync affected configuration.
4. Escalate backend listener/API issues if unresolved.
5. Record incident and mitigation.

Escalation:
- IT -> Admin for operational fallback
- IT -> ACOP for risk communication when service degradation affects release safety

Screenshot slots for IT chapter:
- SS-IT-01: Terminal card showing state and controls
- SS-IT-02: Release groups table with pairing status
- SS-IT-03: Terminal edit drawer with window configuration

## 6C. Admin SOP

### ADM-1 Queue handling (Pending/Approved/Rejected/Archived)

Objective:
Process parent onboarding records quickly and accurately.

Steps:
1. Open /v2/pickup-admin and review Pending count first.
2. Use search/sort to prioritize urgent records.
3. Open each record and validate all required details.
4. Approve eligible records.
5. Reject incomplete or invalid records with clear reason.
6. Track status movement across tabs.

Target SLA:
- Pending reviews completed within same school day.

### ADM-2 Re-enrollment and re-push operations

Objective:
Keep chaperone enrollment current and synchronized across gates.

Steps:
1. Use /v2/chaperones filters for re-enroll due or never enrolled.
2. Select records and run bulk re-enroll campaign when needed.
3. Use per-record re-push if enrollment sync failed.
4. Verify updated status after action.

Expected result:
- No overdue active chaperones without action.

### ADM-3 Chaperone lifecycle controls

Objective:
Maintain accurate active roster and preserve audit trail.

Steps:
1. Search and open chaperone records in /v2/chaperones.
2. Use shadow-delete only with mandatory reason.
3. Restore only after verification and ACOP approval if required.
4. Avoid permanent data removal outside approved retention process.

### ADM-4 Daily incident and override coordination

Objective:
Support officer workflow and ensure exceptions are resolved.

Steps:
1. Monitor /v2/officer-overrides during pickup window.
2. Assist verification for pending flagged cases.
3. Confirm override notes are meaningful and complete.
4. Escalate repeated anomalies to ACOP and IT.

Screenshot slots for Admin chapter:
- SS-ADM-03: Pending queue with bulk actions
- SS-ADM-04: Reject flow with reason input
- SS-ADM-05: Chaperones page with status filters
- SS-ADM-06: Shadow-delete modal with reason

## 6D. Teachers SOP

### TCH-1 Class release readiness

Objective:
Prepare students for safe and orderly release.

Steps:
1. Confirm release timing communication from operations.
2. Keep class release list accessible.
3. Coordinate with pickup desk when exceptions occur.

### TCH-2 Pickup handoff support

Objective:
Assist final handoff while preserving safety checks.

Steps:
1. Release students only after pickup confirmation process is complete.
2. For disputed pickup cases, hold student and escalate immediately.
3. Do not bypass flagged event procedures.

### TCH-3 Exception reporting

Objective:
Ensure every irregular event is escalated and documented.

Steps:
1. Report unknown or disputed pickup attempt to Admin/ACOP at once.
2. Log time, student, and brief context.
3. Confirm closure feedback is received.

Screenshot slots for Teacher chapter:
- SS-TCH-01: Teacher-facing quick reference card (to be designed)
- SS-TCH-02: Escalation contact panel (to be designed)

---

## 7. Incident Playbooks

## 7.1 Unknown chaperone

Trigger:
Event classified as unknown chaperone.

Immediate action:
1. Hold release.
2. Verify identity and authorization context.
3. Use officer override only if policy conditions are met.
4. Record reason and approver in override flow.

Escalation:
- First occurrence: Admin review
- Repeated occurrence: ACOP + IT investigation

## 7.2 Suspended chaperone

Trigger:
Event classified as suspended.

Immediate action:
1. Do not release automatically.
2. Confirm suspension status and reason.
3. Apply manual override only under authorized exception path.
4. Log decision and notify ACOP.

## 7.3 Re-enroll overdue

Trigger:
Event classified as re-enroll overdue.

Immediate action:
1. Hold release and verify urgency context.
2. If override is granted, record full reason.
3. Admin initiates re-enrollment follow-up immediately.

## 7.4 Officer override audit quality

Minimum fields expected for every override:
- Who approved
- Why approved
- When approved
- Which gate/event/chaperone

Quality standard:
No empty or generic notes for safety exceptions.

---

## 8. Compliance and Data Handling

This section must remain aligned with policy text in docs/privacy/privacy-policy-v2.md.

### 8.1 Allowed usage boundaries

- Pickup biometric data is used only for pickup authorization and related audit/security purposes.
- No use for profiling, marketing, or non-pickup surveillance.

### 8.2 Retention highlights (operational summary)

- Pickup event captures (JPEG): 30 days
- Pickup event metadata: 1 year
- Chaperone photos/descriptors: until authorization revoked plus 30 days, or 12 months after last pickup, whichever is sooner
- Consent records: retained per policy legal requirements

### 8.3 Consent and withdrawal handling

- Parent confirms consent path during onboarding.
- Chaperones can request removal through school office process.
- Withdrawal must be actioned within policy timelines.

### 8.4 Access discipline

- Limit dashboard access to approved staff roles.
- Use least-privilege assignment.
- Review access periodically.

---

## 9. Troubleshooting and Escalation

## 9.1 Quick diagnosis matrix

| Symptom | Likely owner | First action | Escalate to |
|---|---|---|---|
| Pending queue not loading | IT/Admin | Refresh and verify account/session | IT |
| Terminal appears closed unexpectedly | IT | Check manual override and window config | IT + ACOP if release impact |
| Many unknown chaperone flags | Admin/ACOP | Validate onboarding quality and enrollment status | ACOP + IT |
| Override panel not updating | IT | Check connectivity and API response | IT |
| Re-enroll overdue spikes | Admin | Start re-enroll campaign and notify ACOP | ACOP |

## 9.2 Escalation ladder

Level 1: Admin/Teacher on-site handling  
Level 2: ACOP operational decision  
Level 3: IT technical intervention  
Level 4: Joint ACOP + IT incident review

---

## 10. KPI and Reporting Pack

Suggested weekly indicators:
- Total pickups processed
- Flagged incident count by kind
- Override count and override ratio
- Median queue resolution time for pending onboarding submissions
- Re-enroll due backlog
- Top gates by incidents

Ownership:
- Admin prepares draft data
- ACOP validates operational conclusions
- IT adds technical reliability notes

---

## 11. Screenshot Manifest Template (for design team)

Use this table format for all screenshots.

| Slot ID | Chapter | Screen route | Role view | Required state/filter | Annotation focus | Filename |
|---|---|---|---|---|---|---|
| SS-ADM-01 | 5.1 | /v2/pickup-admin | Admin | Pending tab selected | Queue count and action buttons | ss-adm-01-pending-queue.png |
| SS-OVR-01 | 5.3 | /v2/officer-overrides | Officer/Admin | Awaiting officer has sample records | Pending exception handling | ss-ovr-01-awaiting-officer.png |
| SS-SEC-01 | 5.4 | /v2/security | ACOP/Admin | 14d or 30d range with visible heat | Time concentration analysis | ss-sec-01-heatmap.png |
| SS-IT-01 | 6B | /v2/terminals | IT | Mixed terminal states | State color and gate controls | ss-it-01-terminal-states.png |

Capture rules:
- Use production-like but privacy-safe sample data.
- Avoid exposing sensitive student/chaperone personal identifiers in final published screenshots.
- Keep UI language and date/time format consistent across chapter sequences.

---

## 12. Chapter-by-Chapter Handoff Checklist

For each chapter before design handoff:
1. SOP objective is clear.
2. Preconditions are listed.
3. Step sequence is complete and unambiguous.
4. Expected result is explicit.
5. Failure mode and escalation are defined.
6. Screenshot slot IDs are present.
7. Role owner has reviewed wording.

---

## 13. Publishing Governance

Required sign-offs for v1.0 publication:
- ACOP lead (process and policy operations)
- IT lead (technical validity)
- Admin lead (daily workflow fit)
- Privacy/compliance reviewer (policy consistency)

Version control section:
- Version number
- Effective date
- Summary of changes
- Approved by

---

## 14. Appendix A - One-Page Quick SOP Cards (Content)

These are condensed extracts from chapters 6A-6D and should be designed as single-page role cards.

### ACOP quick card
- Check daily readiness and unresolved incidents
- Approve high-risk exception closure
- Run weekly trend review and corrective actions

### IT quick card
- Verify terminals and pairing before pickup window
- Resolve gate-state and sync problems quickly
- Escalate safety-impacting technical failures immediately

### Admin quick card
- Clear pending onboarding queue daily
- Run re-enroll and lifecycle cleanup
- Coordinate override desk and complete incident notes

### Teacher quick card
- Release only after confirmation path
- Escalate disputed pickup immediately
- Document and close exceptions through operations

---

## 15. Appendix B - Known Operational Caveats

- Photo quality and recency directly affect pickup verification accuracy.
- Override should remain exceptional, not routine.
- Manual override decisions must always include meaningful reason notes.
- Window and gate configuration changes should be controlled and logged.

---

## 16. Draft Completion Notes

This is the content-complete draft for your team to:
- Add screenshots
- Apply visual design and branding
- Convert into final handbook layout

Suggested output formats after design:
- Print-ready PDF manual
- Internal web manual page
- Role-specific quick reference posters/cards
