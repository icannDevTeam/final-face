# PickupGuard Operations Manual

Version: v1.0  
Date: 2026-06-05  
Audience: Academic Operations (ACOP), IT, Teachers, Admin

---

## 1. Purpose

PickupGuard ensures that only authorized adults can pick up students during dismissal operations.

This manual defines the standard operating model for:
- Parent onboarding and chaperone approval
- Gate operations and exception handling
- Incident monitoring and closure
- Role accountability across ACOP, IT, Admin, and Teachers

---

## 2. Operating Surfaces

Primary routes used by operations:
- Parent onboarding: /pickup/onboarding/[token]
- Pickup admin queue: /v2/pickup-admin
- Chaperone management: /v2/chaperones
- Officer override desk: /v2/officer-overrides
- Security monitoring: /v2/security
- Terminal operations: /v2/terminals
- Release group pairing: /v2/release-groups

---

## 3. Role Accountability

| Activity | ACOP | IT | Admin | Teachers |
|---|---|---|---|---|
| Pickup policy and governance | A | C | C | I |
| Terminal and pairing availability | I | A/R | C | I |
| Onboarding review and daily queue | C | I | A/R | I |
| Chaperone approve/reject decisions | A | I | R | I |
| Override desk operation | A | C | R | I |
| Incident trend review and closure | A | C | R | I |
| Class release coordination | C | I | C | A/R |

Legend: A = Accountable, R = Responsible, C = Consulted, I = Informed

Operating guardrails:
- Teachers do not approve override decisions.
- IT does not make policy approval decisions.
- Admin exceptions require ACOP visibility.

---

## 4. Daily Operating Cycle

Pre-dismissal:
1. IT validates terminal availability and pairing status.
2. Admin clears urgent pending submissions.
3. ACOP confirms officer desk coverage.
4. Teachers receive release readiness update.

During pickup window:
1. Gate scans process normal pickups automatically.
2. Flagged events move to officer override workflow.
3. Teachers release students only after confirmation.

Post-window:
1. Admin reviews unresolved incidents.
2. ACOP signs off high-risk cases.
3. IT documents technical faults and corrective actions.

---

## 5. Core Workflow SOPs

## 5.1 Parent onboarding and approval

Objective:
Register and validate authorized chaperones.

Steps:
1. Parent submits onboarding data via secure token link.
2. Admin reviews records in Pending queue.
3. Admin approves valid submissions.
4. Admin rejects invalid submissions with reason.
5. Approved records become active for gate matching.

Success criteria:
- Valid chaperones are active and auditable.
- Rejections include clear reason text.

Escalation:
- Policy ambiguity to ACOP.
- Technical upload/sync issue to IT.

## 5.2 Gate scan and release

Objective:
Release students to authorized adults while preventing unsafe handoff.

Steps:
1. Chaperone scans at gate terminal.
2. System checks identity and authorization.
3. Normal events proceed to release.
4. Flagged events require officer override action.

Success criteria:
- Every release has a clear decision path and audit trace.

## 5.3 Officer override

Objective:
Handle flagged pickups with controlled manual approval.

Steps:
1. Officer/Admin reviews pending flagged event.
2. Identity and context are verified.
3. 6-digit override code is entered in override desk.
4. Approval note and approver identity are recorded.

Success criteria:
- Manual exceptions are justified and fully auditable.

## 5.4 Incident review and closure

Objective:
Detect risk patterns and close incidents with ownership.

Steps:
1. Review heatmap and trend concentration.
2. Review breakdown by kind and gate.
3. Assign owner and due time for unresolved cases.
4. Mark resolved only after verification.

Success criteria:
- No unresolved high-risk incidents remain unassigned.

---

## 6. Role SOPs

## 6.1 ACOP SOP

Daily:
1. Confirm readiness and active desk coverage.
2. Review unresolved incidents from prior day.
3. Approve or return high-risk closure decisions.

Weekly:
1. Review trend reports and repeat-risk patterns.
2. Approve corrective action plan.

## 6.2 IT SOP

Daily:
1. Verify all active terminals are healthy.
2. Verify release groups and pairing status.
3. Validate gate windows and manual overrides.

On incident:
1. Isolate scope.
2. Restore service.
3. Record root cause and mitigation.

## 6.3 Admin SOP

Daily:
1. Process pending queue.
2. Perform approve/reject actions with documentation.
3. Monitor override desk during pickup window.
4. Run re-enrollment and re-push actions as needed.

Data hygiene:
1. Use shadow-delete with mandatory reason.
2. Preserve audit trail and avoid unapproved data removal.

## 6.4 Teacher SOP

Daily:
1. Follow release timing and checklist.
2. Release only after confirmation path.
3. Escalate disputed pickups immediately.

Never:
- Bypass flagged-event controls.
- Approve overrides directly.

---

## 7. Incident Playbooks

## 7.1 Unknown chaperone

1. Hold release.
2. Verify identity and authorization.
3. Use override only under approved policy condition.
4. Record rationale and approver.

## 7.2 Suspended chaperone

1. Hold release.
2. Confirm suspension reason.
3. Escalate to ACOP for final decision if exception is requested.
4. Log final decision.

## 7.3 Re-enroll overdue

1. Hold release pending verification.
2. If override is granted, record complete note.
3. Admin starts re-enrollment follow-up immediately.

## 7.4 Override note quality standard

Every override entry must include:
- Approver
- Timestamp
- Gate and event context
- Concrete reason for approval

---

## 8. Compliance and Data Handling

Operational compliance rules:
1. Use pickup biometrics only for pickup authorization and related audit/security functions.
2. Do not use pickup data for marketing or profiling.
3. Follow approved retention windows from policy source.
4. Apply least-privilege access to dashboard users.

Retention summary:
- Pickup event capture images: 30 days
- Pickup event metadata: 1 year
- Chaperone biometrics: until revoked plus 30 days, or 12 months after last pickup, whichever is sooner

---

## 9. Troubleshooting and Escalation

| Symptom | First owner | First action | Escalation |
|---|---|---|---|
| Pending queue issue | Admin/IT | Verify session and refresh | IT |
| Terminal state mismatch | IT | Check override and window settings | IT + ACOP |
| High unknown rate | Admin | Validate onboarding and enrollment quality | ACOP + IT |
| Override panel issue | IT | Check connectivity/API status | IT |

Escalation ladder:
1. On-site Admin or Teacher response
2. ACOP operational decision
3. IT technical intervention
4. Joint ACOP and IT incident review

---

## 10. KPI and Reporting

Minimum weekly dashboard pack:
1. Total pickups
2. Flagged incidents by type
3. Override count and ratio
4. Onboarding queue turnaround time
5. Re-enrollment due backlog
6. Top gates by incident frequency

Ownership:
- Admin compiles data
- ACOP validates operational interpretation
- IT appends reliability notes

---

## 11. Publication Controls

Before publishing manual updates:
1. ACOP sign-off (policy and operations)
2. IT sign-off (technical validity)
3. Admin sign-off (workflow fit)
4. Privacy reviewer confirmation (policy alignment)

Document control fields:
- Version
- Effective date
- Change summary
- Approvers

---

## 12. Design and Screenshot Integration Note

This print-oriented file is content-only.
Use the companion capture file for visuals and annotation planning:
- docs/pickupguard-screenshot-capture-sheet.md
