# PickupGuard Screenshot Capture Sheet

Version: v1.0-draft  
Date: 2026-06-05  
Purpose: Production checklist for screenshot capture and design annotation

---

## Student Project Mode (Grade 4 and 5)

This section is the simple version for students.
This now covers the full Pickup system flow, step by step.

Use this rule:
- One small step at a time.
- Click only what is written below.
- Add one screenshot per step.

### Full System Steps (Simple)

### Step 1: Create Parent Invite Link

Goal:
- Make one invite link that can be shared to parents.

Before you start:
- Login as Admin account.
- Open the dashboard.

Click-by-click:
1. Open Pickup Admin page.
2. Open Invites view.
   Suggested route: /v2/pickup-admin?view=invites
3. Click New invite link.
4. In Name, type campaign name.
   Example: Grade 4 and 5 Parent Invite
5. In Description (optional), type short note.
   Example: June student project batch
6. In Valid for, select 30 days or 90 days.
7. In Max submissions, keep empty for Unlimited.
8. Submission window is optional.
   If not needed, leave it blank.
9. Click Generate link.
10. Copy the link URL.
11. Optional: Click QR to show QR code for parents.

Expected result:
- New invite link appears in Onboarding Invite Links list.
- Status is active and ready to share.

If something goes wrong:
1. Click Refresh.
2. Check if Name has at least 2 characters.
3. If using Submission window, Close time must be after Open time.
4. Try Generate link again.

Image placeholders (add screenshots here):
- [PLACEHOLDER IMAGE - STEP 1A - Pickup Admin page open]
- [PLACEHOLDER IMAGE - STEP 1B - Invites view open]
- [PLACEHOLDER IMAGE - STEP 1C - New invite link button]
- [PLACEHOLDER IMAGE - STEP 1D - Form filled]
- [PLACEHOLDER IMAGE - STEP 1E - Generate link button]
- [PLACEHOLDER IMAGE - STEP 1F - New link shown in list]
- [PLACEHOLDER IMAGE - STEP 1G - Copied link or QR preview]

### Step 2: Share Link to Parents

Goal:
- Send the invite link to parents.

Click-by-click:
1. Stay in Invites view.
2. Find your new invite card.
3. Click Copy Link.
4. Paste link into WhatsApp/Email parent group.
5. Send the message.

Image placeholders:
- [PLACEHOLDER IMAGE - STEP 2A - Invite card with Copy Link]
- [PLACEHOLDER IMAGE - STEP 2B - Link pasted in message draft]

### Step 3: Check Parent Form Submission

Goal:
- Confirm parents can open and submit the form.

Click-by-click:
1. Open the shared link in browser.
2. Check form opens correctly.
3. Fill sample data (test data only).
4. Submit form.
5. Check success message.

Image placeholders:
- [PLACEHOLDER IMAGE - STEP 3A - Parent form opened]
- [PLACEHOLDER IMAGE - STEP 3B - Parent form filled]
- [PLACEHOLDER IMAGE - STEP 3C - Parent success message]

### Step 4: Review Parent Request (Admin)

Goal:
- See new request in admin queue.

Click-by-click:
1. Open Pickup Admin.
2. Open Onboarding view (Pending tab).
3. Find the new parent request.
4. Click the record to open details.

Image placeholders:
- [PLACEHOLDER IMAGE - STEP 4A - Pending tab]
- [PLACEHOLDER IMAGE - STEP 4B - Request detail open]

### Step 5: Approve or Reject Request

Goal:
- Complete the decision for each request.

Click-by-click:
1. In request detail, check names and photos.
2. If valid, click Approve.
3. If not valid, click Reject.
4. If Reject, write reason.
5. Confirm action.

Image placeholders:
- [PLACEHOLDER IMAGE - STEP 5A - Approve button]
- [PLACEHOLDER IMAGE - STEP 5B - Reject with reason box]
- [PLACEHOLDER IMAGE - STEP 5C - Approved status in list]

### Step 6: Check Chaperone List

Goal:
- Verify approved person appears in chaperone system.

Click-by-click:
1. Open /v2/chaperones.
2. Search by chaperone name.
3. Check status is active/available.

Image placeholders:
- [PLACEHOLDER IMAGE - STEP 6A - Chaperones page]
- [PLACEHOLDER IMAGE - STEP 6B - Chaperone found in list]

### Step 7: Check Terminal and Gate Setup

Goal:
- Make sure pickup gate devices are ready.

Click-by-click:
1. Open /v2/terminals.
2. Check terminals are enabled.
3. Check gate state (Open/Auto/Close).
4. Open /v2/release-groups.
5. Check terminal is paired to correct group.

Image placeholders:
- [PLACEHOLDER IMAGE - STEP 7A - Terminals status cards]
- [PLACEHOLDER IMAGE - STEP 7B - Release groups paired]

### Step 8: Handle Flagged Pickup (If Happens)

Goal:
- Safely handle exception cases.

Click-by-click:
1. Open /v2/officer-overrides.
2. Check Awaiting officer list.
3. Verify case details.
4. Enter 6-digit code only if approved by policy.
5. Confirm action and check history.

Image placeholders:
- [PLACEHOLDER IMAGE - STEP 8A - Awaiting officer list]
- [PLACEHOLDER IMAGE - STEP 8B - Code input pad]
- [PLACEHOLDER IMAGE - STEP 8C - History updated]

### Step 9: Daily Safety Check (End of Day)

Goal:
- Review incidents and close the day.

Click-by-click:
1. Open /v2/security.
2. Check heatmap and recent incidents.
3. Note any risky pattern.
4. Report summary to ACOP/Admin lead.

Image placeholders:
- [PLACEHOLDER IMAGE - STEP 9A - Security heatmap]
- [PLACEHOLDER IMAGE - STEP 9B - Recent incidents panel]

Teacher notes:
- Do one step per lesson block.
- Keep all screenshots in step order.
- Use test data only for class project work.

---

## 1. Capture Standards

Use these standards for all screenshots:
1. Capture at 1920x1080 (desktop) unless a mobile screen is explicitly required.
2. Keep language as English across the full manual set.
3. Use privacy-safe sample data; avoid real sensitive IDs in published visuals.
4. Show one primary action focus per screenshot.
5. Keep browser zoom at 100% unless a section requires accessibility zoom.
6. Keep date/time format consistent within each chapter sequence.

Filename pattern:
- pickup-<slot-id-lowercase>-<short-label>.png

Example:
- pickup-ss-adm-01-pending-queue.png

---

## 2. Environment Prep Checklist

Before capture session:
1. Login with role-appropriate test account.
2. Seed enough demo records for pending, approved, rejected, and flagged states.
3. Ensure at least one terminal per state: open, closed, manual, disabled.
4. Ensure at least one release group is paired and one is unbound.
5. Ensure security incidents exist in at least 2 categories.
6. Ensure officer override history has at least 3 completed records.

---

## 3. Screenshot Master List

| Slot ID | Chapter | Route | Role | Required State | What to Highlight | Suggested Filename |
|---|---|---|---|---|---|---|
| SS-ONB-01 | 5.1 | /pickup/onboarding/[token] | Parent | Fresh token, empty form | Header, form structure, consent section | pickup-ss-onb-01-onboarding-form.png |
| SS-ONB-02 | 5.1 | /pickup/onboarding/[token] | Parent | Completed successful submit | Confirmation status and next-step text | pickup-ss-onb-02-onboarding-success.png |
| SS-ADM-01 | 5.1 | /v2/pickup-admin | Admin | Pending tab active, queue > 0 | Pending count, search/sort, action buttons | pickup-ss-adm-01-pending-queue.png |
| SS-ADM-02 | 5.1 | /v2/pickup-admin | Admin | Record detail open | Approve/Reject controls and key fields | pickup-ss-adm-02-record-detail.png |
| SS-GATE-01 | 5.2 | /v2/terminals | IT | Terminal in open effective state | State badge, open/auto/close controls | pickup-ss-gate-01-terminal-open.png |
| SS-GATE-02 | 5.2 | /v2/index | Admin | Pickup module view active | Event visibility and status context | pickup-ss-gate-02-pickup-module.png |
| SS-OVR-01 | 5.3 | /v2/officer-overrides | Officer/Admin | Awaiting officer list populated | Pending flagged cards and timers | pickup-ss-ovr-01-awaiting-list.png |
| SS-OVR-02 | 5.3 | /v2/officer-overrides | Officer/Admin | Quick pad visible | Override code entry pad and submit action | pickup-ss-ovr-02-quick-pad.png |
| SS-OVR-03 | 5.3 | /v2/officer-overrides | Admin | History table with entries | Approved by, note, gate, timestamp | pickup-ss-ovr-03-history-table.png |
| SS-SEC-01 | 5.4 | /v2/security | ACOP/Admin | 14d or 30d selected | Heatmap with legend and day/hour pattern | pickup-ss-sec-01-heatmap.png |
| SS-SEC-02 | 5.4 | /v2/security | ACOP/Admin | By-kind panel visible | Incident categories and volume bars | pickup-ss-sec-02-by-kind.png |
| SS-SEC-03 | 5.4 | /v2/security | ACOP/Admin | Recent list contains mixed statuses | Pending/resolved recent incidents | pickup-ss-sec-03-recent-list.png |
| SS-ACOP-01 | 6A | /v2/security | ACOP | 30d view, clear trend | Weekly trend interpretation panel | pickup-ss-acop-01-trend-view.png |
| SS-ACOP-02 | 6A | /v2/officer-overrides | ACOP | History filtered last 7d | Override quality review evidence | pickup-ss-acop-02-override-audit.png |
| SS-IT-01 | 6B | /v2/terminals | IT | Mixed terminal states | Visual state coding and live status | pickup-ss-it-01-terminal-grid.png |
| SS-IT-02 | 6B | /v2/release-groups | IT | Paired + unbound groups visible | Pairing status and gate bindings | pickup-ss-it-02-release-groups.png |
| SS-IT-03 | 6B | /v2/terminals | IT | Edit state for one terminal | Window open/close and override settings | pickup-ss-it-03-terminal-edit.png |
| SS-ADM-03 | 6C | /v2/pickup-admin | Admin | Multi-select rows active | Bulk action controls | pickup-ss-adm-03-bulk-actions.png |
| SS-ADM-04 | 6C | /v2/pickup-admin | Admin | Reject inline reason open | Reject reason field and submit | pickup-ss-adm-04-reject-reason.png |
| SS-ADM-05 | 6C | /v2/chaperones | Admin | Filter by due/never_enrolled | Status filters and row indicators | pickup-ss-adm-05-chaperone-filters.png |
| SS-ADM-06 | 6C | /v2/chaperones | Admin | Shadow-delete modal open | Mandatory reason and warning text | pickup-ss-adm-06-shadow-delete.png |
| SS-TCH-01 | 6D | N/A (design card) | Teacher | Quick card layout mock | Release checklist summary | pickup-ss-tch-01-quick-card.png |
| SS-TCH-02 | 6D | N/A (design card) | Teacher | Escalation card layout mock | Contact ladder and escalation flow | pickup-ss-tch-02-escalation-card.png |

---

## 4. Capture Sequence (Recommended)

Run this sequence to reduce rework:
1. Parent onboarding screens (SS-ONB-01, SS-ONB-02)
2. Admin queue screens (SS-ADM-01, SS-ADM-02, SS-ADM-03, SS-ADM-04)
3. Chaperone lifecycle screens (SS-ADM-05, SS-ADM-06)
4. Terminal and release-group screens (SS-GATE-01, SS-IT-01, SS-IT-02, SS-IT-03)
5. Officer override screens (SS-OVR-01, SS-OVR-02, SS-OVR-03)
6. Security analytics screens (SS-SEC-01, SS-SEC-02, SS-SEC-03, SS-ACOP-01, SS-ACOP-02)
7. Teacher design cards (SS-TCH-01, SS-TCH-02)

---

## 5. Annotation Guidance for Design Team

For each screenshot, add these annotation layers:
1. Primary action: what the user should do first.
2. Decision point: where user decides approve/reject/escalate.
3. Warning/risk point: where wrong action creates safety risk.
4. Evidence point: where audit data is shown.

Keep annotation copy short:
- 3 to 10 words per callout
- Maximum 5 callouts per screenshot

---

## 6. QA Checklist Before Manual Assembly

1. Every slot in section 3 has a captured file.
2. Filenames follow the defined convention.
3. No screenshot contains sensitive personal data in final export.
4. All chapter references match the latest manual version.
5. Visual style is consistent (spacing, arrows, callout style, colors).
6. Final set is approved by ACOP and Admin owner before publishing.
