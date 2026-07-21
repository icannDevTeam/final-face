# Visual Comparison: Student Component vs. Existing System

## 📊 Data Model Alignment

### Student Component Permission Matrix

```
RBAC_GROUPS
├── main
│   ├── main.operations        { read, write }
│   ├── main.devices           { read, write }
│   ├── main.pickupSystem      { read, write }
│   └── main.administration    { read, write }
├── operations
│   ├── operations.dashboard   { read, write }
│   ├── operations.analytics   { read, write }
│   └── operations.reports     { read, write }
├── devices
│   ├── devices.attendanceMonitor
│   ├── devices.datasetCapture
│   ├── devices.mobileEnrollment
│   ├── devices.deviceManager
│   ├── devices.hikvision
│   └── devices.deviceSync
├── pickupSystem
│   ├── pickupSystem.reviewQueue
│   ├── pickupSystem.chaperoneLifecycle
│   ├── pickupSystem.releaseGroups
│   ├── pickupSystem.gateOperations
│   ├── pickupSystem.pickupSettings
│   └── pickupSystem.terminalsAndKiosks
└── administration
    ├── administration.settings
    ├── administration.userManagement
    ├── administration.sensitiveUserAccess
    ├── administration.securityAndAudit
    ├── administration.downloadsHub
    ├── administration.notifications
    └── administration.integrations

Total: 27 distinct permission items
Each item: { read: boolean, write: boolean }
```

### Existing System Permission Model

```
FEATURES (object, each with actions array)
├── dashboard                 { view, view_attendance, view_pickup }
├── analytics                 { view, view_attendance, view_pickup, export, view_pii }
├── reports                   { view, view_attendance, view_pickup, view_forms, export }
├── attendance_monitor        { view, edit }
├── enrollment                { view, edit, delete, enroll, bulk_enroll, export_pii }
├── mobile_enrollment         { view, edit }
├── device_manager            { view, edit, add, remove, factory_reset }
├── hikvision                 { view, edit }
├── device_sync               { view, edit }
├── pickup_admin              { view, approve, reject, bulk_approve, bulk_reject, ... (20 actions) }
├── settings                  { view, edit }
├── user_management           { view, create, edit, suspend, delete, assign_admin, assign_owner, ... }
├── ai_parameters             { view, edit }
├── notifications             { view, edit }
├── integrations              { view, edit }
├── security_audit            { view }
├── sensitive_user_access     { view_rbac, edit_rbac, reset_user_password, view_user_directory, manage_custom_claims }
├── downloads                 { view, download_operational, download_directory, download_security, download_compliance, manage_presets }
└── auth                      { signed_in }

Total: 19 distinct features (+ auth = 20)
Each feature has 1–20 distinct actions
Granular action-based model for fine-grained control
```

---

## 🔄 Conversion Examples

### Example 1: Operations Template

**Student UI Input:**
```javascript
TEMPLATE_OPTIONS[0] = { 
  value: 'operations', 
  label: 'Operations Template', 
  badge: 'Ops' 
};

TEMPLATE_PERMISSIONS['operations'] = {
  'main.operations': { read: true, write: false },
  'operations.dashboard': { read: true, write: false },
  'operations.analytics': { read: true, write: false },
  'operations.reports': { read: true, write: true },
  'pickupSystem.reviewQueue': { read: true, write: false },
  'pickupSystem.chaperoneLifecycle': { read: true, write: false },
  'pickupSystem.releaseGroups': { read: true, write: false },
  'pickupSystem.gateOperations': { read: true, write: false },
  'pickupSystem.pickupSettings': { read: true, write: false },
  'administration.downloadsHub': { read: true, write: true },
  'administration.notifications': { read: true, write: true },
  // (remaining 16 items as false/false)
};
```

**Conversion Process:**
```javascript
// Step 1: UI calls handleAddUser() with template='operations'
const payload = buildUserInvitePayload(
  'John Doe',                    // name
  'john@school.edu',             // email
  'operations',                  // template
  studentPermissions             // permission map
);

// Step 2: buildUserInvitePayload() converts
{
  name: 'John Doe',
  email: 'john@school.edu',
  role: 'viewer',                // ← templateToRole('operations')
  permissions: {}                // ← No custom overrides (template-based)
}

// Step 3: POST to /api/auth/users
// Backend receives { role: 'viewer', permissions: {} }

// Step 4: Firestore storage
{
  uid: 'auto-generated',
  email: 'john@school.edu',
  displayName: 'John Doe',
  role: 'viewer',                // ← Template encoded as role
  permissions: {},
  status: 'ACTIVE',
  customClaims: {
    role: 'viewer',
    permissions: resolvePermissions('viewer', {})  // ← Merged from defaults
  }
}

// Step 5: When UI re-fetches users, conversion back
{
  id: 0,
  name: 'John Doe',
  email: 'john@school.edu',
  template: roleToTemplate('viewer', {})  // ← Infer 'operations' from role
  roleLabel: 'Operations Template',
  summary: 'Reports, Release Groups, Dashboard',
  status: 'Active',
  lastActive: '2 hours ago'
}
```

**Result in Firestore:**
```json
{
  "uid": "...",
  "email": "john@school.edu",
  "displayName": "John Doe",
  "role": "viewer",
  "permissions": {},
  "status": "ACTIVE",
  "customClaims": {
    "role": "viewer",
    "permissions": {
      "dashboard": { "view": true },
      "analytics": { "view": true },
      "reports": { "view": true, "edit": true },
      "attendance_monitor": { "view": true },
      "pickup_admin": { "view": true },
      "settings": { "view": true },
      "downloads": { "view": true, "edit": true },
      "notifications": { "view": true, "edit": true }
    }
  }
}
```

---

### Example 2: Individual Access (Custom Permissions)

**Student UI Input:**
```javascript
selectedTemplate = 'individual'

draftPermissions = {
  'main.operations': { read: true, write: false },
  'operations.dashboard': { read: true, write: false },
  'analytics.reports': { read: true, write: true },
  'devices.attendanceMonitor': { read: true, write: true },
  'devices.hikvision': { read: true, write: false },
  // (rest false/false)
};
```

**Conversion Process:**
```javascript
// Step 1: buildUserInvitePayload() with individual template
{
  name: 'Jane Admin',
  email: 'jane@school.edu',
  role: 'viewer',                           // ← Base role
  permissions: convertStudentPermissionsToFirestore(draftPermissions)  // ← Custom overrides!
}

// Step 2: convertStudentPermissionsToFirestore() maps
convertStudentPermissionsToFirestore({
  'operations.dashboard': { read: true, write: false },
  'operations.reports': { read: true, write: true },
  'devices.attendanceMonitor': { read: true, write: true },
  'devices.hikvision': { read: true, write: false },
  // ...
})
  ↓
{
  'dashboard': { 'view': true },           // ← read: true → view: true
  'reports': { 'view': true, 'edit': true }, // ← read+write → view+edit
  'attendance_monitor': { 'view': true, 'edit': true },
  'hikvision': { 'view': true },
  // ...
}

// Step 3: Firestore storage
{
  uid: 'auto-generated',
  email: 'jane@school.edu',
  displayName: 'Jane Admin',
  role: 'viewer',
  permissions: {
    'dashboard': { 'view': true },
    'reports': { 'view': true, 'edit': true },
    'attendance_monitor': { 'view': true, 'edit': true },
    'hikvision': { 'view': true }
    // (no entry for non-granted features)
  },
  status: 'ACTIVE',
  customClaims: {
    role: 'viewer',
    permissions: resolvePermissions('viewer', {
      'dashboard': { 'view': true },
      'reports': { 'view': true, 'edit': true },
      'attendance_monitor': { 'view': true, 'edit': true },
      'hikvision': { 'view': true }
    })
  }
}

// Step 4: When UI re-fetches, detect as 'individual'
roleToTemplate('viewer', { ...customOverrides })
  ↓ (detects custom overrides differ from defaults)
  ↓
'individual'
```

**Result in UI:**
```javascript
{
  id: 1,
  name: 'Jane Admin',
  email: 'jane@school.edu',
  template: 'individual',              // ← Detected custom permissions
  roleLabel: 'Individual Access',
  summary: '4 read permissions, 2 write permissions',
  status: 'Active',
  lastActive: 'Right now'
}
```

---

## 🎯 Key Mapping Rules

### Template → Role

| Student Template | Existing Role | Why |
|------------------|---------------|-----|
| **Operations** | **viewer** | Read-most, limited write (reports only) |
| **ACOP** | **admin** | Operational + supervision (pickup queue, chaperoneism) |
| **IT** | **owner** | Full system access |
| **Individual** | **viewer + custom** | Base role + custom overrides |

### Permission Format

| Aspect | Student | Existing | Conversion |
|--------|---------|----------|-----------|
| **Read access** | `read: true` | `view: true` | 1:1 mapping |
| **Write access** | `write: true` | `edit: true` | 1:1 mapping (primary action) |
| **Granularity** | 27 items | 19 features (100+ actions) | Simplified for UX |
| **Storage** | In-memory state | Firestore + custom claims | Convert on persist/load |

---

## 🧪 Test Cases (Validation)

### Test 1: Create Operations User

**Input:**
```javascript
handleAddUser({
  draftName: 'Alice Operator',
  draftEmail: 'alice@school.edu',
  selectedTemplate: 'operations'
})
```

**Expected Firestore:**
```
role: 'viewer'
permissions: {} (empty, relies on defaults)
```

**Expected Permissions Resolved:**
```
dashboard: { view: true }
analytics: { view: true }
reports: { view: true, edit: true }
pickup_admin: { view: true }
downloads: { view: true, edit: true }
```

**Test:** Fetch user back, verify `template === 'operations'`

---

### Test 2: Create IT User

**Input:**
```javascript
handleAddUser({
  draftName: 'Bob Admin',
  draftEmail: 'bob@school.edu',
  selectedTemplate: 'it'
})
```

**Expected Firestore:**
```
role: 'owner'
permissions: {} (relies on defaults)
```

**Test:** Fetch user back, verify `template === 'it'`, confirm full access

---

### Test 3: Create Individual User with Custom Perms

**Input:**
```javascript
handleAddUser({
  draftName: 'Carol Custom',
  draftEmail: 'carol@school.edu',
  selectedTemplate: 'individual',
  draftPermissions: {
    'operations.dashboard': { read: true, write: false },
    'operations.reports': { read: true, write: true },
    'devices.attendanceMonitor': { read: true, write: true },
    // (rest false/false)
  }
})
```

**Expected Firestore:**
```
role: 'viewer'
permissions: {
  'dashboard': { 'view': true },
  'reports': { 'view': true, 'edit': true },
  'attendance_monitor': { 'view': true, 'edit': true }
}
```

**Test:** 
- Fetch user back, verify `template === 'individual'`
- Edit individual perms, verify persistence

---

### Test 4: Edit Existing Admin User

**Scenario:** Bob (admin/IT) exists. Admin edits his permissions to remove hikvision access.

**Input:**
```javascript
setEditingPerms({
  email: 'bob@school.edu',
  template: 'it',
  permissions: { /* draftPermissions with hikvision.write: false */ }
})
```

**Expected Firestore:**
```
role: 'owner'
permissions: {
  'hikvision': { 'view': false, 'edit': false }  // Override to remove
}
```

**Test:** Fetch Bob back, verify hikvision access removed while others intact

---

### Test 5: Convert Existing Users (Migration)

**Scenario:** System already has admin users with custom permissions in old format. UI fetches them.

**Setup (Firestore):**
```javascript
{
  uid: 'existing-user-1',
  role: 'admin',
  permissions: {
    'pickup_admin': { 'view': true, 'edit': true },
    'enrollment': { 'view': true },
    'hikvision': { 'view': false }
  }
}
```

**Expected UI Display:**
```javascript
{
  template: 'individual',  // ← Custom perms detected
  roleLabel: 'Individual Access',
  summary: '8 read permissions, 5 write permissions'
}
```

**Test:**
- UI can display existing user correctly
- Editing their perms persists via convert → Firestore → re-convert flow

---

## 🚀 Integration Checklist

### Pre-Merge (Preparation)

- [ ] Read RBAC_REFACTOR_ANALYSIS.md
- [ ] Review lib/rbac-adapter.js for conversion logic
- [ ] Understand template → role mapping
- [ ] Create staging environment copy

### Implementation

- [ ] Copy student component to `pages/v2/settings-new.jsx`
- [ ] Import rbac-adapter.js functions
- [ ] Adapt handleAddUser() to use buildUserInvitePayload()
- [ ] Adapt fetchUsers() to use toManagedUser()
- [ ] Adapt permission editor to use convertStudentPermissionsToFirestore()
- [ ] Add error handling for conversion failures

### Testing

- [ ] Run Test 1 (Operations user)
- [ ] Run Test 2 (IT user)
- [ ] Run Test 3 (Individual user)
- [ ] Run Test 4 (Edit permissions)
- [ ] Run Test 5 (Existing user migration)
- [ ] Verify permission inheritance (admin can't grant owner perms)
- [ ] Check role defaults still apply
- [ ] Test bulk import (if used)

### Validation

- [ ] Firestore schema unchanged
- [ ] API endpoints work as before
- [ ] Access logs recorded correctly
- [ ] Permission resolution functions unchanged
- [ ] UI state persists across refresh
- [ ] Email invitations work

### Deploy

- [ ] Backup current settings.js
- [ ] Swap new file in
- [ ] Restart Next.js server
- [ ] Smoke test in staging
- [ ] Deploy to production
- [ ] Monitor for permission-related errors

---

## 📞 Troubleshooting

| Problem | Likely Cause | Solution |
|---------|-------------|----------|
| User can't access features after creation | Role not mapped correctly | Check templateToRole() output |
| Permissions lost after refresh | Conversion not bidirectional | Verify convertFirestorePermissionsToStudent() |
| "Template unknown" error | Feature not in mapping | Add to studentKeyToFeature() |
| UI displays "No access" for admins | Role inference wrong | Debug roleToTemplate() logic |
| Custom permissions not persisting | Firestore write fails | Check API error response |

---

## 📚 Reference Files

- Student Component: `[provided in user request]`
- Analysis: `RBAC_REFACTOR_ANALYSIS.md`
- Adapter Library: `web-dataset-collector/lib/rbac-adapter.js`
- Existing Permissions: `web-dataset-collector/lib/permissions.js`
- Current Settings: `web-dataset-collector/pages/v2/settings.js`
