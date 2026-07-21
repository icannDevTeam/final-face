# RBAC Integration Architecture Diagram

## System Architecture: Student Component + Existing Backend

```
╔═════════════════════════════════════════════════════════════════════════════╗
║                      STUDENT'S POLISHED UI LAYER                           ║
║                                                                             ║
║   SettingsPage.jsx (Student Component)                                      ║
║  ┌─────────────────────────────────────────────────────────────────────┐   ║
║  │                                                                     │   ║
║  │  ┌─ User Invite Form ──────────────────────────────────────────┐  │   ║
║  │  │  Name: ________________  Email: ______________             │  │   ║
║  │  │  Template: [Operations ▼] | ACOP | IT | Individual        │  │   ║
║  │  │                                                            │  │   ║
║  │  │  ┌─ Permission Matrix (Individual mode) ───────────────┐ │  │   ║
║  │  │  │ Feature           │ Read      │ Write             │ │  │   ║
║  │  │  │ Dashboard         │ ☑ ✓      │ ☐                 │ │  │   ║
║  │  │  │ Attendance Mon    │ ☑ ✓      │ ☑ ✓              │ │  │   ║
║  │  │  │ Pickup Queue      │ ☑ ✓      │ ☐                 │ │  │   ║
║  │  │  │ ...               │          │                   │ │  │   ║
║  │  │  └───────────────────────────────────────────────────┘ │  │   ║
║  │  │  [Reset] [Add User] →                                   │  │   ║
║  │  └─────────────────────────────────────────────────────────┘  │   ║
║  │                                                                     │   ║
║  │  ┌─ Users List Table ───────────────────────────────────────────┐  │   ║
║  │  │ Name          │ Template    │ Access Summary       │ Status  │  │   ║
║  │  │ John Doe      │ Ops [badge] │ Reports, Pickup...  │ Active  │  │   ║
║  │  │ Jane Admin    │ ACOP [badge]│ Full operational...  │ Active  │  │   ║
║  │  │ Bob IT        │ IT [badge]  │ Everything          │ Active  │  │   ║
║  │  └───────────────────────────────────────────────────────────────┘  │   ║
║  │                                                                     │   ║
║  └─────────────────────────────────────────────────────────────────────┘   ║
║                                                                             ║
╚═════════════════════════════════════════════════════════════════════════════╝
                                    ↓
                         (Form submission)
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                     ADAPTER LAYER (NEW)                                     │
│                                                                             │
│  lib/rbac-adapter.js                                                        │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │                                                                    │   │
│  │  buildUserInvitePayload()                                         │   │
│  │  ┌─ INPUT (from UI) ───────────────────────────────────────────┐ │   │
│  │  │ {                                                           │ │   │
│  │  │   name: 'John Doe',                                        │ │   │
│  │  │   email: 'john@school.edu',                               │ │   │
│  │  │   template: 'operations',  ← UI template                  │ │   │
│  │  │   permissions: { 'operations.dashboard': {...} }          │ │   │
│  │  │ }                                                           │ │   │
│  │  └─────────────────────────────────────────────────────────────┘ │   │
│  │                                                                    │   │
│  │  ↓ (conversion)                                                   │   │
│  │                                                                    │   │
│  │  ┌─ OUTPUT (for API) ───────────────────────────────────────────┐ │   │
│  │  │ {                                                            │ │   │
│  │  │   name: 'John Doe',                                         │ │   │
│  │  │   email: 'john@school.edu',                                │ │   │
│  │  │   role: 'viewer',  ← Converted to role                    │ │   │
│  │  │   permissions: {}  ← Empty (relies on role defaults)       │ │   │
│  │  │ }                                                            │ │   │
│  │  └─────────────────────────────────────────────────────────────┘ │   │
│  │                                                                    │   │
│  │ Key functions:                                                    │   │
│  │ • templateToRole()                                               │   │
│  │ • convertStudentPermissionsToFirestore()                        │   │
│  │ • buildUserInvitePayload()                                      │   │
│  │                                                                    │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
                            (API POST call)
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                     EXISTING API LAYER (UNCHANGED)                          │
│                                                                             │
│  /api/auth/users                                                            │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │                                                                    │   │
│  │  POST body:                                                       │   │
│  │  {                                                                │   │
│  │    name: 'John Doe',                                             │   │
│  │    email: 'john@school.edu',                                     │   │
│  │    role: 'viewer',    ← Familiar role-based model               │   │
│  │    permissions: {}    ← Template roles = no custom overrides     │   │
│  │  }                                                                │   │
│  │                                                                    │   │
│  │  Backend validates + creates Firebase Auth user                  │   │
│  │  Backend creates Firestore user doc                              │   │
│  │  Backend resolves permissions from ROLE_DEFAULTS                 │   │
│  │                                                                    │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
                        (Firestore write)
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EXISTING PERMISSION MODEL (UNCHANGED)                   │
│                                                                             │
│  Firestore: users/{uid}                                                     │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │ {                                                                  │   │
│  │   uid: 'user123',                                                 │   │
│  │   email: 'john@school.edu',                                       │   │
│  │   displayName: 'John Doe',                                        │   │
│  │   role: 'viewer',                ← From template mapping          │   │
│  │   permissions: {},               ← Empty (template, no overrides) │   │
│  │   status: 'ACTIVE',                                               │   │
│  │   customClaims: {                                                 │   │
│  │     role: 'viewer',                                               │   │
│  │     permissions: {               ← Merged from ROLE_DEFAULTS     │   │
│  │       dashboard: { view: true },                                  │   │
│  │       analytics: { view: true },                                  │   │
│  │       reports: { view: true, edit: true },                        │   │
│  │       enrollment: { view: true },                                 │   │
│  │       // ... more from ROLE_DEFAULTS['viewer']                   │   │
│  │     }                                                              │   │
│  │   }                                                                │   │
│  │ }                                                                  │   │
│  │                                                                    │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  lib/permissions.js: resolvePermissions(role, overrides)                   │
│  • Merges ROLE_DEFAULTS[role] + custom overrides                           │
│  • Returns final permission object for access checks                       │
│  • No changes needed!                                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
                  (When admin loads settings page again)
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                        REVERSE FLOW (FETCHING)                              │
│                                                                             │
│  GET /api/auth/users                                                        │
│  ↓ (returns Firebase user docs)                                             │
│                                                                             │
│  Adapter: toManagedUser()                                                   │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │                                                                    │   │
│  │  INPUT (from Firestore):                                          │   │
│  │  {                                                                │   │
│  │    uid: 'user123',                                               │   │
│  │    email: 'john@school.edu',                                     │   │
│  │    displayName: 'John Doe',                                      │   │
│  │    role: 'viewer',                                               │   │
│  │    permissions: {},                                              │   │
│  │    status: 'ACTIVE',                                             │   │
│  │    lastSignInTime: '2026-06-09T10:30:00Z'                        │   │
│  │  }                                                                │   │
│  │                                                                    │   │
│  │  ↓ (conversion via roleToTemplate + generateSummary)            │   │
│  │                                                                    │   │
│  │  OUTPUT (for UI):                                                │   │
│  │  {                                                                │   │
│  │    id: 0,                                                        │   │
│  │    name: 'John Doe',                                             │   │
│  │    email: 'john@school.edu',                                     │   │
│  │    template: 'operations',  ← Re-inferred from role             │   │
│  │    roleLabel: 'Operations Template',                             │   │
│  │    summary: 'Reports, Release Groups, Dashboard',               │   │
│  │    status: 'Active',                                             │   │
│  │    lastActive: '3 hours ago'                                     │   │
│  │  }                                                                │   │
│  │                                                                    │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  These objects populate the User List Table ↑                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow for Individual Permissions

```
┌─ Individual User Creation ────────────────────────────────────────────────┐
│                                                                           │
│  Student UI Form:                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ Name: Carol                                                      │   │
│  │ Template: [Individual ▼]                                         │   │
│  │                                                                  │   │
│  │ Permission Matrix (EDITABLE):                                   │   │
│  │ ┌─────────────────────────────────────────────────────────────┐ │   │
│  │ │ Feature                   │ Read      │ Write             │ │   │
│  │ │ main.operations           │ ☑ ✓      │ ☐                 │ │   │
│  │ │ operations.dashboard      │ ☑ ✓      │ ☐                 │ │   │
│  │ │ operations.reports        │ ☑ ✓      │ ☑ ✓              │ │   │
│  │ │ devices.attendanceMonitor │ ☑ ✓      │ ☑ ✓              │ │   │
│  │ │ devices.hikvision         │ ☑ ✓      │ ☐                 │ │   │
│  │ │ (rest)                    │ ☐        │ ☐                 │ │   │
│  │ │ [Save]                                                       │ │   │
│  │ └─────────────────────────────────────────────────────────────┘ │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                           │
│         ↓ handleAddUser() with custom permissions                       │
│                                                                           │
│  buildUserInvitePayload('Carol', 'carol@...', 'individual', {          │
│    'main.operations': { read: true, write: false },                    │
│    'operations.dashboard': { read: true, write: false },               │
│    'operations.reports': { read: true, write: true },                  │
│    'devices.attendanceMonitor': { read: true, write: true },           │
│    'devices.hikvision': { read: true, write: false },                  │
│    // (rest as false/false)                                            │
│  })                                                                     │
│                                                                           │
│         ↓ convertStudentPermissionsToFirestore()                        │
│                                                                           │
│  {                                                                       │
│    dashboard: { view: true },                                           │
│    reports: { view: true, edit: true },                                │
│    attendance_monitor: { view: true, edit: true },                     │
│    hikvision: { view: true }                                           │
│  }                                                                       │
│                                                                           │
│         ↓ POST /api/auth/users                                         │
│                                                                           │
│  {                                                                       │
│    name: 'Carol',                                                       │
│    email: 'carol@school.edu',                                           │
│    role: 'viewer',                                                      │
│    permissions: {                      ← Custom overrides!             │
│      dashboard: { view: true },                                        │
│      reports: { view: true, edit: true },                              │
│      attendance_monitor: { view: true, edit: true },                   │
│      hikvision: { view: true }                                         │
│    }                                                                     │
│  }                                                                       │
│                                                                           │
│         ↓ Firestore write                                              │
│                                                                           │
│  users/{uid} = {                                                        │
│    uid: 'carol123',                                                     │
│    email: 'carol@school.edu',                                           │
│    displayName: 'Carol',                                                │
│    role: 'viewer',                                                      │
│    permissions: {                      ← Stored!                       │
│      dashboard: { view: true },                                        │
│      reports: { view: true, edit: true },                              │
│      attendance_monitor: { view: true, edit: true },                   │
│      hikvision: { view: true }                                         │
│    },                                                                    │
│    status: 'ACTIVE',                                                    │
│    customClaims: {                                                      │
│      role: 'viewer',                                                    │
│      permissions: resolvePermissions('viewer', { ...above })  ← Merged│
│    }                                                                     │
│  }                                                                       │
│                                                                           │
│         ↓ When admin views Carol's permissions later:                  │
│                                                                           │
│  roleToTemplate('viewer', { dashboard: {...}, ... })                   │
│  → Detects custom permissions differ from defaults                     │
│  → Returns 'individual'                                                │
│                                                                           │
│  → UI displays: "Carol | Individual Access | 4R/2W"                    │
│  → Admin can click "Edit" to open permission matrix again              │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Key Transformation Functions (at a glance)

```
┌─ templateToRole() ─────────────────────────┐
│                                            │
│  'operations'  → 'viewer'                  │
│  'acop'        → 'admin'                   │
│  'it'          → 'owner'                   │
│  'individual'  → 'viewer' (+ overrides)    │
│                                            │
└────────────────────────────────────────────┘

┌─ convertStudentPermissionsToFirestore() ───┐
│                                            │
│  {                                         │
│    'operations.dashboard': {               │
│      read: true,                           │
│      write: false                          │
│    }                                       │
│  }                                         │
│              ↓                             │
│  {                                         │
│    'dashboard': {                          │
│      view: true,                           │
│      edit: false                           │
│    }                                       │
│  }                                         │
│                                            │
└────────────────────────────────────────────┘

┌─ roleToTemplate() ─────────────────────────┐
│                                            │
│  ('viewer', {})          → 'operations'    │
│  ('admin', {})           → 'acop'          │
│  ('owner', {})           → 'it'            │
│  ('viewer', { custom })  → 'individual'    │
│       (detects overrides)                  │
│                                            │
└────────────────────────────────────────────┘
```

---

## Integration Testing Loop

```
1. Create User (template: 'operations')
   └─→ buildUserInvitePayload()
       └─→ templateToRole() = 'viewer'
           └─→ POST /api/auth/users { role: 'viewer', permissions: {} }
               └─→ Firestore { role: 'viewer', permissions: {} }

2. Fetch Users
   └─→ GET /api/auth/users
       └─→ toManagedUser(firebaseUser)
           └─→ roleToTemplate('viewer', {}) = 'operations'
               └─→ UI displays { template: 'operations', ... }

3. Edit Permissions (if Individual)
   └─→ draftPermissions = { 'operations.dashboard': { read: true } }
       └─→ togglePermission() modifies state
           └─→ convertStudentPermissionsToFirestore()
               └─→ POST /api/auth/users/[email]/permissions
                   └─→ Firestore permissions updated
                       └─→ customClaims recalculated
                           └─→ Fetch again → toManagedUser()
                               └─→ UI re-renders with new perms
```

---

## Error Handling Flow

```
When something breaks, trace this path:

1. "Template undefined" error
   └─→ Check roleToTemplate() output
       └─→ Verify role parameter exists
           └─→ Check Firestore user doc has 'role' field

2. "Invalid permission format" error
   └─→ Check convertStudentPermissionsToFirestore() output
       └─→ Verify input has { read, write } booleans
           └─→ Verify keys map to valid features

3. "User can't access feature" issue
   └─→ Check Firestore permissions object
       └─→ Verify customClaims.permissions has feature
           └─→ Check resolvePermissions() merges correctly
               └─→ Verify frontend permission checks use customClaims

4. "Permissions lost after refresh" issue
   └─→ Check if stored in Firestore.permissions
       └─→ Verify convertFirestorePermissionsToStudent() works
           └─→ Check bidirectional conversion logic
               └─→ Debug state management in React component
```

---

This visual architecture should help you understand:
1. **Where the conversion happens** (adapter layer)
2. **What data flows where** (UI → adapter → API → Firestore → back)
3. **How template/role mapping works** (template ↔ role ↔ permissions)
4. **How Individual permissions differ** (custom overrides stored in Firestore)

Print this or reference it during implementation! 📋
