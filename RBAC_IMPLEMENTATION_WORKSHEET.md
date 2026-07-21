# RBAC Implementation Worksheet

## Phase 1: Preparation (Before You Start)

### Environment Setup

- [ ] Ensure you're on branch: `git status`
- [ ] Staging environment ready: `npm run dev`
- [ ] Backup current settings: `cp web-dataset-collector/pages/v2/settings.js web-dataset-collector/pages/v2/settings.backup.js`
- [ ] Create feature branch: `git checkout -b feat/rbac-ui-refactor`
- [ ] Review docs:
  - [ ] Read RBAC_REFACTOR_ANALYSIS.md (30 min)
  - [ ] Skim RBAC_ARCHITECTURE_DIAGRAMS.md (10 min)
  - [ ] Read RBAC_COMPARISON_AND_TESTS.md test cases (15 min)

### Dependency Check

- [ ] Verify `lib/permissions.js` exists: `ls web-dataset-collector/lib/permissions.js`
- [ ] Verify `lib/rbac-adapter.js` exists: `ls web-dataset-collector/lib/rbac-adapter.js`
- [ ] Check React version: `grep "react" web-dataset-collector/package.json`
- [ ] Check Node version: `node --version` (must be 16+)

---

## Phase 2: Integration (The Work)

### Step 1: Set Up Files

**Time: 10 minutes**

```bash
# Location: web-dataset-collector/pages/v2/

# Backup current
cp settings.js settings.backup.js

# Create new file (copy student component here)
cat > settings-new.jsx << 'EOF'
[Paste entire student component here]
EOF

# Verify it's there
ls -lh settings-new.jsx
```

**Checklist:**
- [ ] settings.backup.js exists
- [ ] settings-new.jsx contains student component
- [ ] No syntax errors in new file: `npm run lint -- settings-new.jsx`

---

### Step 2: Add Imports to settings-new.jsx

**Time: 10 minutes**

At the **top** of `settings-new.jsx`, add:

```javascript
'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
// ADD THESE IMPORTS:
import { useAuth } from '../../lib/AuthContext';
import {
  templateToRole,
  roleToTemplate,
  buildUserInvitePayload,
  convertStudentPermissionsToFirestore,
  convertFirestorePermissionsToStudent,
  toManagedUser,
  loadTemplatePermissions,
  getEnabledAreas,
  countPermissions,
} from '../../lib/rbac-adapter';

// STUDENT'S EXISTING CODE:
const RBAC_GROUPS = [ ... ];
const TEMPLATE_OPTIONS = [ ... ];
// ... rest of component
```

**Checklist:**
- [ ] Added useAuth import
- [ ] Added all 8 rbac-adapter imports
- [ ] No TypeScript errors

---

### Step 3: Adapt useEffect (Fetch Users)

**Time: 15 minutes**

Find this section in `settings-new.jsx`:

```javascript
const [users, setUsers] = useState(INITIAL_USERS);
// ... rest of state declarations
```

**Replace the user-fetching logic.** Find/add this pattern:

```javascript
const { user, permissions } = useAuth();
const [users, setUsers] = useState([]);
const [loadingUsers, setLoadingUsers] = useState(true);

useEffect(() => {
  async function fetchUsers() {
    setLoadingUsers(true);
    try {
      // Get auth token
      const token = await user.getIdToken();
      const headers = {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`,
      };

      // Fetch users from backend
      const res = await fetch('/api/auth/users', { headers });
      if (res.ok) {
        const data = await res.json();
        
        // CONVERT Firebase format → UI format using adapter
        const managedUsers = (data.users || []).map((firebaseUser, index) => 
          toManagedUser(firebaseUser, index)
        );
        
        setUsers(managedUsers);
      } else {
        console.error('Failed to fetch users:', res.status);
      }
    } catch (err) {
      console.error('Error fetching users:', err);
    } finally {
      setLoadingUsers(false);
    }
  }

  if (user) {
    fetchUsers();
  }
}, [user]);
```

**Checklist:**
- [ ] useEffect added with proper dependencies
- [ ] toManagedUser() used for conversion
- [ ] Error handling in place
- [ ] No console errors when fetching

---

### Step 4: Adapt handleAddUser Function

**Time: 20 minutes**

Find the `handleAddUser()` function in `settings-new.jsx`:

```javascript
function handleAddUser() {
  const trimmedName = draftName.trim();

  if (!trimmedName) {
    return;
  }

  // OLD LOGIC (remove this):
  // const nextUser: ManagedUser = { ... };
  // setUsers((current) => [nextUser, ...current]);

  // NEW LOGIC (replace with this):
  handleAddUserAsync();
}

// NEW FUNCTION - ADD THIS:
async function handleAddUserAsync() {
  const trimmedName = draftName.trim();
  const trimmedEmail = draftEmail.trim();

  if (!trimmedName) {
    console.error('Name is required');
    return;
  }

  try {
    // Get auth token
    const token = await user?.getIdToken();
    if (!token) {
      console.error('Not authenticated');
      return;
    }

    const headers = {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${token}`,
    };

    // BUILD PAYLOAD using adapter
    const payload = buildUserInvitePayload(
      draftName,
      draftEmail,
      selectedTemplate,
      draftPermissions
    );

    // SEND to backend
    const res = await fetch('/api/auth/users', {
      method: 'POST',
      headers,
      body: JSON.stringify(payload),
    });

    if (res.ok) {
      // Success - clear form + refresh list
      setDraftName('');
      setDraftEmail('');
      setSelectedTemplate('operations');
      setDraftPermissions(cloneTemplatePermissions('operations'));
      setIsInviteOpen(false);
      
      // Refresh users list
      await fetchUsers();
    } else {
      const error = await res.json();
      console.error('Failed to add user:', error);
      // Show error to user (integrate with your notify system)
    }
  } catch (err) {
    console.error('Error adding user:', err);
  }
}
```

**Checklist:**
- [ ] handleAddUserAsync() created
- [ ] buildUserInvitePayload() called with correct params
- [ ] POST to /api/auth/users working
- [ ] Form clears after success
- [ ] Users list refreshes after adding

---

### Step 5: Adapt Permission Editor (Optional for Now)

**Time: 15 minutes — SKIP if template-only at first**

If implementing Individual permission editing:

```javascript
// WHEN USER EDITS PERMISSIONS (in togglePermission function):
async function savePermissionOverrides(userEmail, permissionMap) {
  try {
    const token = await user?.getIdToken();
    const headers = {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${token}`,
    };

    // Convert student format → Firestore format
    const converted = convertStudentPermissionsToFirestore(permissionMap);

    // PATCH user permissions
    const res = await fetch(`/api/auth/users/${userEmail}/permissions`, {
      method: 'POST',
      headers,
      body: JSON.stringify({ permissions: converted }),
    });

    if (res.ok) {
      console.log('Permissions saved');
      // Refresh users list to show updated summary
      await fetchUsers();
    } else {
      console.error('Failed to save permissions:', res.status);
    }
  } catch (err) {
    console.error('Error saving permissions:', err);
  }
}

// CALL THIS FROM togglePermission button onClick:
// onClick={() => {
//   togglePermission(item.key, 'read');
//   savePermissionOverrides(editingUserEmail, draftPermissions); // Add this line
// }}
```

**Checklist:**
- [ ] savePermissionOverrides() created (or skip for now)
- [ ] convertStudentPermissionsToFirestore() used
- [ ] Permissions API endpoint called
- [ ] UI updates after save

---

### Step 6: Replace Old File

**Time: 5 minutes**

```bash
cd web-dataset-collector/pages/v2

# Swap files
mv settings.js settings-old.js
mv settings-new.jsx settings.js

# Verify
ls -lh settings.js settings.backup.js settings-old.js
```

**Checklist:**
- [ ] settings.js now has new component
- [ ] Backups exist (settings.backup.js + settings-old.js)
- [ ] No broken imports

---

## Phase 3: Testing (Validation)

### Test Case 1: Create Operations User

**Time: 5 minutes**

```bash
npm run dev
# Navigate to http://localhost:3000/v2/settings

# 1. Click "Add User"
# 2. Enter:
#    Name: "Test Operations"
#    Email: "test-ops@school.edu"
#    Template: "Operations"
# 3. Click "Add User"
# 4. Check browser console for errors
# 5. Verify user appears in table

# 6. Check Firestore:
#    users/[uid] should have:
#    - role: 'viewer'
#    - permissions: {} (empty)

# ✓ PASS if user created with Operations template
```

**Expected Result:**
```
✓ User appears in table
✓ Template shows "Ops [badge]"
✓ Summary shows "Reports, Release Groups, Dashboard"
✓ Status shows "Pending"
✓ Firestore role = 'viewer'
```

**Checklist:**
- [ ] No console errors
- [ ] User appears in table
- [ ] Firestore record created

---

### Test Case 2: Create ACOP User

**Time: 5 minutes**

Same as Test 1, but:
- Name: "Test ACOP"
- Template: "ACOP"

**Expected Result:**
```
✓ Firestore role = 'admin'
✓ Template shows "ACOP [badge]"
✓ Can access Pickup System features
```

**Checklist:**
- [ ] Firestore role = 'admin'
- [ ] Summary reflects ACOP access

---

### Test Case 3: Create IT User

**Time: 5 minutes**

Same as Test 1, but:
- Name: "Test IT"
- Template: "IT"

**Expected Result:**
```
✓ Firestore role = 'owner'
✓ Template shows "IT [badge]"
✓ Can access everything
```

**Checklist:**
- [ ] Firestore role = 'owner'

---

### Test Case 4: Create Individual User (if implemented)

**Time: 10 minutes**

```bash
# 1. Click "Add User"
# 2. Enter:
#    Name: "Test Individual"
#    Email: "test-individual@school.edu"
#    Template: "Individual"
# 3. Permission matrix should now be EDITABLE
# 4. Click on some "Read" + "Write" toggles
# 5. Verify permission summary updates (e.g., "5 Read / 2 Write")
# 6. Click "Add User"

# ✓ PASS if:
#   - User created with custom permissions
#   - Firestore has permissions object with custom entries
#   - Re-loading shows "Individual Access" template
```

**Checklist:**
- [ ] Permission matrix is editable for Individual
- [ ] Summary updates as you toggle
- [ ] Custom permissions persist in Firestore

---

### Test Case 5: Refresh Page & Verify Display

**Time: 5 minutes**

```bash
# After adding a few users:
# 1. Refresh browser (F5)
# 2. Verify all users still display with correct templates
# 3. Check summaries match

# ✓ PASS if all users display correctly after refresh
```

**Checklist:**
- [ ] Users persist
- [ ] Templates correct
- [ ] No stale data

---

### Test Case 6: Verify Permissions Inheritance

**Time: 10 minutes**

```bash
# 1. Create Admin user (ACOP template)
# 2. Try to edit their permissions
# 3. Verify permissions reflect admin defaults:
#    - dashboard: view + edit (operational)
#    - pickup_admin: view + edit (full)
#    - settings: view + edit (configuration)
#    - enrollment: view only (read data, not write)

# ✓ PASS if permission matrix shows correct defaults for template
```

**Checklist:**
- [ ] Admin users have correct permission set
- [ ] Custom overrides don't break role defaults

---

## Phase 4: Deployment

### Pre-Deploy Checklist

- [ ] All 6 test cases pass
- [ ] No console errors in staging
- [ ] No API errors in logs
- [ ] Firestore data looks correct
- [ ] Team reviewed changes (optional but recommended)

### Deploy Commands

```bash
# Stage and commit
git add web-dataset-collector/pages/v2/settings.js
git add web-dataset-collector/lib/rbac-adapter.js
git commit -m "feat: refactor RBAC settings UI with polished template-based design"

# Push to origin
git push origin feat/rbac-ui-refactor

# Create PR (if using GitHub)
# Get approval from team
# Merge to main

# Production deploy
git checkout main
git pull
npm run build
npm run start
# Verify /v2/settings loads
# Monitor logs for errors
```

**Checklist:**
- [ ] Code committed
- [ ] PR created (if applicable)
- [ ] Code reviewed
- [ ] Tests passing in CI
- [ ] Deployed to production
- [ ] No errors in production logs

---

## Phase 5: Monitoring

### First 24 Hours

**Check these hourly:**
- [ ] No permission-related errors in logs
- [ ] No failed API calls to /api/auth/users
- [ ] Firestore write success rate normal
- [ ] No new bugs reported by team

**Sample monitoring queries:**
```bash
# Check logs for errors
grep "error" logs/production.log | grep -i "rbac\|permission\|user"

# Check API stats
# (In your monitoring system, e.g., Grafana)
# - /api/auth/users POST success rate
# - /api/auth/users GET latency
# - Firestore writes per second
```

### Week 1 Metrics

- [ ] Track new user creation rate (should match or exceed previous)
- [ ] Monitor permission denial rates (should stay consistent)
- [ ] Collect team feedback on UI
- [ ] Document any edge cases discovered

**Checklist:**
- [ ] Day 1: No critical errors
- [ ] Day 3: Team feedback positive
- [ ] Week 1: Stable + no regressions

---

## Troubleshooting During Implementation

### Issue: "Cannot find module 'rbac-adapter'"

**Fix:**
```bash
# Verify file exists
ls -l web-dataset-collector/lib/rbac-adapter.js

# Check import path
# Should be: import { ... } from '../../lib/rbac-adapter';
# (from web-dataset-collector/pages/v2/settings.js)
```

### Issue: "toManagedUser is not a function"

**Fix:**
```bash
# Verify export in rbac-adapter.js:
# Should have: export function toManagedUser(firebaseUser, index)

# Verify import in settings.js:
# Should have: import { toManagedUser, ... } from '../../lib/rbac-adapter';
```

### Issue: "User role is undefined after creation"

**Fix:**
```bash
# Check buildUserInvitePayload() output:
console.log('Payload:', buildUserInvitePayload(...));
// Should show: { name, email, role: 'viewer'|'admin'|'owner', permissions: {} }

# Check API response:
console.log('API response:', res.json());
// Verify backend returns role

# Check Firestore document:
// Manually view in Firebase Console
// Verify 'role' field exists
```

### Issue: "Permissions not persisting"

**Fix:**
```bash
# Check conversion function:
console.log('Converted permissions:', convertStudentPermissionsToFirestore(...));
// Should have keys like 'dashboard', 'reports', etc.

# Check API call:
console.log('Saving to:', `/api/auth/users/${userEmail}/permissions`);
// Verify endpoint exists + is POST

# Check Firestore:
// Manually verify 'permissions' field was updated
```

---

## Success Checklist (Final)

When you can check ALL of these, you're done:

- [ ] All 6 test cases pass
- [ ] No console errors
- [ ] Users persist after refresh
- [ ] Templates correctly inferred from roles
- [ ] Individual permissions editable + saveable
- [ ] Firestore schema unchanged
- [ ] API responses correct
- [ ] Team feedback positive
- [ ] Production stable for 24+ hours
- [ ] Documentation updated (optional)

---

## Rollback Plan (If Needed)

```bash
# If major issues discovered:

# Step 1: Revert code
git revert HEAD  # Assuming last commit was the rbac change
git push origin main

# Step 2: Restart server
npm run start

# Step 3: Verify
# Check that old settings page loads
# Verify users still accessible

# Step 4: Investigate
# Look at error logs
# Check Firestore for corruption
# Review commits for issues
```

---

**Good luck! You've got this.** 🚀

*Remember: This is low-risk because it's purely a UI change. Backend is untouched.*

*If stuck, refer to:*
- *RBAC_REFACTOR_ANALYSIS.md (big picture)*
- *RBAC_ARCHITECTURE_DIAGRAMS.md (data flow)*
- *lib/rbac-adapter.js (function reference)*
