#!/usr/bin/env node
/**
 * promote-admin.js — directly upsert a dashboard user as owner.
 *
 *   node backend/scripts/promote-admin.js admin@binus.edu
 *   node backend/scripts/promote-admin.js admin@binus.edu admin
 *
 * Use this when you can't access /v2/admin/rbac because your account
 * has no role yet (cold start) or got accidentally demoted.
 *
 * Reads the Firebase service account JSON from the repo root.
 */
const path = require('path');
const fs = require('fs');
const admin = require('firebase-admin');

const ROOT = path.resolve(__dirname, '..', '..');
const SA_PATH = path.join(ROOT, 'facial-attendance-binus-firebase-adminsdk.json');

if (!fs.existsSync(SA_PATH)) {
  console.error(`ERROR: Service account not found at ${SA_PATH}`);
  process.exit(1);
}

const email = (process.argv[2] || '').toLowerCase().trim();
const role  = (process.argv[3] || 'owner').toLowerCase().trim();

if (!email || !email.includes('@')) {
  console.error('Usage: node promote-admin.js <email> [role=owner|admin]');
  process.exit(1);
}
if (!['owner', 'admin'].includes(role)) {
  console.error(`Invalid role "${role}". Must be owner or admin.`);
  process.exit(1);
}

admin.initializeApp({
  credential: admin.credential.cert(require(SA_PATH)),
});

(async () => {
  const db = admin.firestore();
  const ref = db.collection('dashboard_users').doc(email);
  const snap = await ref.get();

  await ref.set({
    email,
    role,
    addedBy: snap.exists ? (snap.data().addedBy || 'cli') : 'cli',
    addedAt: snap.exists ? (snap.data().addedAt || admin.firestore.FieldValue.serverTimestamp())
                         : admin.firestore.FieldValue.serverTimestamp(),
    promotedAt: admin.firestore.FieldValue.serverTimestamp(),
    bootstrap: !snap.exists,
  }, { merge: true });

  console.log(`✓ ${snap.exists ? 'Updated' : 'Created'} dashboard_users/${email} → role=${role}`);
  process.exit(0);
})().catch((e) => {
  console.error('FAILED:', e.message);
  process.exit(1);
});
