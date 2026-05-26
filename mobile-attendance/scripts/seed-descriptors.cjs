#!/usr/bin/env node
/**
 * seed-descriptors.cjs
 *
 * Downloads student face photos from Firebase Storage, computes 128-d
 * face descriptors using face-api.js + node-canvas, and stores them
 * in Firestore `face_descriptors/{studentId}`.
 *
 * Usage:
 *   cd mobile-attendance
 *   node scripts/seed-descriptors.cjs
 *
 * Prerequisites:
 *   npm install @vladmandic/face-api @tensorflow/tfjs canvas firebase-admin
 */

const path = require('path');
const fs = require('fs');
const admin = require('firebase-admin');

// Force pure-JS TF backend — block native tfjs-node from loading
const Module = require('module');
const origResolveFilename = Module._resolveFilename;
Module._resolveFilename = function (request, ...args) {
  if (request === '@tensorflow/tfjs-node') {
    return origResolveFilename.call(this, '@tensorflow/tfjs', ...args);
  }
  return origResolveFilename.call(this, request, ...args);
};

const tf = require('@tensorflow/tfjs');

// ── Canvas polyfill for Node.js ──
const canvas = require('canvas');
const { Canvas, Image, ImageData } = canvas;

// @vladmandic/face-api with node-canvas support
const faceapi = require('@vladmandic/face-api');

// Patch face-api to use node-canvas
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

// ── Firebase Admin init ──
const SERVICE_ACCOUNT_PATH = path.resolve(
  __dirname,
  '../../facial-attendance-binus-firebase-adminsdk.json'
);

if (!fs.existsSync(SERVICE_ACCOUNT_PATH)) {
  console.error('❌ Service account key not found at:', SERVICE_ACCOUNT_PATH);
  process.exit(1);
}

const serviceAccount = JSON.parse(fs.readFileSync(SERVICE_ACCOUNT_PATH, 'utf8'));

admin.initializeApp({
  credential: admin.credential.cert(serviceAccount),
  storageBucket: 'facial-attendance-binus.firebasestorage.app',
});

const db = admin.firestore();
const bucket = admin.storage().bucket();

// ── Model paths ──
const MODEL_DIR = path.resolve(__dirname, '../public/models');

// ── CLI filters ──
// Usage:
//   node scripts/seed-descriptors.cjs                          # all students
//   node scripts/seed-descriptors.cjs --grade 5                # only Grade 5 (5A, 5B, …)
//   node scripts/seed-descriptors.cjs --homeroom 5A            # only homeroom 5A
//   node scripts/seed-descriptors.cjs --grade 5 --limit 50     # cap to 50 students
//   node scripts/seed-descriptors.cjs --grade 5 --force        # re-seed even if descriptors exist
function parseArgs(argv) {
  const out = { grade: null, homeroom: null, limit: null, force: false };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--grade' || a === '-g')        out.grade = String(argv[++i] || '').trim();
    else if (a === '--homeroom' || a === '-r') out.homeroom = String(argv[++i] || '').trim();
    else if (a === '--limit' || a === '-n')   out.limit = parseInt(argv[++i], 10);
    else if (a === '--force' || a === '-f')   out.force = true;
  }
  return out;
}
const ARGS = parseArgs(process.argv.slice(2));

function matchesFilters(student) {
  const hr = String(student.homeroom || '').trim();
  if (ARGS.homeroom) {
    return hr.toLowerCase() === ARGS.homeroom.toLowerCase();
  }
  if (ARGS.grade) {
    // Match by leading digit(s) of homeroom (e.g. '5A' → grade 5) or by gradeCode field.
    const m = hr.match(/^(\d+)/);
    const fromHr = m ? m[1] : null;
    const fromCode = String(student.gradeCode || student.grade || '').match(/^(\d+)/);
    return fromHr === ARGS.grade || (fromCode && fromCode[1] === ARGS.grade);
  }
  return true;
}

// ── Main ──
async function main() {
  console.log('🧠 Loading face-api.js models...');
  await faceapi.nets.ssdMobilenetv1.loadFromDisk(MODEL_DIR);
  await faceapi.nets.faceLandmark68Net.loadFromDisk(MODEL_DIR);
  await faceapi.nets.faceRecognitionNet.loadFromDisk(MODEL_DIR);
  console.log('✅ Models loaded\n');

  // 1. List all students from Firestore
  console.log('📋 Fetching student list from Firestore...');
  const studentsSnap = await db.collection('students').get();
  const allStudents = [];
  studentsSnap.forEach((doc) => {
    allStudents.push({ id: doc.id, ...doc.data() });
  });
  console.log(`   Found ${allStudents.length} students total`);

  // Apply filters
  let students = allStudents.filter(matchesFilters);
  const filterDesc = ARGS.homeroom
    ? `homeroom=${ARGS.homeroom}`
    : ARGS.grade
    ? `grade=${ARGS.grade}`
    : 'no filter';
  console.log(`   Filter: ${filterDesc} → ${students.length} candidates`);

  if (!ARGS.force) {
    // Skip students that already have descriptors stored. One read per
    // candidate is fine for a trial run — beats redoing minutes of work.
    const existingIds = new Set();
    const batches = [];
    for (let i = 0; i < students.length; i += 10) {
      batches.push(students.slice(i, i + 10));
    }
    for (const batch of batches) {
      const refs = batch.map((s) => db.collection('face_descriptors').doc(s.id));
      const snaps = await db.getAll(...refs);
      snaps.forEach((sn) => { if (sn.exists) existingIds.add(sn.id); });
    }
    if (existingIds.size > 0) {
      console.log(`   Skipping ${existingIds.size} already-seeded (use --force to re-run)`);
      students = students.filter((s) => !existingIds.has(s.id));
    }
  }

  if (ARGS.limit && students.length > ARGS.limit) {
    students = students.slice(0, ARGS.limit);
    console.log(`   --limit applied: capping to ${students.length}`);
  }
  console.log(`   → Processing ${students.length} students\n`);

  if (students.length === 0) {
    console.log('No students found. Nothing to do.');
    return;
  }

  let success = 0;
  let skipped = 0;
  let failed = 0;

  for (const student of students) {
    const { id, name, homeroom, gradeCode } = student;
    console.log(`──────────────────────────────────────`);
    console.log(`👤 ${name} (ID: ${id}, Class: ${homeroom || '?'})`);

    try {
      // 2. List photos in Storage: face_dataset/{homeroom}/{name}/
      const prefix = `face_dataset/${homeroom}/${name}/`;
      const [files] = await bucket.getFiles({ prefix });

      const imageFiles = files.filter((f) =>
        /\.(jpg|jpeg|png)$/i.test(f.name)
      );

      if (imageFiles.length === 0) {
        console.log(`   ⚠️  No photos found at ${prefix}`);
        skipped++;
        continue;
      }

      console.log(`   📁 Found ${imageFiles.length} photos`);

      // 3. Download and compute descriptors for each photo
      const descriptors = [];

      for (const file of imageFiles.slice(0, 5)) {
        // Limit to 5 photos per student for efficiency
        try {
          const [buffer] = await file.download();
          const img = await canvas.loadImage(buffer);

          const detection = await faceapi
            .detectSingleFace(img, new faceapi.SsdMobilenetv1Options({ minConfidence: 0.5 }))
            .withFaceLandmarks()
            .withFaceDescriptor();

          if (detection) {
            descriptors.push(Array.from(detection.descriptor));
            process.stdout.write('   ✓ ');
          } else {
            process.stdout.write('   ✗ ');
          }
        } catch (e) {
          process.stdout.write('   ✗ ');
        }
      }

      console.log(`\n   📊 Got ${descriptors.length}/${imageFiles.length} descriptors`);

      if (descriptors.length === 0) {
        console.log(`   ⚠️  No faces detected in any photo`);
        skipped++;
        continue;
      }

      // 4. Store in Firestore — Firestore doesn't allow nested arrays,
      //    so we store each descriptor as a separate field (descriptor_0, descriptor_1, …)
      const docData = {
        name: name || '',
        homeroom: homeroom || '',
        grade: gradeCode || '',
        descriptorCount: descriptors.length,
        photoCount: imageFiles.length,
        updatedAt: new Date().toISOString(),
      };
      for (let i = 0; i < descriptors.length; i++) {
        docData[`descriptor_${i}`] = descriptors[i]; // flat number[]
      }

      await db.collection('face_descriptors').doc(id).set(docData);

      console.log(`   ✅ Saved ${descriptors.length} descriptors to Firestore`);
      success++;
    } catch (err) {
      console.log(`   ❌ Error: ${err.message}`);
      failed++;
    }
  }

  console.log(`\n══════════════════════════════════════`);
  console.log(`📊 Summary:`);
  console.log(`   ✅ Success: ${success}`);
  console.log(`   ⚠️  Skipped: ${skipped}`);
  console.log(`   ❌ Failed: ${failed}`);
  console.log(`   Total: ${students.length}`);
}

main()
  .then(() => process.exit(0))
  .catch((err) => {
    console.error('Fatal error:', err);
    process.exit(1);
  });
