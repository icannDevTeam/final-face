/**
 * API client for BINUS School & Firebase interactions.
 *
 * CACHING STRATEGY:
 *  - Student lookups:    memory + localStorage (1 hr TTL)
 *  - Student metadata:   memory + localStorage (1 hr TTL)
 *  - Today's check-in:   memory (12 hr TTL) — reset on new day
 *  - Attendance summary:  memory (30 sec TTL) — lightweight refresh
 *
 * This avoids redundant Firestore reads on every navigation / check-in.
 */
import { db } from './firebase';
import {
  doc,
  getDoc,
  setDoc,
  collection,
  query,
  getDocs,
  orderBy,
} from 'firebase/firestore';
import { memGet, memSet, lsGet, lsSet, TTL } from './cache';

// ─── WIB helpers ──────────────────────────────────────────────────────
function getWIBNow() {
  return new Date(Date.now() + 7 * 3600 * 1000);
}

function getWIBDate() {
  return getWIBNow().toISOString().slice(0, 10);
}

function getWIBTime() {
  return getWIBNow().toISOString().slice(11, 19);
}

function getWIBTimestamp() {
  const now = getWIBNow();
  return `${now.toISOString().slice(0, 10)} ${now.toISOString().slice(11, 19)}`;
}

// ─── Student lookup ───────────────────────────────────────────────────

/**
 * Look up a student from the Firestore `students` collection.
 * Returns student metadata if found, null otherwise.
 * Cached in memory + localStorage (1 hr TTL).
 */
export async function lookupStudent(studentId) {
  const cacheKey = `student_${studentId}`;

  // ── Memory cache (instant) ────────────────────────────────────────
  const memCached = memGet(cacheKey, TTL.STUDENT);
  if (memCached) return memCached;

  // ── localStorage cache (survives reload) ──────────────────────────
  const lsCached = lsGet(cacheKey, TTL.STUDENT);
  if (lsCached) {
    memSet(cacheKey, lsCached); // promote to memory
    return lsCached;
  }

  // ── Firestore fetch ───────────────────────────────────────────────
  try {
    const docRef = doc(db, 'students', String(studentId));
    const snap = await getDoc(docRef);
    if (snap.exists()) {
      const result = { id: snap.id, ...snap.data() };
      memSet(cacheKey, result);
      lsSet(cacheKey, result);
      return result;
    }

    // Also check student_metadata collection (keyed by employeeNo = studentId)
    const metaRef = doc(db, 'student_metadata', String(studentId));
    const metaSnap = await getDoc(metaRef);
    if (metaSnap.exists()) {
      const result = { id: metaSnap.id, ...metaSnap.data() };
      memSet(cacheKey, result);
      lsSet(cacheKey, result);
      return result;
    }

    return null;
  } catch (err) {
    console.error('Student lookup failed:', err);
    throw new Error('Could not look up student. Please check your connection.');
  }
}

// ─── Attendance check-in ──────────────────────────────────────────────

/**
 * Record a face-recognition attendance check-in.
 *
 * Writes to Firebase Firestore: attendance/{date}/records/{studentId}
 *
 * DEDUP LOGIC: Checks if the student already has a record for today
 * (from ANY source: mobile, hikvision, etc). If they do, returns the
 * existing record without overwriting. This prevents double-attendance.
 */
export async function checkIn(studentId, studentName, location, metadata = {}) {
  const date = getWIBDate();
  const timestamp = getWIBTimestamp();
  const time = getWIBTime();

  // ── Dedup: check for existing record from ANY source ──────────────
  // Uses memory cache first to avoid Firestore hit on repeated scans
  const existing = await getExistingCheckIn(studentId);
  if (existing) {
    return {
      success: true,
      record: { ...existing, alreadyDone: true },
      alreadyDone: true,
    };
  }

  // Determine if late (after 07:30 WIB)
  const [h, m] = time.split(':').map(Number);
  const isLate = h > 7 || (h === 7 && m > 30);

  const record = {
    name: studentName,
    employeeNo: String(studentId),
    timestamp,
    status: isLate ? 'Late' : 'Present',
    late: isLate,
    homeroom: metadata.homeroom || '',
    grade: metadata.grade || metadata.gradeCode || '',
    source: 'mobile_face',
    location: location.lat ? {
      lat: location.lat,
      lng: location.lng,
      accuracy: location.accuracy,
    } : null,
    distance: location.distance || 0,
    ...(typeof metadata.confidence === 'number' ? { confidence: metadata.confidence / 100 } : {}),
    updatedAt: new Date().toISOString(),
  };

  try {
    const docRef = doc(db, 'attendance', date, 'records', String(studentId));
    await setDoc(docRef, record);

    // Update day summary
    const dayRef = doc(db, 'attendance', date);
    await setDoc(dayRef, { lastUpdated: new Date().toISOString() }, { merge: true });

    // ── Cache the check-in so we skip Firestore on subsequent scans ──
    const checkinKey = `checkin_${date}_${studentId}`;
    memSet(checkinKey, record);
    lsSet(checkinKey, record);

    // ── Also push to BINUS School API (best-effort, non-blocking) ───
    syncToBinusApi(studentId).catch((err) =>
      console.warn('BINUS API sync (non-critical):', err.message)
    );

    return { success: true, record };
  } catch (err) {
    console.error('Check-in failed:', err);
    throw new Error('Failed to record attendance. Please try again.');
  }
}

// ─── BINUS School API sync ────────────────────────────────────────────

/**
 * Look up student_metadata from Firestore and push attendance to BINUS API
 * via the Vercel serverless proxy at /api/binus-attendance.
 * Metadata is cached in memory + localStorage (1 hr TTL).
 */
async function syncToBinusApi(studentId) {
  try {
    // ── Cached metadata lookup ──────────────────────────────────────
    const metaCacheKey = `meta_${studentId}`;
    let meta = memGet(metaCacheKey, TTL.METADATA);

    if (!meta) {
      meta = lsGet(metaCacheKey, TTL.METADATA);
      if (meta) {
        memSet(metaCacheKey, meta); // promote to memory
      }
    }

    if (!meta) {
      const metaRef = doc(db, 'student_metadata', String(studentId));
      const metaSnap = await getDoc(metaRef);

      if (!metaSnap.exists()) {
        console.warn(`No student_metadata for ${studentId} — skipping BINUS API`);
        return;
      }

      meta = metaSnap.data();
      memSet(metaCacheKey, meta);
      lsSet(metaCacheKey, meta);
    }

    const idStudent = meta.idStudent || meta.IdStudent || '';
    const idBinusian = meta.idBinusian || meta.IdBinusian || '';

    if (!idStudent) {
      console.warn(`No IdStudent in metadata for ${studentId} — skipping BINUS API`);
      return;
    }

    const resp = await fetch('/api/binus-attendance', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ IdStudent: idStudent, IdBinusian: idBinusian }),
    });

    const result = await resp.json();
    if (result.success) {
      console.log(`☁️ BINUS API: attendance synced for ${studentId}`);
    } else {
      console.warn(`BINUS API error: ${result.error}`);
    }
  } catch (err) {
    console.warn('BINUS API sync failed:', err.message);
  }
}

// ─── Check existing attendance ────────────────────────────────────────

/**
 * Check if a student has already checked in today.
 * Uses memory + localStorage cache to avoid Firestore hit on repeated scans.
 */
export async function getExistingCheckIn(studentId) {
  const date = getWIBDate();
  const cacheKey = `checkin_${date}_${studentId}`;

  // ── Memory cache ──────────────────────────────────────────────────
  const memCached = memGet(cacheKey, TTL.CHECKIN);
  if (memCached) return memCached;

  // ── localStorage cache ────────────────────────────────────────────
  const lsCached = lsGet(cacheKey, TTL.CHECKIN);
  if (lsCached) {
    memSet(cacheKey, lsCached);
    return lsCached;
  }

  // ── Firestore fetch ───────────────────────────────────────────────
  try {
    const docRef = doc(db, 'attendance', date, 'records', String(studentId));
    const snap = await getDoc(docRef);
    if (snap.exists()) {
      const data = snap.data();
      memSet(cacheKey, data);
      lsSet(cacheKey, data);
      return data;
    }
    return null;
  } catch {
    return null;
  }
}

/**
 * Get today's attendance summary.
 * Cached in memory for 30 seconds to avoid rapid re-fetches.
 */
export async function getTodaySummary() {
  const date = getWIBDate();
  const cacheKey = `summary_${date}`;

  // ── Memory cache (30 sec) ─────────────────────────────────────────
  const cached = memGet(cacheKey, 30_000);
  if (cached) return cached;

  try {
    const colRef = collection(db, 'attendance', date, 'records');
    const q = query(colRef, orderBy('timestamp', 'asc'));
    const snap = await getDocs(q);

    const records = [];
    snap.forEach((doc) => records.push({ id: doc.id, ...doc.data() }));

    const result = {
      date,
      total: records.length,
      present: records.filter((r) => !r.late).length,
      late: records.filter((r) => r.late).length,
      records,
    };

    memSet(cacheKey, result);
    return result;
  } catch {
    return { date, total: 0, present: 0, late: 0, records: [] };
  }
}

/**
 * Get recent attendance records for a specific student (by employeeNo).
 * Scans the last `days` attendance date documents.
 * Returns [{ date, timestamp, status, late, source }] sorted newest-first.
 * Cached in memory for 5 minutes.
 */
export async function getAttendanceHistory(studentId, days = 14) {
  const cacheKey = `history_${studentId}`;
  const cached = memGet(cacheKey, 5 * 60_000);
  if (cached) return cached;

  const results = [];
  const today = getWIBNow();

  for (let i = 0; i < days; i++) {
    const d = new Date(today);
    d.setDate(d.getDate() - i);
    const dateStr = d.toISOString().slice(0, 10);

    try {
      const docRef = doc(db, 'attendance', dateStr, 'records', String(studentId));
      const snap = await getDoc(docRef);
      if (snap.exists()) {
        const data = snap.data();
        results.push({
          date: dateStr,
          timestamp: data.timestamp || dateStr,
          startTime: (data.timestamp || '').split(' ')[1] || '—',
          status: data.status || (data.late ? 'Late' : 'Present'),
          late: !!data.late,
          source: data.source || 'unknown',
        });
      }
    } catch {
      // Skip dates we can't read
    }
  }

  memSet(cacheKey, results);
  return results;
}
