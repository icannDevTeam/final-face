/**
 * API client for BINUS School & Firebase interactions.
 * 
 * In production, these would call a backend proxy to protect API keys.
 * For the prototype, we call Firebase directly from the client.
 */
import { db } from './firebase';
import {
  doc,
  getDoc,
  setDoc,
  collection,
  query,
  where,
  getDocs,
  orderBy,
  serverTimestamp,
} from 'firebase/firestore';

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
 */
export async function lookupStudent(studentId) {
  try {
    const docRef = doc(db, 'students', String(studentId));
    const snap = await getDoc(docRef);
    if (snap.exists()) {
      return { id: snap.id, ...snap.data() };
    }

    // Also check student_metadata collection (keyed by employeeNo = studentId)
    const metaRef = doc(db, 'student_metadata', String(studentId));
    const metaSnap = await getDoc(metaRef);
    if (metaSnap.exists()) {
      return { id: metaSnap.id, ...metaSnap.data() };
    }

    return null;
  } catch (err) {
    console.error('Student lookup failed:', err);
    throw new Error('Could not look up student. Please check your connection.');
  }
}

// ─── Attendance check-in ──────────────────────────────────────────────

/**
 * Record a proximity-based attendance check-in.
 *
 * Writes to Firebase Firestore: attendance/{date}/records/{studentId}
 */
export async function checkIn(studentId, studentName, location, metadata = {}) {
  const date = getWIBDate();
  const timestamp = getWIBTimestamp();
  const time = getWIBTime();

  // Determine if late (after 08:15 WIB)
  const [h, m] = time.split(':').map(Number);
  const isLate = h > 8 || (h === 8 && m > 15);

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
    updatedAt: new Date().toISOString(),
  };

  try {
    const docRef = doc(db, 'attendance', date, 'records', String(studentId));
    await setDoc(docRef, record);

    // Update day summary
    const dayRef = doc(db, 'attendance', date);
    await setDoc(dayRef, { lastUpdated: new Date().toISOString() }, { merge: true });

    return { success: true, record };
  } catch (err) {
    console.error('Check-in failed:', err);
    throw new Error('Failed to record attendance. Please try again.');
  }
}

// ─── Check existing attendance ────────────────────────────────────────

/**
 * Check if a student has already checked in today.
 */
export async function getExistingCheckIn(studentId) {
  const date = getWIBDate();
  try {
    const docRef = doc(db, 'attendance', date, 'records', String(studentId));
    const snap = await getDoc(docRef);
    if (snap.exists()) {
      return snap.data();
    }
    return null;
  } catch {
    return null;
  }
}

/**
 * Get today's attendance summary.
 */
export async function getTodaySummary() {
  const date = getWIBDate();
  try {
    const colRef = collection(db, 'attendance', date, 'records');
    const q = query(colRef, orderBy('timestamp', 'asc'));
    const snap = await getDocs(q);

    const records = [];
    snap.forEach((doc) => records.push({ id: doc.id, ...doc.data() }));

    return {
      date,
      total: records.length,
      present: records.filter((r) => !r.late).length,
      late: records.filter((r) => r.late).length,
      records,
    };
  } catch {
    return { date, total: 0, present: 0, late: 0, records: [] };
  }
}
