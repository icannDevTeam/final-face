/**
 * Caching utilities for the mobile attendance app.
 *
 * Three tiers:
 *  1. Memory cache  — instant, survives within page session
 *  2. localStorage  — fast, persists across reloads (small JSON data)
 *  3. IndexedDB     — large blobs (face descriptors)
 *
 * Every cached value stores { data, ts } where ts = Date.now() at write time.
 * TTL (time-to-live) is checked on read; stale entries return null.
 */

// ─── Memory cache ─────────────────────────────────────────────────────

const _mem = new Map();

/**
 * Get a value from memory cache.
 * Returns null if missing or expired.
 */
export function memGet(key, ttlMs) {
  const entry = _mem.get(key);
  if (!entry) return null;
  if (ttlMs && Date.now() - entry.ts > ttlMs) {
    _mem.delete(key);
    return null;
  }
  return entry.data;
}

/**
 * Set a value in memory cache.
 */
export function memSet(key, data) {
  _mem.set(key, { data, ts: Date.now() });
}

/**
 * Delete from memory cache.
 */
export function memDel(key) {
  _mem.delete(key);
}

// ─── localStorage cache ──────────────────────────────────────────────

const LS_PREFIX = 'binus_cache_';

/**
 * Get a JSON-serialisable value from localStorage.
 * Returns null if missing, expired, or parse error.
 */
export function lsGet(key, ttlMs) {
  try {
    const raw = localStorage.getItem(LS_PREFIX + key);
    if (!raw) return null;
    const entry = JSON.parse(raw);
    if (ttlMs && Date.now() - entry.ts > ttlMs) {
      localStorage.removeItem(LS_PREFIX + key);
      return null;
    }
    return entry.data;
  } catch {
    return null;
  }
}

/**
 * Set a JSON-serialisable value in localStorage.
 */
export function lsSet(key, data) {
  try {
    localStorage.setItem(LS_PREFIX + key, JSON.stringify({ data, ts: Date.now() }));
  } catch {
    // Storage full — silently ignore
  }
}

/**
 * Delete from localStorage.
 */
export function lsDel(key) {
  localStorage.removeItem(LS_PREFIX + key);
}

// ─── IndexedDB cache (for large data like face descriptors) ──────────

const IDB_NAME = 'binus_attendance_cache';
const IDB_STORE = 'cache';
const IDB_VERSION = 1;

function openIDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(IDB_NAME, IDB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(IDB_STORE)) {
        db.createObjectStore(IDB_STORE);
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

/**
 * Get a value from IndexedDB.
 * Returns null if missing, expired, or error.
 */
export async function idbGet(key, ttlMs) {
  try {
    const db = await openIDB();
    return new Promise((resolve) => {
      const tx = db.transaction(IDB_STORE, 'readonly');
      const store = tx.objectStore(IDB_STORE);
      const req = store.get(key);
      req.onsuccess = () => {
        const entry = req.result;
        if (!entry) return resolve(null);
        if (ttlMs && Date.now() - entry.ts > ttlMs) {
          // Stale — clean up in background
          idbDel(key).catch(() => {});
          return resolve(null);
        }
        resolve(entry.data);
      };
      req.onerror = () => resolve(null);
    });
  } catch {
    return null;
  }
}

/**
 * Set a value in IndexedDB.
 */
export async function idbSet(key, data) {
  try {
    const db = await openIDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(IDB_STORE, 'readwrite');
      const store = tx.objectStore(IDB_STORE);
      store.put({ data, ts: Date.now() }, key);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  } catch {
    // IndexedDB not available — silently ignore
  }
}

/**
 * Delete from IndexedDB.
 */
export async function idbDel(key) {
  try {
    const db = await openIDB();
    return new Promise((resolve) => {
      const tx = db.transaction(IDB_STORE, 'readwrite');
      const store = tx.objectStore(IDB_STORE);
      store.delete(key);
      tx.oncomplete = () => resolve();
      tx.onerror = () => resolve();
    });
  } catch {
    // ignore
  }
}

// ─── Cache TTL presets (milliseconds) ────────────────────────────────

export const TTL = {
  DESCRIPTORS: 24 * 60 * 60 * 1000, // 24 hours — face descriptors from Firestore
  STUDENT:     60 * 60 * 1000,    // 1 hour — student profile data
  METADATA:    60 * 60 * 1000,    // 1 hour — student_metadata for BINUS API
  CHECKIN:     12 * 60 * 60 * 1000, // 12 hours — today's check-in dedup
  PROXIMITY:   5 * 1000,          // 5 sec — geolocation proximity result
};
