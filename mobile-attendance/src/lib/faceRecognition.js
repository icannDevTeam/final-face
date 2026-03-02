/**
 * Face recognition engine using face-api.js
 *
 * Workflow:
 *  1. loadModels()        → load SSD + landmark + recognition nets
 *  2. loadDescriptors()   → fetch pre-computed descriptors from Firestore
 *  3. detectAndMatch()    → run detection on a video frame → find best match
 *
 * Face descriptors (128-d Float32Array) are stored in Firestore:
 *   face_descriptors/{studentId} → { name, homeroom, grade, descriptor: number[] }
 *
 * CACHING: Descriptors are cached in IndexedDB (30 min TTL) to avoid
 * re-reading every Firestore doc on each ScanPage mount. Models are
 * cached in memory (loaded once per session) + SW cache-first.
 */
import * as faceapi from 'face-api.js';
import { db } from './firebase';
import { collection, getDocs } from 'firebase/firestore';
import { idbGet, idbSet, memGet, memSet, TTL } from './cache';

const MODEL_URL = '/models';
const DESCRIPTORS_CACHE_KEY = 'face_descriptors';

let modelsLoaded = false;
let labeledDescriptors = []; // array of faceapi.LabeledFaceDescriptors

// ─── Model loading ────────────────────────────────────────────────────

export async function loadModels(onProgress) {
  if (modelsLoaded) return;

  onProgress?.('Loading face detection model…');
  await faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL);

  onProgress?.('Loading landmark model…');
  await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);

  onProgress?.('Loading recognition model…');
  await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);

  modelsLoaded = true;
  onProgress?.('Models ready');
}

export function isModelsLoaded() {
  return modelsLoaded;
}

// ─── Descriptor database (with IndexedDB caching) ────────────────────

/**
 * Serialize labeled descriptors to a plain array for IndexedDB storage.
 * Float32Arrays → regular arrays so they survive structured clone.
 */
function serializeDescriptors(descriptors) {
  return descriptors.map((ld) => ({
    label: ld.label,
    descriptors: ld.descriptors.map((d) => Array.from(d)),
  }));
}

/**
 * Deserialize cached data back into faceapi.LabeledFaceDescriptors.
 */
function deserializeDescriptors(cached) {
  return cached.map((item) =>
    new faceapi.LabeledFaceDescriptors(
      item.label,
      item.descriptors.map((arr) => new Float32Array(arr))
    )
  );
}

/**
 * Load pre-computed face descriptors from Firestore.
 * Uses IndexedDB cache (30 min TTL) to avoid redundant Firestore reads.
 *
 * Each doc in `face_descriptors` has:
 *   { name, homeroom, grade, descriptorCount, descriptor_0: [...], ... }
 */
export async function loadDescriptors(onProgress) {
  // ── Check in-memory first (instant if already loaded this session) ──
  if (labeledDescriptors.length > 0) {
    onProgress?.(`Loaded ${labeledDescriptors.length} students (memory)`);
    return labeledDescriptors.length;
  }

  // ── Check IndexedDB cache (survives page reloads) ──────────────────
  onProgress?.('Loading face database…');

  try {
    const cached = await idbGet(DESCRIPTORS_CACHE_KEY, TTL.DESCRIPTORS);
    if (cached && cached.length > 0) {
      labeledDescriptors = deserializeDescriptors(cached);
      onProgress?.(`Loaded ${labeledDescriptors.length} students (cached)`);
      return labeledDescriptors.length;
    }
  } catch {
    // Cache read failed — fall through to Firestore
  }

  // ── Fetch fresh from Firestore ─────────────────────────────────────
  onProgress?.('Fetching face database from server…');

  const snap = await getDocs(collection(db, 'face_descriptors'));
  labeledDescriptors = [];

  snap.forEach((doc) => {
    const data = doc.data();
    const count = data.descriptorCount || 0;
    if (count === 0) return;

    const descs = [];
    for (let i = 0; i < count; i++) {
      const arr = data[`descriptor_${i}`];
      if (arr && arr.length === 128) {
        descs.push(new Float32Array(arr));
      }
    }

    if (descs.length === 0) return;

    labeledDescriptors.push(
      new faceapi.LabeledFaceDescriptors(
        `${doc.id}|${data.name || ''}|${data.homeroom || ''}|${data.grade || ''}`,
        descs
      )
    );
  });

  // ── Write to IndexedDB for next time ──────────────────────────────
  if (labeledDescriptors.length > 0) {
    idbSet(DESCRIPTORS_CACHE_KEY, serializeDescriptors(labeledDescriptors)).catch(() => {});
  }

  onProgress?.(`Loaded ${labeledDescriptors.length} students`);
  return labeledDescriptors.length;
}

/**
 * Force-refresh descriptors from Firestore (bypasses cache).
 * Call when new students are enrolled.
 */
export async function refreshDescriptors(onProgress) {
  labeledDescriptors = [];
  idbSet(DESCRIPTORS_CACHE_KEY, null).catch(() => {});
  return loadDescriptors(onProgress);
}

export function getLoadedCount() {
  return labeledDescriptors.length;
}

// ─── Face detection & matching ────────────────────────────────────────

/**
 * Parse the label string back into student metadata.
 */
function parseLabel(label) {
  const [id, name, homeroom, grade] = label.split('|');
  return { id, name, homeroom, grade };
}

/**
 * Detect the best face in a video/canvas element and match it against
 * the loaded descriptors.
 *
 * Returns:
 *   { matched: true, student: { id, name, homeroom, grade }, distance, confidence }
 *   or { matched: false, reason: '...' }
 */
export async function detectAndMatch(videoEl, options = {}) {
  const {
    matchThreshold = 0.5, // Euclidean distance threshold (lower = stricter)
  } = options;

  if (!modelsLoaded) throw new Error('Models not loaded');
  if (!labeledDescriptors.length) throw new Error('No face descriptors loaded');

  // Detect single best face with landmarks + descriptor
  const result = await faceapi
    .detectSingleFace(videoEl, new faceapi.SsdMobilenetv1Options({ minConfidence: 0.5 }))
    .withFaceLandmarks()
    .withFaceDescriptor();

  if (!result) {
    return { matched: false, reason: 'no_face', message: 'No face detected' };
  }

  // Match against database
  const matcher = new faceapi.FaceMatcher(labeledDescriptors, matchThreshold);
  const bestMatch = matcher.findBestMatch(result.descriptor);

  if (bestMatch.label === 'unknown') {
    return {
      matched: false,
      reason: 'unknown',
      message: 'Face not recognized',
      distance: bestMatch.distance,
    };
  }

  const student = parseLabel(bestMatch.label);
  const confidence = Math.round((1 - bestMatch.distance) * 100);

  return {
    matched: true,
    student,
    distance: bestMatch.distance,
    confidence,
    detection: result.detection,
  };
}

/**
 * Lightweight face detection only (for showing bounding box overlay).
 */
export async function detectFace(videoEl) {
  if (!modelsLoaded) return null;
  return faceapi.detectSingleFace(
    videoEl,
    new faceapi.SsdMobilenetv1Options({ minConfidence: 0.4 })
  );
}
