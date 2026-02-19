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
 */
import * as faceapi from 'face-api.js';
import { db } from './firebase';
import { collection, getDocs } from 'firebase/firestore';

const MODEL_URL = '/models';
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

// ─── Descriptor database ─────────────────────────────────────────────

/**
 * Load pre-computed face descriptors from Firestore.
 * Each doc in `face_descriptors` has:
 *   { name, homeroom, grade, descriptorCount, descriptor_0: [...], descriptor_1: [...], ... }
 */
export async function loadDescriptors(onProgress) {
  onProgress?.('Loading face database…');

  const snap = await getDocs(collection(db, 'face_descriptors'));
  labeledDescriptors = [];

  snap.forEach((doc) => {
    const data = doc.data();
    const count = data.descriptorCount || 0;
    if (count === 0) return;

    // Reconstruct descriptor arrays from descriptor_0, descriptor_1, … fields
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
        // Label format: "studentId|name|homeroom|grade"
        `${doc.id}|${data.name || ''}|${data.homeroom || ''}|${data.grade || ''}`,
        descs
      )
    );
  });

  onProgress?.(`Loaded ${labeledDescriptors.length} students`);
  return labeledDescriptors.length;
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
