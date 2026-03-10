/**
 * Liveness detection for anti-spoofing.
 *
 * Two complementary checks:
 *  1. Blink detection  — measures Eye Aspect Ratio (EAR) from 68-point landmarks.
 *     A real person blinks naturally; a photo never does.
 *  2. Motion detection — tracks landmark positions across frames.
 *     A real face has micro-movements; a static photo does not.
 *
 * Usage:
 *   const liveness = createLivenessChecker();
 *   // On each frame with landmarks:
 *   const status = liveness.update(landmarks);
 *   // status.passed === true when liveness confirmed
 */
import * as faceapi from 'face-api.js';

// ─── Configuration ────────────────────────────────────────────────────

const EAR_BLINK_THRESHOLD = 0.24; // EAR below this = eye closed
const EAR_OPEN_THRESHOLD = 0.30;  // EAR above this = eye open (debounce)
const REQUIRED_BLINKS = 1;        // minimum blinks needed
const MOTION_FRAMES = 8;          // frames to accumulate for motion check
const MOTION_THRESHOLD = 1.5;     // min avg landmark displacement (px) across frames
const MAX_MOTION = 40;            // reject excessive motion (phone shaking / video playback)
const LIVENESS_TIMEOUT = 10_000;  // ms — fail if not proved alive within this

// 68-point landmark indices for eyes (0-indexed)
// Left eye: points 36–41, Right eye: points 42–47
const LEFT_EYE = [36, 37, 38, 39, 40, 41];
const RIGHT_EYE = [42, 43, 44, 45, 46, 47];

// ─── Eye Aspect Ratio ─────────────────────────────────────────────────

/**
 * Compute Eye Aspect Ratio for one eye.
 * EAR = (|p1-p5| + |p2-p4|) / (2 * |p0-p3|)
 * where p0..p5 are the 6 landmark points around an eye.
 */
function eyeAspectRatio(points, indices) {
  const p = indices.map((i) => points[i]);
  const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

  const vertical1 = dist(p[1], p[5]);
  const vertical2 = dist(p[2], p[4]);
  const horizontal = dist(p[0], p[3]);

  if (horizontal === 0) return 0.3; // avoid division by zero
  return (vertical1 + vertical2) / (2 * horizontal);
}

/**
 * Average EAR of both eyes.
 */
function avgEAR(landmarks) {
  const pts = landmarks.positions;
  const left = eyeAspectRatio(pts, LEFT_EYE);
  const right = eyeAspectRatio(pts, RIGHT_EYE);
  return (left + right) / 2;
}

// ─── Landmark centroid for motion tracking ────────────────────────────

/** Average (x, y) of key stable landmarks: nose bridge + eye corners. */
function landmarkCentroid(landmarks) {
  const pts = landmarks.positions;
  // Use 6 stable points: inner/outer eye corners + nose tip
  const indices = [36, 39, 42, 45, 30, 33];
  let sx = 0, sy = 0;
  for (const i of indices) {
    sx += pts[i].x;
    sy += pts[i].y;
  }
  return { x: sx / indices.length, y: sy / indices.length };
}

// ─── Liveness checker factory ─────────────────────────────────────────

/**
 * Create a liveness checker instance.
 * Call .update(landmarks) on every frame that has detected landmarks.
 * Returns { passed, blinkCount, motionPassed, blinkPassed, prompt }
 */
export function createLivenessChecker() {
  let blinkCount = 0;
  let eyeWasClosed = false;
  let motionHistory = [];  // array of { x, y } centroids
  let motionPassed = false;
  let blinkPassed = false;
  let startTime = null;
  let timedOut = false;

  function reset() {
    blinkCount = 0;
    eyeWasClosed = false;
    motionHistory = [];
    motionPassed = false;
    blinkPassed = false;
    startTime = null;
    timedOut = false;
  }

  function update(landmarks) {
    if (!landmarks) return getStatus();

    if (!startTime) startTime = Date.now();

    // Check timeout
    if (Date.now() - startTime > LIVENESS_TIMEOUT) {
      timedOut = true;
      return getStatus();
    }

    // ── Blink detection ───────────────────────────────────────
    const ear = avgEAR(landmarks);

    if (ear < EAR_BLINK_THRESHOLD && !eyeWasClosed) {
      eyeWasClosed = true;
    } else if (ear > EAR_OPEN_THRESHOLD && eyeWasClosed) {
      // Eye re-opened → complete blink
      eyeWasClosed = false;
      blinkCount++;
      if (blinkCount >= REQUIRED_BLINKS) {
        blinkPassed = true;
      }
    }

    // ── Motion detection ──────────────────────────────────────
    const centroid = landmarkCentroid(landmarks);
    motionHistory.push(centroid);
    if (motionHistory.length > MOTION_FRAMES) {
      motionHistory.shift();
    }

    if (motionHistory.length >= MOTION_FRAMES && !motionPassed) {
      // Compute average displacement between consecutive frames
      let totalDisp = 0;
      for (let i = 1; i < motionHistory.length; i++) {
        totalDisp += Math.hypot(
          motionHistory[i].x - motionHistory[i - 1].x,
          motionHistory[i].y - motionHistory[i - 1].y
        );
      }
      const avgDisp = totalDisp / (motionHistory.length - 1);

      if (avgDisp >= MOTION_THRESHOLD && avgDisp <= MAX_MOTION) {
        motionPassed = true;
      }
    }

    return getStatus();
  }

  function getStatus() {
    const passed = blinkPassed && motionPassed;
    let prompt;

    if (timedOut) {
      prompt = 'Liveness check failed — please try again';
    } else if (passed) {
      prompt = 'Liveness confirmed';
    } else if (!motionPassed && !blinkPassed) {
      prompt = 'Blink naturally and hold still';
    } else if (!blinkPassed) {
      prompt = 'Please blink';
    } else {
      prompt = 'Keep still, verifying…';
    }

    return { passed, blinkCount, blinkPassed, motionPassed, timedOut, prompt };
  }

  return { update, reset, getStatus };
}

// ─── Helper: detect face with landmarks (for liveness overlay loop) ──

/**
 * Detect a face and return its landmarks (needed for liveness checks).
 * Lighter than detectAndMatch — no descriptor computation.
 */
export async function detectWithLandmarks(videoEl) {
  const result = await faceapi
    .detectSingleFace(videoEl, new faceapi.SsdMobilenetv1Options({ minConfidence: 0.4 }))
    .withFaceLandmarks();
  return result || null;
}
