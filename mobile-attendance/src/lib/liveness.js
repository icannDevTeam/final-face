/**
 * Liveness detection for anti-spoofing.
 *
 * Four complementary checks in randomized challenge order:
 *  1. Blink detection    — EAR-based with duration validation (100-500ms)
 *  2. Head turn challenge — yaw estimation from landmark asymmetry
 *  3. Face size stability — bounding box consistency (photo on screen wobbles)
 *  4. Micro-motion check — natural face micro-movements vs. static photo
 *
 * Challenge flow (randomized):
 *   Step 1: Always face-forward stability check (background, ~1s)
 *   Step 2: Random order of [blink, head-turn]
 *   All steps must pass within the timeout.
 *
 * Usage:
 *   const liveness = createLivenessChecker();
 *   const status = liveness.update(landmarks, detectionBox);
 *   // status.passed === true when all challenges completed
 *   // status.challenge === current challenge name
 *   // status.prompt === user-facing instruction
 *   // status.progress === 0-1 overall progress
 */
import * as faceapi from 'face-api.js';

// ─── Configuration ────────────────────────────────────────────────────

// Blink detection
const EAR_BLINK_THRESHOLD = 0.22; // EAR below this = eye closed (tightened)
const EAR_OPEN_THRESHOLD  = 0.28; // EAR above this = eye open
const MIN_BLINK_MS        = 80;   // reject noise shorter than a real blink
const MAX_BLINK_MS        = 500;  // reject holds longer than a real blink
const REQUIRED_BLINKS     = 2;    // need 2 valid blinks

// Head turn (yaw estimation)
const YAW_TURN_THRESHOLD  = 0.62; // ratio > this = turned toward that side
const YAW_NEUTRAL_MIN     = 0.38; // must return to center after turn
const YAW_NEUTRAL_MAX     = 0.62;
const TURN_HOLD_FRAMES    = 4;    // must hold turn for this many frames

// Face size stability
const SIZE_HISTORY_FRAMES = 15;   // frames to collect for size stability
const SIZE_COV_THRESHOLD  = 0.08; // coefficient of variation must be below this (stable)

// Micro-motion (replaces old simple motion check)
const MOTION_FRAMES       = 10;
const MOTION_THRESHOLD    = 1.0;  // min avg displacement (real face micro-movements)
const MAX_MOTION          = 35;   // reject shaking / video playback

const LIVENESS_TIMEOUT    = 15_000; // ms — generous for multi-step challenges

// 68-point landmark indices
const LEFT_EYE  = [36, 37, 38, 39, 40, 41];
const RIGHT_EYE = [42, 43, 44, 45, 46, 47];

// ─── Eye Aspect Ratio ─────────────────────────────────────────────────

function eyeAspectRatio(points, indices) {
  const p = indices.map((i) => points[i]);
  const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
  const vertical1 = dist(p[1], p[5]);
  const vertical2 = dist(p[2], p[4]);
  const horizontal = dist(p[0], p[3]);
  if (horizontal === 0) return 0.3;
  return (vertical1 + vertical2) / (2 * horizontal);
}

function avgEAR(landmarks) {
  const pts = landmarks.positions;
  return (eyeAspectRatio(pts, LEFT_EYE) + eyeAspectRatio(pts, RIGHT_EYE)) / 2;
}

// ─── Head yaw estimation ──────────────────────────────────────────────

/**
 * Estimate head yaw from landmark asymmetry.
 * Returns a ratio 0–1: ~0.5 = facing forward, <0.38 = turned left, >0.62 = turned right.
 * Uses nose tip (30) vs jaw contour edges (0 = left, 16 = right).
 */
function estimateYaw(landmarks) {
  const pts = landmarks.positions;
  const nose  = pts[30];
  const left  = pts[0];   // left jaw edge
  const right = pts[16];  // right jaw edge
  const dLeft  = Math.hypot(nose.x - left.x, nose.y - left.y);
  const dRight = Math.hypot(nose.x - right.x, nose.y - right.y);
  const total = dLeft + dRight;
  if (total === 0) return 0.5;
  // ratio = how far nose is from RIGHT side; high = turned right, low = turned left
  return dRight / total;
}

// ─── Landmark centroid for micro-motion ───────────────────────────────

function landmarkCentroid(landmarks) {
  const pts = landmarks.positions;
  const indices = [36, 39, 42, 45, 30, 33];
  let sx = 0, sy = 0;
  for (const i of indices) { sx += pts[i].x; sy += pts[i].y; }
  return { x: sx / indices.length, y: sy / indices.length };
}

// ─── Challenge generation ─────────────────────────────────────────────

const TURN_DIRECTIONS = ['turn_left', 'turn_right'];

function generateChallenges() {
  const turnDir = TURN_DIRECTIONS[Math.random() < 0.5 ? 0 : 1];
  return ['stabilize', turnDir];
}

// ─── Liveness checker factory ─────────────────────────────────────────

export function createLivenessChecker() {
  let challenges = generateChallenges();
  let challengeIdx = 0;
  let startTime = null;
  let timedOut = false;

  // Blink state
  let blinkCount = 0;
  let eyeCloseStartMs = null;
  let eyeWasClosed = false;

  // Turn state
  let turnHoldCount = 0;
  let turnDetected = false;
  let returnedToCenter = false;

  // Stability state (face size + micro-motion)
  let sizeHistory = [];      // bounding box areas
  let motionHistory = [];    // centroids
  let sizeStable = false;
  let motionOk = false;

  function reset() {
    challenges = generateChallenges();
    challengeIdx = 0;
    startTime = null;
    timedOut = false;
    blinkCount = 0;
    eyeCloseStartMs = null;
    eyeWasClosed = false;
    turnHoldCount = 0;
    turnDetected = false;
    returnedToCenter = false;
    sizeHistory = [];
    motionHistory = [];
    sizeStable = false;
    motionOk = false;
  }

  function currentChallenge() {
    return challengeIdx < challenges.length ? challenges[challengeIdx] : 'done';
  }

  function advanceChallenge() {
    challengeIdx++;
  }

  function update(landmarks, box) {
    if (!landmarks) return getStatus();
    if (!startTime) startTime = Date.now();

    if (Date.now() - startTime > LIVENESS_TIMEOUT) {
      timedOut = true;
      return getStatus();
    }

    const challenge = currentChallenge();
    if (challenge === 'done') return getStatus();

    // ── Always accumulate size + motion (background checks) ─────
    if (box) {
      sizeHistory.push(box.width * box.height);
      if (sizeHistory.length > SIZE_HISTORY_FRAMES) sizeHistory.shift();

      if (sizeHistory.length >= SIZE_HISTORY_FRAMES && !sizeStable) {
        const mean = sizeHistory.reduce((a, b) => a + b, 0) / sizeHistory.length;
        if (mean > 0) {
          const variance = sizeHistory.reduce((s, v) => s + (v - mean) ** 2, 0) / sizeHistory.length;
          const cov = Math.sqrt(variance) / mean;
          if (cov < SIZE_COV_THRESHOLD) sizeStable = true;
        }
      }
    }

    const centroid = landmarkCentroid(landmarks);
    motionHistory.push(centroid);
    if (motionHistory.length > MOTION_FRAMES) motionHistory.shift();

    if (motionHistory.length >= MOTION_FRAMES && !motionOk) {
      let totalDisp = 0;
      for (let i = 1; i < motionHistory.length; i++) {
        totalDisp += Math.hypot(
          motionHistory[i].x - motionHistory[i - 1].x,
          motionHistory[i].y - motionHistory[i - 1].y,
        );
      }
      const avgDisp = totalDisp / (motionHistory.length - 1);
      if (avgDisp >= MOTION_THRESHOLD && avgDisp <= MAX_MOTION) motionOk = true;
    }

    // ── Challenge-specific logic ────────────────────────────────

    if (challenge === 'stabilize') {
      // Wait for face size to stabilize + micro-motion detected
      if (sizeStable && motionOk) advanceChallenge();
    }

    else if (challenge === 'blink') {
      const ear = avgEAR(landmarks);
      const now = Date.now();

      if (ear < EAR_BLINK_THRESHOLD && !eyeWasClosed) {
        eyeWasClosed = true;
        eyeCloseStartMs = now;
      } else if (ear > EAR_OPEN_THRESHOLD && eyeWasClosed) {
        // Eye re-opened — validate blink duration
        const duration = now - (eyeCloseStartMs || now);
        eyeWasClosed = false;
        eyeCloseStartMs = null;

        if (duration >= MIN_BLINK_MS && duration <= MAX_BLINK_MS) {
          blinkCount++;
        }
        // else: reject — too fast (noise) or too slow (held eyes shut)

        if (blinkCount >= REQUIRED_BLINKS) advanceChallenge();
      }
    }

    else if (challenge === 'turn_left' || challenge === 'turn_right') {
      const yaw = estimateYaw(landmarks);

      if (!turnDetected) {
        // Check if user has turned in the requested direction
        const turned = challenge === 'turn_left'
          ? yaw < (1 - YAW_TURN_THRESHOLD) // nose closer to left edge
          : yaw > YAW_TURN_THRESHOLD;       // nose closer to right edge

        if (turned) {
          turnHoldCount++;
          if (turnHoldCount >= TURN_HOLD_FRAMES) {
            turnDetected = true;
            turnHoldCount = 0;
          }
        } else {
          turnHoldCount = 0;
        }
      } else if (!returnedToCenter) {
        // Must return to center (proves it wasn't a tilted photo)
        if (yaw >= YAW_NEUTRAL_MIN && yaw <= YAW_NEUTRAL_MAX) {
          turnHoldCount++;
          if (turnHoldCount >= TURN_HOLD_FRAMES) {
            returnedToCenter = true;
            advanceChallenge();
          }
        } else {
          turnHoldCount = 0;
        }
      }
    }

    return getStatus();
  }

  function getStatus() {
    const challenge = currentChallenge();
    const passed = challenge === 'done';
    const totalSteps = challenges.length;
    const progress = Math.min(challengeIdx / totalSteps, 1);

    let prompt;
    if (timedOut) {
      prompt = 'Liveness check timed out — tap Retry';
    } else if (passed) {
      prompt = 'Liveness confirmed ✓';
    } else if (challenge === 'stabilize') {
      prompt = 'Hold your face steady in the frame';
    } else if (challenge === 'blink') {
      if (blinkCount === 0) {
        prompt = 'Blink naturally';
      } else {
        prompt = `Blink again (${blinkCount}/${REQUIRED_BLINKS})`;
      }
    } else if (challenge === 'turn_left') {
      if (!turnDetected) {
        prompt = '← Turn your head to the left';
      } else {
        prompt = 'Now look back at the camera';
      }
    } else if (challenge === 'turn_right') {
      if (!turnDetected) {
        prompt = 'Turn your head to the right →';
      } else {
        prompt = 'Now look back at the camera';
      }
    }

    return {
      passed,
      timedOut,
      challenge,
      challengeIdx,
      totalSteps,
      progress,
      prompt,
      blinkCount,
      blinkPassed: blinkCount >= REQUIRED_BLINKS,
      sizeStable,
      motionOk,
      turnDetected,
    };
  }

  return { update, reset, getStatus };
}

// ─── Helper: detect face with landmarks (for liveness overlay loop) ──

export async function detectWithLandmarks(videoEl) {
  const result = await faceapi
    .detectSingleFace(videoEl, new faceapi.SsdMobilenetv1Options({ minConfidence: 0.4 }))
    .withFaceLandmarks();
  return result || null;
}
