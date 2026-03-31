/**
 * Geolocation & proximity utilities for campus-based attendance.
 *
 * Uses the Haversine formula to calculate distance between student
 * and the BINUS School Simprug campus. Students must be within the
 * configured radius to clock in.
 *
 * ACCURACY STRATEGY:
 *  - getCurrentPosition forces a FRESH fix (maximumAge: 0) so the
 *    browser cannot serve a stale Wi-Fi/cell-tower position.
 *  - getAccuratePosition uses watchPosition internally, collects
 *    progressive fixes for up to 8 seconds, and resolves with the
 *    most accurate reading. This avoids returning a wildly inaccurate
 *    first fix (common on mobile).
 *
 * CACHING: checkProximity() caches its result for 5 seconds to avoid
 * redundant GPS reads when components mount/re-render quickly.
 */
import { memGet, memSet, TTL } from './cache';
import booleanPointInPolygon from '@turf/boolean-point-in-polygon';
import { point } from '@turf/helpers';
import buffer from '@turf/buffer';
import campusPolygonFeature from '../config/campusPolygon.json';

// ─── Campus coordinates (BINUS School Simprug) ───────────────────────
const CAMPUS_LAT = parseFloat(import.meta.env.VITE_CAMPUS_LAT) || -6.2341;
const CAMPUS_LNG = parseFloat(import.meta.env.VITE_CAMPUS_LNG) || 106.7854;
const CAMPUS_RADIUS_M = parseFloat(import.meta.env.VITE_CAMPUS_RADIUS) || 500; // metres
const CAMPUS_POLYGON_FEATURE =
  campusPolygonFeature?.geometry?.coordinates?.length ? campusPolygonFeature : null;

// Expand polygon by a safety buffer (metres) to account for GPS drift.
// The raw polygon traces the fence; the buffered version adds ~25m padding
// so a student standing just inside doesn't get marked "out of range"
// due to GPS multipath/bounce from nearby buildings.
const GPS_BUFFER_M = parseFloat(import.meta.env.VITE_GPS_BUFFER_M) || 10;
const CAMPUS_POLYGON_BUFFERED = CAMPUS_POLYGON_FEATURE
  ? buffer(CAMPUS_POLYGON_FEATURE, GPS_BUFFER_M, { units: 'meters' })
  : null;

const CAMPUS_POLYGON_COORDS = CAMPUS_POLYGON_FEATURE
  ? CAMPUS_POLYGON_FEATURE.geometry.coordinates.map((ring) =>
      ring.map(([lng, lat]) => [lat, lng])
    )
  : null;
const CAMPUS_NAME = campusPolygonFeature?.properties?.name || 'Campus';

// Accuracy thresholds (metres)
const GOOD_ACCURACY_M   = 20;  // immediately accept a fix this accurate
const MAX_GPS_ACCURACY_M = 50;  // reject readings worse than this (tightened for anti-spoof)
const SETTLE_TIME_MS     = 10_000; // how long to wait for a better fix

// ─── Anti-spoofing configuration ─────────────────────────────────────
const MIN_SAMPLES_FOR_VALIDATION = 3; // need this many GPS samples to validate
const MAX_SPEED_MPS = 30; // 30 m/s ≈ 108 km/h — reject teleportation
const SPOOF_ACCURACY_FLOOR = 3; // suspiciously perfect accuracy (< 3m) on mobile = likely spoofed
const STABILITY_THRESHOLD = 0.0001; // if all samples identical to this precision = synthetic
const GPS_SAMPLE_INTERVAL = 800; // ms between validation samples

// ─── Haversine distance (metres) ─────────────────────────────────────
function haversineMetres(lat1, lon1, lat2, lon2) {
  const R = 6_371_000; // Earth radius in metres
  const toRad = (v) => (v * Math.PI) / 180;
  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) ** 2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

// ─── Get current position (Promise wrapper) ──────────────────────────
function getCurrentPosition(options = {}) {
  return new Promise((resolve, reject) => {
    if (!navigator.geolocation) {
      reject(new Error('Geolocation is not supported by this browser.'));
      return;
    }
    navigator.geolocation.getCurrentPosition(resolve, reject, {
      enableHighAccuracy: true,
      timeout: 15_000,
      maximumAge: 0, // ALWAYS request a fresh fix — never use stale cache
      ...options,
    });
  });
}

/**
 * Use watchPosition to collect progressive GPS fixes and resolve with
 * the most accurate one.  Resolves early if a fix with ≤ GOOD_ACCURACY_M
 * arrives, otherwise resolves after SETTLE_TIME_MS with the best fix seen.
 */
function getAccuratePosition() {
  return new Promise((resolve, reject) => {
    if (!navigator.geolocation) {
      reject(new Error('Geolocation is not supported by this browser.'));
      return;
    }

    let bestPosition = null;
    let settled = false;

    const finish = () => {
      if (settled) return;
      settled = true;
      navigator.geolocation.clearWatch(watchId);
      clearTimeout(timer);
      if (bestPosition) {
        resolve(bestPosition);
      } else {
        reject(new Error('Unable to get an accurate GPS fix. Please ensure location/GPS is enabled.'));
      }
    };

    const watchId = navigator.geolocation.watchPosition(
      (pos) => {
        const acc = pos.coords.accuracy;
        // Keep the most accurate reading seen so far
        if (!bestPosition || acc < bestPosition.coords.accuracy) {
          bestPosition = pos;
        }
        // If we got a really good fix, resolve immediately
        if (acc <= GOOD_ACCURACY_M) {
          finish();
        }
      },
      (err) => {
        // On hard error with no position collected yet, reject
        if (!bestPosition) {
          settled = true;
          clearTimeout(timer);
          navigator.geolocation.clearWatch(watchId);
          reject(err);
        }
        // Otherwise ignore transient errors — we already have a position
      },
      {
        enableHighAccuracy: true,
        timeout: 20_000,
        maximumAge: 0,
      }
    );

    // After settle time, resolve with the best fix we have
    const timer = setTimeout(finish, SETTLE_TIME_MS);
  });
}

function isInsideCampusPolygon(lat, lng) {
  if (!CAMPUS_POLYGON_BUFFERED) return null;
  try {
    // Check against the BUFFERED polygon (original + 25m safety margin)
    // so GPS drift near the boundary doesn't cause false "out of range"
    return booleanPointInPolygon(point([lng, lat]), CAMPUS_POLYGON_BUFFERED, {
      ignoreBoundary: false,
    });
  } catch (err) {
    console.warn('Invalid campus polygon geometry:', err);
    return null;
  }
}

function buildProximityResult(lat, lng, accuracy, spoofScore = 0) {
  const distance = haversineMetres(lat, lng, CAMPUS_LAT, CAMPUS_LNG);
  const polygonCheck = isInsideCampusPolygon(lat, lng);
  const geofenceType = polygonCheck != null ? 'polygon' : 'radius';

  return {
    inRange: polygonCheck != null ? polygonCheck : distance <= CAMPUS_RADIUS_M,
    distance: Math.round(distance),
    accuracy: Math.round(accuracy),
    lat,
    lng,
    campusRadius: CAMPUS_RADIUS_M,
    geofence: geofenceType,
    polygon: CAMPUS_POLYGON_COORDS,
    campusName: CAMPUS_NAME,
    spoofScore, // 0 = confident real, higher = more suspicious
  };
}

// ─── Anti-spoofing: multi-sample GPS validation ──────────────────────

// History of recent GPS fixes for velocity/consistency checking
let _gpsHistory = []; // { lat, lng, accuracy, ts }
const GPS_HISTORY_MAX = 10;

function recordGpsFix(lat, lng, accuracy) {
  const now = Date.now();
  _gpsHistory.push({ lat, lng, accuracy, ts: now });
  if (_gpsHistory.length > GPS_HISTORY_MAX) _gpsHistory.shift();
}

/**
 * Collect multiple GPS samples and check for spoofing indicators:
 * 1. Suspiciously perfect accuracy (< 3m on mobile = unusual)
 * 2. All samples identical (real GPS has natural jitter)
 * 3. Impossible speed between consecutive readings
 * 4. Altitude = 0 exactly (many spoofers don't set altitude)
 *
 * Returns { lat, lng, accuracy, spoofScore, blocked, reason }
 */
async function getValidatedPosition() {
  const samples = [];

  // Collect MIN_SAMPLES_FOR_VALIDATION fresh GPS readings
  for (let i = 0; i < MIN_SAMPLES_FOR_VALIDATION; i++) {
    const pos = await getCurrentPosition({ timeout: 10_000 });
    samples.push({
      lat: pos.coords.latitude,
      lng: pos.coords.longitude,
      accuracy: pos.coords.accuracy,
      altitude: pos.coords.altitude,
      ts: Date.now(),
    });
    // Small delay between samples to detect frozen coordinates
    if (i < MIN_SAMPLES_FOR_VALIDATION - 1) {
      await new Promise((r) => setTimeout(r, GPS_SAMPLE_INTERVAL));
    }
  }

  // Use the most accurate sample as the primary reading
  const best = samples.reduce((a, b) => (a.accuracy < b.accuracy ? a : b));
  let spoofScore = 0;
  const reasons = [];

  // Check 1: Suspiciously perfect accuracy
  if (best.accuracy < SPOOF_ACCURACY_FLOOR) {
    spoofScore += 30;
    reasons.push(`suspiciously perfect accuracy (${best.accuracy.toFixed(1)}m)`);
  }

  // Check 2: All coordinates identical (no natural GPS jitter)
  if (samples.length >= 2) {
    const allIdentical = samples.every(
      (s) =>
        Math.abs(s.lat - samples[0].lat) < STABILITY_THRESHOLD &&
        Math.abs(s.lng - samples[0].lng) < STABILITY_THRESHOLD
    );
    if (allIdentical) {
      spoofScore += 40;
      reasons.push('coordinates unnaturally stable (no GPS jitter)');
    }
  }

  // Check 3: Velocity check against previous readings
  if (_gpsHistory.length > 0) {
    const prev = _gpsHistory[_gpsHistory.length - 1];
    const dist = haversineMetres(prev.lat, prev.lng, best.lat, best.lng);
    const dtSec = (best.ts - prev.ts) / 1000;
    if (dtSec > 0) {
      const speed = dist / dtSec;
      if (speed > MAX_SPEED_MPS) {
        spoofScore += 50;
        reasons.push(`impossible speed (${Math.round(speed)} m/s = ${Math.round(speed * 3.6)} km/h)`);
      }
    }
  }

  // Check 4: All altitudes exactly 0 (common in spoofer apps)
  const allAltZero = samples.every((s) => s.altitude === 0);
  if (allAltZero && samples.length >= 2) {
    spoofScore += 15;
    reasons.push('altitude fixed at 0m');
  }

  // Record this fix for future velocity checks
  recordGpsFix(best.lat, best.lng, best.accuracy);

  // Block if spoof score exceeds threshold
  const blocked = spoofScore >= 50;

  return {
    lat: best.lat,
    lng: best.lng,
    accuracy: best.accuracy,
    spoofScore,
    blocked,
    reason: blocked
      ? `Location appears spoofed: ${reasons.join('; ')}. Please disable mock location apps and try again.`
      : null,
    reasons,
  };
}

// ─── Public API ──────────────────────────────────────────────────────

/**
 * Check if the user is within campus proximity.
 * Returns { inRange, distance, accuracy, lat, lng, campusRadius }
 * Cached for 5 seconds to prevent redundant GPS reads on quick re-renders.
 *
 * Uses getAccuratePosition() which collects progressive GPS fixes for up
 * to 8 seconds and returns the most accurate one. Falls back to a single
 * getCurrentPosition() if the watch-based approach fails.
 */
export async function checkProximity() {
  // ── Short-lived cache to avoid rapid-fire GPS reads ───────────────
  const cached = memGet('proximity', TTL.PROXIMITY);
  if (cached) return cached;

  try {
    // ── Anti-spoofing: collect multiple samples and validate ────────
    const validated = await getValidatedPosition();

    if (validated.blocked) {
      throw new Error(validated.reason);
    }

    const { lat, lng, accuracy, spoofScore } = validated;

    // Reject suspiciously inaccurate GPS readings
    if (accuracy > MAX_GPS_ACCURACY_M) {
      throw new Error(
        `GPS accuracy too low (${Math.round(accuracy)}m). ` +
        'Please ensure GPS is enabled. Moving near a window or outdoors helps.'
      );
    }

    const result = buildProximityResult(lat, lng, accuracy, spoofScore);

    memSet('proximity', result);
    return result;
  } catch (err) {
    // Map GeolocationPositionError codes to user-friendly messages
    const messages = {
      1: 'Location access denied. Please enable location permissions in your browser settings.',
      2: 'Unable to determine your location. Please try again.',
      3: 'Location request timed out. Please ensure you have GPS/location enabled.',
    };
    throw new Error(messages[err.code] || err.message);
  }
}

/**
 * Watch position continuously. Returns a cleanup function.
 * Calls onChange({ inRange, distance, accuracy, lat, lng }) on each update.
 * Calls onError(errorMessage) on failure.
 */
export function watchProximity(onChange, onError) {
  if (!navigator.geolocation) {
    onError?.('Geolocation is not supported.');
    return () => {};
  }

  const id = navigator.geolocation.watchPosition(
    (pos) => {
      const { latitude: lat, longitude: lng, accuracy } = pos.coords;

      // Skip updates with suspiciously low accuracy
      if (accuracy > MAX_GPS_ACCURACY_M) {
        onError?.(`GPS accuracy too low (${Math.round(accuracy)}m). Waiting for better signal.`);
        return;
      }

      onChange(buildProximityResult(lat, lng, accuracy));
    },
    (err) => {
      const messages = {
        1: 'Location access denied.',
        2: 'Unable to determine location.',
        3: 'Location request timed out.',
      };
      onError?.(messages[err.code] || err.message);
    },
    {
      enableHighAccuracy: true,
      timeout: 15_000,
      maximumAge: 0, // never serve stale cached positions
    }
  );

  return () => navigator.geolocation.clearWatch(id);
}

/**
 * Get the campus configuration (for display purposes).
 */
export function getCampusConfig() {
  return {
    lat: CAMPUS_LAT,
    lng: CAMPUS_LNG,
    radius: CAMPUS_RADIUS_M,
    polygon: CAMPUS_POLYGON_COORDS,
    geofence: CAMPUS_POLYGON_FEATURE ? 'polygon' : 'radius',
    name: CAMPUS_NAME,
  };
}
