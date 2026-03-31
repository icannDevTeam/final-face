#!/usr/bin/env node
/**
 * GPS Anti-Spoofing Simulation Test
 *
 * Tests the scoring logic from geolocation.js against various
 * spoofed and legitimate GPS scenarios. Does NOT require a browser.
 *
 * Run: node scripts/test-antispoof.mjs
 */

// ─── Reproduce the scoring logic from geolocation.js ────────────────

const SPOOF_ACCURACY_FLOOR = 3;
const STABILITY_THRESHOLD = 0.000005;
const MAX_SPEED_MPS = 30;

function haversineMetres(lat1, lon1, lat2, lon2) {
  const R = 6_371_000;
  const toRad = (v) => (v * Math.PI) / 180;
  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) ** 2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

/**
 * Score a set of GPS samples for spoofing indicators.
 * Returns { spoofScore, blocked, reasons }
 */
function scoreSamples(samples, previousFix = null) {
  const best = samples.reduce((a, b) => (a.accuracy < b.accuracy ? a : b));
  let spoofScore = 0;
  const reasons = [];

  // Check 1: Suspiciously perfect accuracy
  if (best.accuracy < SPOOF_ACCURACY_FLOOR) {
    spoofScore += 30;
    reasons.push(`perfect accuracy (${best.accuracy.toFixed(1)}m)`);
  }

  // Check 2: All coordinates identical
  if (samples.length >= 3) {
    const allIdentical = samples.every(
      (s) =>
        Math.abs(s.lat - samples[0].lat) < STABILITY_THRESHOLD &&
        Math.abs(s.lng - samples[0].lng) < STABILITY_THRESHOLD
    );
    if (allIdentical) {
      spoofScore += 15;
      reasons.push('coordinates frozen (sub-metre)');
    }
  }

  // Check 3: Velocity check
  if (previousFix) {
    const dist = haversineMetres(previousFix.lat, previousFix.lng, best.lat, best.lng);
    const dtSec = (best.ts - previousFix.ts) / 1000;
    if (dtSec > 0) {
      const speed = dist / dtSec;
      if (speed > MAX_SPEED_MPS) {
        spoofScore += 70;
        reasons.push(`teleportation (${Math.round(speed * 3.6)} km/h)`);
      }
    }
  }

  // Check 4: Altitude analysis
  const allAltZero = samples.every((s) => s.altitude === 0);
  const noAltAccuracy = samples.every((s) => s.altitudeAccuracy === null || s.altitudeAccuracy === undefined);
  const allAltNull = samples.every((s) => s.altitude === null || s.altitude === undefined);

  if (allAltZero) {
    spoofScore += 10;
    reasons.push('altitude fixed at 0m');
  }
  if (noAltAccuracy && !allAltNull) {
    spoofScore += 5;
    reasons.push('no altitudeAccuracy');
  }

  // Check 5: Heading without speed
  const hasHeadingNoSpeed = samples.some(
    (s) => s.heading !== null && s.heading !== undefined && (s.speed === null || s.speed === 0)
  );
  if (hasHeadingNoSpeed) {
    spoofScore += 15;
    reasons.push('heading without speed');
  }

  // Check 6: Response time (instant = suspicious)
  const responseTimes = samples.map((s) => s.responseTime).filter((t) => t > 0);
  if (responseTimes.length >= 3) {
    const allInstant = responseTimes.every((t) => t < 50);
    if (allInstant) {
      spoofScore += 10;
      reasons.push('instant GPS responses (<50ms)');
    }
    const mean = responseTimes.reduce((a, b) => a + b, 0) / responseTimes.length;
    if (mean > 0) {
      const variance = responseTimes.reduce((s, v) => s + (v - mean) ** 2, 0) / responseTimes.length;
      const cov = Math.sqrt(variance) / mean;
      if (cov < 0.1 && responseTimes.length >= 3) {
        spoofScore += 5;
        reasons.push('timing too consistent');
      }
    }
  }

  // Check 7: Accuracy never changes
  if (samples.length >= 3) {
    const allSameAccuracy = samples.every(
      (s) => Math.abs(s.accuracy - samples[0].accuracy) < 0.01
    );
    if (allSameAccuracy) {
      spoofScore += 5;
      reasons.push('accuracy never varies');
    }
  }

  const blocked = spoofScore >= 60;
  return { spoofScore, blocked, reasons, best };
}

// ─── Test scenarios ──────────────────────────────────────────────────

const CAMPUS_LAT = -6.2341;
const CAMPUS_LNG = 106.7854;

function makeTs(baseMs, offsetMs) { return baseMs + offsetMs; }

const now = Date.now();

const scenarios = [
  {
    name: '🟢 REAL GPS — Student at campus (natural jitter)',
    expectBlocked: false,
    samples: [
      { lat: -6.23412, lng: 106.78543, accuracy: 18, altitude: 45.2, altitudeAccuracy: 12, speed: 0.3, heading: null, ts: makeTs(now, 0), responseTime: 340 },
      { lat: -6.23408, lng: 106.78539, accuracy: 15, altitude: 44.8, altitudeAccuracy: 10, speed: 0.1, heading: null, ts: makeTs(now, 1200), responseTime: 280 },
      { lat: -6.23415, lng: 106.78547, accuracy: 12, altitude: 45.5, altitudeAccuracy: 8, speed: 0.2, heading: null, ts: makeTs(now, 2400), responseTime: 520 },
      { lat: -6.23410, lng: 106.78541, accuracy: 14, altitude: 45.0, altitudeAccuracy: 9, speed: 0.0, heading: null, ts: makeTs(now, 5500), responseTime: 190 },
    ],
  },
  {
    name: '🟢 REAL GPS — Walking student (moving)',
    expectBlocked: false,
    samples: [
      { lat: -6.23412, lng: 106.78543, accuracy: 20, altitude: 45.0, altitudeAccuracy: 15, speed: 1.2, heading: 87, ts: makeTs(now, 0), responseTime: 500 },
      { lat: -6.23410, lng: 106.78560, accuracy: 18, altitude: 44.5, altitudeAccuracy: 12, speed: 1.4, heading: 92, ts: makeTs(now, 1200), responseTime: 250 },
      { lat: -6.23408, lng: 106.78575, accuracy: 15, altitude: 45.2, altitudeAccuracy: 10, speed: 1.1, heading: 85, ts: makeTs(now, 2400), responseTime: 780 },
      { lat: -6.23405, lng: 106.78590, accuracy: 22, altitude: 44.8, altitudeAccuracy: 14, speed: 1.3, heading: 90, ts: makeTs(now, 5500), responseTime: 320 },
    ],
  },
  {
    name: '🔴 FAKE GPS — Basic spoofer (frozen coords, no altitude, instant)',
    expectBlocked: true,
    samples: [
      { lat: CAMPUS_LAT, lng: CAMPUS_LNG, accuracy: 1, altitude: 0, altitudeAccuracy: null, speed: 0, heading: null, ts: makeTs(now, 0), responseTime: 10 },
      { lat: CAMPUS_LAT, lng: CAMPUS_LNG, accuracy: 1, altitude: 0, altitudeAccuracy: null, speed: 0, heading: null, ts: makeTs(now, 1200), responseTime: 8 },
      { lat: CAMPUS_LAT, lng: CAMPUS_LNG, accuracy: 1, altitude: 0, altitudeAccuracy: null, speed: 0, heading: null, ts: makeTs(now, 2400), responseTime: 12 },
      { lat: CAMPUS_LAT, lng: CAMPUS_LNG, accuracy: 1, altitude: 0, altitudeAccuracy: null, speed: 0, heading: null, ts: makeTs(now, 5500), responseTime: 9 },
    ],
  },
  {
    name: '� FAKE GPS — "Smart" spoofer (mimics WiFi — caught by liveness/face layers)',
    expectBlocked: false,
    samples: [
      { lat: CAMPUS_LAT, lng: CAMPUS_LNG, accuracy: 20, altitude: 0, altitudeAccuracy: null, speed: null, heading: null, ts: makeTs(now, 0), responseTime: 30 },
      { lat: CAMPUS_LAT, lng: CAMPUS_LNG, accuracy: 20, altitude: 0, altitudeAccuracy: null, speed: null, heading: null, ts: makeTs(now, 1200), responseTime: 28 },
      { lat: CAMPUS_LAT, lng: CAMPUS_LNG, accuracy: 20, altitude: 0, altitudeAccuracy: null, speed: null, heading: null, ts: makeTs(now, 2400), responseTime: 32 },
      { lat: CAMPUS_LAT, lng: CAMPUS_LNG, accuracy: 20, altitude: 0, altitudeAccuracy: null, speed: null, heading: null, ts: makeTs(now, 5500), responseTime: 29 },
    ],
  },
  {
    name: '� FAKE GPS — Spoofer with slight jitter (caught by liveness/face layers)',
    expectBlocked: false,
    samples: [
      { lat: -6.23410, lng: 106.78541, accuracy: 10, altitude: 0, altitudeAccuracy: null, speed: 0, heading: null, ts: makeTs(now, 0), responseTime: 15 },
      { lat: -6.23411, lng: 106.78542, accuracy: 10, altitude: 0, altitudeAccuracy: null, speed: 0, heading: null, ts: makeTs(now, 1200), responseTime: 18 },
      { lat: -6.23409, lng: 106.78540, accuracy: 10, altitude: 0, altitudeAccuracy: null, speed: 0, heading: null, ts: makeTs(now, 2400), responseTime: 14 },
      { lat: -6.23412, lng: 106.78543, accuracy: 10, altitude: 0, altitudeAccuracy: null, speed: 0, heading: null, ts: makeTs(now, 5500), responseTime: 16 },
    ],
  },
  {
    name: '🔴 FAKE GPS — Teleportation (was 10km away 30 seconds ago)',
    expectBlocked: true,
    previousFix: { lat: -6.3000, lng: 106.7854, ts: makeTs(now, -30000) },
    samples: [
      { lat: -6.23412, lng: 106.78543, accuracy: 15, altitude: 45, altitudeAccuracy: 10, speed: 0, heading: null, ts: makeTs(now, 0), responseTime: 300 },
      { lat: -6.23415, lng: 106.78540, accuracy: 18, altitude: 44, altitudeAccuracy: 12, speed: 0, heading: null, ts: makeTs(now, 1200), responseTime: 250 },
      { lat: -6.23410, lng: 106.78545, accuracy: 14, altitude: 46, altitudeAccuracy: 9, speed: 0, heading: null, ts: makeTs(now, 2400), responseTime: 400 },
      { lat: -6.23413, lng: 106.78542, accuracy: 16, altitude: 45, altitudeAccuracy: 11, speed: 0, heading: null, ts: makeTs(now, 5500), responseTime: 320 },
    ],
  },
  {
    name: '🔴 FAKE GPS — Heading reported with zero speed (sensor inconsistency)',
    expectBlocked: true,
    samples: [
      { lat: CAMPUS_LAT, lng: CAMPUS_LNG, accuracy: 15, altitude: 0, altitudeAccuracy: null, speed: 0, heading: 180, ts: makeTs(now, 0), responseTime: 25 },
      { lat: CAMPUS_LAT, lng: CAMPUS_LNG, accuracy: 15, altitude: 0, altitudeAccuracy: null, speed: 0, heading: 180, ts: makeTs(now, 1200), responseTime: 22 },
      { lat: CAMPUS_LAT, lng: CAMPUS_LNG, accuracy: 15, altitude: 0, altitudeAccuracy: null, speed: 0, heading: 180, ts: makeTs(now, 2400), responseTime: 28 },
      { lat: CAMPUS_LAT, lng: CAMPUS_LNG, accuracy: 15, altitude: 0, altitudeAccuracy: null, speed: 0, heading: 180, ts: makeTs(now, 5500), responseTime: 24 },
    ],
  },
  {
    name: '🟡 EDGE — Indoor GPS (poor accuracy but real)',
    expectBlocked: false,
    samples: [
      { lat: -6.23420, lng: 106.78530, accuracy: 40, altitude: 44.0, altitudeAccuracy: 25, speed: null, heading: null, ts: makeTs(now, 0), responseTime: 1200 },
      { lat: -6.23405, lng: 106.78555, accuracy: 35, altitude: 46.0, altitudeAccuracy: 20, speed: null, heading: null, ts: makeTs(now, 1200), responseTime: 800 },
      { lat: -6.23430, lng: 106.78520, accuracy: 45, altitude: 43.0, altitudeAccuracy: 30, speed: null, heading: null, ts: makeTs(now, 2400), responseTime: 2100 },
      { lat: -6.23415, lng: 106.78540, accuracy: 38, altitude: 45.0, altitudeAccuracy: 22, speed: null, heading: null, ts: makeTs(now, 5500), responseTime: 600 },
    ],
  },
  {
    name: '🟢 REAL — Indoor WiFi positioning (stable coords, fixed accuracy, alt 0, fast)',
    expectBlocked: false,
    samples: [
      { lat: -6.23412, lng: 106.78543, accuracy: 20, altitude: 0, altitudeAccuracy: null, speed: null, heading: null, ts: makeTs(now, 0), responseTime: 35 },
      { lat: -6.23412, lng: 106.78543, accuracy: 20, altitude: 0, altitudeAccuracy: null, speed: null, heading: null, ts: makeTs(now, 1200), responseTime: 30 },
      { lat: -6.23413, lng: 106.78544, accuracy: 20, altitude: 0, altitudeAccuracy: null, speed: null, heading: null, ts: makeTs(now, 2400), responseTime: 40 },
      { lat: -6.23412, lng: 106.78543, accuracy: 20, altitude: 0, altitudeAccuracy: null, speed: null, heading: null, ts: makeTs(now, 5500), responseTime: 32 },
    ],
  },
  {
    name: '🟢 REAL — Indoor WiFi standing completely still',
    expectBlocked: false,
    samples: [
      { lat: -6.23412, lng: 106.78543, accuracy: 25, altitude: null, altitudeAccuracy: null, speed: null, heading: null, ts: makeTs(now, 0), responseTime: 45 },
      { lat: -6.23412, lng: 106.78543, accuracy: 25, altitude: null, altitudeAccuracy: null, speed: null, heading: null, ts: makeTs(now, 1200), responseTime: 38 },
      { lat: -6.23412, lng: 106.78543, accuracy: 25, altitude: null, altitudeAccuracy: null, speed: null, heading: null, ts: makeTs(now, 2400), responseTime: 50 },
      { lat: -6.23412, lng: 106.78543, accuracy: 25, altitude: null, altitudeAccuracy: null, speed: null, heading: null, ts: makeTs(now, 5500), responseTime: 42 },
    ],
  },
];

// ─── Run tests ───────────────────────────────────────────────────────

console.log('═══════════════════════════════════════════════════════════════');
console.log('  GPS ANTI-SPOOFING SIMULATION TEST');
console.log('═══════════════════════════════════════════════════════════════\n');

let passed = 0;
let failed = 0;

for (const scenario of scenarios) {
  const result = scoreSamples(scenario.samples, scenario.previousFix || null);
  const ok = result.blocked === scenario.expectBlocked;

  if (ok) passed++;
  else failed++;

  const status = ok ? '✅ PASS' : '❌ FAIL';
  console.log(`${status}  ${scenario.name}`);
  console.log(`       Score: ${result.spoofScore}/60 threshold → ${result.blocked ? 'BLOCKED' : 'ALLOWED'}`);
  if (result.reasons.length > 0) {
    console.log(`       Flags: ${result.reasons.join(' | ')}`);
  }
  if (!ok) {
    console.log(`       EXPECTED: ${scenario.expectBlocked ? 'BLOCKED' : 'ALLOWED'}`);
  }
  console.log();
}

console.log('═══════════════════════════════════════════════════════════════');
console.log(`  Results: ${passed} passed, ${failed} failed out of ${scenarios.length} scenarios`);
console.log('═══════════════════════════════════════════════════════════════');

process.exit(failed > 0 ? 1 : 0);
