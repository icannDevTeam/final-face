/**
 * Geolocation & proximity utilities for campus-based attendance.
 *
 * Uses the Haversine formula to calculate distance between student
 * and the BINUS School Simprug campus. Students must be within the
 * configured radius to clock in.
 */

// ─── Campus coordinates (BINUS School Simprug) ───────────────────────
const CAMPUS_LAT = parseFloat(import.meta.env.VITE_CAMPUS_LAT) || -6.2307;
const CAMPUS_LNG = parseFloat(import.meta.env.VITE_CAMPUS_LNG) || 106.7865;
const CAMPUS_RADIUS_M = parseFloat(import.meta.env.VITE_CAMPUS_RADIUS) || 200; // meters

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
      timeout: 10_000,
      maximumAge: 30_000,
      ...options,
    });
  });
}

// ─── Public API ──────────────────────────────────────────────────────

/**
 * Check if the user is within campus proximity.
 * Returns { inRange, distance, accuracy, lat, lng, campusRadius }
 */
export async function checkProximity() {
  try {
    const pos = await getCurrentPosition();
    const { latitude: lat, longitude: lng, accuracy } = pos.coords;
    const distance = haversineMetres(lat, lng, CAMPUS_LAT, CAMPUS_LNG);

    return {
      inRange: distance <= CAMPUS_RADIUS_M,
      distance: Math.round(distance),
      accuracy: Math.round(accuracy),
      lat,
      lng,
      campusRadius: CAMPUS_RADIUS_M,
    };
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
      const distance = haversineMetres(lat, lng, CAMPUS_LAT, CAMPUS_LNG);
      onChange({
        inRange: distance <= CAMPUS_RADIUS_M,
        distance: Math.round(distance),
        accuracy: Math.round(accuracy),
        lat,
        lng,
        campusRadius: CAMPUS_RADIUS_M,
      });
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
      maximumAge: 10_000,
    }
  );

  return () => navigator.geolocation.clearWatch(id);
}

/**
 * Get the campus configuration (for display purposes).
 */
export function getCampusConfig() {
  return { lat: CAMPUS_LAT, lng: CAMPUS_LNG, radius: CAMPUS_RADIUS_M };
}
