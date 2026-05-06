/**
 * Multi-tenancy helpers (mobile/browser side) — Phase A1.
 *
 * Mirrors backend/tenancy.py + web-dataset-collector/lib/tenancy.js but
 * tailored for browser bundles (no process.env access — uses Vite's
 * `import.meta.env` with sensible defaults).
 */

const DEFAULT_TENANT_ID = 'binus-simprug';

export function getTenantId(explicit) {
  if (explicit) return explicit;
  // Vite injects VITE_* envs at build time
  if (typeof import.meta !== 'undefined' && import.meta.env?.VITE_TENANT_ID) {
    return import.meta.env.VITE_TENANT_ID;
  }
  return DEFAULT_TENANT_ID;
}

function envFlag(key, fallback) {
  if (typeof import.meta === 'undefined' || !import.meta.env) return fallback;
  const raw = import.meta.env[key];
  if (raw === undefined) return fallback;
  return ['true', '1', 'yes'].includes(String(raw).toLowerCase());
}

export const tenantAwareEnabled = () => envFlag('VITE_TENANT_AWARE', false);
export const legacyPathsEnabled = () => envFlag('VITE_LEGACY_PATHS', true);

export const tenantDoc = (t) => `tenants/${getTenantId(t)}`;
export const studentsPath = (t) => `${tenantDoc(t)}/students`;
export const studentMetadataPath = (t) => `${tenantDoc(t)}/student_metadata`;
export const attendanceDayDoc = (date, t) => `${tenantDoc(t)}/attendance/${date}`;
export const attendanceRecordPath = (date, employeeNo, t) =>
  `${attendanceDayDoc(date, t)}/records/${employeeNo}`;
export const spoofAttemptPath = (date, logId, t) =>
  `${tenantDoc(t)}/spoof_attempts/${date}/logs/${logId}`;
