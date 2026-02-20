/**
 * Vercel Serverless Function — Proxy for BINUS School Attendance API (B.2)
 *
 * The mobile app calls this endpoint after a successful face-recognition
 * clock-in. This function:
 *   1. Authenticates with the BINUS School API (gets a bearer token)
 *   2. Inserts the attendance record via POST bss-add-simprug-attendance-fr
 *
 * Environment variables (set in Vercel dashboard):
 *   BINUS_API_KEY — Base64 API key for Basic auth
 *   BINUS_USER_ACTION — UserAction field (default: "TEACHER7")
 */

// BINUS API only serves on HTTP port 80 (HTTPS port 443 returns 404)
const BINUS_BASE = 'http://binusian.ws';
const BINUS_AUTH_URL = `${BINUS_BASE}/binusschool/auth/token`;
const BINUS_ATTENDANCE_URL = `${BINUS_BASE}/binusschool/bss-add-simprug-attendance-fr`;

async function getBinusToken(apiKey) {
  // Use node-fetch or https.get with agent for SSL bypass
  const fetch = globalThis.fetch || (await import('node-fetch')).default;
  const resp = await fetch(BINUS_AUTH_URL, {
    method: 'GET',
    headers: { Authorization: `Basic ${apiKey}` },
  });

  if (!resp.ok) {
    throw new Error(`BINUS auth failed: HTTP ${resp.status}`);
  }

  const data = await resp.json();
  if (data.resultCode === 200 && data.data?.token) {
    return data.data.token;
  }
  throw new Error(`BINUS auth error: ${data.errorMessage || 'unknown'}`);
}

export default async function handler(req, res) {
  // CORS — restrict to known origins in production
  const allowedOrigins = [
    process.env.CORS_ORIGIN,                    // explicit env override
    'https://mobile-attendance.vercel.app',      // production
    'http://localhost:5173',                      // Vite dev
    'http://localhost:4173',                      // Vite preview
  ].filter(Boolean);
  const origin = req.headers.origin || '';
  const corsOrigin = allowedOrigins.includes(origin) ? origin : allowedOrigins[0] || '';
  res.setHeader('Access-Control-Allow-Origin', corsOrigin);
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  res.setHeader('Vary', 'Origin');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const { IdStudent, IdBinusian } = req.body || {};

  if (!IdStudent) {
    return res.status(400).json({ error: 'IdStudent is required' });
  }

  const apiKey = process.env.BINUS_API_KEY;
  if (!apiKey) {
    return res.status(500).json({ error: 'BINUS_API_KEY not configured' });
  }

  try {
    // 1. Get bearer token
    const token = await getBinusToken(apiKey);

    // 2. Insert attendance
    const body = {
      IdStudent: String(IdStudent),
      IdBinusian: String(IdBinusian || ''),
      ImageDesc: '-',
      UserAction: process.env.BINUS_USER_ACTION || 'TEACHER7',
    };

    const resp = await fetch(BINUS_ATTENDANCE_URL, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });

    const result = await resp.json();

    // Check for success (handles both UAT and production response formats)
    const afr = result.attendanceFaceRecognitionResponse || {};
    const isSuccess =
      result.isSuccess === true ||
      result.statusCode === 200 ||
      result.resultCode === 200 ||
      afr.success === true;

    if (isSuccess) {
      return res.status(200).json({ success: true, message: afr.msg || result.message || 'OK' });
    }

    const errorMsg = result.message || result.errorMessage || result.errors || 'Unknown error';
    console.error('BINUS attendance insert failed:', errorMsg);
    return res.status(502).json({ error: errorMsg });
  } catch (err) {
    console.error('BINUS API proxy error:', err);
    return res.status(500).json({ error: err.message });
  }
}
