/**
 * Vercel Serverless Function — Proxy for BINUS School Attendance API (B.2)
 *
 * The mobile app calls this endpoint after a successful face-recognition
 * clock-in. This function:
 *   1. Verifies Firebase ID token (anonymous auth)
 *   2. Rate-limits by IP (10 req/min)
 *   3. Authenticates with the BINUS School API (gets a bearer token)
 *   4. Inserts the attendance record via POST bss-add-simprug-attendance-fr
 *
 * Environment variables (set in Vercel dashboard):
 *   BINUS_API_KEY — Base64 API key for Basic auth
 *   BINUS_USER_ACTION — UserAction field (default: "TEACHER7")
 *   FIREBASE_PROJECT_ID, FIREBASE_CLIENT_EMAIL, FIREBASE_PRIVATE_KEY — Admin SDK
 */

import process from 'node:process';
import admin from 'firebase-admin';

// ─── Firebase Admin init (singleton) ──────────────────────────────────
function initFirebaseAdmin() {
  if (admin.apps.length > 0) return;

  let privateKey = process.env.FIREBASE_PRIVATE_KEY || '';
  if ((privateKey.startsWith('"') && privateKey.endsWith('"')) ||
      (privateKey.startsWith("'") && privateKey.endsWith("'"))) {
    privateKey = privateKey.slice(1, -1);
  }
  privateKey = privateKey.replace(/\\\\n/g, '\n').replace(/\\n/g, '\n');

  admin.initializeApp({
    credential: admin.credential.cert({
      type: 'service_account',
      project_id: process.env.FIREBASE_PROJECT_ID,
      private_key: privateKey,
      client_email: process.env.FIREBASE_CLIENT_EMAIL,
    }),
  });
}

// ─── In-memory rate limiter (per serverless instance) ─────────────────
const RATE_LIMIT_WINDOW_MS = 60_000; // 1 minute
const RATE_LIMIT_MAX = 10;           // 10 requests per window per IP
const rateLimitMap = new Map();

function isRateLimited(ip) {
  const now = Date.now();
  const entry = rateLimitMap.get(ip);
  if (!entry || now - entry.windowStart > RATE_LIMIT_WINDOW_MS) {
    rateLimitMap.set(ip, { windowStart: now, count: 1 });
    return false;
  }
  entry.count++;
  if (entry.count > RATE_LIMIT_MAX) return true;
  return false;
}

// Clean up stale entries periodically (prevent memory leak in long-lived instances)
setInterval(() => {
  const now = Date.now();
  for (const [ip, entry] of rateLimitMap) {
    if (now - entry.windowStart > RATE_LIMIT_WINDOW_MS * 2) rateLimitMap.delete(ip);
  }
}, RATE_LIMIT_WINDOW_MS * 2);

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
    'https://final-face-ten.vercel.app',         // production
    'https://mobile-attendance.vercel.app',      // legacy
    'http://localhost:5173',                      // Vite dev
    'http://localhost:5174',                      // Vite dev alt port
    'http://localhost:4173',                      // Vite preview
  ].filter(Boolean);
  const origin = req.headers.origin || '';
  const corsOrigin = allowedOrigins.includes(origin) ? origin : allowedOrigins[0] || '';
  res.setHeader('Access-Control-Allow-Origin', corsOrigin);
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  res.setHeader('Vary', 'Origin');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  // ── Rate limiting ────────────────────────────────────────────────
  const clientIp = req.headers['x-forwarded-for']?.split(',')[0]?.trim() || req.socket?.remoteAddress || 'unknown';
  if (isRateLimited(clientIp)) {
    return res.status(429).json({ error: 'Too many requests. Try again in a minute.' });
  }

  // ── Firebase token verification ──────────────────────────────────
  const authHeader = req.headers.authorization || '';
  const idToken = authHeader.startsWith('Bearer ') ? authHeader.slice(7) : '';
  if (!idToken) {
    return res.status(401).json({ error: 'Missing Authorization header' });
  }

  try {
    initFirebaseAdmin();
    await admin.auth().verifyIdToken(idToken);
  } catch (authErr) {
    console.error('Firebase token verification failed:', authErr.message);
    return res.status(401).json({ error: 'Invalid or expired token' });
  }

  const { IdStudent, IdBinusian } = req.body || {};

  if (!IdStudent) {
    return res.status(400).json({ error: 'IdStudent is required' });
  }

  // Validate IdStudent format — numeric only, 7-12 digits
  if (!/^\d{7,12}$/.test(String(IdStudent))) {
    return res.status(400).json({ error: 'Invalid IdStudent format' });
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
    return res.status(502).json({ error: 'Attendance submission failed' });
  } catch (err) {
    console.error('BINUS API proxy error:', err);
    return res.status(500).json({ error: 'Internal server error' });
  }
}
