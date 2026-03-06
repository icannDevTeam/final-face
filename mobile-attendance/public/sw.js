const CACHE_NAME = 'binus-attendance-v10';
const STATIC_CACHE = 'binus-static-v10';
const MODEL_CACHE = 'binus-models-v1';   // ML models rarely change — long-lived

const PRECACHE_URLS = [
  '/',
  '/index.html',
  '/manifest.json',
];

// face-api.js model files — cache-first, never expire
const MODEL_FILES = [
  '/models/ssd_mobilenetv1_model-weights_manifest.json',
  '/models/ssd_mobilenetv1_model-shard1',
  '/models/ssd_mobilenetv1_model-shard2',
  '/models/face_landmark_68_model-weights_manifest.json',
  '/models/face_landmark_68_model-shard1',
  '/models/face_recognition_model-weights_manifest.json',
  '/models/face_recognition_model-shard1',
  '/models/face_recognition_model-shard2',
];

// Install — cache shell + precache ML models
self.addEventListener('install', (event) => {
  event.waitUntil(
    Promise.all([
      caches.open(STATIC_CACHE).then((cache) => cache.addAll(PRECACHE_URLS)),
      caches.open(MODEL_CACHE).then((cache) => cache.addAll(MODEL_FILES)),
    ])
  );
  self.skipWaiting();
});

// Listen for SKIP_WAITING message from the client
self.addEventListener('message', (event) => {
  if (event.data && event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }
});

// Activate — clean old caches
self.addEventListener('activate', (event) => {
  const keepCaches = [CACHE_NAME, STATIC_CACHE, MODEL_CACHE];
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => !keepCaches.includes(k)).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

// Fetch — network-first for API, stale-while-revalidate for static assets
self.addEventListener('fetch', (event) => {
  // Skip non-GET requests
  if (event.request.method !== 'GET') return;

  const url = new URL(event.request.url);

  // API calls: always network-first (attendance data must be fresh)
  if (url.pathname.startsWith('/api/')) {
    event.respondWith(
      fetch(event.request)
        .then((response) => {
          if (response.ok) {
            const clone = response.clone();
            caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
          }
          return response;
        })
        .catch(() => caches.match(event.request))
    );
    return;
  }

  // ML model files: cache-first — these are large binary blobs that never change
  if (url.pathname.startsWith('/models/')) {
    event.respondWith(
      caches.match(event.request).then((cached) => {
        if (cached) return cached;
        return fetch(event.request).then((response) => {
          if (response.ok) {
            const clone = response.clone();
            caches.open(MODEL_CACHE).then((cache) => cache.put(event.request, clone));
          }
          return response;
        });
      })
    );
    return;
  }

  // Map tiles (OSM, MapTiler, Mapbox, Google, Carto, etc.): always fetch
  // from network, bypassing browser HTTP cache. Do NOT cache tiles in SW —
  // tile CDNs handle caching; storing them locally causes stale map issues.
  const tileHosts = ['tile.openstreetmap.org', 'tile.osm.org', 'api.maptiler.com',
    'api.mapbox.com', 'tiles.mapbox.com', 'mt0.google.com', 'mt1.google.com',
    'basemaps.cartocdn.com', 'server.arcgisonline.com'];
  if (tileHosts.some((h) => url.hostname.includes(h)) || url.pathname.match(/\/\d+\/\d+\/\d+/)) {
    event.respondWith(
      fetch(event.request, { cache: 'no-store' })
        .catch(() => caches.match(event.request))
    );
    return;
  }

  // Hashed assets (/assets/index-CceGdpPY.js): cache-first is safe
  // because Vite appends a content hash — new deploys = new filenames
  if (url.pathname.startsWith('/assets/') && url.pathname.match(/-[A-Za-z0-9_-]{8,}\./)) {
    event.respondWith(
      caches.match(event.request).then((cached) => {
        if (cached) return cached;
        return fetch(event.request).then((response) => {
          if (response.ok) {
            const clone = response.clone();
            caches.open(STATIC_CACHE).then((cache) => cache.put(event.request, clone));
          }
          return response;
        });
      })
    );
    return;
  }

  // Navigation requests: network-first with offline fallback
  if (event.request.mode === 'navigate') {
    event.respondWith(
      fetch(event.request)
        .then((response) => {
          if (response.ok) {
            const clone = response.clone();
            caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
          }
          return response;
        })
        .catch(() => caches.match('/index.html'))
    );
    return;
  }

  // Default: network-first (everything else, including non-hashed static files)
  event.respondWith(
    fetch(event.request)
      .then((response) => {
        if (response.ok) {
          const clone = response.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
        }
        return response;
      })
      .catch(() => caches.match(event.request))
  );
});
