import { useEffect, useRef, useMemo } from 'react';
import { MapContainer, TileLayer, Marker, Circle, Polygon, Polyline, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// ── Tile layer — CartoDB Voyager is cleaner for mobile ────────────────
const TILE_URL =
  import.meta.env.VITE_MAP_TILE_URL ||
  'https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png';
const TILE_ATTRIBUTION =
  import.meta.env.VITE_MAP_TILE_ATTRIBUTION ||
  '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>';

// Fix default marker icon issue in bundlers
delete L.Icon.Default.prototype._getIconUrl;

// ── Inject map-marker CSS animations once ─────────────────────────────
if (typeof document !== 'undefined' && !document.getElementById('livemap-css')) {
  const style = document.createElement('style');
  style.id = 'livemap-css';
  style.textContent = `
    /* ── User pulsing ring ────────────────── */
    @keyframes lmPulse {
      0%   { transform: scale(0.5); opacity: 0.7; }
      100% { transform: scale(2.8); opacity: 0; }
    }
    @keyframes lmPulse2 {
      0%   { transform: scale(0.5); opacity: 0.45; }
      100% { transform: scale(3.6); opacity: 0; }
    }
    @keyframes lmBounceIn {
      0%   { transform: scale(0) translateY(8px); opacity: 0; }
      50%  { transform: scale(1.2) translateY(-2px); opacity: 1; }
      100% { transform: scale(1) translateY(0); opacity: 1; }
    }
    @keyframes lmFloat {
      0%, 100% { transform: translateY(0); }
      50%      { transform: translateY(-4px); }
    }
    @keyframes lmDash {
      to { stroke-dashoffset: -24; }
    }
    @keyframes lmGlow {
      0%, 100% { box-shadow: 0 0 0 0 rgba(0,84,166,.3); }
      50%      { box-shadow: 0 0 0 10px rgba(0,84,166,0); }
    }
    .lm-user-ring1 {
      animation: lmPulse 2s cubic-bezier(0,.5,.5,1) infinite;
    }
    .lm-user-ring2 {
      animation: lmPulse2 2s cubic-bezier(0,.5,.5,1) 0.4s infinite;
    }
    .lm-campus-marker {
      animation: lmBounceIn 0.6s cubic-bezier(.34,1.56,.64,1) both;
    }
    .lm-campus-label {
      animation: lmBounceIn 0.6s cubic-bezier(.34,1.56,.64,1) 0.2s both;
    }
    .lm-campus-pin {
      animation: lmGlow 2.5s ease-in-out infinite;
    }
    .lm-user-marker {
      animation: lmBounceIn 0.5s cubic-bezier(.34,1.56,.64,1) 0.1s both;
    }
    .lm-route-line {
      animation: lmDash 1s linear infinite;
    }
    /* hide leaflet zoom controls nicely */
    .leaflet-control-zoom {
      border: none !important;
      box-shadow: 0 2px 12px rgba(0,0,0,.1) !important;
      border-radius: 12px !important;
      overflow: hidden;
      margin-right: 12px !important;
      margin-bottom: 12px !important;
    }
    .leaflet-control-zoom a {
      width: 36px !important;
      height: 36px !important;
      line-height: 36px !important;
      font-size: 16px !important;
      color: #1a1a1a !important;
      background: rgba(255,255,255,0.95) !important;
      backdrop-filter: blur(8px) !important;
    }
    .leaflet-control-zoom a:hover {
      background: #f1f5f9 !important;
    }
  `;
  document.head.appendChild(style);
}

// ── Campus marker — large pin with label ──────────────────────────────
const campusIcon = new L.DivIcon({
  className: '',
  html: `
    <div class="lm-campus-marker" style="position:relative;width:56px;height:68px;filter:drop-shadow(0 4px 12px rgba(0,20,60,.25));">
      <!-- Pin body -->
      <div class="lm-campus-pin" style="
        position:absolute;top:0;left:8px;width:40px;height:40px;
        border-radius:50% 50% 50% 0;
        transform:rotate(-45deg);
        background:linear-gradient(135deg,#0054A6 0%,#003d7a 100%);
        border:3px solid #fff;
      "></div>
      <!-- Building icon -->
      <div style="
        position:absolute;top:6px;left:14px;width:28px;height:28px;
        display:flex;align-items:center;justify-content:center;
        color:#fff;font-size:16px;
        z-index:2;
      ">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
          <path d="M3 21h18"/>
          <path d="M5 21V7l8-4v18"/>
          <path d="M19 21V11l-6-4"/>
          <path d="M9 9v.01"/><path d="M9 12v.01"/><path d="M9 15v.01"/><path d="M9 18v.01"/>
        </svg>
      </div>
      <!-- Pin tip -->
      <div style="
        position:absolute;top:36px;left:22px;width:12px;height:12px;
        background:linear-gradient(135deg,#003d7a,#002a56);
        clip-path:polygon(0 0,100% 0,50% 100%);
      "></div>
    </div>
  `,
  iconSize: [56, 68],
  iconAnchor: [28, 52],
});

// Campus label (separate overlay below pin)
const campusLabelIcon = new L.DivIcon({
  className: '',
  html: `
    <div class="lm-campus-label" style="
      white-space:nowrap;
      background:#fff;
      border:1px solid #e2e8f0;
      border-radius:8px;
      padding:3px 10px;
      font-size:11px;
      font-weight:700;
      color:#0054A6;
      box-shadow:0 2px 8px rgba(0,20,60,.12);
      letter-spacing:0.02em;
      font-family:Inter,-apple-system,sans-serif;
    ">BINUS School</div>
  `,
  iconSize: [100, 24],
  iconAnchor: [50, -4],
});

// ── User marker — large pulsing dot with rings ───────────────────────
function makeUserIcon(inRange) {
  const color = inRange ? '#10B981' : '#F97316';
  const ringColor = inRange ? 'rgba(16,185,129,' : 'rgba(249,115,22,';
  return new L.DivIcon({
    className: '',
    html: `
      <div class="lm-user-marker" style="position:relative;width:64px;height:64px;">
        <!-- Outer pulse rings -->
        <div class="lm-user-ring1" style="
          position:absolute;inset:0;border-radius:50%;
          background:${ringColor}0.12);
        "></div>
        <div class="lm-user-ring2" style="
          position:absolute;inset:4px;border-radius:50%;
          background:${ringColor}0.08);
        "></div>
        <!-- Accuracy halo -->
        <div style="
          position:absolute;inset:10px;border-radius:50%;
          background:${ringColor}0.18);
          border:1.5px solid ${ringColor}0.25);
        "></div>
        <!-- Core dot -->
        <div style="
          position:absolute;top:22px;left:22px;width:20px;height:20px;
          border-radius:50%;
          background:${color};
          border:3px solid #fff;
          box-shadow:0 2px 10px ${ringColor}0.5);
        "></div>
      </div>
    `,
    iconSize: [64, 64],
    iconAnchor: [32, 32],
  });
}

// ── Generate curved path between two points ───────────────────────────
function getCurvedRoute(from, to, numPoints = 40) {
  if (!from || !to) return [];
  const midLat = (from[0] + to[0]) / 2;
  const midLng = (from[1] + to[1]) / 2;
  // Offset the midpoint perpendicular to the line for a nice curve
  const dLat = to[0] - from[0];
  const dLng = to[1] - from[1];
  const dist = Math.sqrt(dLat * dLat + dLng * dLng);
  const curvature = Math.min(dist * 0.25, 0.003); // scale curve to distance
  const ctrlLat = midLat + dLng * (curvature / (dist || 1));
  const ctrlLng = midLng - dLat * (curvature / (dist || 1));

  const points = [];
  for (let i = 0; i <= numPoints; i++) {
    const t = i / numPoints;
    const lat = (1 - t) * (1 - t) * from[0] + 2 * (1 - t) * t * ctrlLat + t * t * to[0];
    const lng = (1 - t) * (1 - t) * from[1] + 2 * (1 - t) * t * ctrlLng + t * t * to[1];
    points.push([lat, lng]);
  }
  return points;
}

// ── Animated route overlay (SVG dashed line) ──────────────────────────
function AnimatedRoute({ from, to, inRange }) {
  const map = useMap();
  const svgRef = useRef(null);
  const positions = useMemo(() => getCurvedRoute(from, to), [from, to]);

  useEffect(() => {
    if (!positions.length || !map) return;

    // Create an SVG overlay for the animated dashed route
    const svg = L.svg();
    svg.addTo(map);
    svgRef.current = svg;

    return () => {
      if (svgRef.current) {
        map.removeLayer(svgRef.current);
      }
    };
  }, [map, positions]);

  if (!positions.length) return null;

  const color = inRange ? '#10B981' : '#F97316';

  return (
    <>
      {/* Soft glow line underneath */}
      <Polyline
        positions={positions}
        pathOptions={{
          color: inRange ? 'rgba(16,185,129,0.2)' : 'rgba(249,115,22,0.2)',
          weight: 8,
          lineCap: 'round',
          lineJoin: 'round',
        }}
      />
      {/* Main dashed animated line */}
      <Polyline
        positions={positions}
        pathOptions={{
          color,
          weight: 3,
          dashArray: '8 12',
          lineCap: 'round',
          lineJoin: 'round',
          className: 'lm-route-line',
        }}
      />
      {/* Start dot (user end) */}
      <Circle
        center={positions[0]}
        radius={3}
        pathOptions={{
          color,
          fillColor: color,
          fillOpacity: 1,
          weight: 0,
        }}
      />
      {/* End dot (campus end) */}
      <Circle
        center={positions[positions.length - 1]}
        radius={3}
        pathOptions={{
          color,
          fillColor: color,
          fillOpacity: 1,
          weight: 0,
        }}
      />
    </>
  );
}

// ── Map auto-center with smooth fly animation ─────────────────────────
function MapAutoCenter({ userPos, campusPos, campusPolygon, fullScreen }) {
  const map = useMap();
  const hasZoomed = useRef(false);

  useEffect(() => {
    const polygonPoints = campusPolygon?.flatMap((ring) => ring) || [];
    const anchors = polygonPoints.length ? polygonPoints : [[campusPos.lat, campusPos.lng]];

    if (!hasZoomed.current) {
      const boundsPoints = userPos ? [...anchors, [userPos.lat, userPos.lng]] : anchors;
      if (boundsPoints.length === 1) {
        map.flyTo(boundsPoints[0], fullScreen ? 17 : 16, {
          duration: 1.5,
          easeLinearity: 0.25,
        });
      } else {
        const bounds = L.latLngBounds(boundsPoints).pad(fullScreen ? 0.3 : 0.2);
        map.flyToBounds(bounds, {
          maxZoom: fullScreen ? 18 : 17,
          duration: 1.5,
          easeLinearity: 0.25,
          paddingBottomRight: fullScreen ? [20, 260] : [0, 0], // avoid bottom card
        });
      }
      hasZoomed.current = true;
    } else if (userPos) {
      map.panTo([userPos.lat, userPos.lng], { animate: true, duration: 0.8 });
    }
  }, [userPos, campusPos, campusPolygon, map, fullScreen]);

  return null;
}

/**
 * LiveMap — renders a Leaflet map with animated markers, route line,
 * and geofence visualization.
 */
export default function LiveMap({
  campusLat,
  campusLng,
  campusRadius,
  campusPolygon,
  userLat,
  userLng,
  inRange,
  geofence,
  fullScreen = false,
}) {
  const campusPos = { lat: campusLat, lng: campusLng };
  const userPos = userLat != null && userLng != null ? { lat: userLat, lng: userLng } : null;
  const polygonPositions = campusPolygon?.length ? campusPolygon : null;
  const zoneColor = '#0054A6';
  const zoneFill = 'rgba(0,84,166,0.10)';
  const userMarkerIcon = useMemo(() => makeUserIcon(inRange), [inRange]);

  const mapEl = (
    <MapContainer
      center={[campusLat, campusLng]}
      zoom={fullScreen ? 17 : 16}
      scrollWheelZoom={fullScreen}
      dragging={true}
      zoomControl={false}
      attributionControl={false}
      style={{ height: fullScreen ? '100%' : '200px', width: '100%' }}
    >
      <TileLayer
        url={TILE_URL}
        attribution={TILE_ATTRIBUTION}
        maxZoom={19}
      />

      {/* Geofence shape */}
      {polygonPositions ? (
        <Polygon
          positions={polygonPositions}
          pathOptions={{
            color: zoneColor,
            fillColor: zoneFill,
            fillOpacity: 0.3,
            weight: 2.5,
            dashArray: fullScreen ? '6 8' : '8 5',
          }}
        />
      ) : (
        <Circle
          center={[campusLat, campusLng]}
          radius={campusRadius}
          pathOptions={{
            color: zoneColor,
            fillColor: zoneFill,
            fillOpacity: 0.12,
            weight: 2.5,
            dashArray: '6 8',
          }}
        />
      )}

      {/* Animated route line between user and campus */}
      {userPos && fullScreen && (
        <AnimatedRoute
          from={[userPos.lat, userPos.lng]}
          to={[campusLat, campusLng]}
          inRange={inRange}
        />
      )}

      {/* Campus centre pin */}
      <Marker position={[campusLat, campusLng]} icon={campusIcon} />
      {/* Campus label */}
      {fullScreen && (
        <Marker position={[campusLat, campusLng]} icon={campusLabelIcon} />
      )}

      {/* Student live position */}
      {userPos && (
        <Marker position={[userPos.lat, userPos.lng]} icon={userMarkerIcon} />
      )}

      <MapAutoCenter
        userPos={userPos}
        campusPos={campusPos}
        campusPolygon={polygonPositions}
        fullScreen={fullScreen}
      />
    </MapContainer>
  );

  // Full-screen mode: no card wrapper, just the raw map
  if (fullScreen) {
    return mapEl;
  }

  // Card mode: wrapped with label header
  return (
    <div style={{
      borderRadius: 'var(--radius-2xl, 2rem)',
      overflow: 'hidden',
      boxShadow: 'var(--glass-shadow, 0 8px 32px rgba(0,40,85,.08))',
      border: '1px solid var(--glass-border, rgba(255,255,255,0.25))',
      margin: '0 1rem 1rem',
      background: 'var(--glass-bg, rgba(255,255,255,0.72))',
      backdropFilter: 'blur(12px)',
      WebkitBackdropFilter: 'blur(12px)',
    }}>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0.65rem 1rem',
        borderBottom: '1px solid var(--glass-border, rgba(255,255,255,0.25))',
        fontSize: '0.72rem',
        fontWeight: 600,
        color: 'var(--text-secondary, #5A6B82)',
        letterSpacing: '0.04em',
        textTransform: 'uppercase',
      }}>
        <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <span style={{ fontSize: '0.85rem' }}>📍</span> Live Location
        </span>
        {userPos && (
          <span style={{
            fontWeight: 600,
            fontSize: '0.68rem',
            color: inRange ? 'var(--success-dark, #047857)' : 'var(--warning-dark, #B45309)',
            background: inRange ? 'var(--success-bg, #ECFDF5)' : 'var(--warning-bg, #FFFBEB)',
            padding: '0.2rem 0.6rem',
            borderRadius: 'var(--radius-full, 9999px)',
            textTransform: 'none',
            letterSpacing: '0',
            border: inRange ? '1px solid rgba(16,185,129,0.15)' : '1px solid rgba(245,158,11,0.15)',
          }}>
            {inRange ? '● In range' : '○ Out of range'} · {geofence === 'polygon' ? 'Polygon lock' : 'Radius lock'}
          </span>
        )}
      </div>
      {mapEl}
    </div>
  );
}
