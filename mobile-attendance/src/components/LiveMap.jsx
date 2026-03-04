import { useEffect, useRef } from 'react';
import { MapContainer, TileLayer, Marker, Circle, Polygon, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
const TILE_URL = import.meta.env.VITE_MAP_TILE_URL || 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png';
const TILE_ATTRIBUTION =
  import.meta.env.VITE_MAP_TILE_ATTRIBUTION ||
  '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors';


// Fix default marker icon issue in bundlers
delete L.Icon.Default.prototype._getIconUrl;

// Campus marker (blue pin)
const campusIcon = new L.DivIcon({
  className: 'campus-marker',
  html: `<div style="position:relative;width:24px;height:24px;">
    <div style="
      position:absolute;inset:0;border-radius:50%;
      background:rgba(0,84,166,.15);
    "></div>
    <div style="
      position:absolute;top:4px;left:4px;width:16px;height:16px;
      border-radius:50%;background:#0054A6;
      border:2.5px solid #fff;
      box-shadow:0 2px 8px rgba(0,20,60,.3);
    "></div>
  </div>`,
  iconSize: [24, 24],
  iconAnchor: [12, 12],
});

// User marker (pulsing teal dot)
const userIcon = new L.DivIcon({
  className: 'user-marker',
  html: `<div style="position:relative;width:24px;height:24px;">
    <div style="
      position:absolute;inset:0;border-radius:50%;
      background:rgba(0,163,224,.2);
      animation:userPulse 2s ease-in-out infinite;
    "></div>
    <div style="
      position:absolute;top:5px;left:5px;width:14px;height:14px;
      border-radius:50%;background:#00A3E0;
      border:2.5px solid #fff;box-shadow:0 2px 8px rgba(0,20,60,.25);
    "></div>
  </div>`,
  iconSize: [24, 24],
  iconAnchor: [12, 12],
});

// Inject keyframe animation once
if (typeof document !== 'undefined' && !document.getElementById('user-pulse-css')) {
  const style = document.createElement('style');
  style.id = 'user-pulse-css';
  style.textContent = `
    @keyframes userPulse {
      0%,100%{transform:scale(1);opacity:.6}
      50%{transform:scale(1.6);opacity:0}
    }
  `;
  document.head.appendChild(style);
}

/**
 * Inner component that pans the map when user position changes.
 */
function MapAutoCenter({ userPos, campusPos, campusPolygon }) {
  const map = useMap();
  const hasZoomed = useRef(false);

  useEffect(() => {
    const polygonPoints = campusPolygon?.flatMap((ring) => ring) || [];
    const anchors = polygonPoints.length ? polygonPoints : [[campusPos.lat, campusPos.lng]];

    if (!hasZoomed.current) {
      const boundsPoints = userPos ? [...anchors, [userPos.lat, userPos.lng]] : anchors;
      if (boundsPoints.length === 1) {
        map.setView(boundsPoints[0], 17);
      } else {
        const bounds = L.latLngBounds(boundsPoints).pad(0.2);
        map.fitBounds(bounds, { maxZoom: 18 });
      }
      hasZoomed.current = true;
    } else if (userPos) {
      map.panTo([userPos.lat, userPos.lng], { animate: true });
    }
  }, [userPos, campusPos, campusPolygon, map]);

  return null;
}

/**
 * LiveMap — renders a Leaflet map showing:
 *  • Campus center pin + geofence circle
 *  • Student's live GPS position
 *
 * Props:
 *  campusLat, campusLng  — campus centre
 *  campusRadius          — geofence radius in metres
 *  userLat, userLng      — student's current position (null when unknown)
 *  inRange               — whether the student is inside the geofence
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
}) {
  const campusPos = { lat: campusLat, lng: campusLng };
  const userPos = userLat != null && userLng != null ? { lat: userLat, lng: userLng } : null;
  const polygonPositions = campusPolygon?.length ? campusPolygon : null;
  const zoneColor = inRange ? '#10B981' : '#F59E0B';
  const zoneFill = inRange ? 'rgba(16,185,129,0.15)' : 'rgba(245,158,11,0.15)';

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
      {/* Map label */}
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
      <MapContainer
        center={[campusLat, campusLng]}
        zoom={16}
        scrollWheelZoom={false}
        dragging={true}
        zoomControl={false}
        attributionControl={false}
        style={{ height: '200px', width: '100%' }}
      >
        <TileLayer
          url={TILE_URL}
          attribution={TILE_ATTRIBUTION}
        />

        {/* Geofence shape */}
        {polygonPositions ? (
          <Polygon
            positions={polygonPositions}
            pathOptions={{
              color: zoneColor,
              fillColor: zoneFill,
              fillOpacity: 0.35,
              weight: 2,
            }}
          />
        ) : (
          <Circle
            center={[campusLat, campusLng]}
            radius={campusRadius}
            pathOptions={{
              color: zoneColor,
              fillColor: zoneFill,
              fillOpacity: 0.25,
              weight: 2,
              dashArray: '8 5',
            }}
          />
        )}

        {/* Campus centre pin */}
        <Marker position={[campusLat, campusLng]} icon={campusIcon} />

        {/* Student live position */}
        {userPos && (
          <Marker position={[userPos.lat, userPos.lng]} icon={userIcon} />
        )}

        <MapAutoCenter userPos={userPos} campusPos={campusPos} campusPolygon={polygonPositions} />
      </MapContainer>
    </div>
  );
}
