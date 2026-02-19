import { useEffect, useRef } from 'react';
import { MapContainer, TileLayer, Marker, Circle, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

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
function MapAutoCenter({ userPos, campusPos }) {
  const map = useMap();
  const hasZoomed = useRef(false);

  useEffect(() => {
    if (!userPos) return;
    if (!hasZoomed.current) {
      // Fit both markers on first position fix
      const bounds = L.latLngBounds([
        [campusPos.lat, campusPos.lng],
        [userPos.lat, userPos.lng],
      ]).pad(0.3);
      map.fitBounds(bounds, { maxZoom: 17 });
      hasZoomed.current = true;
    } else {
      map.panTo([userPos.lat, userPos.lng], { animate: true });
    }
  }, [userPos, campusPos, map]);

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
  userLat,
  userLng,
  inRange,
}) {
  const campusPos = { lat: campusLat, lng: campusLng };
  const userPos = userLat != null && userLng != null ? { lat: userLat, lng: userLng } : null;

  return (
    <div style={{
      borderRadius: 'var(--radius-lg, 1.25rem)',
      overflow: 'hidden',
      boxShadow: 'var(--shadow-md, 0 4px 16px rgba(0,60,120,.07))',
      border: '1px solid var(--border-light, #F1F5F9)',
      margin: '0 1rem 1rem',
      background: '#fff',
    }}>
      {/* Map label */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0.6rem 0.9rem',
        borderBottom: '1px solid var(--border-light, #F1F5F9)',
        fontSize: '0.72rem',
        fontWeight: 600,
        color: 'var(--text-secondary, #5A6B82)',
        letterSpacing: '0.04em',
        textTransform: 'uppercase',
      }}>
        <span>📍 Live Location</span>
        {userPos && (
          <span style={{
            fontWeight: 500,
            fontSize: '0.7rem',
            color: inRange ? 'var(--success-dark, #047857)' : 'var(--warning-dark, #B45309)',
            background: inRange ? 'var(--success-bg, #ECFDF5)' : 'var(--warning-bg, #FFFBEB)',
            padding: '0.15rem 0.5rem',
            borderRadius: 'var(--radius-full, 9999px)',
            textTransform: 'none',
            letterSpacing: '0',
          }}>
            {inRange ? '● In range' : '○ Out of range'}
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
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {/* Geofence radius circle */}
        <Circle
          center={[campusLat, campusLng]}
          radius={campusRadius}
          pathOptions={{
            color: inRange ? '#10B981' : '#F59E0B',
            fillColor: inRange ? '#ECFDF5' : '#FFFBEB',
            fillOpacity: 0.2,
            weight: 2,
            dashArray: '8 5',
          }}
        />

        {/* Campus centre pin */}
        <Marker position={[campusLat, campusLng]} icon={campusIcon} />

        {/* Student live position */}
        {userPos && (
          <Marker position={[userPos.lat, userPos.lng]} icon={userIcon} />
        )}

        <MapAutoCenter userPos={userPos} campusPos={campusPos} />
      </MapContainer>
    </div>
  );
}
