import { useEffect, useRef } from 'react';
import { MapContainer, TileLayer, Marker, Circle, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// Fix default marker icon issue in bundlers
delete L.Icon.Default.prototype._getIconUrl;

// Campus marker (blue pin)
const campusIcon = new L.DivIcon({
  className: 'campus-marker',
  html: `<div style="
    width:18px;height:18px;border-radius:50%;
    background:#0054A6;border:3px solid #fff;
    box-shadow:0 2px 6px rgba(0,0,0,.35);
  "></div>`,
  iconSize: [18, 18],
  iconAnchor: [9, 9],
});

// User marker (pulsing teal dot)
const userIcon = new L.DivIcon({
  className: 'user-marker',
  html: `<div style="position:relative;width:20px;height:20px;">
    <div style="
      position:absolute;inset:0;border-radius:50%;
      background:rgba(0,163,224,.25);
      animation:userPulse 2s ease-in-out infinite;
    "></div>
    <div style="
      position:absolute;top:4px;left:4px;width:12px;height:12px;
      border-radius:50%;background:#00A3E0;
      border:2.5px solid #fff;box-shadow:0 1px 4px rgba(0,0,0,.3);
    "></div>
  </div>`,
  iconSize: [20, 20],
  iconAnchor: [10, 10],
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
      borderRadius: '1rem',
      overflow: 'hidden',
      boxShadow: '0 2px 12px rgba(0,0,0,.08)',
      border: '1px solid var(--border, #e5e7eb)',
      margin: '0 1rem 1rem',
    }}>
      <MapContainer
        center={[campusLat, campusLng]}
        zoom={16}
        scrollWheelZoom={false}
        dragging={true}
        zoomControl={false}
        attributionControl={false}
        style={{ height: '220px', width: '100%' }}
      >
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {/* Geofence radius circle */}
        <Circle
          center={[campusLat, campusLng]}
          radius={campusRadius}
          pathOptions={{
            color: inRange ? '#16a34a' : '#ea580c',
            fillColor: inRange ? '#dcfce7' : '#fef3c7',
            fillOpacity: 0.25,
            weight: 2,
            dashArray: '6 4',
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
