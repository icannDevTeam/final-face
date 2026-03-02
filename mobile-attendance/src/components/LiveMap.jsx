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
 * LiveMap — renders a Leaflet map with campus geofence + student position.
 */
export default function LiveMap({ proximity }) {
  const campus = {
    lat: proximity?.campusLat ?? -6.2307,
    lng: proximity?.campusLng ?? 106.7865,
    radius: proximity?.campusRadius ?? 500,
  };
  const userPos =
    proximity?.lat != null && proximity?.lng != null
      ? { lat: proximity.lat, lng: proximity.lng }
      : null;
  const inRange = proximity?.inRange ?? false;

  return (
    <div style={{
      borderRadius: 'var(--radius-2xl, 1.25rem)',
      overflow: 'hidden',
      boxShadow: 'var(--glass-shadow, 0 4px 16px rgba(0,60,120,.07))',
      border: '1px solid var(--glass-border, #F1F5F9)',
      margin: '0 1rem 1rem',
      background: 'var(--glass-bg, #fff)',
      backdropFilter: 'blur(12px)',
      WebkitBackdropFilter: 'blur(12px)',
    }}>
      {/* Map label */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0.6rem 0.9rem',
        borderBottom: '1px solid var(--glass-border, #F1F5F9)',
        fontSize: '0.72rem',
        fontWeight: 600,
        color: 'var(--text-secondary, #5A6B82)',
        letterSpacing: '0.04em',
        textTransform: 'uppercase',
      }}>
        <span style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
          <span>📍</span> Live Location
        </span>
        {userPos && (
          <span style={{
            fontWeight: 600,
            fontSize: '0.7rem',
            color: inRange ? 'var(--success-dark, #047857)' : 'var(--warning-dark, #B45309)',
            background: inRange ? 'var(--success-bg, #ECFDF5)' : 'var(--warning-bg, #FFFBEB)',
            padding: '0.15rem 0.5rem',
            borderRadius: 'var(--radius-full, 9999px)',
            textTransform: 'none',
            letterSpacing: '0',
            border: `1px solid ${inRange ? 'rgba(16,185,129,0.15)' : 'rgba(245,158,11,0.15)'}`,
          }}>
            {inRange ? '● In range' : '○ Out of range'}
          </span>
        )}
      </div>
      <MapContainer
        center={[campus.lat, campus.lng]}
        zoom={16}
        scrollWheelZoom={false}
        dragging={true}
        zoomControl={false}
        attributionControl={false}
        style={{ height: '200px', width: '100%' }}
      >
        <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />

        <Circle
          center={[campus.lat, campus.lng]}
          radius={campus.radius}
          pathOptions={{
            color: inRange ? '#10B981' : '#F59E0B',
            fillColor: inRange ? '#ECFDF5' : '#FFFBEB',
            fillOpacity: 0.2,
            weight: 2,
            dashArray: '8 5',
          }}
        />

        <Marker position={[campus.lat, campus.lng]} icon={campusIcon} />

        {userPos && (
          <Marker position={[userPos.lat, userPos.lng]} icon={userIcon} />
        )}

        <MapAutoCenter userPos={userPos} campusPos={campus} />
      </MapContainer>
    </div>
  );
}
