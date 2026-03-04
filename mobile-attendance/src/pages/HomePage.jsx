import { useState, useEffect, useCallback, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { checkProximity, watchProximity, getCampusConfig } from '../lib/geolocation';
import styles from '../styles/Home.module.css';

// Dynamically import LiveMap only when needed (Leaflet is heavy)
let LiveMapComponent = null;

const STATUS = {
  OUT: 'out',
  IN: 'in',
  CHECKING: 'checking',
  ERROR: 'error',
};

// ─── Pulse rings ─────────────────────────────────────────────────────

function PulseRing({ status }) {
  return (
    <div className={styles.fingerprintWrap} style={{ position: 'absolute', inset: 0 }}>
      {[0, 1, 2].map((i) => (
        <div
          key={i}
          className={`${styles.pulseRing} ${styles[status]}`}
          style={{
            width: `${80 + i * 30}%`,
            height: `${80 + i * 30}%`,
            opacity: 0.15 - i * 0.04,
            animationDelay: `${i * 0.4}s`,
          }}
        />
      ))}
    </div>
  );
}

// ─── Fingerprint icon ────────────────────────────────────────────────

function FingerprintIcon({ status }) {
  const isChecking = status === STATUS.CHECKING;

  return (
    <div className={styles.fingerprintWrap}>
      <PulseRing status={status} />
      <div className={`${styles.fingerprintCircle} ${styles[status]}`}>
        <svg className={styles.fingerprintSvg} viewBox="0 0 44 44" fill="none">
          <path d="M22 8C14.268 8 8 14.268 8 22" stroke="white" strokeWidth="2.5" strokeLinecap="round" />
          <path d="M22 8C29.732 8 36 14.268 36 22" stroke="white" strokeWidth="2.5" strokeLinecap="round" />
          <path d="M14 22C14 17.582 17.582 14 22 14C26.418 14 30 17.582 30 22C30 26.418 26.418 30 22 30" stroke="white" strokeWidth="2.5" strokeLinecap="round" />
          <path d="M22 14C22 14 22 22 22 26C22 28.209 20.209 30 18 30" stroke="white" strokeWidth="2.5" strokeLinecap="round" />
          <path d="M26 20C26 17.791 24.209 16 22 16" stroke="white" strokeWidth="2.5" strokeLinecap="round" />
          <path d="M18 22C18 19.791 19.791 18 22 18C24.209 18 26 19.791 26 22C26 24.209 24.209 26 22 26" stroke="white" strokeWidth="2.5" strokeLinecap="round" />
          <circle cx="22" cy="22" r="2" fill="white" />
          {isChecking && (
            <circle
              className={styles.spinnerRing}
              cx="22" cy="22" r="18"
              stroke="white" strokeWidth="2"
              strokeDasharray="28 84"
              strokeLinecap="round"
            />
          )}
        </svg>
      </div>
    </div>
  );
}

// ─── Static map preview ──────────────────────────────────────────────

function MapPreview({ status, distance, accuracy, geofence, campusName }) {
  const isOut = status === STATUS.OUT || status === STATUS.ERROR;
  const safeCampusName = campusName || 'Campus';

  const circleColor = isOut ? 'rgba(153,27,27,0.2)' : 'rgba(0,84,166,0.2)';
  const strokeColor = isOut ? '#991B1B' : '#0054A6';
  const lineColor   = isOut ? '#991B1B' : '#0054A6';

  const distText = distance != null
    ? distance >= 1000
      ? `📍 ${(distance / 1000).toFixed(1)} km away`
      : `📍 ${distance}m away`
    : '📍 Measuring…';
  const zoneText = geofence === 'polygon'
    ? (isOut ? `Outside ${safeCampusName} zone` : `${safeCampusName} zone locked`)
    : null;

  return (
    <div className={styles.mapPreview}>
      {/* Fake map grid */}
      <svg className={styles.mapGrid}>
        {[...Array(8)].map((_, i) => (
          <line key={`h${i}`} x1="0" y1={i * 28} x2="100%" y2={i * 28} stroke="#94a3b8" strokeWidth="0.5" />
        ))}
        {[...Array(12)].map((_, i) => (
          <line key={`v${i}`} x1={i * 32} y1="0" x2={i * 32} y2="100%" stroke="#94a3b8" strokeWidth="0.5" />
        ))}
        {/* Fake roads */}
        <path d="M0,80 Q120,70 200,100 T400,90" stroke="#c8d6e5" strokeWidth="3" fill="none" />
        <path d="M0,130 Q150,120 300,140 T450,130" stroke="#c8d6e5" strokeWidth="3" fill="none" />
        <path d="M160,0 Q170,80 155,200" stroke="#c8d6e5" strokeWidth="3" fill="none" />
        <path d="M280,0 Q290,100 275,200" stroke="#b8cfe0" strokeWidth="5" fill="none" />
      </svg>

      {/* Radius circle */}
      <div
        className={styles.radiusCircle}
        style={{
          width: 110,
          height: 110,
          background: circleColor,
          borderColor: strokeColor,
        }}
      />

      {/* Line to user */}
      <svg className={styles.lineToUser}>
        <line
          x1="44%" y1="38%"
          x2={isOut ? '72%' : '58%'}
          y2={isOut ? '68%' : '55%'}
          stroke={lineColor}
          strokeWidth="2"
          strokeDasharray="4 3"
        />
      </svg>

      {/* Campus pin with BINUS logo */}
      <div className={styles.campusPin}>
        <div className={styles.campusPinLabel}>
          <img src="/logo.jpe" alt="BINUS" className={styles.campusPinLogo} />
          BINUS
        </div>
        <div className={styles.campusPinStem} />
        <div className={styles.campusPinDot} />
      </div>

      {/* User pin */}
      <div
        className={styles.userPin}
        style={{
          top: isOut ? '62%' : '50%',
          left: isOut ? '68%' : '55%',
        }}
      />

      {/* Distance label */}
      <div className={`${styles.distLabel} ${isOut ? styles.out : styles.in}`}>
        {geofence === 'polygon'
            ? zoneText || distText
          : (!isOut ? '📍 Within range' : distText)}
      </div>

      {/* Accuracy badge */}
      {accuracy != null && (
        <div className={styles.accBadge}>±{accuracy}m accuracy</div>
      )}
    </div>
  );
}

// ─── BINUS Spirit Values ticker ──────────────────────────────────────

const SPIRIT_VALUES = [
  { short: 'S', word: 'Striving', desc: 'for excellence' },
  { short: 'P', word: 'Perseverance', desc: 'in every challenge' },
  { short: 'I', word: 'Integrity', desc: 'in every action' },
  { short: 'R', word: 'Respect', desc: 'for all' },
  { short: 'I', word: 'Innovation', desc: 'for the future' },
  { short: 'T', word: 'Teamwork', desc: 'to achieve together' },
];

function SpiritTicker() {
  return (
    <div className={styles.tickerWrap}>
      <div className={styles.tickerLabel}>
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
          <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
        </svg>
        BINUS SPIRIT
      </div>
      <div className={styles.tickerTrack}>
        <div className={styles.tickerScroll}>
          {/* Double the items for seamless loop */}
          {[...SPIRIT_VALUES, ...SPIRIT_VALUES].map((v, i) => (
            <span key={i} className={styles.tickerItem}>
              <span className={styles.tickerLetter}>{v.short}</span>
              <span className={styles.tickerWord}>{v.word}</span>
              <span className={styles.tickerDesc}>{v.desc}</span>
              <span className={styles.tickerDot}>·</span>
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}

// ─── Main component ──────────────────────────────────────────────────

export default function HomePage() {
  const navigate = useNavigate();

  const [status, setStatus]       = useState(STATUS.CHECKING);
  const [gpsData, setGpsData]     = useState(null);      // { inRange, distance, accuracy, lat, lng, campusRadius }
  const [gpsError, setGpsError]   = useState(null);
  const [mapExpanded, setMapExpanded] = useState(false);
  const [useLiveMap, setUseLiveMap]   = useState(false);
  const [clock, setClock]         = useState(() => new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }));

  const cleanupRef = useRef(null);
  const campusCfg  = getCampusConfig();

  // ── Clock tick ─────────────────────────────────────────
  useEffect(() => {
    const id = setInterval(() => {
      setClock(new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }));
    }, 30_000);
    return () => clearInterval(id);
  }, []);

  // ── Initial proximity check + continuous watch ──────────
  // watchProximity fires immediately on first position, so we
  // combine both into one effect to avoid double GPS reads.
  useEffect(() => {
    let initialDone = false;

    const stop = watchProximity(
      (data) => {
        setGpsData(data);
        setStatus(data.inRange ? STATUS.IN : STATUS.OUT);
        setGpsError(null);
        initialDone = true;
      },
      (errMsg) => {
        setGpsError(errMsg);
        if (!initialDone) setStatus(STATUS.ERROR);
      }
    );
    cleanupRef.current = stop;

    // Fallback: if watchProximity hasn't fired in 5s, do a one-shot
    const fallback = setTimeout(() => {
      if (!initialDone) {
        checkProximity()
          .then((data) => {
            setGpsData(data);
            setStatus(data.inRange ? STATUS.IN : STATUS.OUT);
            setGpsError(null);
          })
          .catch((err) => {
            setGpsError(err.message);
            setStatus(STATUS.ERROR);
          });
      }
    }, 5000);

    return () => {
      stop();
      clearTimeout(fallback);
    };
  }, []);

  // ── Lazy-load LiveMap when map expands ─────────────────
  const handleMapToggle = useCallback(async () => {
    const next = !mapExpanded;
    setMapExpanded(next);

    if (next && !LiveMapComponent) {
      try {
        const mod = await import('../components/LiveMap');
        LiveMapComponent = mod.default;
        setUseLiveMap(true);
      } catch {
        // Leaflet load failed — fall back to static preview
        setUseLiveMap(false);
      }
    }
  }, [mapExpanded]);

  // ── Navigate to face scan ──────────────────────────────
  const handlePrimaryAction = useCallback(() => {
    if (status === STATUS.CHECKING) return;

    if (status === STATUS.IN) {
      navigate('/scan');
    } else {
      // Re-check proximity
      setStatus(STATUS.CHECKING);
      setGpsError(null);
      checkProximity()
        .then((data) => {
          setGpsData(data);
          setStatus(data.inRange ? STATUS.IN : STATUS.OUT);
        })
        .catch((err) => {
          setGpsError(err.message);
          setStatus(STATUS.ERROR);
        });
    }
  }, [status, navigate]);

  // ── Derived display values ─────────────────────────────
  const isChecking = status === STATUS.CHECKING;
  const isError    = status === STATUS.ERROR;
  const isOut      = status === STATUS.OUT;
  const isIn       = status === STATUS.IN;

  const geofenceType = gpsData?.geofence || campusCfg.geofence;
  const campusName = gpsData?.campusName || campusCfg.name;

  const distDisplay = gpsData
    ? gpsData.distance >= 1000
      ? `${(gpsData.distance / 1000).toFixed(1)} km`
      : `${gpsData.distance} m`
    : '—';

  const radiusDisplay = gpsData
    ? gpsData.campusRadius >= 1000
      ? `${(gpsData.campusRadius / 1000).toFixed(1)} km`
      : `${gpsData.campusRadius} m`
    : `${campusCfg.radius} m`;

  const zoneLabel = geofenceType === 'polygon' ? 'Attendance Zone' : 'Allowed Radius';
  const zoneValue = geofenceType === 'polygon' ? campusName : radiusDisplay;

  const accDisplay = gpsData ? `±${gpsData.accuracy} m` : '—';

  const statusLabel = isChecking
    ? 'Checking…'
    : isError
    ? 'GPS error'
    : isOut
    ? 'Out of range'
    : 'In range';

  const titleText = isChecking
    ? 'Detecting Location…'
    : isError
    ? 'Location Unavailable'
    : isOut
    ? 'Out of Attendance Zone'
    : 'Within Attendance Zone';

  const subText = isChecking
    ? 'Please wait while we check your GPS position'
    : isError
    ? gpsError || 'Enable location services and try again'
    : isOut
    ? 'Move closer to campus to mark attendance'
    : 'You\'re within the allowed area. Ready to check in!';

  const btnText = isChecking
    ? 'Checking location…'
    : isIn
    ? '✓ Mark Attendance'
    : 'Check Again';

  return (
    <div className={styles.page}>
      <div className={styles.card}>
        {/* ── Header ───────────────────────────────────── */}
        <div className={styles.header}>
          <div className={styles.headerLeft}>
            <div className={styles.logoBox}>
              <img src="/logo.jpe" alt="BINUS" className={styles.logoImg} />
            </div>
            <div>
              <p className={styles.headerTitle}>BINUS Attendance</p>
              <p className={styles.headerTime}>{clock}</p>
            </div>
          </div>

          <div className={`${styles.statusBadge} ${styles[status]}`}>
            <div className={`${styles.statusDot} ${styles[status]}`} />
            {statusLabel}
          </div>
        </div>

        {/* ── Body ─────────────────────────────────────── */}
        <div className={styles.body}>
          {/* Status card */}
          <div className={`${styles.statusCard} ${styles[status]}`}>
            <FingerprintIcon status={status} />

            <div className={styles.statusTextWrap} key={status}>
              <h2 className={`${styles.statusTitle} ${styles[status]}`}>{titleText}</h2>
              <p className={styles.statusSub}>{subText}</p>
            </div>

            {/* Stats row */}
            <div className={styles.statsRow}>
              <div className={styles.statBox}>
                <div className={styles.statIcon}>
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 1 1 16 0Z" />
                    <circle cx="12" cy="10" r="3" />
                  </svg>
                </div>
                <div className={styles.statValue}>{distDisplay}</div>
                <div className={styles.statLabel}>Distance</div>
              </div>
              <div className={styles.statBox}>
                <div className={styles.statIcon}>
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M12 2l7 4v6c0 5-3 9-7 10-4-1-7-5-7-10V6l7-4Z" />
                    <circle cx="12" cy="11" r="2" />
                  </svg>
                </div>
                <div className={styles.statValue}>{zoneValue}</div>
                <div className={styles.statLabel}>{zoneLabel}</div>
              </div>
              <div className={styles.statBox}>
                <div className={styles.statIcon}>
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <circle cx="12" cy="12" r="10" />
                    <circle cx="12" cy="12" r="6" />
                    <circle cx="12" cy="12" r="2" />
                  </svg>
                </div>
                <div className={styles.statValue}>{accDisplay}</div>
                <div className={styles.statLabel}>GPS Accuracy</div>
              </div>
            </div>
          </div>

          {/* Map toggle */}
          <button className={styles.mapToggle} onClick={handleMapToggle}>
            <div className={styles.mapToggleLeft}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polygon points="3 6 9 3 15 6 21 3 21 18 15 21 9 18 3 21" />
                <line x1="9" y1="3" x2="9" y2="18" />
                <line x1="15" y1="6" x2="15" y2="21" />
              </svg>
              <span>View Map</span>
            </div>
            <span className={`${styles.mapChevron} ${mapExpanded ? styles.open : ''}`}>▾</span>
          </button>

          {/* Map content */}
          {mapExpanded && (
            <div className={styles.mapSlideUp}>
              {useLiveMap && LiveMapComponent && gpsData ? (
                <div className={styles.realMap}>
                  <LiveMapComponent
                    campusLat={campusCfg.lat}
                    campusLng={campusCfg.lng}
                    campusRadius={campusCfg.radius}
                    campusPolygon={gpsData?.polygon || campusCfg.polygon}
                    userLat={gpsData.lat}
                    userLng={gpsData.lng}
                    inRange={gpsData.inRange}
                    geofence={geofenceType}
                  />
                </div>
              ) : (
                <MapPreview
                  status={status}
                  distance={gpsData?.distance}
                  accuracy={gpsData?.accuracy}
                  geofence={geofenceType}
                  campusName={campusName}
                />
              )}
            </div>
          )}

          {/* Primary action button */}
          <button
            className={`${styles.primaryBtn} ${styles[status]}`}
            onClick={handlePrimaryAction}
            disabled={isChecking}
          >
            {btnText}
          </button>

          {/* Success banner */}
          {isIn && !isChecking && (
            <div className={styles.successBanner}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{display:'inline',verticalAlign:'middle',marginRight:4}}>
                <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
                <polyline points="22 4 12 14.01 9 11.01" />
              </svg>
              Inside attendance zone · System verified
            </div>
          )}

          {/* Error detail */}
          {isError && gpsError && (
            <div className={styles.errorDetail}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{display:'inline',verticalAlign:'middle',marginRight:4}}>
                <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
                <line x1="12" y1="9" x2="12" y2="13" />
                <line x1="12" y1="17" x2="12.01" y2="17" />
              </svg>
              {gpsError}
            </div>
          )}

          {/* BINUS Spirit Values Ticker */}
          <SpiritTicker />

          {/* Copyright */}
          <p className={styles.copyright}>© 2026 AI Club · BINUS School Simprug</p>
        </div>
      </div>
    </div>
  );
}
