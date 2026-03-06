import { useState, useEffect, useCallback, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { checkProximity, watchProximity, getCampusConfig } from '../lib/geolocation';
import { getAttendanceHistory } from '../lib/api';
import styles from '../styles/Home.module.css';

// Dynamically import LiveMap only when needed (Leaflet is heavy)
let LiveMapComponent = null;

const STATUS = {
  OUT: 'out',
  IN: 'in',
  CHECKING: 'checking',
  ERROR: 'error',
};

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

  // view: 'dashboard' → clean splash,  'map' → full-screen map + GPS
  const [view, setView]           = useState('dashboard');
  const [status, setStatus]       = useState(STATUS.CHECKING);
  const [gpsData, setGpsData]     = useState(null);
  const [gpsError, setGpsError]   = useState(null);
  const [mapReady, setMapReady]   = useState(false);
  const [clock, setClock]         = useState(() =>
    new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
  );

  // Attendance log state
  const [attendanceLog, setAttendanceLog] = useState([]);
  const [lastClockIn, setLastClockIn]     = useState(null);
  const [logLoading, setLogLoading]       = useState(false);

  const cleanupRef = useRef(null);
  const campusCfg  = getCampusConfig();

  // ── Today's formatted date ────────────────────────────
  const todayDate = new Date().toLocaleDateString('en-US', {
    weekday: 'long',
    day: '2-digit',
    month: 'long',
    year: 'numeric',
  });

  // ── Load last clock-in from localStorage & fetch history ─
  useEffect(() => {
    try {
      const raw = localStorage.getItem('lastClockIn');
      if (raw) {
        const parsed = JSON.parse(raw);
        setLastClockIn(parsed);
        if (parsed.studentId) {
          setLogLoading(true);
          getAttendanceHistory(parsed.studentId, 7)
            .then((records) => setAttendanceLog(records))
            .catch(() => {})
            .finally(() => setLogLoading(false));
        }
      }
    } catch { /* ignore */ }
  }, []);

  // ── Clock tick ─────────────────────────────────────────
  useEffect(() => {
    const id = setInterval(() => {
      setClock(new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }));
    }, 30_000);
    return () => clearInterval(id);
  }, []);

  // ── GPS watch — only active when view === 'map' ────────
  useEffect(() => {
    if (view !== 'map') return;

    setStatus(STATUS.CHECKING);
    setGpsError(null);
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
      },
    );
    cleanupRef.current = stop;

    // Fallback one-shot after 5 s
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
  }, [view]);

  // ── Lazy-load LiveMap when entering map view ───────────
  useEffect(() => {
    if (view !== 'map' || LiveMapComponent) {
      if (LiveMapComponent) setMapReady(true);
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const mod = await import('../components/LiveMap');
        if (!cancelled) {
          LiveMapComponent = mod.default;
          setMapReady(true);
        }
      } catch {
        if (!cancelled) setMapReady(false);
      }
    })();
    return () => { cancelled = true; };
  }, [view]);

  // ── Handlers ───────────────────────────────────────────
  const handleMarkAttendance = useCallback(() => setView('map'), []);

  const handleBack = useCallback(() => {
    if (cleanupRef.current) { cleanupRef.current(); cleanupRef.current = null; }
    setView('dashboard');
  }, []);

  const handleClockIn = useCallback(() => {
    if (status !== STATUS.IN) return;
    navigate('/scan');
  }, [status, navigate]);

  const handleRetry = useCallback(() => {
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
  }, []);

  // ── Derived values ─────────────────────────────────────
  const isChecking = status === STATUS.CHECKING;
  const isIn       = status === STATUS.IN;
  const isOut      = status === STATUS.OUT;
  const isError    = status === STATUS.ERROR;

  const geofenceType = gpsData?.geofence || campusCfg.geofence;

  const distDisplay = gpsData
    ? gpsData.distance >= 1000
      ? `${(gpsData.distance / 1000).toFixed(1)} km`
      : `${gpsData.distance} m`
    : '—';

  const accDisplay = gpsData ? `±${gpsData.accuracy} m` : null;

  const todayClockIn = lastClockIn?.date === new Date().toISOString().slice(0, 10) ? lastClockIn : null;

  const welcomeName = lastClockIn?.name?.split(' ')[0] || null;

  // ═══════════════════════════════════════════════════════
  //  Dashboard view — splash / home
  // ═══════════════════════════════════════════════════════
  if (view === 'dashboard') {
    return (
      <div className={styles.page}>
        <div className={styles.card}>
          {/* ── Header ─────────────────────────────────── */}
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
            {todayClockIn && (
              <div className={`${styles.statusBadge} ${styles.in}`}>
                <div className={`${styles.statusDot} ${styles.in}`} />
                Clocked in
              </div>
            )}
          </div>

          {/* ── Body ───────────────────────────────────── */}
          <div className={styles.body}>
            {/* Welcome + date */}
            <div className={styles.welcomeSection}>
              <p className={styles.welcomeDate}>{todayDate}</p>
              <h1 className={styles.welcomeHeading}>
                {welcomeName ? `Welcome, ${welcomeName}` : "Today's Summary"}
              </h1>
            </div>

            {/* Today's clock-in card */}
            <div className={styles.summaryCard}>
              <div className={styles.summaryRow}>
                <div className={styles.summaryItem}>
                  <span className={styles.summaryLabel}>Clock In</span>
                  <span className={styles.summaryValue}>{todayClockIn?.timestamp || '—  —'}</span>
                </div>
                <div className={styles.summaryDivider} />
                <div className={styles.summaryItem}>
                  <span className={styles.summaryLabel}>Status</span>
                  <span className={`${styles.summaryValue} ${todayClockIn?.status === 'Late' ? styles.summaryLate : ''}`}>
                    {todayClockIn?.status || '—  —'}
                  </span>
                </div>
              </div>
            </div>

            {/* Mark Attendance CTA */}
            <button className={styles.markBtn} onClick={handleMarkAttendance}>
              <div className={styles.markBtnIcon}>
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 1 1 16 0Z" />
                  <circle cx="12" cy="10" r="3" />
                </svg>
              </div>
              <div className={styles.markBtnText}>
                <span className={styles.markBtnTitle}>
                  {todayClockIn ? 'View Location' : 'Mark Attendance'}
                </span>
                <span className={styles.markBtnSub}>Open map &amp; verify GPS location</span>
              </div>
              <svg className={styles.markBtnChevron} width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M9 18l6-6-6-6" />
              </svg>
            </button>

            {/* Attendance log */}
            {(attendanceLog.length > 0 || logLoading) && (
              <div className={styles.logSection}>
                <div className={styles.logHeader}>
                  <span className={styles.logTitle}>Attendance Log</span>
                  <span className={styles.logCount}>{attendanceLog.length} days</span>
                </div>
                {/* Table header */}
                <div className={`${styles.logEntry} ${styles.logTableHead}`}>
                  <div className={styles.logDate}>Date</div>
                  <div className={styles.logTime}>Clock In</div>
                  <span className={styles.logStatus}>Status</span>
                </div>
                <div className={styles.logList}>
                  {attendanceLog.map((entry) => (
                    <div key={entry.date} className={styles.logEntry}>
                      <div className={styles.logDate}>
                        {new Date(entry.date + 'T00:00:00').toLocaleDateString('en-US', { weekday: 'short', day: '2-digit', month: 'short' })}
                      </div>
                      <div className={styles.logTime}>{entry.startTime}</div>
                      <span className={`${styles.logStatus} ${entry.late ? styles.logStatusLate : ''}`}>
                        {entry.status}
                      </span>
                    </div>
                  ))}
                </div>
                {logLoading && <p className={styles.logLoading}>Loading…</p>}
              </div>
            )}

            {/* Spirit ticker */}
            <SpiritTicker />

            <p className={styles.copyright}>© 2026 AI Club · BINUS School Simprug</p>
          </div>
        </div>
      </div>
    );
  }

  // ═══════════════════════════════════════════════════════
  //  Map view — full-screen map + floating bottom card
  // ═══════════════════════════════════════════════════════
  return (
    <div className={styles.mapPage}>
      {/* Floating header */}
      <div className={styles.mapHeader}>
        <button className={styles.mapBackBtn} onClick={handleBack}>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M15 18l-6-6 6-6" />
          </svg>
        </button>
        <span className={styles.mapHeaderTitle}>Mark Attendance</span>
        <div className={`${styles.mapHeaderBadge} ${styles[status]}`}>
          <div className={`${styles.statusDot} ${styles[status]}`} />
          {isChecking ? 'Checking…' : isIn ? 'In range' : isOut ? 'Out of range' : 'Error'}
        </div>
      </div>

      {/* Full-screen map */}
      <div className={styles.mapFull}>
        {mapReady && LiveMapComponent && gpsData ? (
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
        ) : (
          <div className={styles.mapLoading}>
            <div className={styles.mapSpinner} />
            <p>Loading map…</p>
          </div>
        )}
      </div>

      {/* Floating bottom card */}
      <div className={styles.mapBottomCard}>
        {/* Location info */}
        <div className={styles.mapInfoRow}>
          <div className={styles.mapInfoItem}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 1 1 16 0Z" />
              <circle cx="12" cy="10" r="3" />
            </svg>
            <div>
              <span className={styles.mapInfoLabel}>Distance</span>
              <span className={styles.mapInfoValue}>{distDisplay}</span>
            </div>
          </div>
          {accDisplay && (
            <div className={styles.mapInfoItem}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="10" />
                <circle cx="12" cy="12" r="6" />
                <circle cx="12" cy="12" r="2" />
              </svg>
              <div>
                <span className={styles.mapInfoLabel}>Accuracy</span>
                <span className={styles.mapInfoValue}>{accDisplay}</span>
              </div>
            </div>
          )}
        </div>

        {/* Status message */}
        {isChecking && (
          <p className={styles.mapStatusMsg}>
            <span className={styles.mapSpinnerInline} /> Verifying your location…
          </p>
        )}
        {isIn && (
          <p className={`${styles.mapStatusMsg} ${styles.mapStatusIn}`}>
            ✓ You are within the attendance zone
          </p>
        )}
        {isOut && (
          <p className={`${styles.mapStatusMsg} ${styles.mapStatusOut}`}>
            You are outside the attendance zone ({distDisplay} away)
          </p>
        )}
        {isError && (
          <p className={`${styles.mapStatusMsg} ${styles.mapStatusError}`}>
            {gpsError || 'Unable to determine your location'}
          </p>
        )}

        {/* Action button */}
        {isIn && (
          <button className={styles.clockInBtn} onClick={handleClockIn}>
            Clock In
          </button>
        )}
        {isOut && (
          <button className={styles.retryBtn} onClick={handleRetry}>
            Retry Location
          </button>
        )}
        {isError && (
          <button className={styles.retryBtn} onClick={handleRetry}>
            Try Again
          </button>
        )}
        {isChecking && (
          <button className={styles.checkingBtn} disabled>
            <span className={styles.mapSpinnerInline} /> Checking…
          </button>
        )}
      </div>
    </div>
  );
}
