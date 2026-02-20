import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Clock, Fingerprint, CheckCircle2, MapPin, MapPinOff, AlertTriangle } from 'lucide-react';
import { loadModels, loadDescriptors, isModelsLoaded, getLoadedCount } from '../lib/faceRecognition';
import { watchProximity, getCampusConfig } from '../lib/geolocation';
import LiveMap from '../components/LiveMap';
import styles from '../styles/Home.module.css';

export default function HomePage() {
  const navigate = useNavigate();
  const [time, setTime] = useState(new Date());
  const [status, setStatus] = useState('idle'); // idle | loading | ready | error
  const [statusMsg, setStatusMsg] = useState('');

  // Proximity state
  const [location, setLocation] = useState(null);   // { inRange, distance, accuracy }
  const [locError, setLocError] = useState(null);    // error message string

  // Live clock
  useEffect(() => {
    const t = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  // Watch proximity continuously
  useEffect(() => {
    const stopWatch = watchProximity(
      (loc) => { setLocation(loc); setLocError(null); },
      (err) => { setLocError(err); setLocation(null); },
    );
    return stopWatch;
  }, []);

  // Load models + descriptors on mount
  useEffect(() => {
    let cancelled = false;

    async function init() {
      setStatus('loading');
      try {
        await loadModels((msg) => !cancelled && setStatusMsg(msg));
        await loadDescriptors((msg) => !cancelled && setStatusMsg(msg));
        if (!cancelled) {
          setStatus('ready');
          setStatusMsg('System ready');
        }
      } catch (err) {
        if (!cancelled) {
          setStatus('error');
          setStatusMsg(err.message);
        }
      }
    }

    init();

    return () => { cancelled = true; };
  }, []);

  // Format WIB time
  const wib = new Date(time.getTime() + 7 * 3600 * 1000);
  const hours = wib.getUTCHours().toString().padStart(2, '0');
  const minutes = wib.getUTCMinutes().toString().padStart(2, '0');
  const seconds = wib.getUTCSeconds().toString().padStart(2, '0');
  const dateStr = wib.toLocaleDateString('en-GB', {
    timeZone: 'Asia/Jakarta',
    weekday: 'long',
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  });

  const modelsReady = status === 'ready';
  const inRange = location?.inRange === true;
  const canClockIn = modelsReady && inRange;

  // Button label logic
  let btnLabel = 'Clock In';
  if (status === 'loading') btnLabel = 'Preparing…';
  else if (!inRange && modelsReady) btnLabel = 'Out of Range';

  return (
    <div className={styles.container}>
      {/* Top bar */}
      <div className={styles.topBar}>
        <div className={styles.brandGroup}>
          <img src="/logo.jpe" alt="BINUS" className={styles.logo} />
          <span className={styles.brand}>BINUS Attendance</span>
        </div>
      </div>

      {/* Clock */}
      <div className={styles.clockSection}>
        <p className={styles.date}>{dateStr}</p>
        <div className={styles.clock}>
          <span className={styles.clockDigit}>{hours}</span>
          <span className={styles.clockSep}>:</span>
          <span className={styles.clockDigit}>{minutes}</span>
          <span className={styles.clockSep}>:</span>
          <span className={styles.clockDigitSec}>{seconds}</span>
        </div>
        <p className={styles.timezone}>WIB (UTC+7)</p>
      </div>

      {/* Clock In Button */}
      <div className={styles.actionSection}>
        <button
          className={`${styles.clockInBtn} ${canClockIn ? styles.clockInReady : ''}`}
          onClick={() => canClockIn && navigate('/scan')}
          disabled={!canClockIn}
        >
          <div className={styles.clockInIcon}>
            <Fingerprint size={48} />
          </div>
          <span className={styles.clockInLabel}>{btnLabel}</span>
        </button>

        <p className={styles.statusLine}>
          {status === 'loading' && (
            <span className={styles.statusLoading}>{statusMsg}</span>
          )}
          {status === 'ready' && (
            <span className={styles.statusReady}>
              <CheckCircle2 size={14} /> {statusMsg}
            </span>
          )}
          {status === 'error' && (
            <span className={styles.statusError}>{statusMsg}</span>
          )}
        </p>
      </div>

      {/* Proximity indicator */}
      <div className={styles.locationBar}>
        {locError && (
          <div className={`${styles.locationStatus} ${styles.locationError}`}>
            <MapPinOff size={16} />
            <span>{locError}</span>
          </div>
        )}
        {location && location.inRange && (
          <div className={`${styles.locationStatus} ${styles.locationOk}`}>
            <MapPin size={16} />
            <span>On campus — {location.distance}m from school</span>
          </div>
        )}
        {location && !location.inRange && (
          <div className={`${styles.locationStatus} ${styles.locationFar}`}>
            <AlertTriangle size={16} />
            <span>Too far — {location.distance}m away (max {location.campusRadius}m)</span>
          </div>
        )}
        {!location && !locError && (
          <div className={`${styles.locationStatus} ${styles.locationLoading}`}>
            <MapPin size={16} />
            <span>Checking location…</span>
          </div>
        )}
      </div>

      {/* Live Map */}
      {(() => {
        const campus = getCampusConfig();
        return (
          <LiveMap
            campusLat={campus.lat}
            campusLng={campus.lng}
            campusRadius={campus.radius}
            userLat={location?.lat ?? null}
            userLng={location?.lng ?? null}
            inRange={inRange}
          />
        );
      })()}

      {/* SPIRIT Values Scrolling Banner */}
      <div className={styles.spiritBanner}>
        <div className={styles.spiritTrack}>
          {[...Array(2)].map((_, i) => (
            <div className={styles.spiritSet} key={i}>
              {[
                { letter: 'S', value: 'Striving for Excellence' },
                { letter: 'P', value: 'Perseverance' },
                { letter: 'I', value: 'Integrity' },
                { letter: 'R', value: 'Respect' },
                { letter: 'I', value: 'Innovation' },
                { letter: 'T', value: 'Teamwork' },
              ].map((item, j) => (
                <span className={styles.spiritItem} key={j}>
                  <span className={styles.spiritLetter}>{item.letter}</span>
                  <span className={styles.spiritValue}>{item.value}</span>
                </span>
              ))}
            </div>
          ))}
        </div>
      </div>

      {/* Copyright */}
      <footer className={styles.copyright}>
        <span>© 2026 BINUS School AI Club</span>
      </footer>

    </div>
  );
}
