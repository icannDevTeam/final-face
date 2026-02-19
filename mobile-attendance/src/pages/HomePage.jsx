import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Clock, Fingerprint, CheckCircle2, Users } from 'lucide-react';
import { loadModels, loadDescriptors, isModelsLoaded, getLoadedCount } from '../lib/faceRecognition';
import { getTodaySummary } from '../lib/api';
import styles from '../styles/Home.module.css';

export default function HomePage() {
  const navigate = useNavigate();
  const [time, setTime] = useState(new Date());
  const [status, setStatus] = useState('idle'); // idle | loading | ready | error
  const [statusMsg, setStatusMsg] = useState('');
  const [summary, setSummary] = useState(null);

  // Live clock
  useEffect(() => {
    const t = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  // Load models + descriptors on mount
  useEffect(() => {
    let cancelled = false;

    async function init() {
      setStatus('loading');
      try {
        await loadModels((msg) => !cancelled && setStatusMsg(msg));
        const count = await loadDescriptors((msg) => !cancelled && setStatusMsg(msg));
        if (!cancelled) {
          setStatus('ready');
          setStatusMsg(`${count} student${count !== 1 ? 's' : ''} enrolled`);
        }
      } catch (err) {
        if (!cancelled) {
          setStatus('error');
          setStatusMsg(err.message);
        }
      }
    }

    init();

    // Also fetch today's summary
    getTodaySummary().then((s) => !cancelled && setSummary(s));

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

  const canClockIn = status === 'ready';

  return (
    <div className={styles.container}>
      {/* Top bar */}
      <div className={styles.topBar}>
        <div className={styles.brandGroup}>
          <img src="/logo.jpe" alt="BINUS" className={styles.logo} />
          <span className={styles.brand}>BINUS Attendance</span>
        </div>
        {summary && (
          <span className={styles.todayBadge}>
            <Users size={14} />
            {summary.total} today
          </span>
        )}
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
          <span className={styles.clockInLabel}>
            {status === 'loading' ? 'Preparing…' : 'Clock In'}
          </span>
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

      {/* Today summary */}
      {summary && summary.total > 0 && (
        <div className={styles.summaryBar}>
          <div className={styles.summaryItem}>
            <span className={styles.summaryNum}>{summary.present}</span>
            <span className={styles.summaryLabel}>Present</span>
          </div>
          <div className={styles.summaryDivider} />
          <div className={styles.summaryItem}>
            <span className={`${styles.summaryNum} ${styles.summaryLate}`}>{summary.late}</span>
            <span className={styles.summaryLabel}>Late</span>
          </div>
        </div>
      )}
    </div>
  );
}
