import { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  detectAndMatch,
  detectFace,
  isModelsLoaded,
  getLoadedCount,
} from '../lib/faceRecognition';
import { checkIn } from '../lib/api';
import { checkProximity } from '../lib/geolocation';
import {
  ArrowLeft,
  Camera,
  Loader2,
  CheckCircle2,
  XCircle,
  ScanFace,
} from 'lucide-react';
import styles from '../styles/Scan.module.css';

const SCAN_INTERVAL = 600;   // ms between recognition attempts
const CONFIRM_DELAY = 2000;  // ms to hold match before auto-clocking
const MATCH_STREAK = 3;      // consecutive matches needed

export default function ScanPage() {
  const navigate = useNavigate();
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const animRef = useRef(null);
  const scanTimerRef = useRef(null);

  const [phase, setPhase] = useState('starting'); // starting | scanning | matched | clocking | done | error
  const [statusMsg, setStatusMsg] = useState('Starting camera…');
  const [matchResult, setMatchResult] = useState(null);
  const [clockResult, setClockResult] = useState(null);
  const [faceDetected, setFaceDetected] = useState(false);

  // Streak tracking
  const streakRef = useRef({ id: null, count: 0 });

  // ─── Camera setup ──────────────────────────────────────────

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (animRef.current) cancelAnimationFrame(animRef.current);
    if (scanTimerRef.current) clearInterval(scanTimerRef.current);
  }, []);

  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'user',
          width: { ideal: 640 },
          height: { ideal: 480 },
        },
        audio: false,
      });
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      setPhase('scanning');
      setStatusMsg('Look at the camera');
    } catch (err) {
      setPhase('error');
      setStatusMsg('Camera access denied. Please allow camera permissions.');
    }
  }, []);

  // ─── Face overlay drawing ──────────────────────────────────

  const drawOverlay = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current) return;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const det = await detectFace(video);
    if (det) {
      setFaceDetected(true);
      const { x, y, width, height } = det.box;

      ctx.strokeStyle = '#00A3E0';
      ctx.lineWidth = 3;
      ctx.setLineDash([8, 4]);
      ctx.strokeRect(x, y, width, height);

      // Corner accents
      const corner = 16;
      ctx.setLineDash([]);
      ctx.lineWidth = 4;
      ctx.strokeStyle = '#0054A6';
      // Top-left
      ctx.beginPath(); ctx.moveTo(x, y + corner); ctx.lineTo(x, y); ctx.lineTo(x + corner, y); ctx.stroke();
      // Top-right
      ctx.beginPath(); ctx.moveTo(x + width - corner, y); ctx.lineTo(x + width, y); ctx.lineTo(x + width, y + corner); ctx.stroke();
      // Bottom-left
      ctx.beginPath(); ctx.moveTo(x, y + height - corner); ctx.lineTo(x, y + height); ctx.lineTo(x + corner, y + height); ctx.stroke();
      // Bottom-right
      ctx.beginPath(); ctx.moveTo(x + width - corner, y + height); ctx.lineTo(x + width, y + height); ctx.lineTo(x + width, y + height - corner); ctx.stroke();
    } else {
      setFaceDetected(false);
    }

    animRef.current = requestAnimationFrame(drawOverlay);
  }, []);

  // ─── Face recognition scan loop ────────────────────────────

  const runRecognitionLoop = useCallback(() => {
    scanTimerRef.current = setInterval(async () => {
      if (!videoRef.current || !isModelsLoaded() || getLoadedCount() === 0) return;

      try {
        const result = await detectAndMatch(videoRef.current);

        if (result.matched) {
          const streak = streakRef.current;
          if (streak.id === result.student.id) {
            streak.count++;
          } else {
            streak.id = result.student.id;
            streak.count = 1;
          }

          if (streak.count >= MATCH_STREAK) {
            // Confirmed match!
            clearInterval(scanTimerRef.current);
            setMatchResult(result);
            setPhase('matched');
            setStatusMsg(`Identified: ${result.student.name}`);

            // Auto clock-in after brief delay
            setTimeout(() => performClockIn(result), CONFIRM_DELAY);
          }
        } else {
          // Reset streak on miss
          streakRef.current = { id: null, count: 0 };
        }
      } catch {
        // Ignore transient errors
      }
    }, SCAN_INTERVAL);
  }, []);

  // ─── Clock in ──────────────────────────────────────────────

  const performClockIn = useCallback(async (result) => {
    setPhase('clocking');
    setStatusMsg('Verifying location…');

    try {
      // Re-verify proximity at clock-in time (prevent spoofing from home page)
      let loc;
      try {
        loc = await checkProximity();
        if (!loc.inRange) {
          setPhase('error');
          setStatusMsg(`You are ${loc.distance}m from campus (max ${loc.campusRadius}m). Move closer to clock in.`);
          return;
        }
      } catch (locErr) {
        setPhase('error');
        setStatusMsg(locErr.message);
        return;
      }

      setStatusMsg('Recording attendance…');

      // checkIn() now handles dedup internally — it returns alreadyDone
      // if a record already exists from any source (mobile or hikvision)
      const result2 = await checkIn(
        result.student.id,
        result.student.name,
        { lat: loc.lat, lng: loc.lng, accuracy: loc.accuracy, distance: loc.distance },
        {
          homeroom: result.student.homeroom,
          grade: result.student.grade,
        }
      );

      setClockResult(result2.record);
      setPhase('done');
      setStatusMsg(result2.alreadyDone ? 'Already clocked in today' : 'Attendance recorded!');
    } catch (err) {
      setPhase('error');
      setStatusMsg(err.message);
    }
  }, []);

  // ─── Lifecycle ─────────────────────────────────────────────

  useEffect(() => {
    startCamera();
    return () => stopCamera();
  }, [startCamera, stopCamera]);

  // Start scanning & overlay once video is playing
  useEffect(() => {
    if (phase === 'scanning') {
      drawOverlay();
      runRecognitionLoop();
    }
    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, [phase, drawOverlay, runRecognitionLoop]);

  // Auto-navigate home after success
  useEffect(() => {
    if (phase === 'done') {
      const t = setTimeout(() => navigate('/'), 4000);
      return () => clearTimeout(t);
    }
  }, [phase, navigate]);

  // ─── Render ────────────────────────────────────────────────

  return (
    <div className={styles.container}>
      {/* Header */}
      <div className={styles.header}>
        <button className={styles.backBtn} onClick={() => { stopCamera(); navigate('/'); }}>
          <ArrowLeft size={22} />
        </button>
        <img src="/logo.jpe" alt="BINUS" className={styles.headerLogo} />
        <h2 className={styles.title}>Face Recognition</h2>
      </div>

      {/* Camera viewport */}
      <div className={styles.viewport}>
        <video
          ref={videoRef}
          className={styles.video}
          playsInline
          muted
        />
        <canvas ref={canvasRef} className={styles.overlay} />

        {/* Scan frame guide */}
        {phase === 'scanning' && (
          <div className={styles.scanGuide}>
            <div className={`${styles.scanFrame} ${faceDetected ? styles.scanFrameActive : ''}`} />
          </div>
        )}

        {/* Matched overlay */}
        {(phase === 'matched' || phase === 'clocking') && matchResult && (
          <div className={styles.matchOverlay}>
            <div className={styles.matchCard}>
              <ScanFace size={36} className={styles.matchIcon} />
              <h3>{matchResult.student.name}</h3>
              <p>{matchResult.student.homeroom} · {matchResult.student.grade}</p>
              <p className={styles.matchConf}>{matchResult.confidence}% confidence</p>
              {phase === 'clocking' && (
                <div className={styles.clockingSpinner}>
                  <Loader2 size={20} className={styles.spinner} />
                  <span>Recording…</span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Done overlay */}
        {phase === 'done' && clockResult && (
          <div className={styles.doneOverlay}>
            <div className={styles.doneCard}>
              <CheckCircle2 size={48} className={styles.doneIcon} />
              <h3>{clockResult.alreadyDone ? 'Already Clocked In' : 'Clocked In!'}</h3>
              <p className={styles.doneName}>{clockResult.name}</p>
              <p className={styles.doneStatus}>
                {clockResult.status || (clockResult.late ? 'Late' : 'Present')}
              </p>
              <p className={styles.doneTime}>{clockResult.timestamp}</p>
            </div>
          </div>
        )}

        {/* Error overlay */}
        {phase === 'error' && (
          <div className={styles.errorOverlay}>
            <XCircle size={48} />
            <p>{statusMsg}</p>
            <button className={styles.retryBtn} onClick={() => { stopCamera(); navigate('/'); }}>
              Go Back
            </button>
          </div>
        )}
      </div>

      {/* Bottom status */}
      <div className={styles.statusBar}>
        {phase === 'starting' && (
          <p className={styles.statusText}><Loader2 size={16} className={styles.spinner} /> {statusMsg}</p>
        )}
        {phase === 'scanning' && (
          <p className={styles.statusText}>
            <Camera size={16} /> {faceDetected ? 'Analyzing face…' : 'Position your face in frame'}
          </p>
        )}
        {phase === 'done' && (
          <p className={styles.statusTextDone}>Returning to home…</p>
        )}
        <p className={styles.scanCopyright}>© 2026 BINUS School Simprug AI Club</p>
      </div>
    </div>
  );
}
