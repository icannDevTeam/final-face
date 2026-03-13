import { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  loadModels,
  loadDescriptors,
  detectAndMatch,
  isModelsLoaded,
  getLoadedCount,
} from '../lib/faceRecognition';
import { createLivenessChecker, detectWithLandmarks } from '../lib/liveness';
import { checkIn } from '../lib/api';
import { checkProximity } from '../lib/geolocation';
import {
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
  const [modelsReady, setModelsReady] = useState(false); // means models + descriptors loaded
  const [cameraReady, setCameraReady] = useState(false);
  const [livenessStatus, setLivenessStatus] = useState(null); // { passed, prompt, blinkPassed, motionPassed }
  const [livenessOk, setLivenessOk] = useState(false);

  // Streak tracking
  const streakRef = useRef({ id: null, count: 0 });
  const livenessRef = useRef(null);

  // Ref to hold performClockIn so runRecognitionLoop can access it
  // without a temporal dead zone / circular useCallback dependency
  const clockInRef = useRef(null);

  // ─── Load face-api models + descriptors ────────────────────

  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        // Wait for Firebase anonymous auth before reading Firestore
        setStatusMsg('Preparing face engine…');
        const firebaseModule = await import('../lib/firebase');
        const authPromise = firebaseModule.authReady;

        setStatusMsg('Loading face detection models…');
        const modelPromise = loadModels((msg) => !cancelled && setStatusMsg(msg));

        await Promise.all([authPromise, modelPromise]);

        setStatusMsg('Loading face database…');
        const count = await loadDescriptors((msg) => !cancelled && setStatusMsg(msg));

        if (!cancelled) {
          if (count === 0) {
            setPhase('error');
            setStatusMsg('No face descriptors found in database. Please enroll students first.');
          } else {
            setModelsReady(true);
          }
        }
      } catch (err) {
        if (!cancelled) {
          setPhase('error');
          setStatusMsg(`Model load failed: ${err.message}`);
        }
      }
    })();

    return () => { cancelled = true; };
  }, []);

  // ─── Camera setup ──────────────────────────────────────────

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    setCameraReady(false);
    if (animRef.current) cancelAnimationFrame(animRef.current);
    if (scanTimerRef.current) clearInterval(scanTimerRef.current);
  }, []);

  const startCamera = useCallback(async () => {
    try {
      // Camera requires HTTPS (except localhost)
      if (!navigator.mediaDevices) {
        setPhase('error');
        setStatusMsg('Camera requires HTTPS. Please access this page via a secure connection.');
        return;
      }

      setStatusMsg('Requesting camera access…');

      // Timeout wrapper — getUserMedia can hang on some mobile browsers
      // (permission prompt stuck behind PWA, silently blocked, etc.)
      const stream = await Promise.race([
        navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: 'user',
            width: { ideal: 640 },
            height: { ideal: 480 },
          },
          audio: false,
        }),
        new Promise((_, reject) =>
          setTimeout(() => reject(Object.assign(new Error('Camera request timed out. Please check your browser permissions and reload.'), { name: 'TimeoutError' })), 15000)
        ),
      ]);
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        // video.play() can also hang on some mobile browsers
        await Promise.race([
          videoRef.current.play(),
          new Promise((_, reject) =>
            setTimeout(() => reject(Object.assign(new Error('Camera stream failed to start. Please reload and try again.'), { name: 'PlayError' })), 8000)
          ),
        ]);
      }
      setCameraReady(true);
    } catch (err) {
      console.error('Camera init failed:', err);
      // Release any acquired stream on failure
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
      }
      setPhase('error');
      setStatusMsg(
        err.name === 'NotAllowedError'
          ? 'Camera access denied. Please allow camera permissions in your browser settings and reload.'
          : err.name === 'NotFoundError'
          ? 'No camera found on this device.'
          : err.name === 'NotReadableError'
          ? 'Camera is in use by another app. Please close it and try again.'
          : err.name === 'TimeoutError'
          ? 'Camera request timed out. Please check your permissions and reload.'
          : `Camera failed: ${err.message}`
      );
    }
  }, []);

  // ─── Face overlay drawing + liveness ───────────────────────

  const drawOverlay = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current) return;
    try {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      if (!isModelsLoaded()) {
        animRef.current = requestAnimationFrame(drawOverlay);
        return;
      }

      const det = await detectWithLandmarks(video);
      if (det) {
        setFaceDetected(true);
        const { x, y, width, height } = det.detection.box;
        const cx = x + width / 2;
        const cy = y + height / 2;
        const rx = width / 2 + 15;
        const ry = height / 2 + 25;

        // Feed landmarks + box to liveness checker
        if (!livenessRef.current) {
          livenessRef.current = createLivenessChecker();
        }
        const lStatus = livenessRef.current.update(det.landmarks, det.detection.box);
        setLivenessStatus(lStatus);

        // Handle liveness timeout — let user retry
        if (lStatus.timedOut && !livenessOk) {
          // Don't auto-fail, user can tap Retry
        } else if (lStatus.passed && !livenessOk) {
          setLivenessOk(true);
        }

        // Elliptical face outline — color reflects challenge state
        const isActive = lStatus.challenge === 'turn_left' || lStatus.challenge === 'turn_right';
        const ellipseColor = lStatus.passed ? '#10B981' : isActive ? '#F59E0B' : '#00A3E0';
        ctx.strokeStyle = ellipseColor;
        ctx.lineWidth = 3;
        ctx.setLineDash([8, 4]);
        ctx.beginPath();
        ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
        ctx.stroke();

        // Solid inner ring
        ctx.setLineDash([]);
        ctx.lineWidth = 2;
        ctx.strokeStyle = lStatus.passed
          ? 'rgba(16,185,129,0.5)'
          : isActive ? 'rgba(245,158,11,0.35)' : 'rgba(0,84,166,0.5)';
        ctx.beginPath();
        ctx.ellipse(cx, cy, rx - 8, ry - 8, 0, 0, Math.PI * 2);
        ctx.stroke();

        // Progress arc around ellipse (shows liveness completion)
        if (!lStatus.passed && lStatus.progress > 0) {
          ctx.strokeStyle = '#10B981';
          ctx.lineWidth = 3;
          ctx.setLineDash([]);
          ctx.beginPath();
          ctx.ellipse(cx, cy, rx + 6, ry + 8, 0, -Math.PI / 2, -Math.PI / 2 + lStatus.progress * Math.PI * 2);
          ctx.stroke();
        }
      } else {
        setFaceDetected(false);
      }

    animRef.current = requestAnimationFrame(drawOverlay);
    } catch (err) {
      console.error('Overlay draw error:', err);
      animRef.current = requestAnimationFrame(drawOverlay);
    }
  }, [livenessOk]);

  // ─── Face recognition scan loop (starts only after liveness) ─

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
            setTimeout(() => clockInRef.current?.(result), CONFIRM_DELAY);
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
          confidence: result.confidence,
        }
      );

      setClockResult(result2.record);
      setPhase('done');
      setStatusMsg(result2.alreadyDone ? 'Already clocked in today' : 'Attendance recorded!');

      // Persist last clock-in for HomePage display
      try {
        localStorage.setItem('lastClockIn', JSON.stringify({
          studentId: result.student.id,
          name: result.student.name,
          homeroom: result.student.homeroom,
          grade: result.student.grade,
          timestamp: result2.record?.timestamp || new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          status: result2.record?.status || (result2.record?.late ? 'Late' : 'Present'),
          date: new Date().toISOString().slice(0, 10),
          alreadyDone: result2.alreadyDone,
        }));
      } catch { /* localStorage not available */ }
    } catch (err) {
      setPhase('error');
      setStatusMsg(err.message);
    }
  }, []);

  // Keep the ref in sync so runRecognitionLoop always calls the latest version
  useEffect(() => { clockInRef.current = performClockIn; }, [performClockIn]);

  // ─── Lifecycle ─────────────────────────────────────────────

  useEffect(() => {
    startCamera();
    return () => stopCamera();
  }, [startCamera, stopCamera]);

  // Once both camera and models/descriptors are ready, enter scanning phase
  useEffect(() => {
    if (cameraReady && modelsReady) {
      setPhase('scanning');
      setStatusMsg('Look at the camera');
    } else if (cameraReady && !modelsReady) {
      setStatusMsg('Camera ready. Loading face engine…');
    } else if (!cameraReady && modelsReady) {
      setStatusMsg('Waiting for camera access…');
    }
  }, [cameraReady, modelsReady]);

  // Start overlay once scanning; recognition only after liveness passes
  useEffect(() => {
    if (phase === 'scanning') {
      drawOverlay();
    }
    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, [phase, drawOverlay]);

  // Start recognition loop once liveness confirmed
  useEffect(() => {
    if (phase === 'scanning' && livenessOk) {
      runRecognitionLoop();
    }
    return () => {
      if (scanTimerRef.current) clearInterval(scanTimerRef.current);
    };
  }, [phase, livenessOk, runRecognitionLoop]);

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
          <XCircle size={20} />
        </button>
        <h2 className={styles.title}>Take your photo</h2>
        <div style={{ width: 36 }} /> {/* spacer for centering */}
      </div>

      {/* Instruction bar — shows liveness challenge prompt */}
      {phase === 'scanning' && (
        <div className={`${styles.instructionBar} ${livenessOk ? styles.instructionBarSuccess : livenessStatus?.timedOut ? styles.instructionBarError : ''}`}>
          <span>{livenessStatus?.prompt || 'Hold your face steady in the frame'}</span>
          {!livenessOk && !livenessStatus?.timedOut && livenessStatus?.progress > 0 && (
            <div className={styles.challengeProgress}>
              <div className={styles.challengeProgressFill} style={{ width: `${(livenessStatus.progress * 100).toFixed(0)}%` }} />
            </div>
          )}
          {livenessStatus?.timedOut && (
            <button
              className={styles.retryLivenessBtn}
              onClick={() => {
                if (livenessRef.current) livenessRef.current.reset();
                setLivenessOk(false);
                setLivenessStatus(null);
              }}
            >
              Retry
            </button>
          )}
        </div>
      )}

      {/* Camera viewport */}
      <div className={styles.viewport}>
        <video
          ref={videoRef}
          className={styles.video}
          playsInline
          autoPlay
          muted
        />
        <canvas ref={canvasRef} className={styles.overlay} />

        {/* Circular scan frame guide */}
        {phase === 'scanning' && (
          <div className={styles.scanGuide}>
            <div className={`${styles.scanFrame} ${faceDetected ? styles.scanFrameActive : ''}`} />
            {faceDetected && <div className={styles.scanLine} />}
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
        {phase === 'done' && (
          <div className={styles.doneOverlay}>
            <div className={styles.doneCard}>
              <CheckCircle2 size={56} className={styles.doneIcon} />
              <h3>{clockResult?.alreadyDone ? 'Already Clocked In' : 'Clock in successful'}</h3>
              <p className={styles.doneMsg}>
                {clockResult?.alreadyDone
                  ? 'You have already clocked in today.'
                  : 'Great job! Your clock in has been successfully saved.'}
              </p>
              {clockResult && (
                <>
                  <p className={styles.doneName}>{clockResult.name}</p>
                  <p className={styles.doneStatus}>
                    {clockResult.status || (clockResult.late ? 'Late' : 'Present')}
                  </p>
                  <p className={styles.doneTime}>{clockResult.timestamp}</p>
                </>
              )}
              <button className={styles.doneHomeBtn} onClick={() => { stopCamera(); navigate('/'); }}>
                Back to Home
              </button>
            </div>
          </div>
        )}

        {/* Error overlay */}
        {phase === 'error' && (
          <div className={styles.errorOverlay}>
            <XCircle size={48} />
            <p>{statusMsg}</p>
            <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap', justifyContent: 'center' }}>
              <button className={styles.retryBtn} onClick={() => {
                setPhase('starting');
                setStatusMsg('Retrying camera…');
                setCameraReady(false);
                stopCamera();
                startCamera();
              }}>
                Retry Camera
              </button>
              <button className={styles.retryBtn} onClick={() => { stopCamera(); navigate('/'); }}>
                Go Back
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Face guidelines */}
      {(phase === 'scanning' || phase === 'starting') && (
        <div className={styles.guidelines}>
          <div className={styles.guideItem}>
            <CheckCircle2 size={16} className={styles.guideCheck} />
            <span>Ensure that your forehead, ears, and chin are fully visible within the frame.</span>
          </div>
          <div className={styles.guideItem}>
            <CheckCircle2 size={16} className={styles.guideCheck} />
            <span>Please do not wear glasses, a mask, or any other accessories that might cover your face.</span>
          </div>
        </div>
      )}

      {/* Bottom status */}
      <div className={styles.statusBar}>
        {phase === 'starting' && (
          <p className={styles.statusText}><Loader2 size={16} className={styles.spinner} /> {statusMsg}</p>
        )}
        {phase === 'scanning' && (
          <p className={styles.statusText}>
            <Camera size={16} />{' '}
            {!faceDetected
              ? 'Position your face in frame'
              : livenessOk
                ? 'Recognizing face…'
                : livenessStatus?.timedOut
                  ? 'Liveness timed out — tap Retry above'
                  : `Verifying liveness… Step ${(livenessStatus?.challengeIdx || 0) + 1}/${livenessStatus?.totalSteps || 3}`}
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
