import { useState, useEffect } from 'react';
import { Download, X, Share } from 'lucide-react';

/**
 * PWA Install Prompt — shows a banner encouraging users to add the app
 * to their home screen. Handles both Android (beforeinstallprompt) and
 * iOS (manual instructions) flows.
 */
export default function InstallPrompt() {
  const [deferredPrompt, setDeferredPrompt] = useState(null);
  const [showBanner, setShowBanner] = useState(false);
  const [isIOS, setIsIOS] = useState(false);
  const [isInstalled, setIsInstalled] = useState(false);
  const [dismissed, setDismissed] = useState(false);

  useEffect(() => {
    // Check if already installed as PWA
    const isStandalone =
      window.matchMedia('(display-mode: standalone)').matches ||
      window.navigator.standalone === true;

    if (isStandalone) {
      setIsInstalled(true);
      return;
    }

    // Check if user previously dismissed (respect for 7 days)
    const dismissedAt = localStorage.getItem('pwa-install-dismissed');
    if (dismissedAt) {
      const daysSince = (Date.now() - parseInt(dismissedAt, 10)) / (1000 * 60 * 60 * 24);
      if (daysSince < 7) {
        setDismissed(true);
        return;
      }
    }

    // Detect iOS
    const ua = navigator.userAgent;
    const isiOS = /iPad|iPhone|iPod/.test(ua) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
    setIsIOS(isiOS);

    // Android / Chrome — capture the beforeinstallprompt event
    const handler = (e) => {
      e.preventDefault();
      setDeferredPrompt(e);
      setShowBanner(true);
    };

    window.addEventListener('beforeinstallprompt', handler);

    // On iOS, show the banner after a short delay (no native prompt)
    if (isiOS) {
      const timer = setTimeout(() => setShowBanner(true), 3000);
      return () => {
        clearTimeout(timer);
        window.removeEventListener('beforeinstallprompt', handler);
      };
    }

    return () => window.removeEventListener('beforeinstallprompt', handler);
  }, []);

  const handleInstall = async () => {
    if (!deferredPrompt) return;
    deferredPrompt.prompt();
    const { outcome } = await deferredPrompt.userChoice;
    if (outcome === 'accepted') {
      setIsInstalled(true);
    }
    setDeferredPrompt(null);
    setShowBanner(false);
  };

  const handleDismiss = () => {
    setShowBanner(false);
    setDismissed(true);
    localStorage.setItem('pwa-install-dismissed', String(Date.now()));
  };

  if (isInstalled || dismissed || !showBanner) return null;

  return (
    <div style={styles.overlay}>
      <div style={styles.banner}>
        <button onClick={handleDismiss} style={styles.closeBtn} aria-label="Close">
          <X size={18} />
        </button>

        <div style={styles.iconRow}>
          <img src="/icon-96.png" alt="BINUS" style={styles.appIcon} />
          <div>
            <p style={styles.title}>Install BINUS Attendance</p>
            <p style={styles.subtitle}>Add to your home screen for quick access</p>
          </div>
        </div>

        {isIOS ? (
          <div style={styles.iosInstructions}>
            <p style={styles.iosStep}>
              <span style={styles.stepNum}>1</span>
              Tap <Share size={16} style={{ verticalAlign: 'middle' }} /> <strong>Share</strong> in the browser toolbar
            </p>
            <p style={styles.iosStep}>
              <span style={styles.stepNum}>2</span>
              Scroll down and tap <strong>"Add to Home Screen"</strong>
            </p>
            <p style={styles.iosStep}>
              <span style={styles.stepNum}>3</span>
              Tap <strong>"Add"</strong> to confirm
            </p>
          </div>
        ) : (
          <button onClick={handleInstall} style={styles.installBtn}>
            <Download size={18} />
            <span>Install App</span>
          </button>
        )}
      </div>
    </div>
  );
}

const styles = {
  overlay: {
    position: 'fixed',
    bottom: 0,
    left: 0,
    right: 0,
    zIndex: 9999,
    padding: '12px',
    pointerEvents: 'none',
  },
  banner: {
    position: 'relative',
    background: 'linear-gradient(135deg, #1e293b 0%, #0f172a 100%)',
    border: '1px solid rgba(34, 197, 94, 0.3)',
    borderRadius: '16px',
    padding: '16px',
    boxShadow: '0 -4px 24px rgba(0,0,0,0.4)',
    pointerEvents: 'all',
    animation: 'slideUp 0.4s ease-out',
  },
  closeBtn: {
    position: 'absolute',
    top: '8px',
    right: '8px',
    background: 'none',
    border: 'none',
    color: '#94a3b8',
    cursor: 'pointer',
    padding: '4px',
    borderRadius: '8px',
  },
  iconRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    marginBottom: '12px',
  },
  appIcon: {
    width: '48px',
    height: '48px',
    borderRadius: '12px',
  },
  title: {
    margin: 0,
    color: '#f8fafc',
    fontSize: '15px',
    fontWeight: 700,
  },
  subtitle: {
    margin: '2px 0 0',
    color: '#94a3b8',
    fontSize: '13px',
  },
  installBtn: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '8px',
    width: '100%',
    padding: '12px',
    background: 'linear-gradient(135deg, #22c55e 0%, #16a34a 100%)',
    color: '#fff',
    border: 'none',
    borderRadius: '12px',
    fontSize: '15px',
    fontWeight: 600,
    cursor: 'pointer',
  },
  iosInstructions: {
    background: 'rgba(255,255,255,0.05)',
    borderRadius: '12px',
    padding: '12px',
  },
  iosStep: {
    color: '#cbd5e1',
    fontSize: '13px',
    margin: '6px 0',
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  },
  stepNum: {
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: '22px',
    height: '22px',
    borderRadius: '50%',
    background: '#22c55e',
    color: '#fff',
    fontSize: '12px',
    fontWeight: 700,
    flexShrink: 0,
  },
};
