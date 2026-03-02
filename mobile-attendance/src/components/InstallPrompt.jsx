import { useState, useEffect } from 'react';

/**
 * PWA install prompt — shows a banner when the browser fires
 * the `beforeinstallprompt` event (Chromium-based browsers).
 * No-op on iOS / unsupported browsers.
 */
export default function InstallPrompt() {
  const [deferredPrompt, setDeferredPrompt] = useState(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const handler = (e) => {
      e.preventDefault();
      setDeferredPrompt(e);
      setVisible(true);
    };

    window.addEventListener('beforeinstallprompt', handler);
    return () => window.removeEventListener('beforeinstallprompt', handler);
  }, []);

  const handleInstall = async () => {
    if (!deferredPrompt) return;
    deferredPrompt.prompt();
    const { outcome } = await deferredPrompt.userChoice;
    if (outcome === 'accepted') {
      setVisible(false);
    }
    setDeferredPrompt(null);
  };

  if (!visible) return null;

  return (
    <div
      style={{
        position: 'fixed',
        bottom: '1rem',
        left: '1rem',
        right: '1rem',
        zIndex: 9999,
        background: 'linear-gradient(135deg, #1e3a5f 0%, #1a3252 100%)',
        color: '#ffffff',
        borderRadius: '1rem',
        padding: '0.875rem 1.25rem',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: '0.75rem',
        boxShadow: '0 8px 32px rgba(0, 0, 0, 0.25)',
        fontFamily: "'Sora', -apple-system, sans-serif",
        fontSize: '0.8rem',
      }}
    >
      <div>
        <strong>Install BINUS Attend</strong>
        <div style={{ opacity: 0.7, fontSize: '0.7rem', marginTop: 2 }}>
          Add to home screen for quick access
        </div>
      </div>
      <div style={{ display: 'flex', gap: '0.5rem', flexShrink: 0 }}>
        <button
          onClick={() => setVisible(false)}
          style={{
            background: 'rgba(255,255,255,0.15)',
            border: '1px solid rgba(255,255,255,0.2)',
            color: '#fff',
            borderRadius: '0.5rem',
            padding: '0.4rem 0.75rem',
            cursor: 'pointer',
            fontSize: '0.75rem',
          }}
        >
          Later
        </button>
        <button
          onClick={handleInstall}
          style={{
            background: '#22c55e',
            border: 'none',
            color: '#fff',
            borderRadius: '0.5rem',
            padding: '0.4rem 0.75rem',
            cursor: 'pointer',
            fontWeight: 600,
            fontSize: '0.75rem',
          }}
        >
          Install
        </button>
      </div>
    </div>
  );
}
