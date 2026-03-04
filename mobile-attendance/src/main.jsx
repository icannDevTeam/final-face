import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)

// Register service worker for PWA — force update check every load
if ('serviceWorker' in navigator) {
  window.addEventListener('load', async () => {
    try {
      const reg = await navigator.serviceWorker.register('/sw.js', {
        updateViaCache: 'none', // bypass HTTP cache for SW script
      });

      // Check for updates immediately
      reg.update().catch(() => {});

      // When a new SW is waiting, tell it to skip waiting and take over
      const onNewSW = (sw) => {
        sw.addEventListener('statechange', () => {
          if (sw.state === 'activated') {
            // New SW activated — reload to use fresh assets
            window.location.reload();
          }
        });
        sw.postMessage({ type: 'SKIP_WAITING' });
      };

      if (reg.waiting) {
        onNewSW(reg.waiting);
      }
      reg.addEventListener('updatefound', () => {
        const newSW = reg.installing;
        if (newSW) {
          newSW.addEventListener('statechange', () => {
            if (newSW.state === 'installed' && navigator.serviceWorker.controller) {
              onNewSW(newSW);
            }
          });
        }
      });
    } catch (err) {
      console.warn('SW registration failed:', err);
    }
  });
}
