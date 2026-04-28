import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { useEffect, lazy, Suspense } from 'react';
import HomePage from './pages/HomePage';
import InstallPrompt from './components/InstallPrompt';
import ErrorBoundary from './components/ErrorBoundary';

// Lazy-load ScanPage so face-api.js (~640KB chunk) isn't fetched until /scan is opened.
const ScanPage = lazy(() => import('./pages/ScanPage'));

export default function App() {
  // Defer face-api preload until after first paint so the ~6.5MB chunk doesn't block initial render.
  useEffect(() => {
    const schedule = window.requestIdleCallback || ((cb) => setTimeout(cb, 200));
    const handle = schedule(() => {
      import('./lib/faceRecognition').then((m) => m.preload()).catch(() => {});
    });
    return () => {
      if (window.cancelIdleCallback && typeof handle === 'number') {
        window.cancelIdleCallback(handle);
      } else {
        clearTimeout(handle);
      }
    };
  }, []);

  return (
    <ErrorBoundary>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route
            path="/scan"
            element={
              <ErrorBoundary>
                <Suspense fallback={<div style={{ padding: 24, textAlign: 'center', color: '#64748b' }}>Loading scanner…</div>}>
                  <ScanPage />
                </Suspense>
              </ErrorBoundary>
            }
          />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
        <InstallPrompt />
      </BrowserRouter>
    </ErrorBoundary>
  );
}
