import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import HomePage from './pages/HomePage';
import ScanPage from './pages/ScanPage';
import InstallPrompt from './components/InstallPrompt';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/scan" element={<ScanPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
      <InstallPrompt />
    </BrowserRouter>
  );
}
