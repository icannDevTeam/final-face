/**
 * Firebase configuration for the mobile attendance app.
 * Reuses the same Firebase project as the main system.
 */
import { initializeApp } from 'firebase/app';
import { getFirestore } from 'firebase/firestore';

const firebaseConfig = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY || '',
  authDomain: 'facial-attendance-binus.firebaseapp.com',
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID || 'facial-attendance-binus',
  storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET || 'facial-attendance-binus.firebasestorage.app',
  messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID || '',
  appId: import.meta.env.VITE_FIREBASE_APP_ID || '',
};

const app = initializeApp(firebaseConfig);
export const db = getFirestore(app);
export default app;
