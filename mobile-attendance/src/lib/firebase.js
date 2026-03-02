/**
 * Firebase configuration for the mobile attendance app.
 * Reuses the same Firebase project as the main system.
 * Initializes anonymous auth so Firestore rules (request.auth != null) pass.
 */
import { initializeApp } from 'firebase/app';
import { getFirestore } from 'firebase/firestore';
import { getAuth, signInAnonymously, onAuthStateChanged } from 'firebase/auth';

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
export const auth = getAuth(app);

/**
 * Sign in anonymously so Firestore security rules pass.
 * Returns a promise that resolves once the user is authenticated.
 * Safe to call multiple times — only signs in once.
 */
export const authReady = new Promise((resolve) => {
  onAuthStateChanged(auth, (user) => {
    if (user) {
      resolve(user);
    } else {
      signInAnonymously(auth).catch((err) => {
        console.error('Anonymous auth failed:', err.message);
        resolve(null); // resolve anyway so app doesn't hang
      });
    }
  });
});

export default app;
