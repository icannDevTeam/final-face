/**
 * Firebase configuration for the mobile attendance app.
 * Reuses the same Firebase project as the main system.
 * Initializes anonymous auth so Firestore rules (request.auth != null) pass.
 */
import { initializeApp } from 'firebase/app';
import { getFirestore } from 'firebase/firestore';
import { getAuth, signInAnonymously, onAuthStateChanged } from 'firebase/auth';

const firebaseConfig = {
  apiKey: 'AIzaSyClDQe4e2NpfVw4nvLG10vzK8wmdGCHJwk',
  authDomain: 'facial-attendance-binus.firebaseapp.com',
  projectId: 'facial-attendance-binus',
  storageBucket: 'facial-attendance-binus.firebasestorage.app',
  messagingSenderId: '866005352235',
  appId: '1:866005352235:web:90f5c63b84892bdf774f6e',
  measurementId: 'G-RBKGEX9RHF',
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
