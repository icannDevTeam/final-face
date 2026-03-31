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
 * Rejects with a clear message if anonymous auth is not enabled.
 * Safe to call multiple times — only signs in once.
 */
export const authReady = new Promise((resolve, reject) => {
  onAuthStateChanged(auth, (user) => {
    if (user) {
      resolve(user);
    } else {
      signInAnonymously(auth).catch((err) => {
        console.error('Anonymous auth failed:', err.message);
        reject(new Error(
          'Authentication failed. Anonymous sign-in may not be enabled in Firebase Console. ' +
          'Please ask your administrator to enable it (Firebase Console → Authentication → Sign-in method → Anonymous).'
        ));
      });
    }
  });
});

export default app;
