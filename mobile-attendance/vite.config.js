import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { fileURLToPath } from 'url'
import { dirname } from 'path'

const __dirname = dirname(fileURLToPath(import.meta.url))

// https://vite.dev/config/
export default defineConfig({
  root: __dirname,
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          'face-api': ['face-api.js'],
          'firebase': ['firebase/app', 'firebase/firestore', 'firebase/auth'],
          'leaflet': ['leaflet', 'react-leaflet'],
          'turf': [
            '@turf/buffer',
            '@turf/bbox-polygon',
            '@turf/boolean-point-in-polygon',
            '@turf/helpers',
          ],
        },
      },
    },
  },
})
