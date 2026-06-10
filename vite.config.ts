import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: 'build',
  },
  optimizeDeps: {
    entries: ['index.html'],
  },
  // Bake the backend URLs into the bundle at build time. The app reads
  // process.env.REACT_APP_API_URL / REACT_APP_WS_URL (src/api.ts); replacing
  // the whole `process.env` object with a literal both injects those values
  // AND keeps any other process.env.* access from throwing in the browser
  // (it just reads undefined). docker-compose passes these as build args
  // derived from FASTAPI_PORT, so the UI tracks the backend's host port.
  define: {
    'process.env': JSON.stringify({
      REACT_APP_API_URL:
        process.env.REACT_APP_API_URL || 'http://localhost:8000',
      REACT_APP_WS_URL:
        process.env.REACT_APP_WS_URL || 'ws://localhost:8000',
    }),
  },
});
