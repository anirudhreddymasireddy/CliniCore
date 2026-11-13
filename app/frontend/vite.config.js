import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// The backend (FastAPI, read-only) runs on 127.0.0.1:8765; /api is proxied in dev and preview.
const api = { '/api': { target: 'http://127.0.0.1:8765', changeOrigin: false } }

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { host: '127.0.0.1', port: 5173, strictPort: true, proxy: api },
  preview: { host: '127.0.0.1', port: 5173, strictPort: true, proxy: api },
})
