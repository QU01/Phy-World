import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// En desarrollo (npm run dev) la API la sirve `python world4_cli.py --serve` en el puerto 8765.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { "/api": "http://127.0.0.1:8765" } },
  build: { outDir: "dist", chunkSizeWarningLimit: 1500 },
});
