// vite.config.js
// What: Vite build tool configuration.
// Why:  Tells Vite to use the React plugin for JSX support.

import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,  // Frontend runs on this port
    open: true,  // Auto-open browser when you run npm run dev
  },
});
