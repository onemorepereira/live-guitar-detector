/// <reference types="vitest" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import basicSsl from "@vitejs/plugin-basic-ssl";

// HTTPS dev mode is required for `getUserMedia` from a phone over the
// LAN — browsers only allow camera access on secure contexts, and
// `localhost` is the only insecure exception. The self-signed cert
// produced by basic-ssl triggers a one-time browser warning; click
// "Advanced → Proceed" once per device and the WebSocket + WebRTC paths
// inherit the secure context.
export default defineConfig({
  plugins: [react(), basicSsl()],
  server: {
    host: true,
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      "/ws": {
        target: "ws://localhost:8000",
        ws: true,
      },
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./tests/setup.ts"],
    css: false,
  },
});
