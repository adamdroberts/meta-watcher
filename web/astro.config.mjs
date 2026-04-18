import { defineConfig } from "astro/config";

const API_TARGET = process.env.META_WATCHER_BACKEND ?? "http://127.0.0.1:8765";

export default defineConfig({
  output: "static",
  outDir: "../meta_watcher/web/static",
  server: {
    host: "127.0.0.1",
    port: 4321,
  },
  vite: {
    server: {
      proxy: {
        "/api": { target: API_TARGET, changeOrigin: false },
        "/stream.mjpg": { target: API_TARGET, changeOrigin: false, ws: false },
        "/frame.jpg": { target: API_TARGET, changeOrigin: false },
      },
    },
  },
});
