/// <reference types="vitest/config" />
import path from "node:path";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(import.meta.dirname, "./src"),
    },
  },
  server: {
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
    css: true,
    // Vitest's 5s default is tight for a jsdom + Recharts + Framer Motion
    // suite running files in parallel: several tests pass comfortably in
    // isolation but intermittently time out under full-suite load. This is
    // a slow-environment allowance, not a licence for slow tests — nothing
    // here is expected to take anywhere near 20s.
    testTimeout: 20000,
  },
});
