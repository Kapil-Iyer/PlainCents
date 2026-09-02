import { defineConfig, devices } from "@playwright/test";

/**
 * Build Plan Phase 10 / §12: exactly four E2E flows, run against the
 * packaged reviewer-mode app (see backend/main.py + tests/e2e/global-setup.ts),
 * each against its own isolated temp SQLite database (tests/e2e/helpers/backend.ts).
 * workers: 1 keeps each flow's subprocess lifecycle simple and deterministic
 * on a single dev machine — there are only 4 flows, so parallelism isn't
 * worth the added flakiness risk here.
 */
export default defineConfig({
  testDir: "./tests/e2e",
  timeout: 60_000,
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: "list",
  globalSetup: require.resolve("./tests/e2e/global-setup.ts"),
  use: {
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
});
