import { defineConfig, devices } from "@playwright/test";

/**
 * Build Plan Phase 10 / §12 (+ Phase 12B flow 5): five E2E flows, run against
 * the packaged reviewer-mode app (see backend/main.py + tests/e2e/global-setup.ts),
 * each against its own isolated temp SQLite database (tests/e2e/helpers/backend.ts).
 * Flow 5 (six-bank import) is one parameterized spec file covering
 * RBC/Scotiabank/CIBC/TD-headerless/unsupported-format/explicit-mismatch as
 * sub-tests, deliberately not six duplicated spec files — everything else
 * about per-bank row classification is already proven at the backend
 * parser/integration layer (tests/backend/services/test_ingest_six_bank.py).
 * workers: 1 keeps each flow's subprocess lifecycle simple and deterministic
 * on a single dev machine — there are only 5 flows, so parallelism isn't
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
