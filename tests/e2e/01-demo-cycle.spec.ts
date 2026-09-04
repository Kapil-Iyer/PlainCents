import { expect, test } from "@playwright/test";

import { startBackend, type E2EBackend } from "./helpers/backend";

/**
 * FLOW 1 — Demo cycle (Build Plan Phase 10 / §12).
 * clean app -> load demo -> verify DEMO mode + populated state -> clear
 * demo -> verify EMPTY state again.
 *
 * There is no standalone "reset" button in the product — the only sanctioned
 * UI path to clear demo data is the DemoConflictDialog shown when a real
 * import is attempted while DEMO is active (PRD §10a/§19). This flow uses
 * exactly that path, then cancels the resulting import preview without
 * confirming it, so the app settles back into EMPTY rather than REAL.
 */
const PORT = 8111;
let backend: E2EBackend;

test.use({ baseURL: `http://127.0.0.1:${PORT}` });

test.beforeAll(async () => {
  backend = await startBackend(PORT);
});

test.afterAll(async () => {
  await backend.stop();
});

test("clean app -> load demo -> populated DEMO state -> clear demo -> EMPTY again", async ({ page }) => {
  // 1. Clean app: EMPTY onboarding state.
  await page.goto("/dashboard");
  await expect(page.getByText("Welcome to PlainCents")).toBeVisible();
  await expect(page.getByRole("link", { name: /Import real data/ })).toBeVisible();
  await expect(page.getByRole("button", { name: /Load demo data/ })).toBeVisible();

  // 2. Load demo.
  await page.getByRole("button", { name: /Load demo data/ }).click();
  await expect(page.getByText("Demo Data — everything you see is sample data, not your own.")).toBeVisible();

  // Representative populated product state: onboarding empty state is gone,
  // dashboard shows real summary content instead.
  await expect(page.getByText("Welcome to PlainCents")).not.toBeVisible();
  await expect(page.getByRole("heading", { name: "Dashboard" })).toBeVisible();

  await page.goto("/transactions");
  await expect(page.getByText("No transactions yet")).not.toBeVisible();
  await expect(page.locator("table")).toBeVisible();

  // 3. Clear demo, via the only real product path: attempt a real import.
  await page.goto("/import");
  const fileInput = page.locator('input[type="file"]');
  await fileInput.setInputFiles("tests/fixtures/td_csv/clean_valid.csv");
  await page.getByRole("button", { name: "Upload & preview" }).click();

  await expect(page.getByText("Demo data is currently loaded")).toBeVisible();
  await page.getByRole("button", { name: /Clear demo data & retry/ }).click();

  // Retry re-uploads the same file for preview, but we deliberately do NOT
  // confirm it — cancel instead, so the app settles at EMPTY, not REAL.
  await expect(page.getByRole("heading", { name: "Preview" })).toBeVisible();
  await page.getByRole("button", { name: "Cancel" }).click();

  // 4. Verify EMPTY state again.
  await page.goto("/dashboard");
  await expect(page.getByText("Welcome to PlainCents")).toBeVisible();
  await expect(page.getByText("Demo Data — everything you see is sample data, not your own.")).not.toBeVisible();
});

test("clearing demo from a non-Dashboard page still leaves a discoverable way to reload it, repeatedly", async ({
  page,
}) => {
  // Regression test for an observed bug: DemoBanner's own "Clear demo data"
  // button is global chrome, reachable from every route -- but only
  // Dashboard.tsx ever rendered a "Load demo data" action, so clearing demo
  // from anywhere else left the user on a non-Dashboard EMPTY surface with
  // no visible way back into Demo mode. Fixed by DemoReentryBanner (shown on
  // every EMPTY-mode route except Dashboard, which already has its own
  // richer onboarding CTA). This must also work repeatedly, not just once.
  for (let cycle = 0; cycle < 2; cycle++) {
    await page.goto("/forecast");

    // Load demo (first pass) — either the fresh-EMPTY onboarding banner
    // (this exact regression's fix) or, on the second pass, the same banner
    // reappearing after the previous cycle's clear.
    await expect(page.getByRole("button", { name: "Load demo data" })).toBeVisible();
    await page.getByRole("button", { name: "Load demo data" }).click();
    await expect(page.getByText("Demo Data — everything you see is sample data, not your own.")).toBeVisible();

    // Clear demo directly from this same (non-Dashboard) page via
    // DemoBanner's own button -- the actual path that exposed the bug.
    await page.getByRole("button", { name: "Clear demo data" }).click();
    await page.getByRole("dialog").getByRole("button", { name: "Clear demo data" }).click();

    // The bug: nothing here previously offered a way back into Demo mode.
    await expect(page.getByText("Demo Data — everything you see is sample data, not your own.")).not.toBeVisible();
    await expect(page.getByRole("button", { name: "Load demo data" })).toBeVisible();
  }
});
