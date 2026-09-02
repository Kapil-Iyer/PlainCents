import { expect, test } from "@playwright/test";

import { startBackend, type E2EBackend } from "./helpers/backend";

/**
 * FLOW 3 — Forecast staleness (Build Plan Phase 10 / §12).
 * begin from a state with an existing forecast -> edit/correct a
 * transaction category -> verify the prior forecast remains visible but is
 * marked stale -> explicitly refresh -> verify stale clears and a refreshed
 * persisted forecast is shown.
 *
 * "Begin from a state with an existing forecast": demo load seeds 12 months
 * of transaction history plus a prebuilt forecast run (backend/services/
 * demo_seed_data.py:generate_demo_forecast), which is the fastest
 * deterministic way to reach that starting state without fabricating a
 * separate cold-start bypass.
 */
const PORT = 8113;
let backend: E2EBackend;

test.use({ baseURL: `http://127.0.0.1:${PORT}` });

test.beforeAll(async () => {
  backend = await startBackend(PORT);
});

test.afterAll(async () => {
  await backend.stop();
});

test("editing a transaction's category stales the existing forecast; explicit refresh clears it", async ({
  page,
}) => {
  // Start from a state with an existing forecast: load demo.
  await page.goto("/dashboard");
  await page.getByRole("button", { name: /Load demo data/ }).click();
  await expect(page.getByText("Demo Data — everything you see is sample data, not your own.")).toBeVisible();

  await page.goto("/forecast");
  await expect(page.getByRole("heading", { name: "Forecast", exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: /Refresh forecast/ })).toBeVisible();
  await expect(page.getByText(/out of date/)).not.toBeVisible();

  // Correct a transaction's category — a staleness-triggering mutation.
  await page.goto("/transactions");
  const firstRow = page.locator("table tbody tr").first();
  await firstRow.getByRole("button", { name: /^Edit/ }).click();

  await expect(page.getByRole("heading", { name: "Edit transaction" })).toBeVisible();
  await page.getByLabel("Category (optional override)").click();
  await page.getByRole("option", { name: "Entertainment" }).click();
  await page.getByRole("button", { name: "Save changes" }).click();
  await expect(page.getByRole("heading", { name: "Edit transaction" })).not.toBeVisible();

  // Prior forecast remains readable but is now marked stale.
  await page.goto("/forecast");
  await expect(page.getByText(/out of date/)).toBeVisible();
  await expect(page.getByRole("button", { name: /Refresh forecast/ })).toBeVisible();
  // The prior prediction content is still shown underneath the warning.
  await expect(page.getByText(/Generated/)).toBeVisible();

  // Explicit refresh recomputes it.
  await page.getByRole("button", { name: /Refresh forecast/ }).click();
  await expect(page.getByRole("button", { name: /Generating/ })).toBeVisible();
  await expect(page.getByText(/out of date/)).not.toBeVisible({ timeout: 20_000 });
  await expect(page.getByRole("button", { name: /Refresh forecast/ })).toBeVisible();
});
