import { expect, test } from "@playwright/test";

import { startBackend, type E2EBackend } from "./helpers/backend";

/**
 * FLOW 4 — Portfolio refresh (Build Plan Phase 10 / §12, §2A).
 * create holding -> verify opening/reading Portfolio does NOT trigger a
 * market network refresh -> explicitly invoke Refresh Prices -> verify
 * price and timestamp populate through the real application flow.
 *
 * Determinism (§2A): the real browser -> React -> FastAPI ->
 * PortfolioService -> repository/DB flow is exercised unmodified. Only the
 * external boundary (pipeline/portfolio.py's `import yfinance`) is swapped
 * for a deterministic offline double via PYTHONPATH, done entirely in test
 * infra (tests/e2e/helpers/backend.ts) — no production code changed. GET
 * /api/holdings still never touches that boundary at all; this flow proves
 * that by reloading the page after creating a holding and asserting the
 * price stays "Not yet refreshed" until Refresh Prices is clicked.
 */
const PORT = 8114;
let backend: E2EBackend;

test.use({ baseURL: `http://127.0.0.1:${PORT}` });

test.beforeAll(async () => {
  backend = await startBackend(PORT);
});

test.afterAll(async () => {
  await backend.stop();
});

test("GET holdings never fetches a price; explicit Refresh Prices populates price + timestamp", async ({
  page,
}) => {
  await page.goto("/portfolio");
  await expect(page.getByRole("heading", { name: "Portfolio" })).toBeVisible();

  // No holdings yet -> both the header action and the empty-state action
  // render "Add holding"; either opens the same dialog.
  await page.getByRole("button", { name: "Add holding" }).first().click();
  const dialog = page.getByRole("dialog");
  await expect(dialog.getByRole("heading", { name: "Add holding" })).toBeVisible();
  await dialog.getByLabel("Ticker").fill("E2ETEST");
  await dialog.getByLabel("Shares").fill("10");
  await dialog.getByLabel("Average cost per share").fill("50");
  await dialog.getByRole("button", { name: "Add holding" }).click();
  await expect(dialog).not.toBeVisible();

  const row = page.locator("table tbody tr", { hasText: "E2ETEST" });
  await expect(row).toBeVisible();
  await expect(row.getByText("Not yet refreshed")).toBeVisible();

  // Reloading (a plain GET /api/holdings) must not have triggered a fetch —
  // price stays unrefreshed after a reload, proving GET alone never hits
  // the external market-data boundary.
  await page.reload();
  const rowAfterReload = page.locator("table tbody tr", { hasText: "E2ETEST" });
  await expect(rowAfterReload.getByText("Not yet refreshed")).toBeVisible();

  // Explicit refresh.
  await page.getByRole("button", { name: /Refresh prices/i }).click();
  // exact: true — the toast's own text and its aria-live announcement
  // duplicate region ("Notification Prices refreshed") both otherwise match
  // a loose substring query, which is a strict-mode violation in Playwright.
  await expect(page.getByText("Prices refreshed", { exact: true }).first()).toBeVisible({ timeout: 10_000 });

  const rowAfterRefresh = page.locator("table tbody tr", { hasText: "E2ETEST" });
  await expect(rowAfterRefresh.getByText("Not yet refreshed")).not.toBeVisible();
  await expect(rowAfterRefresh.getByText(/as of/)).toBeVisible();
  // tests/e2e/fixtures/fake_yfinance's deterministic price for "E2ETEST" is
  // 100 + (sum of char codes % 50) = $108.00.
  await expect(rowAfterRefresh.getByText("$108.00")).toBeVisible();
});
