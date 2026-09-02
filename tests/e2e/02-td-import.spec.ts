import { expect, test } from "@playwright/test";

import { startBackend, type E2EBackend } from "./helpers/backend";

/**
 * FLOW 2 — TD import (Build Plan Phase 10 / §12).
 * clean app -> upload tests/fixtures/td_csv/clean_valid.csv -> preview ->
 * confirm -> verify transactions are visible with predicted categories.
 */
const PORT = 8112;
let backend: E2EBackend;

test.use({ baseURL: `http://127.0.0.1:${PORT}` });

test.beforeAll(async () => {
  backend = await startBackend(PORT);
});

test.afterAll(async () => {
  await backend.stop();
});

test("upload TD CSV -> preview -> confirm -> transactions show predicted categories", async ({ page }) => {
  await page.goto("/import");
  await expect(page.getByRole("heading", { name: "Import", exact: true })).toBeVisible();

  const fileInput = page.locator('input[type="file"]');
  await fileInput.setInputFiles("tests/fixtures/td_csv/clean_valid.csv");
  await page.getByRole("button", { name: "Upload & preview" }).click();

  // Preview stage: sample rows with a predicted category each.
  await expect(page.getByRole("heading", { name: "Preview" })).toBeVisible();
  await expect(page.getByText("Valid rows")).toBeVisible();
  const previewTable = page.locator("table").first();
  await expect(previewTable.locator("tbody tr").first()).toBeVisible();
  // At least one predicted category cell is populated (not the "—" fallback).
  await expect(previewTable.locator("tbody tr").first().locator("td").nth(3)).not.toHaveText("—");

  await page.getByRole("button", { name: "Confirm import" }).click();

  // Result stage.
  await expect(page.getByText("Import complete")).toBeVisible();
  await page.getByRole("link", { name: "View transactions" }).click();

  // Transactions list: rows visible with a predicted-category badge each
  // (dashed/muted styling per CategoryBadge — not yet user-confirmed).
  await expect(page).toHaveURL(/\/transactions$/);
  const rows = page.locator("table tbody tr");
  await expect(rows.first()).toBeVisible();
  // Category column (3rd) renders a CategoryBadge — predicted (not yet
  // user-confirmed) transactions show the dashed/muted "predicted" variant.
  const firstRowCategoryBadge = rows.first().locator("td").nth(2).locator('[title="Predicted by the categorization model"]');
  await expect(firstRowCategoryBadge).toBeVisible();
});
