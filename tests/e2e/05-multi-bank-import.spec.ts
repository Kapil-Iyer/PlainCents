import { expect, test } from "@playwright/test";

import { startBackend, type E2EBackend } from "./helpers/backend";

/**
 * FLOW 5 — six-bank import (Phase 12A.5/12B). One parameterized flow rather
 * than six duplicated spec files (per the six-bank design's own "prefer a
 * parameterized test" guidance) covering: RBC/Scotiabank/CIBC via
 * Auto-detect, TD's headerless fallback via Auto-detect (the specific
 * Phase 12A blocker this phase fixes — headerless TD must work when
 * Auto-detect is selected, not just explicit bank="TD"), an unsupported
 * format rejection, and one explicit-bank mismatch. Everything the backend
 * parser/integration layer already proves (row-level classification,
 * dedup, exclusion counts) is not re-proven here — this only proves the
 * frontend wiring (bank selector, detected-format display, rejection UI)
 * end-to-end.
 */
const PORT = 8115;
let backend: E2EBackend;

test.use({ baseURL: `http://127.0.0.1:${PORT}` });

test.beforeAll(async () => {
  backend = await startBackend(PORT);
});

test.afterAll(async () => {
  await backend.stop();
});

async function selectBank(page: import("@playwright/test").Page, bank: string) {
  await page.getByRole("combobox").click();
  await page.getByRole("option", { name: bank, exact: true }).click();
}

const AUTO_DETECT_CASES: { bank: string; fixture: string; validRows: number }[] = [
  { bank: "RBC", fixture: "tests/fixtures/rbc_csv/clean_valid.csv", validRows: 3 },
  { bank: "Scotiabank", fixture: "tests/fixtures/scotia_csv/clean_valid.csv", validRows: 4 },
  { bank: "CIBC", fixture: "tests/fixtures/cibc_csv/clean_valid.csv", validRows: 2 },
];

for (const { bank, fixture, validRows } of AUTO_DETECT_CASES) {
  test(`Auto-detect correctly identifies ${bank} and previews it`, async ({ page }) => {
    await page.goto("/import");
    await page.locator('input[type="file"]').setInputFiles(fixture);
    await page.getByRole("button", { name: "Upload & preview" }).click();

    await expect(page.getByRole("heading", { name: "Preview" })).toBeVisible();
    await expect(page.getByText(`Detected format: ${bank}`)).toBeVisible();
    await expect(page.getByText("Valid rows")).toBeVisible();
    const validRowsStat = page.getByText("Valid rows").locator("..").getByText(String(validRows));
    await expect(validRowsStat).toBeVisible();
  });
}

test("Auto-detect reaches TD's headerless fallback (Phase 12A blocker fix)", async ({ page }) => {
  await page.goto("/import");
  await page.locator('input[type="file"]').setInputFiles("tests/fixtures/td_csv/headerless_positional.csv");
  await page.getByRole("button", { name: "Upload & preview" }).click();

  await expect(page.getByRole("heading", { name: "Preview" })).toBeVisible();
  await expect(page.getByText("Detected format: TD")).toBeVisible();
  await expect(page.getByText("Credits excluded")).toBeVisible(); // the deposit-only row
});

test("an unsupported CSV format is rejected with a clear message, not silently imported", async ({ page }) => {
  await page.goto("/import");
  await page.locator('input[type="file"]').setInputFiles("tests/fixtures/td_csv/unrecognized_format.csv");
  await page.getByRole("button", { name: "Upload & preview" }).click();

  await expect(page.getByText("Import failed", { exact: true }).first()).toBeVisible();
  // Never silently lands on the Preview step for a genuinely unsupported file.
  await expect(page.getByRole("heading", { name: "Preview" })).not.toBeVisible();
});

test("a BMO-shaped file is never misclassified as TD (explicit selection required)", async ({ page }) => {
  await page.goto("/import");
  await page.locator('input[type="file"]').setInputFiles("tests/fixtures/shared_csv/blocked_balance_format.csv");
  await page.getByRole("button", { name: "Upload & preview" }).click();

  await expect(page.getByText("Import failed", { exact: true }).first()).toBeVisible();
  await expect(page.getByRole("heading", { name: "Preview" })).not.toBeVisible();
});

test("the bank selector shows BMO and National Bank as disabled Coming Soon options", async ({ page }) => {
  await page.goto("/import");
  await page.getByRole("combobox").click();

  await expect(page.getByRole("option", { name: "RBC", exact: true })).toBeEnabled();
  const bmo = page.getByRole("option", { name: "BMO — Coming Soon" });
  await expect(bmo).toHaveAttribute("aria-disabled", "true");
  const national = page.getByRole("option", { name: "National Bank — Coming Soon" });
  await expect(national).toHaveAttribute("aria-disabled", "true");
});

test("explicit bank selection never silently reinterprets a mismatched file", async ({ page }) => {
  await page.goto("/import");
  await selectBank(page, "RBC");
  await page.locator('input[type="file"]').setInputFiles("tests/fixtures/scotia_csv/clean_valid.csv");
  await page.getByRole("button", { name: "Upload & preview" }).click();

  await expect(page.getByText("Import failed", { exact: true }).first()).toBeVisible();
  await expect(page.getByRole("heading", { name: "Preview" })).not.toBeVisible();
});
