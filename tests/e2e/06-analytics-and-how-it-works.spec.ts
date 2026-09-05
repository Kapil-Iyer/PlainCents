import { expect, test, type Page } from "@playwright/test";

import { startBackend, type E2EBackend } from "./helpers/backend";

/**
 * FLOW 6 — the analytics surfaces and the rebuilt How It Works page.
 *
 * Driven through demo mode rather than an import, because these views need
 * a year of history to have anything to say, and the demo seed is the one
 * deterministic source of that. What is being proven here is the frontend
 * wiring and the empty/populated behaviour of the new surfaces — the
 * aggregation itself is already proven directly at the service layer
 * (tests/backend/services/test_analytics_service.py), and is not re-proven
 * through a browser.
 */
const PORT = 8116;
let backend: E2EBackend;

test.beforeAll(async () => {
  backend = await startBackend(PORT);
});

test.afterAll(async () => {
  await backend?.stop();
});

test.use({ baseURL: `http://127.0.0.1:${PORT}` });

/**
 * Every test in this file shares one backend process and therefore one
 * database, so demo data survives between them. This is idempotent: it loads
 * the demo only when the app is still EMPTY, and otherwise just confirms the
 * populated dashboard is up.
 */
async function loadDemo(page: Page) {
  await page.goto("/dashboard");
  const loadButton = page.getByRole("button", { name: "Load demo data" });
  if (await loadButton.isVisible().catch(() => false)) {
    await loadButton.click();
  }
  await expect(page.getByRole("heading", { name: "Spending pace" })).toBeVisible();
}

test.describe("analytics", () => {
  test("EMPTY state shows the onboarding screen, not empty charts", async ({ page }) => {
    await page.goto("/dashboard");

    // No chart should be drawn from nothing.
    await expect(page.getByRole("heading", { name: "Spending pace" })).toHaveCount(0);
    await expect(page.getByRole("button", { name: "Load demo data" })).toBeVisible();
  });

  test("dashboard shows spending pace and what changed once data exists", async ({ page }) => {
    await loadDemo(page);

    await expect(page.getByRole("heading", { name: "Spending pace" })).toBeVisible();
    await expect(page.getByRole("heading", { name: "What changed this month" })).toBeVisible();
    // Both cards render real content rather than their error/empty fallbacks.
    await expect(page.getByText("Couldn't load spending pace")).toHaveCount(0);
  });

  test("Transactions Insights tab shows category trend and top merchants", async ({ page }) => {
    await loadDemo(page);
    await page.goto("/transactions");

    await expect(page.getByRole("tab", { name: "Transactions" })).toHaveAttribute(
      "aria-selected",
      "true",
    );

    await page.getByRole("tab", { name: "Insights" }).click();

    await expect(page.getByRole("heading", { name: "Category trend" })).toBeVisible();
    await expect(page.getByRole("heading", { name: "Top merchants" })).toBeVisible();
    await expect(page.getByText("No merchants in this period")).toHaveCount(0);
  });

  test("category trend switches between stacked and line views and time ranges", async ({
    page,
  }) => {
    await loadDemo(page);
    await page.goto("/transactions");
    await page.getByRole("tab", { name: "Insights" }).click();
    await expect(page.getByRole("heading", { name: "Category trend" })).toBeVisible();

    const chartType = page.getByRole("radiogroup", { name: "Chart type" });
    await chartType.getByRole("radio", { name: "Lines" }).click();
    await expect(chartType.getByRole("radio", { name: "Lines" })).toHaveAttribute(
      "aria-checked",
      "true",
    );

    const range = page.getByRole("radiogroup", { name: "Time range" }).first();
    await range.getByRole("radio", { name: "24m" }).click();
    await expect(range.getByRole("radio", { name: "24m" })).toHaveAttribute("aria-checked", "true");
  });

  test("forecast accuracy says why it has nothing to show, rather than showing nothing", async ({
    page,
  }) => {
    await loadDemo(page);
    await page.goto("/forecast");

    // Demo mode seeds a forecast, so the accuracy card renders — and its
    // honest empty state is what should appear: the seeded run was not
    // generated before a month that has since completed.
    await expect(page.getByText("No forecast history yet")).toBeVisible();
    await expect(page.getByText(/won't re-run today's model on old months/)).toBeVisible();
  });
});

test.describe("How It Works", () => {
  test("opens on the product premise and reaches every section", async ({ page }) => {
    await page.goto("/how-it-works");

    await expect(
      page.getByRole("heading", {
        name: /reads your bank statements and tells you where the money actually went/i,
      }),
    ).toBeVisible();

    for (const id of ["walkthrough", "video", "categorization", "memory", "forecasting", "evaluation", "limitations"]) {
      await expect(page.locator(`#${id}`)).toHaveCount(1);
    }
  });

  test("the app walkthrough steps forward", async ({ page }) => {
    await page.goto("/how-it-works");
    const walkthrough = page.locator("#walkthrough");

    await expect(walkthrough.getByText("Step 1 of 10")).toBeVisible();
    await walkthrough.getByRole("button", { name: "Next" }).click();
    await expect(walkthrough.getByText("Step 2 of 10")).toBeVisible();
  });

  test("the video section reports honestly that no recording is present", async ({ page }) => {
    await page.goto("/how-it-works");

    await expect(page.getByText("The walkthrough hasn't been recorded yet")).toBeVisible();
    await expect(
      page.getByText("frontend/public/media/plaincents-walkthrough.mp4"),
    ).toBeVisible();
  });

  test("the deep link from Transactions lands on the categorization section", async ({ page }) => {
    await page.goto("/transactions");
    await page.getByRole("link", { name: "How was this predicted?" }).click();

    await expect(page).toHaveURL(/how-it-works#categorization/);
    await expect(page.getByText("What the system decided")).toBeVisible();
  });

  test("the forecast explainer shows its arithmetic and updates with the preset", async ({
    page,
  }) => {
    await page.goto("/how-it-works");
    const forecasting = page.locator("#forecasting");

    await expect(forecasting.getByText(/\$300\.00 \+ \$450\.00 \+ \$600\.00\) ÷ 3 =/)).toBeVisible();
    await forecasting.getByRole("radio", { name: "One big month" }).click();
    await expect(forecasting.getByText(/\$280\.00 \+ \$310\.00 \+ \$900\.00\) ÷ 3 =/)).toBeVisible();
  });
});

test.describe("responsive", () => {
  test("no page overflows horizontally on a phone-sized viewport", async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await loadDemo(page);

    for (const path of ["/dashboard", "/transactions", "/forecast", "/import", "/how-it-works"]) {
      await page.goto(path);
      // Let charts finish their initial layout pass before measuring.
      await page.waitForTimeout(300);
      const overflow = await page.evaluate(
        () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
      );
      // A couple of pixels of rounding is tolerable; a scrollbar's worth is
      // a layout bug. Wide content (tables, charts) must scroll inside its
      // own container, never the page.
      expect(overflow, `${path} overflows horizontally by ${overflow}px`).toBeLessThanOrEqual(2);
    }
  });

  test("the Insights tab is reachable and readable on a phone", async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await loadDemo(page);
    await page.goto("/transactions");

    await page.getByRole("tab", { name: "Insights" }).click();
    await expect(page.getByRole("heading", { name: "Top merchants" })).toBeVisible();
  });
});
