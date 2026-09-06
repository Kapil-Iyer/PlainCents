/**
 * Guided tour step definitions (Build Plan Phase 10 follow-up, PATCH C).
 *
 * Each step names a real route and a real, always-rendered element on that
 * page (via a `data-tour` attribute already present in the live app's own
 * JSX) — the tour spotlights the actual product, not a mockup. Targets are
 * chosen to render unconditionally regardless of app state (EMPTY/DEMO/REAL,
 * loading, error, or zero rows), so the tour never gets stuck waiting for a
 * data-dependent element that may not exist yet.
 */
export interface TourStep {
  id: string;
  /** Route to navigate to before looking for `target` (a no-op if already there). */
  route: string;
  /** The `data-tour` attribute value of the real element to spotlight. */
  target: string;
  title: string;
  body: string;
}

export const TOUR_STEPS: TourStep[] = [
  {
    id: "welcome",
    route: "/dashboard",
    target: "topnav-mode-badge",
    title: "Welcome to PlainCents",
    body: "This badge always shows which of the three states the app is in: no data yet, sample Demo data, or your own Real data. The two are never mixed — let's walk through what each screen does.",
  },
  {
    id: "import",
    route: "/import",
    target: "page-header",
    title: "Bring in a bank CSV",
    body: "Upload a Canadian bank export, preview every row before anything is saved, and the categorization model suggests a category for each one. Nothing is written to your data until you confirm the preview.",
  },
  {
    id: "transactions",
    route: "/transactions",
    target: "transactions-tabs",
    title: "Review, correct, and analyze",
    body: "The list shows what the model predicted for each transaction. Correct one and it's marked confirmed — that correction is remembered for the same merchant next time. Insights holds the category and merchant analytics for these same rows.",
  },
  {
    id: "dashboard",
    route: "/dashboard",
    target: "page-header",
    title: "See spending at a glance",
    body: "A month-over-month summary, spending pace, category movers, and a spending trend — all computed live from your persisted transactions, sharing one selected month across every card.",
  },
  {
    id: "forecast",
    route: "/forecast",
    target: "page-header",
    title: "Forecasts, generated on demand",
    body: "A category-level spending forecast you generate explicitly — never silently retrained — and marked stale the moment your underlying data changes, so you always know whether what you're looking at is current.",
  },
  {
    id: "portfolio-holdings",
    route: "/portfolio",
    target: "portfolio-add-holding",
    title: "Holdings, refreshed on request",
    body: "Track a ticker and how many shares you hold. Average cost is optional — add it now, later, or let PlainCents calculate it from your purchases. Prices only ever update when you click Refresh Prices.",
  },
  {
    id: "portfolio-analytics",
    route: "/portfolio",
    target: "portfolio-analytics",
    title: "Portfolio analytics",
    body: "Total value, allocation by holding, and gain/loss by holding — all computed from your current holdings. Unrealized P&L only ever uses holdings where a cost basis is actually known; it's never guessed for the rest.",
  },
  {
    id: "portfolio-how-it-works",
    route: "/portfolio",
    target: "portfolio-how-it-works",
    title: "How your portfolio works",
    body: "The exact math behind value, cost basis, and P&L — including what happens when a cost basis is unknown. Portfolio tracking is entirely separate from spending: it never touches your transaction totals or forecasts.",
  },
  {
    id: "powerbi",
    route: "/dashboard",
    target: "export-powerbi",
    title: "Take it further in Power BI",
    body: "Download a Power BI-ready snapshot of your current data, then use the included setup guide to load it into Power BI Desktop. It's a snapshot, not a live connection — download a fresh one and refresh Power BI whenever your data changes.",
  },
  {
    id: "how-it-works",
    route: "/how-it-works",
    target: "page-header",
    title: "Want the full picture?",
    body: "This page walks through exactly how categorization, corrections, forecasting, portfolio math, and the Power BI export all work under the hood — including their honest limits. That's the whole tour — take a look around, or dive in here.",
  },
];
