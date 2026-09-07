/**
 * Guided tour step definitions (Build Plan Phase 10 follow-up, PATCH C; PATCH D
 * granularity pass).
 *
 * Each step names a real route and a real, always-rendered element on that
 * page (via a `data-tour` attribute already present in the live app's own
 * JSX) — the tour spotlights the actual product, not a mockup. Chart/analytics
 * targets (dashboard-summary onward through portfolio-how-it-works) only
 * render once Demo or Real data exists -- the tour's own entry points
 * (OnboardingEmptyState's "Load demo data & start tour", TopNav's "Replay
 * tour") always ensure Demo data is loaded first, so a step is never started
 * against an empty app with nothing to spotlight.
 *
 * ONE STEP = ONE VISUAL IDEA: the Dashboard's five cards (summary, Spending
 * Pace, What Changed, Spending by Category, Spending Trend) and the Forecast
 * chart each get their own step and their own target, rather than one step
 * gesturing at the whole page. Portfolio's allocation and P&L numbers stay
 * merged into one "portfolio-analytics" step, since they're already one
 * physical card in the product (PortfolioAnalytics), not two.
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
    body: "This badge always shows the app's mode — no data yet, sample Demo data, or your own Real data. The two are never mixed. Let's walk through what each screen does.",
  },
  {
    id: "import",
    route: "/import",
    target: "page-header",
    title: "Bring in a bank CSV",
    body: "Upload a Canadian bank export and preview every row before anything is saved. The categorization model suggests a category for each one — nothing is written until you confirm.",
  },
  {
    id: "transactions",
    route: "/transactions",
    target: "transactions-tabs",
    title: "Review, correct, and analyze",
    body: "Correct a prediction once and it's remembered for that merchant next time. Insights holds the category and merchant analytics for these same rows.",
  },
  {
    id: "dashboard-summary",
    route: "/dashboard",
    target: "dashboard-summary",
    title: "This month, at a glance",
    body: "This month's totals against last month, for the period picked above. Every chart below shares this same analysis month.",
  },
  {
    id: "spending-pace",
    route: "/dashboard",
    target: "spending-pace",
    title: "Spending Pace",
    body: "Two cumulative lines, this month against last, day by day — tells you at a glance whether you're ahead of or behind last month's pace.",
  },
  {
    id: "category-movers",
    route: "/dashboard",
    target: "category-movers",
    title: "What Changed",
    body: "A diverging bar chart split left and right around zero, showing exactly which categories drove the change from last month.",
  },
  {
    id: "category-breakdown",
    route: "/dashboard",
    target: "category-breakdown",
    title: "Spending by Category",
    body: "Where this month's spending actually went, broken down by category.",
  },
  {
    id: "spending-trend",
    route: "/dashboard",
    target: "spending-trend",
    title: "Spending Trend",
    body: "A 6-month area chart of total spending, so a single month's number always has its recent history right beside it.",
  },
  {
    id: "forecast",
    route: "/forecast",
    target: "forecast-chart",
    title: "Forecasts, generated on demand",
    body: "Your last three actual months next to the next three predicted ones, per category. Generated explicitly — never silently retrained — and marked stale the moment your data changes.",
  },
  {
    id: "portfolio-holdings",
    route: "/portfolio",
    target: "portfolio-add-holding",
    title: "Holdings, refreshed on request",
    body: "Track a ticker and how many shares you hold. Average cost is optional. Prices come from Yahoo Finance, cached for up to an hour, and only ever update when you click Refresh Prices.",
  },
  {
    id: "portfolio-analytics",
    route: "/portfolio",
    target: "portfolio-analytics",
    title: "Portfolio analytics",
    body: "Total value, allocation, and gain/loss — all computed from your current holdings. Unrealized P&L only uses holdings with a known cost basis; it's never guessed for the rest.",
  },
  {
    id: "portfolio-how-it-works",
    route: "/portfolio",
    target: "portfolio-how-it-works",
    title: "How your portfolio works",
    body: "The exact math behind value, cost basis, and P&L, plus the price source, cache freshness, and what Refresh Prices actually changes. Portfolio tracking never touches your spending totals or forecasts.",
  },
  {
    id: "powerbi",
    route: "/dashboard",
    target: "export-powerbi",
    title: "Take it further in Power BI",
    body: "Download a Power BI-ready snapshot of your current data, plus a setup guide. It's a snapshot, not a live connection — download a fresh one whenever your data changes.",
  },
  {
    id: "how-it-works",
    route: "/how-it-works",
    target: "page-header",
    title: "Want the full picture?",
    body: "This page explains exactly how categorization, forecasting, portfolio math, and the Power BI export work — including their honest limits. That's the whole tour — take a look around, or dive in here.",
  },
];
