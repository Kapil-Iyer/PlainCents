import { Suspense, lazy } from "react";
import { Navigate, createBrowserRouter } from "react-router-dom";

import { AppShell } from "@/components/layout/AppShell";
import { Skeleton } from "@/components/ui/skeleton";

// Phase 10: route-level code splitting. Dashboard stays eagerly loaded since
// it's the default/EMPTY-onboarding route (and carries the walkthrough) —
// splitting it would just delay first paint, not save anything. The other
// four pages are lazy so the initial bundle doesn't ship Recharts/dialog
// forms for pages the reviewer may never open in a given session.
import { DashboardPage } from "@/pages/Dashboard";
const ForecastPage = lazy(() => import("@/pages/Forecast").then((m) => ({ default: m.ForecastPage })));
const ImportPage = lazy(() => import("@/pages/Import").then((m) => ({ default: m.ImportPage })));
const PortfolioPage = lazy(() => import("@/pages/Portfolio").then((m) => ({ default: m.PortfolioPage })));
const TransactionsPage = lazy(() =>
  import("@/pages/Transactions").then((m) => ({ default: m.TransactionsPage })),
);

function RouteFallback() {
  return (
    <div className="flex flex-col gap-4">
      <Skeleton className="h-8 w-48" />
      <Skeleton className="h-64 w-full" />
    </div>
  );
}

function lazyRoute(Component: React.LazyExoticComponent<() => React.JSX.Element>) {
  return (
    <Suspense fallback={<RouteFallback />}>
      <Component />
    </Suspense>
  );
}

/** TRD §9.6: five routes matching PRD §11.1's five sections exactly. */
export const router = createBrowserRouter([
  {
    path: "/",
    element: <AppShell />,
    children: [
      { index: true, element: <Navigate to="/dashboard" replace /> },
      { path: "dashboard", element: <DashboardPage /> },
      { path: "transactions", element: lazyRoute(TransactionsPage) },
      { path: "import", element: lazyRoute(ImportPage) },
      { path: "forecast", element: lazyRoute(ForecastPage) },
      { path: "portfolio", element: lazyRoute(PortfolioPage) },
    ],
  },
]);
