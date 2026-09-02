import { Navigate, createBrowserRouter } from "react-router-dom";

import { AppShell } from "@/components/layout/AppShell";
import { DashboardPage } from "@/pages/Dashboard";
import { ForecastPage } from "@/pages/Forecast";
import { ImportPage } from "@/pages/Import";
import { PortfolioPage } from "@/pages/Portfolio";
import { TransactionsPage } from "@/pages/Transactions";

/** TRD §9.6: five routes matching PRD §11.1's five sections exactly. */
export const router = createBrowserRouter([
  {
    path: "/",
    element: <AppShell />,
    children: [
      { index: true, element: <Navigate to="/dashboard" replace /> },
      { path: "dashboard", element: <DashboardPage /> },
      { path: "transactions", element: <TransactionsPage /> },
      { path: "import", element: <ImportPage /> },
      { path: "forecast", element: <ForecastPage /> },
      { path: "portfolio", element: <PortfolioPage /> },
    ],
  },
]);
