import { Navigate, createBrowserRouter } from "react-router-dom";
import { LayoutDashboard, LineChart, Wallet } from "lucide-react";

import { AppShell } from "@/components/layout/AppShell";
import { PlaceholderPage } from "@/pages/Placeholder";
import { ImportPage } from "@/pages/Import";
import { TransactionsPage } from "@/pages/Transactions";

/** TRD §9.6: five routes matching PRD §11.1's five sections exactly. */
export const router = createBrowserRouter([
  {
    path: "/",
    element: <AppShell />,
    children: [
      { index: true, element: <Navigate to="/dashboard" replace /> },
      {
        path: "dashboard",
        element: <PlaceholderPage title="Dashboard" icon={LayoutDashboard} phase={6} />,
      },
      { path: "transactions", element: <TransactionsPage /> },
      { path: "import", element: <ImportPage /> },
      {
        path: "forecast",
        element: <PlaceholderPage title="Forecast" icon={LineChart} phase={7} />,
      },
      {
        path: "portfolio",
        element: <PlaceholderPage title="Portfolio" icon={Wallet} phase={8} />,
      },
    ],
  },
]);
