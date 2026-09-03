import { NavLink } from "react-router-dom";
import {
  ArrowLeftRight,
  LayoutDashboard,
  LineChart,
  Sparkles,
  UploadCloud,
  Wallet,
} from "lucide-react";

import { cn } from "@/lib/utils";

const NAV_ITEMS = [
  { to: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { to: "/transactions", label: "Transactions", icon: ArrowLeftRight },
  { to: "/import", label: "Import", icon: UploadCloud },
  { to: "/forecast", label: "Forecast", icon: LineChart },
  { to: "/portfolio", label: "Portfolio", icon: Wallet },
  { to: "/how-it-works", label: "How It Works", icon: Sparkles, isNew: true },
];

export function Sidebar() {
  return (
    <aside className="hidden w-60 shrink-0 flex-col border-r border-border bg-card md:flex">
      <div className="flex h-14 items-center gap-2.5 border-b border-border px-5">
        <div className="flex h-7 w-7 items-center justify-center rounded-md bg-primary text-sm font-bold text-primary-foreground shadow-sm shadow-primary/30">
          P
        </div>
        <span className="text-sm font-semibold tracking-tight">PlainCents</span>
      </div>
      <nav className="flex flex-1 flex-col gap-0.5 p-3">
        {NAV_ITEMS.map(({ to, label, icon: Icon, isNew }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              cn(
                "group flex items-center gap-2.5 rounded-md px-3 py-2 text-sm font-medium text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground",
                isActive && "bg-primary/10 text-primary hover:bg-primary/10 hover:text-primary",
              )
            }
          >
            {({ isActive }) => (
              <>
                <span
                  className={cn(
                    "flex h-7 w-7 items-center justify-center rounded-md transition-colors",
                    isActive ? "bg-primary/15 text-primary" : "text-muted-foreground group-hover:text-accent-foreground",
                  )}
                >
                  <Icon className="h-4 w-4" />
                </span>
                <span className="flex-1">{label}</span>
                {isNew && (
                  <span className="rounded-full bg-primary/15 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-primary">
                    New
                  </span>
                )}
              </>
            )}
          </NavLink>
        ))}
      </nav>
    </aside>
  );
}
