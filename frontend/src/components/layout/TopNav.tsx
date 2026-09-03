import { useQuery } from "@tanstack/react-query";
import { AlertTriangle, Sparkles } from "lucide-react";
import { NavLink } from "react-router-dom";

import { getHealth } from "@/api/health";
import { Badge } from "@/components/ui/badge";
import { useAppState } from "@/context/AppStateContext";
import { cn } from "@/lib/utils";

const MODE_LABEL: Record<string, string> = {
  EMPTY: "No data yet",
  DEMO: "Demo",
  REAL: "Live",
};

export function TopNav() {
  const { mode } = useAppState();
  const { data: health } = useQuery({
    queryKey: ["health"],
    queryFn: getHealth,
    refetchInterval: 60_000,
  });

  return (
    <header className="flex h-14 shrink-0 items-center justify-between border-b border-border bg-card px-4 sm:px-5">
      {/* Sidebar is hidden below `md` — How It Works stays reachable on
       * mobile via this compact link instead of a full nav duplication. */}
      <NavLink
        to="/how-it-works"
        className={({ isActive }) =>
          cn(
            "flex items-center gap-1.5 rounded-md px-2 py-1 text-xs font-medium text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground md:hidden",
            isActive && "bg-primary/10 text-primary",
          )
        }
      >
        <Sparkles className="h-3.5 w-3.5" />
        How It Works
      </NavLink>
      <div className="hidden md:block" />
      <div className="flex items-center gap-3">
        {health?.categorization_model && health.categorization_model !== "loaded" && (
          <span className="flex items-center gap-1.5 text-xs font-medium text-warning">
            <AlertTriangle className="h-3.5 w-3.5" />
            Categorization model unavailable
          </span>
        )}
        <Badge variant={mode === "DEMO" ? "warning" : mode === "REAL" ? "success" : "outline"}>
          {MODE_LABEL[mode] ?? mode}
        </Badge>
      </div>
    </header>
  );
}
