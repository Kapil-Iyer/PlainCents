import { useQuery } from "@tanstack/react-query";
import { AlertTriangle } from "lucide-react";

import { getHealth } from "@/api/health";
import { Badge } from "@/components/ui/badge";
import { useAppState } from "@/context/AppStateContext";

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
    <header className="flex h-14 shrink-0 items-center justify-between border-b border-border bg-card px-5">
      <div />
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
