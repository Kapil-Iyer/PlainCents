import { FlaskConical } from "lucide-react";

import { useAppState } from "@/context/AppStateContext";

/**
 * Build Plan §2.5 / TRD §9.7 corrected mode table: DEMO -> banner shown,
 * EMPTY -> no banner (onboarding empty-state instead), REAL -> no banner.
 */
export function DemoBanner() {
  const { mode } = useAppState();

  if (mode !== "DEMO") return null;

  return (
    <div className="flex items-center justify-center gap-2 bg-warning/15 px-4 py-2 text-sm font-medium text-warning">
      <FlaskConical className="h-4 w-4" />
      Demo Data — everything you see is sample data, not your own.
    </div>
  );
}
