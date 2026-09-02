import { Loader2, RefreshCw } from "lucide-react";

import { Button } from "@/components/ui/button";

interface RefreshPricesButtonProps {
  isPending: boolean;
  onClick: () => void;
}

/** PRD §9.7: refresh is manual only — this button is the only thing that
 * ever triggers POST /api/holdings/refresh-prices. */
export function RefreshPricesButton({ isPending, onClick }: RefreshPricesButtonProps) {
  return (
    <Button variant="outline" onClick={onClick} disabled={isPending}>
      {isPending ? (
        <>
          <Loader2 className="h-4 w-4 animate-spin" />
          Refreshing…
        </>
      ) : (
        <>
          <RefreshCw className="h-4 w-4" />
          Refresh prices
        </>
      )}
    </Button>
  );
}
