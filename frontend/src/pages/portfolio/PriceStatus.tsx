import { formatCurrency, formatDate } from "@/lib/utils";
import type { HoldingResponse } from "@/types/holding";

interface PriceStatusProps {
  holding: HoldingResponse;
}

/** TRD §13.3: cached price + its timestamp show regardless of age; a
 * never-refreshed holding gets an honest "not yet refreshed" state instead
 * of a fabricated price (Build Plan Phase 8 price-state rule #1). */
export function PriceStatus({ holding }: PriceStatusProps) {
  if (holding.current_price === null) {
    return <span className="text-sm text-muted-foreground">Not yet refreshed</span>;
  }

  return (
    <div className="flex flex-col">
      <span className="font-medium tabular-nums">{formatCurrency(holding.current_price)}</span>
      {holding.price_last_updated && (
        <span className="text-xs text-muted-foreground">
          as of {formatDate(holding.price_last_updated.slice(0, 10))}
        </span>
      )}
    </div>
  );
}
