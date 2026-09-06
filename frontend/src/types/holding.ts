/** Mirrors backend/schemas/holding.py (TRD §6, §5.7, §13). */

export interface HoldingResponse {
  id: number;
  ticker: string;
  shares: number;
  /** null means "cost basis not recorded yet" -- never fabricated from
   * current_price or a demo value. `pnl` is also null whenever this is. */
  avg_cost: number | null;
  current_price: number | null;
  current_value: number | null;
  pnl: number | null;
  price_last_updated: string | null;
  /** True only when `price_last_updated` is the fixed sentinel a demo load
   * stamps on a never-actually-fetched seeded price -- lets PriceStatus show
   * an honest "Demo snapshot" label instead of implying a real (if old)
   * fetch. False for a real holding and for a demo holding refreshed since. */
  price_is_demo_snapshot: boolean;
  created_at: string;
  updated_at: string;
}

export interface HoldingCreate {
  ticker: string;
  shares: number;
  /** Optional -- a holding can be tracked ("10 MSFT shares") without a
   * known cost basis yet. Omit or send null, never a fabricated 0. */
  avg_cost?: number | null;
}

export interface HoldingUpdate {
  shares?: number;
  /** null explicitly clears a previously-known cost basis; omitting the
   * key entirely leaves it unchanged (see backend/schemas/holding.py). */
  avg_cost?: number | null;
}

export interface RefreshedTicker {
  ticker: string;
  price: number;
}

export interface FailedTicker {
  ticker: string;
  error: string;
}

export interface RefreshPricesResponse {
  refreshed: RefreshedTicker[];
  failed: FailedTicker[];
}
