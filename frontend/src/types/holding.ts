/** Mirrors backend/schemas/holding.py (TRD §6, §5.7, §13). */

export interface HoldingResponse {
  id: number;
  ticker: string;
  shares: number;
  avg_cost: number;
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
  avg_cost: number;
}

export interface HoldingUpdate {
  shares?: number;
  avg_cost?: number;
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
