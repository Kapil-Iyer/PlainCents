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
