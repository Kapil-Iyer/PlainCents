/** Mirrors backend/schemas/dashboard.py (TRD §6 DashboardSummary, §5.8). */
import type { TransactionResponse } from "@/types/transaction";
import type { DataMode } from "@/types/common";

export interface DashboardPeriod {
  current: string;
  previous: string;
}

export interface CategoryBreakdownItem {
  category: string;
  total_spend: number;
  pct_of_total: number;
}

export interface SpendingTrendPoint {
  month: string;
  total_spend: number;
}

export interface DashboardSummaryResponse {
  period: DashboardPeriod;
  total_spend_current: number;
  total_spend_previous: number;
  /** Previous month's spend capped at `comparable_day` — the fair basis
   * `change_pct` is computed against. `total_spend_previous` above stays
   * the full previous calendar month, a separate honest standalone figure. */
  total_spend_previous_to_date: number;
  comparable_day: number;
  change_pct: number | null;
  category_breakdown: CategoryBreakdownItem[];
  spending_trend: SpendingTrendPoint[];
  recent_transactions: TransactionResponse[];
  /** Absent until Phase 7's ForecastService lands (Build Plan Phase 6, item 12). */
  forecast_summary: Record<string, unknown> | null;
  /** Absent until Phase 8's PortfolioService lands. */
  portfolio_summary: Record<string, unknown> | null;
  data_mode: DataMode;
}
