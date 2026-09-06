/**
 * Analytics response types. Mirror backend/schemas/analytics.py exactly.
 *
 * Every one of these is a live aggregation over stored transactions grouped
 * by `effective_category` (the user's correction where one exists, otherwise
 * the system's prediction), so a manual correction moves every chart the
 * moment it is saved.
 */

export interface CategoryTrendPoint {
  month: string;
  total_spend: number;
  by_category: Record<string, number>;
}

export interface CategoryTrendResponse {
  months: string[];
  categories: string[];
  points: CategoryTrendPoint[];
}

export interface TopMerchantItem {
  merchant: string;
  merchant_key: string | null;
  total_spend: number;
  transaction_count: number;
  average_transaction: number;
  category: string | null;
  last_seen: string | null;
  pct_of_total: number;
}

export interface TopMerchantsResponse {
  items: TopMerchantItem[];
  total_spend: number;
  distinct_merchants: number;
  top_n_share_pct: number;
  months: number;
}

export interface CategoryMover {
  category: string;
  current: number;
  previous: number;
  change: number;
  /** null when the previous month was zero — there is no percentage of nothing. */
  change_pct: number | null;
}

export interface CategoryMoversResponse {
  current_month: string;
  previous_month: string;
  /** True when `current_month` is still in progress. When true, both totals
   * below are capped at the same day-of-month (`comparable_day`); when
   * false (a completed historical month was selected), both are full
   * calendar months, uncapped. */
  is_current_incomplete: boolean;
  total_current: number;
  total_previous: number;
  total_change: number;
  /** Both totals above are capped at this same day-of-month. */
  comparable_day: number;
  movers: CategoryMover[];
}

export interface SpendPacePoint {
  day: number;
  /** null past today, or past the previous month's real length — a genuine gap. */
  current_cumulative: number | null;
  previous_cumulative: number | null;
}

export interface SpendPaceResponse {
  current_month: string;
  previous_month: string;
  /** True when `current_month` is still in progress. When false (a
   * completed historical month was selected), both curves and to-date
   * figures run to their own full real length — there is no "today" to
   * stop at and no comparable-vs-context split. */
  is_current_incomplete: boolean;
  day_of_month: number;
  /** The previous month's own length may be shorter than day_of_month
   * (e.g. day 30/31 vs. February) — use this for any label naming the
   * previous period's day range. */
  comparable_day: number;
  current_to_date: number;
  previous_same_point: number;
  difference: number;
  points: SpendPacePoint[];
}

export interface ForecastAccuracyItem {
  forecast_month: string;
  category: string;
  predicted: number;
  actual: number;
  error: number;
  generated_at: string | null;
}

export interface ForecastAccuracyResponse {
  /**
   * False until a forecast run exists that was generated BEFORE the month it
   * predicted, for a month that has since completed. Past predictions are
   * never recomputed after the fact and shown as history.
   */
  available: boolean;
  reason: string | null;
  items: ForecastAccuracyItem[];
  months_covered: string[];
  total_predicted: number;
  total_actual: number;
  wape: number | null;
}
