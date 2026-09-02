/**
 * Mirrors backend `config.CATEGORIES` (root config.py). GET /api/categories
 * is NOT implemented yet (TRD §5.5 is aspirational) — Build Plan Phase 5
 * explicitly says to mirror the fixed 8-label taxonomy here instead of
 * adding a backend route. If root config.py's CATEGORIES list ever changes,
 * this file must be updated to match by hand.
 */
export const CATEGORIES = [
  "Food & Dining",
  "Transport",
  "Rent & Utilities",
  "Entertainment",
  "Healthcare",
  "Shopping",
  "Subscriptions",
  "Other",
] as const;

export type Category = (typeof CATEGORIES)[number];
