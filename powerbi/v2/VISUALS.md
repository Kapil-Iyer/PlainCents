# Suggested visual layout

A starting point, not a requirement — arrange these however you like once
the four tables are loaded (see [SCHEMA.md](./SCHEMA.md)). Two report pages
keep spending and portfolio separate, mirroring how the app itself never
mixes the two domains.

## Page 1 — Spending

| Visual | Fields | Notes |
|---|---|---|
| KPI card | `category_summary[total_spend]` (current month, filtered) | Total spend for the selected month. |
| Line/area chart | `category_summary[month]` (axis), `total_spend` (values), split by `category` | Mirrors the app's Spending Trend. |
| Bar chart (horizontal) | `category_summary[category]` (axis), `total_spend` (values) | Mirrors the app's Spending by Category. |
| Table or matrix | `transactions[date, merchant, amount, category, is_manual_override]` | The row-level detail, filterable by the visuals above. |
| Slicer | `category_summary[category]` | Cross-filters every visual on the page. |
| Slicer | `category_summary[month]` | Month-by-month drill-down. |

**Forecast overlay**: a combo chart with `category_summary[total_spend]`
(actuals, as columns) and `forecast[predicted_amount]` (forecast, as a
line), both filtered to the same `category` and aligned by month, shows
actual-vs-forecast the same way the app's own Forecast page does. Consider
a visual-level filter or a legend on `forecast[is_stale]` so a stale
forecast is visibly marked, not presented as current.

## Page 2 — Portfolio

| Visual | Fields | Notes |
|---|---|---|
| KPI card | `SUM(portfolio[current_value])` | Total market value. Power BI's SUM already skips blanks, so an unpriced holding is correctly excluded, not treated as 0. |
| KPI card | A measure summing `pnl` only where it is not blank (e.g. `SUMX(FILTER(portfolio, NOT ISBLANK(portfolio[pnl])), portfolio[pnl])`) | Unrealized P&L — matches the app's "only holdings with a known cost basis" rule. Label it that way on the visual itself. |
| Donut or bar chart | `portfolio[ticker]` (legend/axis), `current_value` (values) | Allocation by market value — mirrors the app's Portfolio Analytics allocation chart. |
| Bar chart (diverging) | `portfolio[ticker]` (axis), `pnl` (values), conditional formatting (red below 0, green above) | Gain/loss by holding — mirrors the app's own signed P&L chart. A blank `pnl` naturally renders as no bar; do not force it to 0. |
| Table | `portfolio[ticker, shares, avg_cost, current_price, current_value, pnl, price_last_updated]` | The row-level detail. |

**No time-series/performance-over-time visual for the portfolio.** The
export is a point-in-time snapshot — there is no historical portfolio-value
data to chart, in the app or in this export. Building one here would
require fabricating history PlainCents does not have.

## Applying the color theme

`plaincents_theme.json` (in this same folder) sets Power BI's default
categorical color palette to match the app's own chart colors. Apply it via
**View → Themes → Browse for themes** in Power BI Desktop, then select the
file. This is entirely optional and purely cosmetic — every visual above
works with Power BI's default theme too.
