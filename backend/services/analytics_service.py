"""
AnalyticsService -- spending intelligence beyond the Dashboard summary.

Every number here is computed by live SQL aggregation over stored
transactions. Nothing is fabricated, extrapolated, or back-filled. In
particular:

  * All category grouping uses `effective_category`
    (COALESCE(confirmed_category, predicted_category), from
    v_transactions_effective), so a manual correction propagates into every
    chart the moment it is saved -- it is not a separate "corrected" view
    bolted on beside a "predicted" one.
  * Merchant analytics group by the stable merchant identity
    (`merchant_key`) rather than the raw description, because the raw text
    carries a different card suffix / store number on every transaction and
    grouping by it would split one merchant into dozens of one-row entries.
    A human-readable label is chosen deterministically from the actual stored
    text (see _merchant_label), never invented.
  * Forecast-vs-actual uses only GENUINE historical snapshots: a forecast run
    counts only if it was generated strictly BEFORE the month it predicted
    began, and only for months that have since completed. Old predictions are
    never recomputed and presented as if they had been made at the time.
    Until real snapshots exist, the endpoint honestly reports that it has
    nothing to show.

Each method answers one specific user question, named in its docstring. A
chart that does not answer a question a person would actually ask is not
worth the pixels.
"""
from __future__ import annotations

import sqlite3
from datetime import date

from backend.repositories.money import round_money

# Trailing window defaults. Not spec values -- product choices, kept here so
# the routes and tests share one definition.
DEFAULT_TREND_MONTHS = 12
DEFAULT_TOP_MERCHANTS = 8
MAX_TREND_MONTHS = 36
MAX_TOP_MERCHANTS = 25


def _shift_month(year: int, month: int, delta: int) -> tuple[int, int]:
    zero_based = (month - 1) + delta
    return year + zero_based // 12, zero_based % 12 + 1


def _month_str(year: int, month: int) -> str:
    return f"{year:04d}-{month:02d}"


def _month_start(year: int, month: int) -> str:
    return f"{year:04d}-{month:02d}-01"


def _mode_clause(data_mode: str | None, params: list) -> str:
    """Every query in this module is scoped by data_mode so DEMO and REAL
    rows can never mix into one chart. `None` (the EMPTY app state) matches
    nothing by construction, because EMPTY has no rows at all."""
    if data_mode is None:
        return " AND 1 = 0"
    params.append(data_mode)
    return " AND data_mode = ?"


class AnalyticsService:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    # -- 1. Category trend over time -------------------------------------

    def category_trend(
        self,
        data_mode: str | None,
        months: int = DEFAULT_TREND_MONTHS,
        reference_date: date | None = None,
    ) -> dict:
        """"Which categories are growing or shrinking, and since when?"

        Monthly spend per effective category over a trailing window, zero
        filled so a month with genuinely no spend in a category reads as $0
        rather than breaking the line. Only categories that appear at least
        once in the window are returned -- rendering six flat zero lines for
        categories the user has never spent in is noise, not information.
        """
        months = max(1, min(int(months), MAX_TREND_MONTHS))
        today = reference_date or date.today()
        start_y, start_m = _shift_month(today.year, today.month, -(months - 1))

        params: list = [_month_start(start_y, start_m)]
        sql = (
            "SELECT substr(date, 1, 7) AS month, effective_category AS category, "
            "SUM(amount) AS total_spend, COUNT(*) AS transaction_count "
            "FROM v_transactions_effective WHERE date >= ?"
        )
        sql += _mode_clause(data_mode, params)
        sql += " GROUP BY month, category ORDER BY month, category"
        rows = [dict(r) for r in self._conn.execute(sql, params).fetchall()]

        window = [
            _month_str(*_shift_month(today.year, today.month, -offset))
            for offset in range(months - 1, -1, -1)
        ]
        window_set = set(window)
        rows = [r for r in rows if r["month"] in window_set]

        categories = sorted({r["category"] for r in rows})
        by_month: dict[str, dict[str, float]] = {m: {} for m in window}
        for r in rows:
            by_month[r["month"]][r["category"]] = round_money(r["total_spend"])

        points = [
            {
                "month": m,
                "total_spend": round_money(sum(by_month[m].values())),
                "by_category": {c: by_month[m].get(c, 0.0) for c in categories},
            }
            for m in window
        ]
        return {"months": window, "categories": categories, "points": points}

    # -- 2. Top merchants -------------------------------------------------

    def top_merchants(
        self,
        data_mode: str | None,
        limit: int = DEFAULT_TOP_MERCHANTS,
        months: int = DEFAULT_TREND_MONTHS,
        reference_date: date | None = None,
    ) -> dict:
        """"Which specific places take the biggest share of my money?"

        Grouped by stable merchant identity, not raw description -- see the
        module docstring. Returns each merchant's total, transaction count,
        average transaction, dominant effective category, and share of the
        window's total spend, plus the combined share of the returned top N
        (merchant concentration), which is the number that actually tells a
        user whether their spending is concentrated or diffuse.
        """
        limit = max(1, min(int(limit), MAX_TOP_MERCHANTS))
        months = max(1, min(int(months), MAX_TREND_MONTHS))
        today = reference_date or date.today()
        start_y, start_m = _shift_month(today.year, today.month, -(months - 1))
        start = _month_start(start_y, start_m)

        params: list = [start]
        sql = (
            "SELECT COALESCE(merchant_key, 'RAW:' || merchant) AS group_key, "
            "       MIN(merchant) AS label, "
            "       SUM(amount) AS total_spend, "
            "       COUNT(*) AS transaction_count, "
            "       MAX(date) AS last_seen "
            "FROM v_transactions_effective WHERE date >= ?"
        )
        sql += _mode_clause(data_mode, params)
        sql += " GROUP BY group_key ORDER BY total_spend DESC, label ASC"
        rows = [dict(r) for r in self._conn.execute(sql, params).fetchall()]

        total_params: list = [start]
        total_sql = "SELECT SUM(amount) FROM v_transactions_effective WHERE date >= ?"
        total_sql += _mode_clause(data_mode, total_params)
        total_row = self._conn.execute(total_sql, total_params).fetchone()
        total_spend = round_money(total_row[0] or 0.0)

        top = rows[:limit]
        items = []
        for r in top:
            category = self._dominant_category(r["group_key"], start, data_mode)
            total = round_money(r["total_spend"])
            items.append({
                "merchant": _merchant_label(r["label"]),
                "merchant_key": None if r["group_key"].startswith("RAW:") else r["group_key"],
                "total_spend": total,
                "transaction_count": int(r["transaction_count"]),
                "average_transaction": round_money(total / r["transaction_count"]),
                "category": category,
                "last_seen": r["last_seen"],
                "pct_of_total": round(total / total_spend * 100, 1) if total_spend else 0.0,
            })

        top_share = round(sum(i["total_spend"] for i in items) / total_spend * 100, 1) if total_spend else 0.0
        return {
            "items": items,
            "total_spend": total_spend,
            "distinct_merchants": len(rows),
            "top_n_share_pct": top_share,
            "months": months,
        }

    def _dominant_category(self, group_key: str, start: str, data_mode: str | None) -> str | None:
        """The effective category this merchant's spend mostly falls in.

        Deterministic: highest total spend wins, ties broken alphabetically,
        so the same data always produces the same label. A merchant whose
        rows genuinely span categories (a big-box store) gets its largest
        one, which is honest -- the number beside it is still that merchant's
        full total.
        """
        params: list = [start]
        if group_key.startswith("RAW:"):
            sql = ("SELECT effective_category, SUM(amount) AS s FROM v_transactions_effective "
                   "WHERE date >= ? AND merchant_key IS NULL AND merchant = ?")
            params.append(group_key[4:])
        else:
            sql = ("SELECT effective_category, SUM(amount) AS s FROM v_transactions_effective "
                   "WHERE date >= ? AND merchant_key = ?")
            params.append(group_key)
        sql += _mode_clause(data_mode, params)
        sql += " GROUP BY effective_category ORDER BY s DESC, effective_category ASC LIMIT 1"
        row = self._conn.execute(sql, params).fetchone()
        return row[0] if row else None

    # -- 3. Month-over-month category movers ------------------------------

    def category_movers(
        self, data_mode: str | None, reference_date: date | None = None
    ) -> dict:
        """"Why did I spend more (or less) than last month?"

        Decomposes the change in total spend into per-category contributions.
        The category deltas sum exactly to the total delta, so this reads as
        an explanation rather than a second, unrelated chart -- that additive
        property is the whole point, and there is a test asserting it holds.
        """
        today = reference_date or date.today()
        current = _month_str(today.year, today.month)
        prev_y, prev_m = _shift_month(today.year, today.month, -1)
        previous = _month_str(prev_y, prev_m)

        params: list = [_month_start(prev_y, prev_m)]
        sql = (
            "SELECT substr(date, 1, 7) AS month, effective_category AS category, "
            "SUM(amount) AS total_spend FROM v_transactions_effective WHERE date >= ?"
        )
        sql += _mode_clause(data_mode, params)
        sql += " GROUP BY month, category"
        rows = [dict(r) for r in self._conn.execute(sql, params).fetchall()]

        cur_by_cat: dict[str, float] = {}
        prev_by_cat: dict[str, float] = {}
        for r in rows:
            if r["month"] == current:
                cur_by_cat[r["category"]] = r["total_spend"]
            elif r["month"] == previous:
                prev_by_cat[r["category"]] = r["total_spend"]

        movers = []
        for category in sorted(set(cur_by_cat) | set(prev_by_cat)):
            c = round_money(cur_by_cat.get(category, 0.0))
            p = round_money(prev_by_cat.get(category, 0.0))
            movers.append({
                "category": category,
                "current": c,
                "previous": p,
                "change": round_money(c - p),
                # Undefined against a zero baseline -- there is no meaningful
                # percentage of nothing. None, never a fabricated 100%.
                "change_pct": None if p == 0 else round((c - p) / p * 100, 1),
            })
        # Largest absolute movement first: what a user wants to see is what
        # moved most, in either direction, not what is alphabetically first.
        movers.sort(key=lambda m: (-abs(m["change"]), m["category"]))

        total_current = round_money(sum(cur_by_cat.values()))
        total_previous = round_money(sum(prev_by_cat.values()))
        return {
            "current_month": current,
            "previous_month": previous,
            "total_current": total_current,
            "total_previous": total_previous,
            "total_change": round_money(total_current - total_previous),
            "movers": movers,
        }

    # -- 4. Cumulative spend pace -----------------------------------------

    def spend_pace(
        self, data_mode: str | None, reference_date: date | None = None
    ) -> dict:
        """"Am I ahead of or behind where I was this time last month?"

        Cumulative spend by day-of-month for the current month against the
        previous one. The previous month's curve runs to its own real length;
        the current month's stops at today, so the comparison is like-for-like
        up to the day the user is actually on rather than implying the month
        is already over.
        """
        today = reference_date or date.today()
        current = _month_str(today.year, today.month)
        prev_y, prev_m = _shift_month(today.year, today.month, -1)
        previous = _month_str(prev_y, prev_m)

        params: list = [_month_start(prev_y, prev_m)]
        sql = (
            "SELECT substr(date, 1, 7) AS month, CAST(substr(date, 9, 2) AS INTEGER) AS day, "
            "SUM(amount) AS total_spend FROM v_transactions_effective WHERE date >= ?"
        )
        sql += _mode_clause(data_mode, params)
        sql += " GROUP BY month, day ORDER BY month, day"
        rows = [dict(r) for r in self._conn.execute(sql, params).fetchall()]

        cur_daily = {r["day"]: r["total_spend"] for r in rows if r["month"] == current}
        prev_daily = {r["day"]: r["total_spend"] for r in rows if r["month"] == previous}

        day_of_month = today.day
        max_prev_day = max(prev_daily) if prev_daily else 0
        n_days = max(day_of_month, max_prev_day, 1)

        points = []
        cur_running = prev_running = 0.0
        for day in range(1, n_days + 1):
            cur_running += cur_daily.get(day, 0.0)
            prev_running += prev_daily.get(day, 0.0)
            points.append({
                "day": day,
                # Beyond today the current month has not happened yet. None
                # (a gap in the line), never a flat continuation that would
                # read as "spent nothing".
                "current_cumulative": round_money(cur_running) if day <= day_of_month else None,
                "previous_cumulative": round_money(prev_running) if day <= max_prev_day else None,
            })

        cur_to_date = round_money(sum(v for d, v in cur_daily.items() if d <= day_of_month))
        prev_to_date = round_money(sum(v for d, v in prev_daily.items() if d <= day_of_month))
        return {
            "current_month": current,
            "previous_month": previous,
            "day_of_month": day_of_month,
            "current_to_date": cur_to_date,
            "previous_same_point": prev_to_date,
            "difference": round_money(cur_to_date - prev_to_date),
            "points": points,
        }

    # -- 5. Forecast vs actual (only on genuine snapshots) ----------------

    def forecast_accuracy(
        self, data_mode: str | None, reference_date: date | None = None
    ) -> dict:
        """"Were my past forecasts any good?"

        Answered ONLY from genuine historical snapshots. A forecast row is
        eligible when both hold:

          * the run that produced it was generated STRICTLY BEFORE the first
            day of the month it predicted -- so it was a real prediction at
            the time, not hindsight, and
          * that month has since completed, so an actual exists to compare
            against.

        Old predictions are never recomputed against present-day data and
        presented as historical. When no eligible snapshot exists yet, this
        returns available=False with a plain reason, and the UI says so
        instead of drawing an empty chart. On a fresh install that is the
        normal state, and it stays true until the user has generated a
        forecast and then let a month pass.
        """
        today = reference_date or date.today()
        first_of_this_month = _month_start(today.year, today.month)

        if data_mode is None:
            return _no_accuracy("no_data")

        rows = self._conn.execute(
            """
            SELECT p.category      AS category,
                   p.forecast_month AS forecast_month,
                   p.predicted_amount AS predicted_amount,
                   r.generated_at  AS generated_at,
                   r.id            AS run_id
            FROM forecast_predictions p
            JOIN forecast_runs r ON r.id = p.forecast_run_id
            WHERE r.data_mode = ?
              AND p.is_available = 1
              AND p.predicted_amount IS NOT NULL
              AND p.forecast_month < ?
              AND date(r.generated_at) < (p.forecast_month || '-01')
            ORDER BY p.forecast_month, p.category, r.generated_at DESC, r.id DESC
            """,
            (data_mode, today.strftime("%Y-%m")),
        ).fetchall()
        # `p.forecast_month < 'YYYY-MM'` keeps only months strictly before the
        # current one, i.e. already complete. first_of_this_month is not used
        # in the SQL for that reason; it stays documented here so the
        # completeness rule is explicit rather than implied by a substring
        # comparison.
        assert first_of_this_month  # documents intent; the filter is above

        if not rows:
            return _no_accuracy("no_snapshots_yet")

        # Most recent eligible run per (forecast_month, category): a user may
        # have regenerated the forecast several times before the month began,
        # and the last one made before it started is the prediction that
        # actually stood.
        latest: dict[tuple[str, str], dict] = {}
        for r in rows:
            key = (r["forecast_month"], r["category"])
            latest.setdefault(key, dict(r))

        params: list = [data_mode]
        actual_rows = self._conn.execute(
            "SELECT substr(date, 1, 7) AS month, effective_category AS category, "
            "SUM(amount) AS actual FROM v_transactions_effective "
            "WHERE data_mode = ? GROUP BY month, category",
            params,
        ).fetchall()
        actuals = {(r["month"], r["category"]): r["actual"] for r in actual_rows}

        items = []
        for (month, category), pred in sorted(latest.items()):
            actual = actuals.get((month, category))
            if actual is None:
                # The category recorded no spend that month. That is a real
                # zero, and comparing a forecast against it is meaningful.
                actual = 0.0
            predicted = round_money(pred["predicted_amount"])
            actual = round_money(actual)
            items.append({
                "forecast_month": month,
                "category": category,
                "predicted": predicted,
                "actual": actual,
                "error": round_money(predicted - actual),
                "generated_at": pred["generated_at"],
            })

        total_predicted = round_money(sum(i["predicted"] for i in items))
        total_actual = round_money(sum(i["actual"] for i in items))
        abs_error = sum(abs(i["error"]) for i in items)
        return {
            "available": True,
            "reason": None,
            "items": items,
            "months_covered": sorted({i["forecast_month"] for i in items}),
            "total_predicted": total_predicted,
            "total_actual": total_actual,
            # WAPE: total absolute error over total actual. Chosen over MAPE
            # because a category with a near-zero actual makes MAPE explode
            # to a meaningless number, which is exactly the failure the ML-C
            # forecasting metrics already documented.
            "wape": round(abs_error / total_actual, 4) if total_actual else None,
        }


def _no_accuracy(reason: str) -> dict:
    return {
        "available": False,
        "reason": reason,
        "items": [],
        "months_covered": [],
        "total_predicted": 0.0,
        "total_actual": 0.0,
        "wape": None,
    }


def _merchant_label(raw: str) -> str:
    """A readable label from the merchant text actually stored.

    Only ever a substring of real stored text with trailing reference noise
    trimmed -- never an invented or prettified merchant name. Title-casing is
    display formatting, applied by the frontend, not here.
    """
    from ml.categorization.text_normalize_v2 import normalize_deployment_text_v2

    cleaned = normalize_deployment_text_v2(raw or "")
    return cleaned or (raw or "").strip() or "Unknown"
