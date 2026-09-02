"""
Monetary value policy (Build Plan §4, item 7): all amounts are rounded to 2
decimal places at the point of insertion/computation, matching V1's existing
convention (round(x, 2) throughout pipeline/forecast.py and
pipeline/portfolio.py). Defined once here so every repository follows it
consistently rather than each re-deciding rounding behavior.
"""


def round_money(value: float | int | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 2)
