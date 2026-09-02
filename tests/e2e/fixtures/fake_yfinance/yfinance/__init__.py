"""
Deterministic offline stand-in for the real `yfinance` package, used ONLY by
the Playwright E2E Portfolio flow (Build Plan Phase 10 / TRD §17.3's "assert
via a mocked/blocked yfinance client").

Why this seam and not something else:
- Production Portfolio semantics are untouched: `pipeline/portfolio.py` and
  `backend/services/portfolio_service.py` are not modified at all, and
  `GET /api/holdings` still never imports or calls anything network-related
  (confirmed: it only reads `HoldingRepository`/`PriceCacheRepository`).
- The only thing that ever imports the real `yfinance` package is
  `pipeline/portfolio.py:fetch_price()`, called exclusively from the explicit
  `POST /api/holdings/refresh-prices` route. That import is the narrowest
  point where "the real yfinance client" can be swapped for a deterministic
  double without touching any product code path.
- This package is made importable in place of the real `yfinance` purely by
  prepending this directory to PYTHONPATH when the E2E backend process is
  launched (see tests/e2e/global-setup.ts) — `import yfinance` then resolves
  here instead of site-packages. No monkeypatching of application code, no
  test-only branches in production files, no env-var-gated behavior in
  `pipeline/portfolio.py` itself.
- Manual verification of the real yfinance path remains possible any time by
  simply not setting that PYTHONPATH override (i.e. running normally).

Prices are deterministic per ticker (stable across runs) so an E2E assertion
on a specific rendered price/timestamp never flakes.
"""
from __future__ import annotations


def _deterministic_price(ticker: str) -> float:
    # Stable, ticker-derived price — not random, not time-based.
    return round(100.0 + (sum(ord(c) for c in ticker) % 50), 2)


class _FastInfo(dict):
    """Mimics yfinance's fast_info: supports both dict-style ["lastPrice"]
    and attribute-style .last_price access, matching pipeline/portfolio.py's
    fallback handling."""

    def __init__(self, price: float):
        super().__init__(lastPrice=price)
        self.last_price = price


class Ticker:
    def __init__(self, ticker: str):
        self.ticker = ticker

    @property
    def fast_info(self) -> _FastInfo:
        return _FastInfo(_deterministic_price(self.ticker))
