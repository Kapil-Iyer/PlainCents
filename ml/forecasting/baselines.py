"""
ML Spec Section 11, candidates 1-2: Naive and Seasonal Naive baselines.

Neither baseline has a meaningful "multi-step strategy" distinction
(Section 11.1) the way RF/Ridge do:
  - Naive always predicts the single last-observed value, for every
    horizon -- there is nothing to "feed back" recursively, since a
    recursive re-application of "repeat the last observation" produces the
    exact same value at every horizon as the non-recursive version.
  - Seasonal Naive always looks up a fixed calendar-relative real
    observation (same month, prior year) directly from history, never from
    another baseline's own prediction -- again nothing for a recursive
    variant to change.
This is documented explicitly (not a shortcut) in reports/ml -- Section
11.1's strategy comparison is scoped to the two candidates whose feature
construction actually depends on a strategy choice (RF, Ridge).
"""
from __future__ import annotations

import numpy as np


def naive_predict(spend_history: np.ndarray) -> float:
    """Section 11 candidate 1: next month's spend = lag_1 (latest observed)."""
    return float(spend_history[-1])


def rolling_mean_predict(spend_history: np.ndarray, window: int) -> float:
    """ML-F forecast re-evaluation Candidate C: predicted spend for every
    future month = the mean of the most recent `window` observed months
    (same value reused at every horizon, exactly like Naive -- there is
    nothing for a recursive variant to change here either, since the
    predicted value never depends on a prior *prediction*). Uses however
    much history is actually available if shorter than `window` (never
    raises), matching Naive's own "use what you have" floor rather than
    imposing a harder eligibility gate than the product's."""
    if len(spend_history) == 0:
        raise ValueError("rolling_mean_predict requires at least one observed month")
    return float(np.mean(spend_history[-window:]))


def ewma_predict(spend_history: np.ndarray, alpha: float) -> float:
    """ML-F forecast re-evaluation Candidate D: exponentially weighted
    moving average, predicted spend = the EWMA of the full observed history
    with smoothing factor `alpha` (weight on the most recent observation).
    Same value reused at every horizon (no recursive variant needed, for the
    same reason as rolling_mean_predict). Standard recursive EWMA
    definition: s_1 = x_1; s_t = alpha * x_t + (1 - alpha) * s_(t-1)."""
    if len(spend_history) == 0:
        raise ValueError("ewma_predict requires at least one observed month")
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1]")
    s = float(spend_history[0])
    for x in spend_history[1:]:
        s = alpha * float(x) + (1 - alpha) * s
    return s


def seasonal_naive_predict(spend_history: np.ndarray, horizon: int) -> tuple[float | None, bool]:
    """
    Section 11 candidate 2: next month's spend = same calendar month, prior
    year -- i.e. the observation exactly 12 months before the target month.

    `spend_history` is chronological, ending at the fold's last TRAIN month
    (length L). The target month is `horizon` months after the last TRAIN
    month, so the prior-year observation sits at history index
    `L - 13 + horizon` (derivation: index L-1 = last train month = "origin";
    index L-1-k = "origin minus k months"; we want the month at
    origin + horizon - 12, i.e. k = 12 - horizon, giving index
    L-1-(12-horizon) = L-13+horizon).

    Eligibility (Section 11: "where >=13 months of history exist") requires
    that index to be >= 0. Returns (value_or_None, eligible) -- when
    ineligible, value is None (never fabricated), and the caller must record
    this as "not eligible for seasonal naive at this fold/horizon," not as a
    zero or a silently-skipped row.
    """
    L = len(spend_history)
    idx = L - 13 + horizon
    if idx < 0:
        return None, False
    return float(spend_history[idx]), True
