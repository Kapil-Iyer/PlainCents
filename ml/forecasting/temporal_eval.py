"""
ML Spec Section 12: explicit calendar-month-boundary expanding-window
temporal validation. Section 12's own worked example is implemented
literally: TRAIN grows one calendar month at a time; a month's rows (all
categories) move from VALIDATE to TRAIN as a whole unit, never split.

Section 12.1: the most recent `n_final_reserved_months` calendar months are
reserved as the FINAL UNTOUCHED TEST temporal period and structurally
excluded from every fold this module generates -- they never appear as a
TRAIN month or a target month in any VALIDATION fold. `reserved_final_period`
reports only which calendar months were reserved (structural fact), never
any value/metric computed from them.
"""
from __future__ import annotations

from dataclasses import dataclass

HORIZONS = (1, 2, 3)


@dataclass
class Fold:
    origin_index: int          # index into `dev_months` of the last TRAIN month
    train_months: list[str]    # dev_months[0 : origin_index+1]
    target_months: dict[int, str]  # {horizon: month} for horizons that fit inside dev_months


def build_folds(
    all_months: list[str],
    min_train_months: int = 7,
    horizons: tuple[int, ...] = HORIZONS,
    n_final_reserved_months: int = 3,
) -> tuple[list[Fold], dict]:
    """
    Parameters
    ----------
    all_months : chronologically sorted list of every calendar month present
        in the dataset (e.g. ["2023-01", ..., "2024-12"]).
    min_train_months : a fold's TRAIN window must contain at least this many
        months (matches pipeline/forecast.py's walk_forward_validate's own
        `len(train_months) < 7: continue` threshold -- reused, not
        re-invented, per Section 12's instruction to reuse
        walk_forward_validate's proven-correct loop style).
    horizons : which +N horizons to produce targets for.
    n_final_reserved_months : how many of the MOST RECENT calendar months are
        reserved as the untouched final temporal period (Section 12.1).

    Returns
    -------
    (folds, reserved_final_period)
      folds : list[Fold], each entirely within the non-reserved "development"
        region (all_months[: len(all_months) - n_final_reserved_months]).
      reserved_final_period : {"months": [...], "n_months": int} -- structural
        only, safe to log/report without violating the FINAL TEST seal.
    """
    if n_final_reserved_months >= len(all_months):
        raise ValueError("n_final_reserved_months must leave at least one development month")

    dev_months = all_months[: len(all_months) - n_final_reserved_months]
    reserved_final_period = {
        "months": all_months[len(all_months) - n_final_reserved_months:],
        "n_months": n_final_reserved_months,
    }

    max_horizon = max(horizons)
    folds: list[Fold] = []
    for origin_index in range(min_train_months - 1, len(dev_months)):
        targets = {h: dev_months[origin_index + h] for h in horizons if origin_index + h < len(dev_months)}
        if not targets:
            continue
        folds.append(Fold(
            origin_index=origin_index,
            train_months=dev_months[: origin_index + 1],
            target_months=targets,
        ))

    return folds, reserved_final_period


def assert_no_reserved_month_used(folds: list[Fold], reserved_final_period: dict) -> None:
    """Defense in depth, also exercised directly by tests/ml: no fold may
    reference a reserved-final-period month anywhere (train or target)."""
    reserved = set(reserved_final_period["months"])
    for fold in folds:
        assert not (set(fold.train_months) & reserved), f"fold {fold.origin_index} trains on a reserved final month"
        assert not (set(fold.target_months.values()) & reserved), f"fold {fold.origin_index} targets a reserved final month"
