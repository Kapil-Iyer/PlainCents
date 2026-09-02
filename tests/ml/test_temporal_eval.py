"""ML Spec Section 12/12.1 calendar-boundary chronology tests."""
import pytest

from ml.forecasting.temporal_eval import assert_no_reserved_month_used, build_folds


def _months(n=24, start_year=2023):
    months = []
    y, m = start_year, 1
    for _ in range(n):
        months.append(f"{y}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def test_reserved_final_months_never_appear_in_any_fold():
    all_months = _months(24)
    folds, reserved = build_folds(all_months, min_train_months=7, n_final_reserved_months=3)
    assert reserved["months"] == all_months[-3:]
    assert_no_reserved_month_used(folds, reserved)  # raises if violated


def test_folds_respect_min_train_months():
    all_months = _months(24)
    folds, _ = build_folds(all_months, min_train_months=7, n_final_reserved_months=3)
    for fold in folds:
        assert len(fold.train_months) >= 7


def test_train_months_are_strictly_before_every_target_month():
    all_months = _months(24)
    folds, _ = build_folds(all_months, min_train_months=7, n_final_reserved_months=3)
    for fold in folds:
        last_train = fold.train_months[-1]
        for h, target in fold.target_months.items():
            assert target > last_train, f"target {target} not after last train month {last_train}"


def test_no_month_split_between_train_and_target_within_a_fold():
    """A month is either fully in TRAIN or fully a target -- never both, and
    the whole-month-as-a-unit invariant (Section 12's core requirement)."""
    all_months = _months(24)
    folds, _ = build_folds(all_months, min_train_months=7, n_final_reserved_months=3)
    for fold in folds:
        train_set = set(fold.train_months)
        target_set = set(fold.target_months.values())
        assert train_set.isdisjoint(target_set)


def test_expanding_not_sliding_window():
    """Each successive fold's TRAIN window is a strict superset of the
    previous fold's (expanding), never a same-size sliding window."""
    all_months = _months(24)
    folds, _ = build_folds(all_months, min_train_months=7, n_final_reserved_months=3)
    for prev, cur in zip(folds, folds[1:]):
        assert set(prev.train_months).issubset(set(cur.train_months))
        assert len(cur.train_months) > len(prev.train_months)


def test_horizon_3_target_never_exceeds_development_region():
    all_months = _months(24)
    folds, reserved = build_folds(all_months, min_train_months=7, n_final_reserved_months=3)
    dev_months = set(all_months) - set(reserved["months"])
    for fold in folds:
        for target in fold.target_months.values():
            assert target in dev_months


def test_raises_if_reserved_months_exceed_available_history():
    with pytest.raises(ValueError):
        build_folds(["2024-01"], n_final_reserved_months=5)


def test_at_least_one_fold_produced_for_24_months_of_history():
    all_months = _months(24)
    folds, _ = build_folds(all_months, min_train_months=7, n_final_reserved_months=3)
    assert len(folds) > 0
    # 24 months, 3 reserved -> 21 dev months (indices 0..20), min_train=7 ->
    # origins 6..19 produce at least one in-range horizon (origin 20 produces
    # none, since even +1 would land outside the dev region, and is
    # correctly skipped) => 14 folds total, with the last few folds carrying
    # fewer than 3 horizons (partial folds near the development-region tail,
    # not discarded -- Section 12 doesn't require throwing away a fold just
    # because +3 no longer fits).
    assert len(folds) == 14
    assert len(folds[-1].target_months) == 1  # origin 19: only +1 (target idx 20) fits
    assert len(folds[0].target_months) == 3   # origin 6: all three horizons fit
