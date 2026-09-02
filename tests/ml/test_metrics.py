"""ML Spec Section 7/13 metric-calculation correctness tests, especially WAPE."""
import numpy as np

from ml.common.metrics import (
    categorization_metric_bundle,
    mae,
    mape_safe,
    rmse,
    wape,
)


def test_wape_zero_error_is_zero():
    actual = np.array([100.0, 200.0, 50.0])
    assert wape(actual, actual) == 0.0


def test_wape_matches_hand_calculation():
    actual = np.array([100.0, 0.0, 50.0])
    predicted = np.array([80.0, 10.0, 60.0])
    # sum|actual-predicted| = 20 + 10 + 10 = 40; sum|actual| = 150
    assert abs(wape(actual, predicted) - 40 / 150) < 1e-9


def test_wape_undefined_when_all_actuals_zero():
    actual = np.array([0.0, 0.0])
    predicted = np.array([5.0, 3.0])
    assert np.isnan(wape(actual, predicted))


def test_wape_small_actual_does_not_dominate_like_mape_would():
    # A near-zero actual with a modest absolute miss should barely move WAPE,
    # unlike MAPE where it would look enormous.
    actual = np.array([1000.0, 0.01])
    predicted = np.array([1000.0, 5.0])
    w = wape(actual, predicted)
    assert w < 0.01  # (0 + 4.99) / 1000.01 ~= 0.00499


def test_mape_safe_flags_near_zero_actuals_separately():
    actual = np.array([0.0, 0.5, 100.0])
    predicted = np.array([5.0, 0.6, 105.0])
    result = mape_safe(actual, predicted)
    assert result["n_near_zero_actual"] == 2  # 0.0 and 0.5 are both <= 1.00
    assert result["n_rows"] == 3
    # mape_all uses the epsilon guard and will be huge for the first row
    assert result["mape_all"] > 1000
    # mape_nonzero_only excludes both near-zero rows, leaving only row 3
    assert abs(result["mape_nonzero_only"] - 5.0) < 1e-6


def test_mae_and_rmse_basic():
    actual = np.array([10.0, 20.0])
    predicted = np.array([12.0, 16.0])
    assert abs(mae(actual, predicted) - 3.0) < 1e-9
    assert abs(rmse(actual, predicted) - np.sqrt((4 + 16) / 2)) < 1e-9


def test_categorization_bundle_perfect_predictions():
    labels = ["A", "B", "C"]
    y_true = ["A", "B", "C", "A", "B"]
    y_pred = ["A", "B", "C", "A", "B"]
    bundle = categorization_metric_bundle(y_true, y_pred, labels)
    assert bundle["accuracy"] == 1.0
    assert bundle["macro_f1"] == 1.0
    for cat in labels:
        assert bundle["per_category"][cat]["f1"] == 1.0 or bundle["per_category"][cat]["support"] == 0


def test_categorization_bundle_confusion_matrix_shape():
    labels = ["A", "B"]
    y_true = ["A", "A", "B", "B"]
    y_pred = ["A", "B", "B", "B"]
    bundle = categorization_metric_bundle(y_true, y_pred, labels)
    assert bundle["confusion_matrix"]["A"]["A"] == 1
    assert bundle["confusion_matrix"]["A"]["B"] == 1
    assert bundle["confusion_matrix"]["B"]["B"] == 2
    assert bundle["accuracy"] == 0.75
