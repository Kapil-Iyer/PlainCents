"""CategorizationService unit tests (Build Plan Phase 3, item 8): loaded,
missing, and error paths, using the checked-in test model artifact."""
from pathlib import Path

import pytest

from backend.api.errors import CategorizationUnavailableError
from backend.services.categorization_service import CategorizationService
from config import CATEGORIES

FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "fixtures"
TEST_MODEL_PATH = FIXTURES_DIR / "kmeans_model_test.pkl"


def test_loads_successfully_and_reports_loaded_status():
    service = CategorizationService(TEST_MODEL_PATH)
    assert service.status == "loaded"


def test_missing_model_reports_missing_status(tmp_path):
    service = CategorizationService(tmp_path / "does_not_exist.pkl")
    assert service.status == "missing"


def test_corrupt_model_reports_error_status(tmp_path):
    corrupt = tmp_path / "corrupt.pkl"
    corrupt.write_bytes(b"not a real pickle")
    service = CategorizationService(corrupt)
    assert service.status == "error"


def test_predict_returns_a_known_category():
    service = CategorizationService(TEST_MODEL_PATH)
    result = service.predict({"merchant": "TIM HORTONS", "amount": 4.50, "date": "2026-01-15"})
    assert set(result.keys()) == {"predicted_category"}
    assert result["predicted_category"] in CATEGORIES


def test_predict_batch_returns_one_result_per_row_in_order():
    service = CategorizationService(TEST_MODEL_PATH)
    rows = [
        {"merchant": "TIM HORTONS", "amount": 4.50, "date": "2026-01-15"},
        {"merchant": "NETFLIX", "amount": 16.99, "date": "2026-01-16"},
        {"merchant": "UBER", "amount": 22.10, "date": "2026-01-17"},
    ]
    results = service.predict_batch(rows)
    assert len(results) == len(rows)
    for result in results:
        assert result["predicted_category"] in CATEGORIES


def test_predict_batch_empty_list_returns_empty_list():
    service = CategorizationService(TEST_MODEL_PATH)
    assert service.predict_batch([]) == []


def test_predict_raises_categorization_unavailable_when_missing(tmp_path):
    service = CategorizationService(tmp_path / "does_not_exist.pkl")
    with pytest.raises(CategorizationUnavailableError):
        service.predict({"merchant": "TIM HORTONS", "amount": 4.50, "date": "2026-01-15"})


def test_predict_batch_raises_categorization_unavailable_when_error(tmp_path):
    corrupt = tmp_path / "corrupt.pkl"
    corrupt.write_bytes(b"not a real pickle")
    service = CategorizationService(corrupt)
    with pytest.raises(CategorizationUnavailableError):
        service.predict_batch([{"merchant": "TIM HORTONS", "amount": 4.50, "date": "2026-01-15"}])
