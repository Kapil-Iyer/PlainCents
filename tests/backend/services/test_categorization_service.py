"""CategorizationService unit tests (Build Plan Phase 3, item 8; ML-D
Production Integration): loaded, missing, and error paths, plus ML-D-specific
checks that the selected production family/version is identifiable and that
inference never fits anything — using the checked-in test model artifact
(TF-IDF + Logistic Regression, ML-C selected recipe)."""
from pathlib import Path
from unittest.mock import patch

import pytest

from backend.api.errors import CategorizationUnavailableError
from backend.services.categorization_service import CategorizationService
from config import CATEGORIES

FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "fixtures"
TEST_MODEL_PATH = FIXTURES_DIR / "logreg_model_test.pkl"


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


# -- ML-D: selected production recipe identification -------------------------


def test_loaded_service_identifies_selected_family_and_version():
    service = CategorizationService(TEST_MODEL_PATH)
    assert service.model_impl_version == "tfidf_logreg_v1"
    assert service.metadata is not None
    assert service.metadata["family"] == "TF-IDF + Logistic Regression"
    assert service.metadata["candidate_name"] == "tfidf_logreg"


def test_missing_model_has_no_version_or_metadata(tmp_path):
    service = CategorizationService(tmp_path / "does_not_exist.pkl")
    assert service.model_impl_version is None
    assert service.metadata is None


# -- ML-D: inference never fits anything --------------------------------------


def test_predict_never_fits_the_underlying_model():
    service = CategorizationService(TEST_MODEL_PATH)
    with patch.object(service._model, "fit") as mock_fit, \
         patch.object(service._vectorizer, "fit") as mock_vectorizer_fit, \
         patch.object(service._vectorizer, "fit_transform") as mock_fit_transform:
        service.predict({"merchant": "TIM HORTONS", "amount": 4.50, "date": "2026-01-15"})
    mock_fit.assert_not_called()
    mock_vectorizer_fit.assert_not_called()
    mock_fit_transform.assert_not_called()


def test_predict_batch_never_fits_the_underlying_model():
    service = CategorizationService(TEST_MODEL_PATH)
    with patch.object(service._model, "fit") as mock_fit, \
         patch.object(service._vectorizer, "fit") as mock_vectorizer_fit, \
         patch.object(service._vectorizer, "fit_transform") as mock_fit_transform:
        service.predict_batch(
            [
                {"merchant": "TIM HORTONS", "amount": 4.50, "date": "2026-01-15"},
                {"merchant": "NETFLIX", "amount": 16.99, "date": "2026-01-16"},
            ]
        )
    mock_fit.assert_not_called()
    mock_vectorizer_fit.assert_not_called()
    mock_fit_transform.assert_not_called()
