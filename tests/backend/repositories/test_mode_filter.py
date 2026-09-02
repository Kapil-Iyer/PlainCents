"""Phase 1 test: EMPTY/DEMO/REAL read-mapping pure function. Requirement #13."""
import pytest

from backend.repositories.mode_filter import resolve_data_mode_filter


def test_empty_maps_to_none():
    assert resolve_data_mode_filter("EMPTY") is None


def test_demo_maps_to_demo():
    assert resolve_data_mode_filter("DEMO") == "demo"


def test_real_maps_to_real():
    assert resolve_data_mode_filter("REAL") == "real"


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        resolve_data_mode_filter("SOMETHING_ELSE")
