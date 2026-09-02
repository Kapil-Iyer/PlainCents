"""
Build Plan Phase 0 smoke tests.

Verifies the test-fixture bootstrap (deterministic K-Means test artifact)
and the TD CSV fixtures are valid, before any backend code exists to consume
them (Phase 3+).
"""
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = ROOT / "tests" / "fixtures"
TD_CSV_DIR = FIXTURES_DIR / "td_csv"
TEST_ARTIFACT_PATH = FIXTURES_DIR / "kmeans_model_test.pkl"


@pytest.fixture(scope="module", autouse=True)
def build_test_artifact():
    """Run the bootstrap script once for this test module."""
    result = subprocess.run(
        [sys.executable, str(FIXTURES_DIR / "build_test_kmeans_model.py")],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert result.returncode == 0, f"bootstrap script failed:\n{result.stdout}\n{result.stderr}"
    yield


def test_test_artifact_is_loadable_with_expected_keys():
    assert TEST_ARTIFACT_PATH.exists(), "build_test_kmeans_model.py did not produce the expected file"
    payload = joblib.load(TEST_ARTIFACT_PATH)
    assert set(payload.keys()) == {"kmeans", "scaler", "vectorizer", "cluster_to_category"}
    assert len(payload["cluster_to_category"]) == 12


@pytest.mark.parametrize("filename", [
    "clean_valid.csv",
    "unparseable_dates.csv",
    "unrecognized_format.csv",
    "duplicate_rows.csv",
])
def test_td_fixture_is_valid_utf8_csv(filename):
    path = TD_CSV_DIR / filename
    assert path.exists(), f"missing TD fixture: {filename}"
    # Must be plain UTF-8 (not UTF-16, unlike the pre-Phase-0 requirements.txt bug)
    text = path.read_text(encoding="utf-8")
    assert text.strip(), f"{filename} is empty"
    df = pd.read_csv(path)
    assert len(df) > 0
    assert len(df.columns) == 3
