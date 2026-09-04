"""
Import API six-bank tests (Phase 12A.5/12B): auto-detect and explicit-bank
upload via HTTP for RBC/Scotiabank/CIBC, the BMO/National BLOCKED 400, the
upload size limit, and GET history exposing the new exclusion counts.
"""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.api.routes.imports import MAX_IMPORT_FILE_BYTES

FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "fixtures"


def _upload(client: TestClient, subdir: str, filename: str, bank: str | None = None):
    content = (FIXTURES_DIR / subdir / filename).read_bytes()
    data = {} if bank is None else {"bank": bank}
    return client.post(
        "/api/imports",
        files={"file": (filename, content, "text/csv")},
        data=data,
    )


@pytest.mark.parametrize(
    "subdir,bank,expected_valid",
    [("rbc_csv", "RBC", 3), ("scotia_csv", "Scotiabank", 4), ("cibc_csv", "CIBC", 2)],
)
def test_upload_explicit_bank(client: TestClient, subdir, bank, expected_valid):
    response = _upload(client, subdir, "clean_valid.csv", bank=bank)
    assert response.status_code == 200
    body = response.json()
    assert body["detected_bank"] == bank
    assert body["rows_valid"] == expected_valid


@pytest.mark.parametrize(
    "subdir,bank,expected_valid",
    [("rbc_csv", "RBC", 3), ("scotia_csv", "Scotiabank", 4), ("cibc_csv", "CIBC", 2)],
)
def test_upload_auto_detect(client: TestClient, subdir, bank, expected_valid):
    response = _upload(client, subdir, "clean_valid.csv", bank=None)
    assert response.status_code == 200
    body = response.json()
    assert body["detected_bank"] == bank
    assert body["rows_valid"] == expected_valid


def test_upload_bmo_returns_400_not_yet_supported(client: TestClient):
    response = _upload(client, "shared_csv", "blocked_balance_format.csv", bank="BMO")
    assert response.status_code == 400
    assert "not yet supported" in response.json()["message"]


def test_upload_national_bank_returns_400_not_yet_supported(client: TestClient):
    response = _upload(client, "shared_csv", "blocked_balance_format.csv", bank="National Bank")
    assert response.status_code == 400
    assert "not yet supported" in response.json()["message"]


def test_upload_blocked_shaped_file_never_becomes_td_in_auto_mode(client: TestClient):
    response = _upload(client, "shared_csv", "blocked_balance_format.csv", bank=None)
    assert response.status_code == 400
    body = response.json()
    assert body["error"] == "bad_request"


def test_upload_blocked_shaped_file_rejected_under_explicit_td_too(client: TestClient):
    # Phase 12B closure patch (Cursor finding): this specific request was
    # wrongly accepted as a 200 TD preview before this patch.
    response = _upload(client, "shared_csv", "blocked_balance_format.csv", bank="TD")
    assert response.status_code == 400
    assert response.json()["error"] == "bad_request"


def test_upload_explicit_bank_mismatch_returns_400(client: TestClient):
    # Explicitly ask for RBC against a Scotiabank-shaped file -- must never
    # be silently reinterpreted as Scotiabank.
    response = _upload(client, "scotia_csv", "clean_valid.csv", bank="RBC")
    assert response.status_code == 400
    assert "RBC" in response.json()["message"]


def test_upload_oversized_file_rejected(client: TestClient):
    oversized = b"Date,Description,Amount\n" + (b"08/01/2026,PADDING ROW,1.00\n" * 1)
    oversized += b"X" * (MAX_IMPORT_FILE_BYTES + 1)  # pad past the limit inside a comment-like tail
    response = client.post(
        "/api/imports",
        files={"file": ("huge.csv", oversized, "text/csv")},
        data={"bank": "TD"},
    )
    assert response.status_code == 400
    assert "too large" in response.json()["message"].lower()


def test_get_import_history_exposes_exclusion_counts(client: TestClient):
    preview = _upload(client, "rbc_csv", "clean_valid.csv", bank="RBC").json()
    detail = client.get(f"/api/imports/{preview['batch_id']}").json()
    assert detail["bank_source"] == "RBC"
    assert detail["rows_skipped_credit"] == 1
    assert detail["rows_skipped_currency"] == 1


def test_confirm_result_includes_exclusion_counts(client: TestClient):
    preview = _upload(client, "cibc_csv", "clean_valid.csv", bank="CIBC").json()
    result = client.post(f"/api/imports/{preview['batch_id']}/confirm").json()
    assert result["rows_skipped_credit"] == 1
    assert result["rows_skipped_currency"] == 0
