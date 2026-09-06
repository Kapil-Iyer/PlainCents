"""Power BI export route tests (PATCH D)."""
import io
import zipfile

from fastapi.testclient import TestClient


def test_export_returns_a_downloadable_zip_on_empty_database(client: TestClient):
    response = client.get("/api/export/powerbi")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert 'attachment; filename="plaincents_export_' in response.headers["content-disposition"]
    assert response.headers["content-disposition"].endswith('.zip"')

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        assert set(zf.namelist()) == {
            "transactions.csv", "category_summary.csv", "portfolio.csv", "forecast.csv",
        }


def test_export_reflects_real_transactions_via_effective_category(client: TestClient):
    created = client.post(
        "/api/transactions",
        json={"date": "2026-06-05", "merchant": "TIM HORTONS", "amount": 12.5},
    ).json()
    client.patch(f"/api/transactions/{created['id']}", json={"confirmed_category": "Healthcare"})

    response = client.get("/api/export/powerbi")

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        transactions_csv = zf.read("transactions.csv").decode("utf-8")

    assert "TIM HORTONS" in transactions_csv
    assert "Healthcare" in transactions_csv
