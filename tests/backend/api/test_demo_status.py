"""GET /api/demo/status, POST /api/demo/load, DELETE /api/demo/clear
(Build Plan Phase 2, TRD §5.2, §2.5)."""
import sqlite3

from backend.repositories.app_state_repository import AppStateRepository


def test_demo_status_returns_empty_on_fresh_db(client):
    response = client.get("/api/demo/status")

    assert response.status_code == 200
    assert response.json() == {"mode": "EMPTY", "can_load_demo": True}


def test_demo_status_reflects_real_mode(client, conn: sqlite3.Connection):
    AppStateRepository(conn).set_mode("REAL")
    conn.commit()

    response = client.get("/api/demo/status")

    assert response.status_code == 200
    assert response.json() == {"mode": "REAL", "can_load_demo": False}


def test_demo_load_not_implemented(client):
    response = client.post("/api/demo/load")

    assert response.status_code == 501
    body = response.json()
    assert body["error"] == "not_implemented"


def test_demo_clear_not_implemented(client):
    response = client.delete("/api/demo/clear")

    assert response.status_code == 501
    body = response.json()
    assert body["error"] == "not_implemented"
