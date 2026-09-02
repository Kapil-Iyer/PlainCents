"""
Fixtures for FastAPI API tests (Build Plan Phase 2, extended Phase 3).

The app's DB and CategorizationService dependencies are overridden to use
the isolated, temporary `conn` fixture and the checked-in test model artifact
(tests/fixtures/logreg_model_test.pkl) — tests never touch a developer's
real plaincents_v2.db or the production models/tfidf_logreg_v1.pkl, and never
run the lifespan hook's own get_connection()/CategorizationService(...)
against those default paths.
"""
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.api.deps import get_categorization_service, get_db
from backend.main import app
from backend.services.categorization_service import CategorizationService

TEST_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "fixtures" / "logreg_model_test.pkl"
MISSING_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "fixtures" / "does_not_exist.pkl"


@pytest.fixture
def categorization_service() -> CategorizationService:
    """A CategorizationService loaded from the checked-in test artifact."""
    return CategorizationService(TEST_MODEL_PATH)


@pytest.fixture
def client(conn: sqlite3.Connection, categorization_service: CategorizationService) -> TestClient:
    def _override_get_db():
        yield conn

    app.dependency_overrides[get_db] = _override_get_db
    app.dependency_overrides[get_categorization_service] = lambda: categorization_service
    # Not used as a context manager: that would run the lifespan hook, which
    # opens a connection to the real default plaincents_v2.db and loads the
    # production model. Routes only depend on get_db/get_categorization_service
    # (both overridden above), so skipping lifespan is safe here.
    test_client = TestClient(app)
    yield test_client
    app.dependency_overrides.pop(get_db, None)
    app.dependency_overrides.pop(get_categorization_service, None)
