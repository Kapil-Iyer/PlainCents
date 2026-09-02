"""
PlainCents V2 FastAPI app (Build Plan Phase 2, extended in Phase 3).

Run with: uvicorn backend.main:app --reload

The lifespan hook opens the single shared DB connection (migrations applied
once at startup) and loads the CategorizationService's model artifact, both
closed/released at shutdown. This hook's structure was kept stable from
Phase 2 specifically so Phase 3 only adds to it, per Phase 2's design note.
"""
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.error_handlers import register_error_handlers
from backend.api.routes import dashboard, demo, forecasts, health, holdings, imports, transactions
from backend.config import FRONTEND_ORIGIN, KMEANS_MODEL_PATH, V2_DB_PATH
from backend.db.connection import get_connection
from backend.services.categorization_service import CategorizationService


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Startup: open the shared connection, applying any pending migrations.
    app.state.db_connection = get_connection(db_path=V2_DB_PATH)

    # CategorizationService loads its model artifact once here (TRD §11.1) —
    # never per-request. A missing/corrupt artifact does not crash startup;
    # the service just reports "missing"/"error" via /api/health, and
    # prediction-dependent writes get a 503 until a valid model is present.
    app.state.categorization_service = CategorizationService(KMEANS_MODEL_PATH)

    yield

    # Shutdown
    app.state.db_connection.close()


app = FastAPI(title="PlainCents V2 API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_error_handlers(app)

app.include_router(health.router)
app.include_router(demo.router)
app.include_router(transactions.router)
app.include_router(imports.router)
app.include_router(dashboard.router)
app.include_router(forecasts.router)
app.include_router(holdings.router)
