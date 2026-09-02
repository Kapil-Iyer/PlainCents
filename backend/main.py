"""
PlainCents V2 FastAPI app (Build Plan Phase 2, extended in Phase 3, packaged
reviewer mode added in Phase 10).

Dev mode (two servers, HMR):  uvicorn backend.main:app --reload
Reviewer/demo mode (one process, one port): build the frontend first
(`npm run build` in frontend/, or `python -m backend.scripts.run_reviewer`
which does this for you), then run this same command — this file detects the built
`frontend/dist` and serves it alongside the API. See README.md.

The lifespan hook opens the single shared DB connection (migrations applied
once at startup) and loads the CategorizationService's model artifact, both
closed/released at shutdown. This hook's structure was kept stable from
Phase 2 specifically so Phase 3 only adds to it, per Phase 2's design note.

ML-D Production Integration: CategorizationService now loads the ML-C
selected TF-IDF + Logistic Regression artifact (LOGREG_MODEL_PATH,
models/tfidf_logreg_v1.pkl, built by scripts/build_production_logreg_model.py)
rather than the K-Means artifact — see backend/services/categorization_service.py.
"""
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.api.error_handlers import register_error_handlers
from backend.api.routes import dashboard, demo, forecasts, health, holdings, imports, transactions
from backend.config import FRONTEND_ORIGIN, LOGREG_MODEL_PATH, ROOT_DIR, V2_DB_PATH
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
    app.state.categorization_service = CategorizationService(LOGREG_MODEL_PATH)

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

# --- Packaged reviewer/demo mode (TRD §1.7, Build Plan §2.3) ---------------
#
# In dev mode, two servers run: Vite (HMR, :5173) proxies /api to this app
# (:8000), and the browser never hits this app for anything but /api/*, so
# this section is inert. In reviewer/demo mode, the built frontend
# (frontend/dist, produced by `npm run build`) is served from this same
# process/port: static assets directly, and a catch-all SPA fallback so
# client-side deep links like /dashboard or /forecast return index.html
# instead of a 404 (React Router then renders the right page client-side).
# All /api/* routes above are registered first and always take priority.
FRONTEND_DIST = ROOT_DIR / "frontend" / "dist"

if FRONTEND_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str, request: Request) -> FileResponse:
        """Serve index.html for any non-API GET so client-side routes work.

        A real static file under dist/ (e.g. favicon.ico) is served directly
        if present; unknown deep links fall back to index.html, exactly like
        a standard SPA history-mode server. Registered last, so it never
        shadows /api/* — a genuinely unknown /api/... path still 404s
        (this handler explicitly refuses the /api prefix as a second layer
        of protection, in case route registration order ever changes).
        """
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not Found")
        candidate = FRONTEND_DIST / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST / "index.html")
