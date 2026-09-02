# PlainCents V2 Backend

This directory holds the V2 FastAPI application. It is being built per
`docs/V2_BUILD_PLAN.md`, one phase at a time. As of **Phase 0**, this is a
skeleton only — no routes, services, or repositories are implemented yet.

## Local dev setup (Phase 0 state)

```bash
# From the repository root, using the project's existing venv:
venv/Scripts/python.exe -m pip install -r requirements.txt

# Build the deterministic test artifact for CategorizationService tests
# (Phase 3+); safe to (re-)run any time.
venv/Scripts/python.exe tests/fixtures/build_test_kmeans_model.py

# Run tests
venv/Scripts/python.exe -m pytest tests/ -v
```

`.env.example` documents the V2 configuration values (`V2_DB_PATH`,
`FRONTEND_ORIGIN`, `LOG_LEVEL`) that later phases will read; copy it to
`.env` when Phase 2 introduces config loading.

## What exists so far

- `backend/{api,schemas,services,repositories,db}/` — empty package skeleton, no logic (Phase 1+ fills these in).
- `db/migrations/` — empty, populated by Phase 1 with `001_initial_v2.sql`.
- `tests/fixtures/` — TD CSV parser fixtures and the deterministic K-Means test-artifact bootstrap script (see `tests/fixtures/README.md`).

## V1 compatibility

V1's pipeline (`pipeline/`, `db/database.py`, `main.py`) is untouched and
remains fully runnable independent of this backend — see the root README and
`docs/V2_TRD.md` §18 for the compatibility plan.
