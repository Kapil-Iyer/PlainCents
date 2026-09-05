"""
Reviewer/demo packaged-mode launcher (Build Plan Phase 10, TRD §1.7 / §2.3).

This is the ONE normal command a reviewer runs after initial setup:

    python -m backend.scripts.run_reviewer

It does the smallest amount of work needed to turn "two dev servers" into
"one process, one port":

  1. Ensures the frontend production build (frontend/dist) exists and is not
     older than the frontend source — builds it with `npm run build` if not.
     (Dev mode is untouched: `npm run dev` + `uvicorn --reload` still work
     side by side, this script is purely an alternate entrypoint.)
  2. Warns (does not fail) if the ML categorizer artifact is missing, since
     the app already tolerates that per TRD §11.1 (health reports "missing",
     categorization-dependent writes 503) — Explore Demo does not need it.
  3. Launches `uvicorn backend.main:app` on one port, which backend/main.py
     already serves the built frontend from (StaticFiles + SPA fallback).

No Docker, no cloud, no second deployment architecture — just an ordering
helper around the same two tools already used in dev.
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"
FRONTEND_DIST = FRONTEND_DIR / "dist"
FRONTEND_SRC = FRONTEND_DIR / "src"
# ML-G: the production categorizer is the word+char TF-IDF + Logistic
# Regression artifact with the frozen abstention policy (see
# backend/config.py::CATEGORIZER_MODEL_PATH and reports/ml/ML_G_FINAL_REPORT.md).
# Superseded ML-D's K-Means artifact and ML-F's tfidf_logreg_v1/v2 recipes.
MODEL_PATH = ROOT_DIR / "models" / "categorizer_v3.pkl"

HOST = "127.0.0.1"
PORT = "8000"


def _dist_is_stale() -> bool:
    if not FRONTEND_DIST.is_dir() or not (FRONTEND_DIST / "index.html").is_file():
        return True
    dist_mtime = max((p.stat().st_mtime for p in FRONTEND_DIST.rglob("*") if p.is_file()), default=0)
    src_mtime = max((p.stat().st_mtime for p in FRONTEND_SRC.rglob("*") if p.is_file()), default=0)
    return src_mtime > dist_mtime


def _npm_path() -> str:
    npm = shutil.which("npm")
    if npm is None:
        print(
            "ERROR: npm was not found on PATH. Install Node.js (see README's Initial "
            "installation section), then re-run this command.",
            file=sys.stderr,
        )
        sys.exit(1)
    return npm


def ensure_frontend_built() -> None:
    if not _dist_is_stale():
        print("[reviewer] frontend/dist is up to date, skipping build.")
        return

    npm = _npm_path()
    node_modules = FRONTEND_DIR / "node_modules"
    if not node_modules.is_dir():
        print("[reviewer] frontend/node_modules missing — running npm install...")
        subprocess.run([npm, "install"], cwd=FRONTEND_DIR, check=True)

    print("[reviewer] building frontend production bundle (npm run build)...")
    subprocess.run([npm, "run", "build"], cwd=FRONTEND_DIR, check=True)


def warn_if_model_missing() -> None:
    if not MODEL_PATH.is_file():
        print(
            "[reviewer] WARNING: models/categorizer_v3.pkl not found. Real CSV import will "
            "report the categorization model as unavailable. Explore Demo does not need "
            "it and will still work. Build it with: python -m scripts.build_production_categorizer",
            file=sys.stderr,
        )


def main() -> None:
    ensure_frontend_built()
    warn_if_model_missing()

    print(f"[reviewer] starting PlainCents at http://{HOST}:{PORT} ...")
    subprocess.run(
        [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", HOST, "--port", PORT],
        cwd=ROOT_DIR,
        check=True,
    )


if __name__ == "__main__":
    main()
