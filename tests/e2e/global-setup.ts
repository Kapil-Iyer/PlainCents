/**
 * Playwright global setup (Build Plan Phase 10 / §12).
 *
 * Runs ONCE before all 4 E2E flows. Does only the expensive, idempotent,
 * cross-cutting work that every spec needs:
 *
 *   1. Builds the frontend production bundle (frontend/dist) so the
 *      packaged reviewer mode backend/main.py serves has something to
 *      serve — every spec drives the app through this same packaged mode,
 *      not the Vite dev server, so E2E exercises the real reviewer path.
 *   2. Ensures a loadable ML categorizer artifact exists at
 *      models/kmeans_model.pkl, using the repo's EXISTING deterministic
 *      test-artifact bootstrap (tests/fixtures/build_test_kmeans_model.py)
 *      when the real trained one isn't present — this is the same fixture
 *      Phase 0's backend unit tests already rely on, just also placed
 *      where backend/config.py's KMEANS_MODEL_PATH expects it. Never
 *      overwrites a real model artifact if one is already there.
 *
 * Each spec file starts/stops its OWN backend process against its own
 * isolated temp SQLite database and its own port (see
 * tests/e2e/helpers/backend.ts) — global setup never touches
 * plaincents_v2.db and never starts a server itself.
 */
import { spawnSync } from "node:child_process";
import { copyFileSync, existsSync, mkdirSync } from "node:fs";
import path from "node:path";

const ROOT = path.resolve(__dirname, "..", "..");
const MODEL_PATH = path.join(ROOT, "models", "kmeans_model.pkl");
const TEST_MODEL_PATH = path.join(ROOT, "tests", "fixtures", "kmeans_model_test.pkl");
const BUILD_TEST_MODEL_SCRIPT = path.join(ROOT, "tests", "fixtures", "build_test_kmeans_model.py");

function pythonExe(): string {
  const venvPy =
    process.platform === "win32"
      ? path.join(ROOT, "venv", "Scripts", "python.exe")
      : path.join(ROOT, "venv", "bin", "python");
  return existsSync(venvPy) ? venvPy : "python";
}

function ensureFrontendBuilt(): void {
  console.log("[e2e global-setup] building frontend production bundle (npm run build)...");
  const npmCmd = process.platform === "win32" ? "npm.cmd" : "npm";
  const res = spawnSync(npmCmd, ["run", "build"], {
    cwd: path.join(ROOT, "frontend"),
    stdio: "inherit",
    shell: process.platform === "win32",
  });
  if (res.status !== 0) {
    throw new Error("frontend production build failed ahead of the E2E run");
  }
}

function ensureCategorizerArtifact(): void {
  if (existsSync(MODEL_PATH)) {
    console.log("[e2e global-setup] models/kmeans_model.pkl already present, reusing it.");
    return;
  }
  console.log(
    "[e2e global-setup] models/kmeans_model.pkl is missing — bootstrapping it from the " +
      "existing deterministic test-artifact mechanism (tests/fixtures/build_test_kmeans_model.py) " +
      "so the Import E2E flow has a loadable categorizer, without inventing a new artifact workflow.",
  );
  if (!existsSync(TEST_MODEL_PATH)) {
    const res = spawnSync(pythonExe(), [BUILD_TEST_MODEL_SCRIPT], { cwd: ROOT, stdio: "inherit" });
    if (res.status !== 0) {
      throw new Error("building the deterministic test categorizer artifact failed");
    }
  }
  mkdirSync(path.dirname(MODEL_PATH), { recursive: true });
  copyFileSync(TEST_MODEL_PATH, MODEL_PATH);
}

export default async function globalSetup(): Promise<void> {
  ensureFrontendBuilt();
  ensureCategorizerArtifact();
}
