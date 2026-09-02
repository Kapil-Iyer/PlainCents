/**
 * Per-spec isolated backend process (Build Plan Phase 10 / §12).
 *
 * Each E2E flow calls startBackend(port) once in test.beforeAll and stop()
 * in test.afterAll. Every call gets:
 *   - its own fresh temp-directory SQLite file via V2_DB_PATH — the
 *     developer's real plaincents_v2.db is never opened, so nothing here
 *     can mutate normal dev state, and each of the 4 flows starts from a
 *     genuinely clean app regardless of run order.
 *   - the deterministic fake `yfinance` package prepended to PYTHONPATH
 *     (tests/e2e/fixtures/fake_yfinance) — see that package's docstring
 *     for why this is the narrowest safe seam for the Portfolio flow.
 *     Harmless for the other 3 flows, which never call the refresh-prices
 *     route and so never import yfinance at all.
 * The app is served via the same packaged reviewer mode a real reviewer
 * uses (backend/main.py serving frontend/dist), built once in global-setup.
 */
import { spawn, type ChildProcess } from "node:child_process";
import { existsSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

const ROOT = path.resolve(__dirname, "..", "..", "..");
const FAKE_YFINANCE_DIR = path.join(__dirname, "..", "fixtures", "fake_yfinance");

export interface E2EBackend {
  baseURL: string;
  dbPath: string;
  stop: () => Promise<void>;
}

function pythonExe(): string {
  const venvPy =
    process.platform === "win32"
      ? path.join(ROOT, "venv", "Scripts", "python.exe")
      : path.join(ROOT, "venv", "bin", "python");
  return existsSync(venvPy) ? venvPy : "python";
}

export async function startBackend(port: number): Promise<E2EBackend> {
  const dbDir = mkdtempSync(path.join(tmpdir(), "plaincents-e2e-"));
  const dbPath = path.join(dbDir, "plaincents_v2_e2e.db");
  const baseURL = `http://127.0.0.1:${port}`;

  const env: NodeJS.ProcessEnv = {
    ...process.env,
    V2_DB_PATH: dbPath,
    PYTHONPATH: [FAKE_YFINANCE_DIR, process.env.PYTHONPATH].filter(Boolean).join(path.delimiter),
  };

  const proc: ChildProcess = spawn(
    pythonExe(),
    ["-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", String(port)],
    { cwd: ROOT, env, stdio: "pipe" },
  );

  let stderr = "";
  proc.stderr?.on("data", (d) => {
    stderr += d.toString();
  });

  const deadline = Date.now() + 30_000;
  let ready = false;
  while (Date.now() < deadline) {
    try {
      const res = await fetch(`${baseURL}/api/health`);
      if (res.ok) {
        ready = true;
        break;
      }
    } catch {
      // backend not accepting connections yet
    }
    await new Promise((r) => setTimeout(r, 300));
  }
  if (!ready) {
    proc.kill();
    throw new Error(`E2E backend on port ${port} did not become ready in time.\n${stderr}`);
  }

  return {
    baseURL,
    dbPath,
    stop: async () => {
      proc.kill();
    },
  };
}
