"""
ML Spec Section 19 (reproducibility) / prompt's "Experiment Logging"
requirement: every meaningful experiment (categorization candidate run,
forecast candidate/strategy/horizon run, history-length or sparsity probe)
is appended to a single JSONL ledger so a losing or invalid result is never
silently dropped.

One JSON object per line. Append-only within a single ML-B execution
(ml/run_all.py calls `reset_log()` once at the start of a full run so
re-running doesn't silently accumulate duplicate history across sessions).
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOG_PATH = Path(__file__).resolve().parent.parent.parent / "reports" / "ml" / "results" / "experiment_log.jsonl"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent.parent.parent, text=True
        ).strip()
    except Exception:
        return "unknown"


def reset_log() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("")


def log_experiment(
    experiment_id: str,
    dataset_id: str,
    evidence_tier: str,
    seed: int,
    status: str,  # "SUCCESS" | "FAILED" | "INVALID"
    metrics: dict[str, Any] | None = None,
    partition_definition_ref: str | None = None,
    preprocessing: dict[str, Any] | None = None,
    model: str | None = None,
    forecasting_strategy: str | None = None,
    hyperparameters: dict[str, Any] | None = None,
    notes: str = "",
    reason: str = "",
) -> dict:
    if status not in {"SUCCESS", "FAILED", "INVALID"}:
        raise ValueError(f"status must be SUCCESS/FAILED/INVALID, got {status!r}")
    if status in {"FAILED", "INVALID"} and not reason:
        raise ValueError("reason is required when status is FAILED or INVALID")

    record = {
        "experiment_id": experiment_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "dataset_id": dataset_id,
        "evidence_tier": evidence_tier,
        "seed": seed,
        "partition_definition_ref": partition_definition_ref,
        "preprocessing": preprocessing or {},
        "model": model,
        "forecasting_strategy": forecasting_strategy,
        "hyperparameters": hyperparameters or {},
        "validation_metrics": metrics or {},
        "notes": notes,
        "status": status,
        "reason": reason,
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
    return record


def read_log() -> list[dict]:
    if not LOG_PATH.exists():
        return []
    with open(LOG_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]
