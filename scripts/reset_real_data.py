"""
One-off maintenance script: clear REAL-mode data and return the app to
EMPTY, so Load Demo Data becomes available again.

The product now also has an in-app/API path for this: DELETE
/api/demo/clear-real-data (DemoService.clear_real_data(), surfaced via the
"Clear all real data" danger-zone action on the Import page) -- the
user-facing equivalent of what this script does, for anywhere a shell isn't
available (e.g. a deployed environment). This script remains useful as a
terminal-only alternative during local development, mirroring
clear_real_data()'s exact deletion logic (transactions, holdings, forecast
runs) scoped to data_mode='real', then resetting app_state.mode to 'EMPTY'
unconditionally.

This does NOT touch the original CSV files that were imported, only the
rows already written to plaincents_v2.db.

Run:  python -m scripts.reset_real_data          # preview (dry run)
      python -m scripts.reset_real_data --apply   # actually delete
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.db.connection import get_connection  # noqa: E402
from backend.repositories.app_state_repository import AppStateRepository  # noqa: E402
from backend.repositories.forecast_repository import ForecastRepository  # noqa: E402
from backend.repositories.holding_repository import HoldingRepository  # noqa: E402
from backend.repositories.transaction_repository import TransactionRepository  # noqa: E402


def main() -> None:
    apply = "--apply" in sys.argv
    conn = get_connection()

    txn_repo = TransactionRepository(conn)
    holding_repo = HoldingRepository(conn)
    forecast_repo = ForecastRepository(conn)
    app_state_repo = AppStateRepository(conn)

    current_mode = app_state_repo.get_mode()
    real_txns = txn_repo.list(data_mode="real")
    real_holdings = holding_repo.list(data_mode="real")

    print(f"Current app_state.mode: {current_mode}")
    print(f"Real transactions to delete: {len(real_txns)}")
    print(f"Real holdings to delete: {len(real_holdings)}")

    if not apply:
        print("\nDry run only -- re-run with --apply to actually delete.")
        conn.close()
        return

    with conn:
        txns_deleted = txn_repo.delete_by_data_mode("real")
        holdings_deleted = holding_repo.delete_by_data_mode("real")
        forecasts_deleted = forecast_repo.delete_runs_by_data_mode("real")
        app_state_repo.set_mode("EMPTY")

    print(
        f"\nDeleted {txns_deleted} transactions, {holdings_deleted} holdings, "
        f"{forecasts_deleted} forecast runs. app_state.mode -> EMPTY."
    )
    conn.close()


if __name__ == "__main__":
    main()
