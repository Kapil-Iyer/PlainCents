"""
Thin CLI wrapper around DemoService.load_demo() (TRD §14.1; Build Plan
Phase 9). All deterministic seed-generation logic lives in
backend.services.demo_seed_data; all persistence/state-machine logic lives
in backend.services.demo_service.DemoService — this script only opens a
connection and calls it, exactly what the app itself does from
POST /api/demo/load. It is a separate file from V1's db/seed_synthetic_data.py,
not a modification of it (that script remains V1's own, untouched, per
Build Plan Phase 9's "V1 files untouched" note).

Run: python -m backend.scripts.seed_v2_demo_data
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.config import V2_DB_PATH  # noqa: E402
from backend.db.connection import get_connection  # noqa: E402
from backend.services.demo_service import DemoService  # noqa: E402


def main() -> None:
    conn = get_connection(db_path=V2_DB_PATH)
    try:
        result = DemoService(conn).load_demo()
    finally:
        conn.close()

    print(f"Demo data loaded — mode is now {result['mode']}.")
    for key, count in result["summary"].items():
        print(f"  {key}: {count}")


if __name__ == "__main__":
    main()
