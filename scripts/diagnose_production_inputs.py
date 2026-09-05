"""
Read-only diagnostics for the categorizer on real stored transactions.

WHAT THIS IS FOR
----------------
Private bank exports carry no category labels, so accuracy cannot be measured
on them — and this script never pretends otherwise. What it CAN measure is
whether the model can *read* the descriptions it is being given, which is the
failure mode that produced the "everything is Food & Dining" symptom:

    a description that yields no features -> a linear model returns
    argmax(intercept_) -> one fixed category, forever, for every such row.

So this reports representation coverage, the abstention/routing breakdown, and
the distribution of decisions. All aggregate. No transaction description,
amount, merchant or date is ever printed, and nothing is written to the
database.

USAGE
    python -m scripts.diagnose_production_inputs [--db PATH] [--limit N]

Add --show-drift to also compare the decision the CURRENT pipeline would make
against what is already stored, which is how you find out whether stored rows
predate a model change. That comparison is reported as counts only.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import CATEGORIZER_MODEL_PATH, V2_DB_PATH  # noqa: E402
from backend.services.ambiguity import is_structurally_ambiguous  # noqa: E402
from backend.services.categorization_service import CategorizationService  # noqa: E402
from backend.services.merchant_identity import stable_merchant_key  # noqa: E402


def _pct(n: int, total: int) -> str:
    return f"{n / total * 100:5.1f}%" if total else "    —"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(V2_DB_PATH), help="path to the SQLite database")
    parser.add_argument("--limit", type=int, default=None, help="cap rows examined")
    parser.add_argument("--show-drift", action="store_true",
                        help="compare current decisions against stored ones (counts only)")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"No database at {db_path} — nothing to diagnose.")
        return

    service = CategorizationService(CATEGORIZER_MODEL_PATH)
    if service.status != "loaded":
        print(f"Categorization model unavailable ({service.status}) at {CATEGORIZER_MODEL_PATH}.")
        return

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    sql = ("SELECT merchant, bank_source, predicted_category, confirmed_category, data_mode "
           "FROM transactions ORDER BY id")
    if args.limit:
        sql += f" LIMIT {int(args.limit)}"
    rows = conn.execute(sql).fetchall()
    conn.close()

    if not rows:
        print(f"No transactions in {db_path}.")
        return

    total = len(rows)
    print(f"Rows examined: {total}   (source: {db_path})")
    print(f"Model: {service.model_impl_version}  normalizer={service.normalizer_name}  "
          f"min_margin={service.min_margin}")
    print("Nothing below identifies an individual transaction.\n")

    by_mode = Counter(r["data_mode"] for r in rows)
    print("data_mode: " + ", ".join(f"{k}={v}" for k, v in sorted(by_mode.items())))

    ambiguous = [r for r in rows if is_structurally_ambiguous(r["merchant"])]
    classifiable = [r for r in rows if not is_structurally_ambiguous(r["merchant"])]
    print(f"\nStructural routing (no merchant identity in the text):")
    print(f"  routed to Other by rule : {len(ambiguous):5d}  {_pct(len(ambiguous), total)}")
    print(f"  handed to the model     : {len(classifiable):5d}  {_pct(len(classifiable), total)}")

    if classifiable:
        results = service.classify_batch([r["merchant"] for r in classifiable])
        n_active = [x["n_active_features"] for x in results]
        zero = sum(1 for n in n_active if n == 0)
        weak = sum(1 for n in n_active if 0 < n <= 2)
        abstained = [x for x in results if x["abstained"]]

        print(f"\nRepresentation coverage on the {len(classifiable)} model-eligible rows:")
        print(f"  zero-feature (no signal): {zero:5d}  {_pct(zero, len(classifiable))}")
        print(f"  weak (1-2 features)     : {weak:5d}  {_pct(weak, len(classifiable))}")
        print(f"  mean active features    : {sum(n_active) / len(n_active):8.1f}")
        print(f"  mean top-vs-second gap  : "
              f"{sum(x['margin'] for x in results) / len(results):8.3f}")

        print(f"\nAbstention (declined to answer -> Other):")
        print(f"  abstained               : {len(abstained):5d}  {_pct(len(abstained), len(classifiable))}")
        reasons = Counter(x["abstain_reason"] for x in abstained)
        for reason, count in sorted(reasons.items()):
            print(f"    {reason:22s}: {count:5d}")

        print("\nDecision distribution (what the system would serve today):")
        served = Counter(x["category"] for x in results)
        served["Other"] += len(ambiguous)
        for category, count in served.most_common():
            print(f"  {category:18s}: {count:5d}  {_pct(count, total)}")
        # The specific pathology this phase existed to remove: one category
        # absorbing everything because evidence-free rows all collapse into it.
        top_category, top_count = served.most_common(1)[0]
        print(f"\n  Largest single category: {top_category} at {_pct(top_count, total)} of rows.")
        print("  (No accuracy claim: these rows have no ground-truth labels.)")

    keys = [stable_merchant_key(r["merchant"], r["bank_source"]) for r in rows]
    with_key = [k for k in keys if k]
    print(f"\nCorrection-memory identity:")
    print(f"  rows with a stable key  : {len(with_key):5d}  {_pct(len(with_key), total)}")
    print(f"  distinct merchants      : {len(set(with_key)):5d}")
    recurring = sum(1 for _, c in Counter(with_key).items() if c > 1)
    print(f"  merchants seen 2+ times : {recurring:5d}   "
          f"(these are the ones a correction pays off on)")
    confirmed = sum(1 for r in rows if r["confirmed_category"])
    print(f"  rows carrying your own correction: {confirmed:5d}  {_pct(confirmed, total)}")

    if args.show_drift and classifiable:
        stored_vs_now = Counter()
        for r, result in zip(classifiable, results):
            if r["predicted_category"] is None:
                continue
            stored_vs_now["same" if r["predicted_category"] == result["category"] else "differs"] += 1
        for r in ambiguous:
            if r["predicted_category"] is not None:
                stored_vs_now["same" if r["predicted_category"] == "Other" else "differs"] += 1
        same, differs = stored_vs_now["same"], stored_vs_now["differs"]
        print(f"\nDrift between stored decisions and the current pipeline:")
        print(f"  unchanged : {same:5d}  {_pct(same, same + differs)}")
        print(f"  changed   : {differs:5d}  {_pct(differs, same + differs)}")
        if differs:
            print("  Stored rows keep the decision made when they were imported. To adopt the\n"
                  "  current model on rows you never corrected, run:\n"
                  "      python -m scripts.recategorize_stored_transactions --apply")


if __name__ == "__main__":
    main()
