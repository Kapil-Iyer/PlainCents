"""
ML-B full run: executes Part A (categorization bake-off) and Part B
(forecasting bake-off + history-length + sparsity analyses) in sequence and
resets the experiment ledger first, so a rerun doesn't silently accumulate
duplicate history across sessions (ML Spec Section 19 reproducibility).

Usage: python -m ml.run_all
"""
from __future__ import annotations

from ml.categorization import run_bakeoff as cat_bakeoff
from ml.common.experiment_log import reset_log
from ml.forecasting import history_sensitivity, run_bakeoff as fc_bakeoff, sparsity_analysis


def main() -> None:
    reset_log()

    print("=== Part A: Categorization bake-off ===")
    cat_results = cat_bakeoff.run()
    for name, r in cat_results["candidates"].items():
        v = r["validation"]
        print(f"  {name}: VALIDATION macro_f1={v['macro_f1']:.4f} accuracy={v['accuracy']:.4f}")

    print("\n=== Part B: Forecasting bake-off ===")
    fc_metrics = fc_bakeoff.run()
    for key, entry in fc_metrics["by_candidate_strategy"].items():
        c = entry["combined"]
        if c:
            print(f"  {key}: combined WAPE={c['wape']:.4f} MAE={c['mae']:.2f} n={c['n']}")

    print("\n=== Part B: History-length sensitivity (Section 15) ===")
    hist_result = history_sensitivity.run()
    print(f"  status: {hist_result['status']}")

    print("\n=== Part B: Sparsity analysis (Section 16) ===")
    sparsity_result = sparsity_analysis.run()
    for category, info in sparsity_result["per_category_sparsity"].items():
        print(f"  {category}: {info['sparsity_bucket']}")

    print("\nAll ML-B results written to reports/ml/results/")


if __name__ == "__main__":
    main()
