"""
ML-C Part B: Forecast fold-level stability analysis.

ML-B stored prediction-level results across 14 expanding-window VALIDATION
origins but pooled the headline forecasting metrics (reports/ml/results/
forecasting_metrics.json's "combined"/"by_horizon" blocks). This module
performs the fold-level stability review the ML-C brief requires: per
origin, per candidate/strategy, per horizon WAPE/MAE, then summarizes
mean/median/spread, best/worst origins, how often each candidate beats
Naive (and Seasonal Naive where eligible) origin-by-origin, and whether
horizon rankings are stable.

Does NOT create a new split or re-run any model. Reads only the already-
committed ML-B artifact reports/ml/results/forecasting_predictions_long.csv
(itself produced by ml/forecasting/run_bakeoff.py) and the pooled metrics
file, both sealed against the reserved FINAL period already.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CSV_PATH = REPO_ROOT / "reports" / "ml" / "results" / "forecasting_predictions_long.csv"
METRICS_PATH = REPO_ROOT / "reports" / "ml" / "results" / "forecasting_metrics.json"
OUT_PATH = REPO_ROOT / "reports" / "ml" / "ML_C_FOLD_STABILITY.json"


def _wape(actual: pd.Series, predicted: pd.Series) -> float:
    num = (actual - predicted).abs().sum()
    den = actual.abs().sum()
    if den == 0:
        return float("nan")
    return num / den


def _mae(actual: pd.Series, predicted: pd.Series) -> float:
    return (actual - predicted).abs().mean()


def _rmse(actual: pd.Series, predicted: pd.Series) -> float:
    return math.sqrt(((actual - predicted) ** 2).mean())


def _origin_to_month_map(df: pd.DataFrame) -> dict[int, str]:
    """Derived directly from the data (origin_month = the +1 target month
    minus one calendar month), not by assuming list-order correspondence
    with the pooled metrics file's fold_origin_months."""
    h1 = df[df["horizon"] == 1][["origin_index", "target_month"]].drop_duplicates()
    h1 = h1.copy()
    h1["target_month"] = pd.to_datetime(h1["target_month"])
    h1["origin_month"] = (h1["target_month"] - pd.DateOffset(months=1)).dt.strftime("%Y-%m")
    return dict(zip(h1["origin_index"], h1["origin_month"]))


def compute_per_origin_records(df: pd.DataFrame, origin_indices: list[int], origin_to_month: dict[int, str]) -> dict:
    """(candidate, strategy) -> list of per-origin {origin_index, origin_month,
    n, wape, mae, rmse, by_horizon} records, in origin_index order."""
    per_origin_records = {}
    for (candidate, strategy), g in df.groupby(["candidate", "strategy"], dropna=False):
        g_eligible = g[g["eligible"] == True]  # noqa: E712
        records = []
        for origin_idx in origin_indices:
            og = g_eligible[g_eligible["origin_index"] == origin_idx]
            if og.empty:
                records.append({"origin_index": int(origin_idx), "origin_month": origin_to_month[origin_idx],
                                 "n": 0, "wape": None, "mae": None, "rmse": None, "by_horizon": {}})
                continue
            actual = og["actual"].astype(float)
            predicted = og["predicted"].astype(float)
            by_horizon = {}
            for h in sorted(og["horizon"].unique()):
                hg = og[og["horizon"] == h]
                ha, hp = hg["actual"].astype(float), hg["predicted"].astype(float)
                hw = _wape(ha, hp)
                by_horizon[str(int(h))] = {"n": int(len(hg)), "wape": None if pd.isna(hw) else hw, "mae": float(_mae(ha, hp))}
            w = _wape(actual, predicted)
            records.append({
                "origin_index": int(origin_idx), "origin_month": origin_to_month[origin_idx],
                "n": int(len(og)), "wape": None if pd.isna(w) else w,
                "mae": float(_mae(actual, predicted)), "rmse": float(_rmse(actual, predicted)),
                "by_horizon": by_horizon,
            })
        per_origin_records[(candidate, strategy)] = records
    return per_origin_records


def summarize_wape_series(records: list[dict]) -> dict | None:
    vals = [r["wape"] for r in records if r["wape"] is not None and not math.isnan(r["wape"])]
    if not vals:
        return None
    vals_sorted = sorted(vals)
    n = len(vals_sorted)
    mean = sum(vals_sorted) / n
    median = vals_sorted[n // 2] if n % 2 == 1 else (vals_sorted[n // 2 - 1] + vals_sorted[n // 2]) / 2
    return {
        "n_origins_with_data": n, "mean": mean, "median": median,
        "min": vals_sorted[0], "max": vals_sorted[-1],
        "std": (sum((v - mean) ** 2 for v in vals_sorted) / n) ** 0.5,
        "range": vals_sorted[-1] - vals_sorted[0],
    }


def build_summary(per_origin_records: dict) -> dict:
    naive_records = {r["origin_index"]: r for r in per_origin_records[("naive", "n/a")]}
    seasonal_records = {r["origin_index"]: r for r in per_origin_records[("seasonal_naive", "n/a")]}

    summary = {}
    for (candidate, strategy), records in per_origin_records.items():
        key = f"{candidate}__{strategy}"
        wape_summary = summarize_wape_series(records)

        beats_naive, total_vs_naive = 0, 0
        for r in records:
            nr = naive_records.get(r["origin_index"])
            if r["wape"] is None or nr is None or nr["wape"] is None or math.isnan(r["wape"]) or math.isnan(nr["wape"]):
                continue
            total_vs_naive += 1
            beats_naive += int(r["wape"] < nr["wape"])

        beats_seasonal, total_vs_seasonal = 0, 0
        for r in records:
            sr = seasonal_records.get(r["origin_index"])
            if r["wape"] is None or sr is None or sr["wape"] is None or sr["n"] == 0:
                continue
            if math.isnan(r["wape"]) or math.isnan(sr["wape"]):
                continue
            total_vs_seasonal += 1
            beats_seasonal += int(r["wape"] < sr["wape"])

        per_horizon_beat_naive = {}
        for h in ["1", "2", "3"]:
            beat, total = 0, 0
            for r in records:
                nr = naive_records.get(r["origin_index"])
                rh, nh = r["by_horizon"].get(h), (nr["by_horizon"].get(h) if nr else None)
                if not rh or not nh or rh["wape"] is None or nh["wape"] is None:
                    continue
                total += 1
                beat += int(rh["wape"] < nh["wape"])
            per_horizon_beat_naive[h] = {"beats": int(beat), "total": int(total), "rate": (beat / total) if total else None}

        defined = [r for r in records if r["wape"] is not None and not math.isnan(r["wape"])]
        best = min(defined, key=lambda r: r["wape"], default=None)
        worst = max(defined, key=lambda r: r["wape"], default=None)

        summary[key] = {
            "candidate": candidate, "strategy": strategy,
            "wape_stats_across_origins": wape_summary,
            "beats_naive_origin_count": beats_naive, "beats_naive_origin_total": total_vs_naive,
            "beats_naive_origin_rate": (beats_naive / total_vs_naive) if total_vs_naive else None,
            "beats_seasonal_naive_origin_count": beats_seasonal, "beats_seasonal_naive_origin_total": total_vs_seasonal,
            "beats_seasonal_naive_origin_rate": (beats_seasonal / total_vs_seasonal) if total_vs_seasonal else None,
            "per_horizon_beats_naive": per_horizon_beat_naive,
            "best_origin": ({"origin_index": best["origin_index"], "origin_month": best["origin_month"], "wape": best["wape"], "n": best["n"]} if best else None),
            "worst_origin": ({"origin_index": worst["origin_index"], "origin_month": worst["origin_month"], "wape": worst["wape"], "n": worst["n"]} if worst else None),
        }
    return summary


def build_horizon_rankings(pooled_metrics: dict, candidates: list[tuple[str, str]]) -> dict:
    pooled_by_horizon = {}
    for candidate, strategy in candidates:
        key = f"{candidate}__{strategy}"
        ck = pooled_metrics["by_candidate_strategy"].get(key)
        if ck is None:
            continue
        pooled_by_horizon[key] = {h: ck["by_horizon"][h]["wape"] for h in ["1", "2", "3"]}

    rankings = {}
    for h in ["1", "2", "3"]:
        ranked = sorted(
            ((k, v[h]) for k, v in pooled_by_horizon.items() if v[h] is not None and not (isinstance(v[h], float) and math.isnan(v[h]))),
            key=lambda kv: kv[1],
        )
        rankings[h] = [k for k, _ in ranked]
    return rankings


def run(csv_path: Path = CSV_PATH, metrics_path: Path = METRICS_PATH, out_path: Path = OUT_PATH) -> dict:
    df = pd.read_csv(csv_path)
    df["strategy"] = df["strategy"].fillna("n/a")

    with open(metrics_path, encoding="utf-8") as f:
        pooled_metrics = json.load(f)

    fold_origin_months = pooled_metrics["fold_origin_months"]
    if len(fold_origin_months) != 14:
        raise RuntimeError(f"Expected 14 frozen ML-B folds, found {len(fold_origin_months)}")

    origin_indices = sorted(df["origin_index"].unique())
    if len(origin_indices) != 14:
        raise RuntimeError(f"Expected 14 origin indices in predictions CSV, found {len(origin_indices)}")

    origin_to_month = _origin_to_month_map(df)
    if set(origin_to_month.values()) != set(fold_origin_months):
        raise RuntimeError("Derived origin->month mapping does not match pooled metrics fold_origin_months")

    candidates = sorted(df.groupby(["candidate", "strategy"], dropna=False).groups.keys())
    per_origin_records = compute_per_origin_records(df, origin_indices, origin_to_month)
    summary = build_summary(per_origin_records)
    rankings = build_horizon_rankings(pooled_metrics, candidates)

    out = {
        "methodology": (
            "Computed directly from the frozen ML-B prediction-level artifact "
            "reports/ml/results/forecasting_predictions_long.csv (14 expanding-window "
            "VALIDATION origins, dev-region only, 2024-10/11/12 reserved and untouched). "
            "No new split was created; this is a derived summary of existing sealed-FINAL "
            "TRAIN+VALIDATION evidence, per ML Spec Section 12 / ML-C Part B instructions."
        ),
        "n_folds": 14,
        "fold_origin_months": fold_origin_months,
        "per_candidate_strategy_summary": summary,
        "horizon_ranking_by_wape": rankings,
        "per_origin_detail": {f"{c}__{s}": recs for (c, s), recs in per_origin_records.items()},
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, allow_nan=False, default=str)

    return out


if __name__ == "__main__":
    result = run()
    print("Wrote", OUT_PATH)
    print()
    print("=== Per-candidate/strategy WAPE stability across 14 origins ===")
    for key, s in result["per_candidate_strategy_summary"].items():
        ws = s["wape_stats_across_origins"]
        if ws is None:
            print(f"{key}: no defined-WAPE origins")
            continue
        print(f"{key}: mean={ws['mean']:.4f} median={ws['median']:.4f} std={ws['std']:.4f} "
              f"min={ws['min']:.4f} max={ws['max']:.4f} n={ws['n_origins_with_data']} "
              f"| beats_naive {s['beats_naive_origin_count']}/{s['beats_naive_origin_total']} "
              f"| beats_seasonal {s['beats_seasonal_naive_origin_count']}/{s['beats_seasonal_naive_origin_total']}")
        print(f"    per-horizon beat-naive rate: {s['per_horizon_beats_naive']}")
    print()
    print("=== Horizon ranking (best to worst WAPE) ===")
    for h, ranking in result["horizon_ranking_by_wape"].items():
        print(f"+{h}: {ranking}")
