"""
Phase 3: Random Forest spending forecast.
Walk-forward validation (expanding window), 3-month horizon per category.
PRD: MAPE < 15%, never shuffle time series, never use future data in features.
"""
import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder

from config import CATEGORIES, RF_MODEL_PATH

logger = logging.getLogger(__name__)


def aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate labeled transactions to monthly category totals.

    Parameters
    ----------
    df : DataFrame
        Must have columns: date (YYYY-MM-DD str), amount, category.

    Returns
    -------
    DataFrame with columns: month (YYYY-MM str), category, total_spend.
    """
    required = {"date", "amount", "category"}
    if required - set(df.columns):
        raise ValueError(f"DataFrame must have columns: {required}")

    df = df.copy()
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)
    monthly = df.groupby(["month", "category"], as_index=False)["amount"].sum()
    monthly = monthly.rename(columns={"amount": "total_spend"})
    monthly = monthly.sort_values(["category", "month"]).reset_index(drop=True)

    n_months = monthly["month"].nunique()
    if n_months < 12:
        logger.warning("Only %d months of data (need 12). Forecasting may be unreliable.", n_months)
        raise ValueError(f"Need 12 months minimum for forecasting. Found {n_months}.")

    return monthly


def build_forecast_features(monthly_df: pd.DataFrame, le: LabelEncoder | None = None, fit_le: bool = True):
    """
    Build features for Random Forest from monthly category totals.
    PRD Section 6: month_num, category_encoded, rolling_3m_avg, rolling_6m_avg,
    rolling_std, is_december, is_summer.

    Features only use data from months BEFORE the target month.
    Rows with insufficient history (< 6 prior months) are dropped.

    Parameters
    ----------
    monthly_df : DataFrame
        Columns: month, category, total_spend. Sorted by (category, month).
    le : LabelEncoder or None
        If provided and fit_le=False, use existing encoder.
    fit_le : bool
        If True, fit a new LabelEncoder on categories.

    Returns
    -------
    X : DataFrame of features
    y : Series of targets (total_spend)
    le : fitted LabelEncoder
    """
    df = monthly_df.copy()
    df["month_dt"] = pd.to_datetime(df["month"])
    df["month_num"] = df["month_dt"].dt.month

    if fit_le:
        le = LabelEncoder()
        le.fit(CATEGORIES)
    df["category_encoded"] = le.transform(df["category"])

    df = df.sort_values(["category", "month_dt"]).reset_index(drop=True)

    rolling_3m = []
    rolling_6m = []
    rolling_std_list = []
    lag_1_list = []
    for _, group in df.groupby("category"):
        spend = group["total_spend"].values
        r3 = []
        r6 = []
        rstd = []
        lag1 = []
        for i in range(len(spend)):
            if i >= 3:
                r3.append(np.mean(spend[i - 3:i]))
            else:
                r3.append(np.nan)
            if i >= 6:
                r6.append(np.mean(spend[i - 6:i]))
            else:
                r6.append(np.nan)
            if i >= 3:
                rstd.append(np.std(spend[i - 3:i], ddof=1))
            else:
                rstd.append(np.nan)
            if i >= 1:
                lag1.append(spend[i - 1])
            else:
                lag1.append(np.nan)
        rolling_3m.extend(r3)
        rolling_6m.extend(r6)
        rolling_std_list.extend(rstd)
        lag_1_list.extend(lag1)

    df["rolling_3m_avg"] = rolling_3m
    df["rolling_6m_avg"] = rolling_6m
    df["rolling_std"] = rolling_std_list
    df["lag_1_spend"] = lag_1_list

    df["is_december"] = (df["month_num"] == 12).astype(int)
    df["is_summer"] = df["month_num"].isin([6, 7, 8]).astype(int)

    df = df.dropna(subset=["rolling_3m_avg", "rolling_6m_avg", "rolling_std", "lag_1_spend"]).reset_index(drop=True)

    feature_cols = [
        "month_num", "category_encoded", "rolling_3m_avg",
        "rolling_6m_avg", "rolling_std", "is_december", "is_summer",
        "lag_1_spend",
    ]
    X = df[feature_cols]
    y = df["total_spend"]

    return X, y, le


def walk_forward_validate(monthly_df: pd.DataFrame) -> dict:
    """
    Walk-forward validation with expanding window. Never shuffle.

    For each test month M (from earliest feasible onward):
        train = all months before M
        test = month M
        Refit RF on train, predict test, compute APE.

    Returns dict with overall_mape, per_category_mape, predictions_df.
    """
    df = monthly_df.copy()
    df["month_dt"] = pd.to_datetime(df["month"])
    all_months = sorted(df["month_dt"].unique())

    le = LabelEncoder()
    le.fit(CATEGORIES)

    feature_cols = [
        "month_num", "category_encoded", "rolling_3m_avg",
        "rolling_6m_avg", "rolling_std", "is_december", "is_summer",
        "lag_1_spend",
    ]
    results = []

    for test_idx in range(len(all_months)):
        test_month = all_months[test_idx]
        train_months = [m for m in all_months if m < test_month]
        if len(train_months) < 7:
            continue

        train_df = df[df["month_dt"].isin(train_months)][["month", "category", "total_spend"]]
        test_df_raw = df[df["month_dt"] == test_month][["month", "category", "total_spend"]]

        if train_df.empty or test_df_raw.empty:
            continue

        X_train, y_train, _ = build_forecast_features(train_df, le=le, fit_le=False)
        if len(X_train) < 3:
            continue

        rf = RandomForestRegressor(
            n_estimators=100, max_depth=3, min_samples_leaf=3, random_state=42,
        )
        rf.fit(X_train, y_train)

        test_month_str = pd.Timestamp(test_month).strftime("%Y-%m")
        test_month_num = pd.Timestamp(test_month).month
        is_dec = 1 if test_month_num == 12 else 0
        is_sum = 1 if test_month_num in [6, 7, 8] else 0

        for cat in CATEGORIES:
            cat_test = test_df_raw[test_df_raw["category"] == cat]
            if cat_test.empty:
                continue
            actual = cat_test["total_spend"].values[0]

            cat_history = train_df[train_df["category"] == cat].sort_values("month")
            spend = cat_history["total_spend"].values

            if len(spend) < 3:
                continue

            r3 = np.mean(spend[-3:])
            rstd = np.std(spend[-3:], ddof=1)
            r6 = np.mean(spend[-6:]) if len(spend) >= 6 else np.mean(spend)
            lag1 = spend[-1]

            cat_encoded = le.transform([cat])[0]
            x_test = pd.DataFrame([{
                "month_num": test_month_num,
                "category_encoded": cat_encoded,
                "rolling_3m_avg": r3,
                "rolling_6m_avg": r6,
                "rolling_std": rstd,
                "is_december": is_dec,
                "is_summer": is_sum,
                "lag_1_spend": lag1,
            }])
            pred_val = rf.predict(x_test[feature_cols])[0]
            ape = abs(actual - pred_val) / max(abs(actual), 1e-9) * 100
            results.append({
                "month": test_month_str,
                "category": cat,
                "actual": actual,
                "predicted": pred_val,
                "ape": ape,
            })

    if not results:
        logger.warning("No walk-forward results. Insufficient data for validation.")
        return {"overall_mape": float("nan"), "per_category_mape": {}, "predictions_df": pd.DataFrame()}

    results_df = pd.DataFrame(results)
    overall_mape = results_df["ape"].mean()
    per_category_mape = results_df.groupby("category")["ape"].mean().to_dict()

    logger.info("Walk-forward MAPE: %.1f%%", overall_mape)
    for cat, mape in sorted(per_category_mape.items()):
        logger.info("  %s: %.1f%%", cat, mape)

    if overall_mape > 15:
        logger.warning("MAPE %.1f%% exceeds 15%% target. Consider GridSearchCV tuning.", overall_mape)

    return {
        "overall_mape": overall_mape,
        "per_category_mape": per_category_mape,
        "predictions_df": results_df,
    }


def fit_and_forecast(df: pd.DataFrame) -> tuple[pd.DataFrame, float, dict]:
    """
    Main entry point: aggregate, validate, fit final model, produce 3-month forecasts.

    Parameters
    ----------
    df : DataFrame
        Labeled transactions (date, merchant, amount, category).

    Returns
    -------
    forecast_df : DataFrame
        Columns: category, month_offset, forecast_month, predicted_amount.
    overall_mape : float
    per_category_mape : dict
    """
    monthly_df = aggregate_monthly(df)

    val_results = walk_forward_validate(monthly_df)
    overall_mape = val_results["overall_mape"]
    per_category_mape = val_results["per_category_mape"]

    X_all, y_all, le = build_forecast_features(monthly_df)

    rf_params = {
        "n_estimators": 100, "max_depth": 10,
        "min_samples_leaf": 5, "random_state": 42,
    }

    if overall_mape > 15:
        logger.info("MAPE %.1f%% > 15%% — running GridSearchCV per PRD spec.", overall_mape)
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 10],
            "min_samples_leaf": [3, 5, 10],
        }
        tscv = TimeSeriesSplit(n_splits=3)
        grid = GridSearchCV(
            RandomForestRegressor(random_state=42),
            param_grid,
            cv=tscv,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
        )
        grid.fit(X_all, y_all)
        rf_params.update(grid.best_params_)
        logger.info("GridSearchCV best params: %s", grid.best_params_)
        logger.info("GridSearchCV best MAE: %.2f", -grid.best_score_)

    rf = RandomForestRegressor(**rf_params)
    rf.fit(X_all, y_all)

    last_month = pd.to_datetime(monthly_df["month"]).max()
    last_month_str = last_month.strftime("%Y-%m")

    feature_cols = [
        "month_num", "category_encoded", "rolling_3m_avg",
        "rolling_6m_avg", "rolling_std", "is_december", "is_summer",
        "lag_1_spend",
    ]

    forecasts = []
    for offset in [1, 2, 3]:
        future_month = last_month + pd.DateOffset(months=offset)
        future_month_num = future_month.month
        is_dec = 1 if future_month_num == 12 else 0
        is_sum = 1 if future_month_num in [6, 7, 8] else 0
        forecast_month_str = future_month.strftime("%Y-%m")

        for cat in CATEGORIES:
            cat_data = monthly_df[monthly_df["category"] == cat].sort_values("month")
            spend_history = cat_data["total_spend"].values

            if len(spend_history) >= 3:
                r3 = np.mean(spend_history[-3:])
                rstd = np.std(spend_history[-3:], ddof=1)
            else:
                r3 = np.mean(spend_history) if len(spend_history) > 0 else 0.0
                rstd = 0.0

            if len(spend_history) >= 6:
                r6 = np.mean(spend_history[-6:])
            else:
                r6 = np.mean(spend_history) if len(spend_history) > 0 else 0.0

            lag1 = spend_history[-1] if len(spend_history) > 0 else 0.0

            cat_encoded = le.transform([cat])[0]
            row = pd.DataFrame([{
                "month_num": future_month_num,
                "category_encoded": cat_encoded,
                "rolling_3m_avg": r3,
                "rolling_6m_avg": r6,
                "rolling_std": rstd,
                "is_december": is_dec,
                "is_summer": is_sum,
                "lag_1_spend": lag1,
            }])
            pred = rf.predict(row[feature_cols])[0]
            forecasts.append({
                "category": cat,
                "month_offset": offset,
                "forecast_month": forecast_month_str,
                "predicted_amount": round(pred, 2),
            })

    forecast_df = pd.DataFrame(forecasts)

    RF_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": rf, "label_encoder": le, "feature_names": feature_cols}, RF_MODEL_PATH)
    logger.info("Saved RF model to %s", RF_MODEL_PATH)

    return forecast_df, overall_mape, per_category_mape


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from pipeline.ingest import load_and_clean
    from pipeline.cluster import predict_categories

    df = load_and_clean("synthetic_24mo.csv", bank="TD")
    df = predict_categories(df)

    forecast_df, mape, cat_mape = fit_and_forecast(df)

    print(f"Overall MAPE: {mape:.1f}%")
    print("Per-category MAPE:")
    for cat, m in sorted(cat_mape.items()):
        print(f"  {cat}: {m:.1f}%")
    print()
    print("3-month forecast:")
    print(forecast_df.to_string(index=False))

    print("\n=== DIAGNOSTIC: True label forecast ===")
    from pipeline.cluster import _get_true_labels
    df2 = load_and_clean("synthetic_24mo.csv", bank="TD")
    df2["category"] = _get_true_labels(df2)
    forecast_df2, mape2, cat_mape2 = fit_and_forecast(df2)
    print(f"True-label MAPE: {mape2:.1f}%")
    print("Per-category MAPE (true labels):")
    for cat, m in sorted(cat_mape2.items()):
        print(f"  {cat}: {m:.1f}%")
