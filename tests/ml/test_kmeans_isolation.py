"""
ML Spec Section 6.1: K-Means scaler/vectorizer/model fitting and the
cluster->category mapping must use TRAIN labels/data only. VALIDATION/FINAL
rows may be transformed and predicted against the frozen TRAIN artifacts,
but must never influence fitting or the mapping.
"""
import pandas as pd

from ml.categorization.candidates import KMeansCandidate


def _synthetic_df(n=220, seed=0):
    import numpy as np
    rng = np.random.RandomState(seed)
    categories = ["Food & Dining", "Transport", "Shopping"]
    merchants = {
        "Food & Dining": ["STARBUCKS COFFEE", "PIZZA PIZZA", "WENDYS"],
        "Transport": ["OC TRANSPO", "LYFT TRIP", "CHEVRON GAS"],
        "Shopping": ["WALMART SUPERCENTRE", "COSTCO WHOLESALE", "DOLLARAMA"],
    }
    rows = []
    dates = pd.date_range("2024-01-01", periods=400, freq="D")
    for i in range(n):
        cat = categories[i % 3]
        merchant = merchants[cat][i % 3]
        rows.append({
            "date": dates[i % len(dates)].strftime("%Y-%m-%d"),
            "merchant": merchant,
            "amount": float(10 + rng.rand() * 50),
            "true_category": cat,
        })
    return pd.DataFrame(rows)


def test_fit_object_identity_unchanged_by_predict_calls():
    """Calling .predict() on new data must not refit/replace the scaler,
    vectorizer, or kmeans model objects."""
    df = _synthetic_df()
    train_df = df.iloc[:150].reset_index(drop=True)
    other_df = df.iloc[150:].reset_index(drop=True)

    candidate = KMeansCandidate().fit(train_df)
    scaler_before, vectorizer_before, kmeans_before = candidate._scaler, candidate._vectorizer, candidate._kmeans
    mapping_before = dict(candidate._cluster_to_category)

    candidate.predict(other_df)  # must be transform-only

    assert candidate._scaler is scaler_before
    assert candidate._vectorizer is vectorizer_before
    assert candidate._kmeans is kmeans_before
    assert candidate._cluster_to_category == mapping_before


def test_mapping_built_only_from_train_labels_not_from_predict_calls():
    """If cluster->category mapping construction ever accidentally consulted
    a later dataset's labels, feeding a wildly different-labeled dataset to
    .predict() (which has no label column at all) would raise or change
    behavior. It must not: predict() only needs merchant/amount/date."""
    df = _synthetic_df()
    train_df = df.iloc[:150].reset_index(drop=True)
    unlabeled_val_df = df.iloc[150:][["date", "merchant", "amount"]].reset_index(drop=True)

    candidate = KMeansCandidate().fit(train_df)
    # Must not raise even though unlabeled_val_df has no true_category column --
    # proves predict() never looks at VALIDATION labels.
    preds = candidate.predict(unlabeled_val_df)
    assert len(preds) == len(unlabeled_val_df)


def test_two_candidates_fit_on_same_train_produce_identical_mapping():
    """Determinism check: same TRAIN data + same random_state must yield the
    same cluster->category mapping every time (Section 19 reproducibility)."""
    df = _synthetic_df()
    train_df = df.iloc[:150].reset_index(drop=True)

    c1 = KMeansCandidate().fit(train_df)
    c2 = KMeansCandidate().fit(train_df)
    assert c1._cluster_to_category == c2._cluster_to_category


def test_predict_on_final_test_like_rows_never_calls_fit():
    """Explicit contract check via monkeypatching: KMeans.fit/fit_predict
    must be called exactly once (during .fit()), never again during any
    number of subsequent .predict() calls simulating VALIDATION and
    FINAL_TEST partitions."""
    from sklearn.cluster import KMeans
    df = _synthetic_df()
    train_df = df.iloc[:150].reset_index(drop=True)
    val_df = df.iloc[150:185].reset_index(drop=True)
    final_df = df.iloc[185:].reset_index(drop=True)

    original_fit_predict = KMeans.fit_predict
    call_count = {"n": 0}

    def counting_fit_predict(self, *args, **kwargs):
        call_count["n"] += 1
        return original_fit_predict(self, *args, **kwargs)

    KMeans.fit_predict = counting_fit_predict
    try:
        candidate = KMeansCandidate().fit(train_df)
        assert call_count["n"] == 1
        candidate.predict(val_df)
        candidate.predict(final_df)
        assert call_count["n"] == 1, "predict() must never call fit_predict again"
    finally:
        KMeans.fit_predict = original_fit_predict
