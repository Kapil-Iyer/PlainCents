"""
Phase 2: Feature engineering for K-Means clustering.
Amount (StandardScaler), TF-IDF merchant (top 50), day-of-week, is_weekend.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, normalize


def build_feature_matrix(
    df: pd.DataFrame,
    scaler: StandardScaler | None = None,
    vectorizer: TfidfVectorizer | None = None,
    fit: bool = True,
):
    """
    Build feature matrix for clustering.
    Order: amount_scaled, merchant_tfidf(50), day_of_week, is_weekend.

    Parameters
    ----------
    df : DataFrame
        Must have columns: date (YYYY-MM-DD str), merchant, amount.
    scaler : StandardScaler or None
        If provided and fit=False, use to transform amount. If fit=True, fit new one.
    vectorizer : TfidfVectorizer or None
        If provided and fit=False, use to transform merchant. If fit=True, fit new one.
    fit : bool
        If True, fit scaler and vectorizer on this data. If False, use provided scaler/vectorizer.

    Returns
    -------
    X : np.ndarray
        Dense matrix for KMeans.
    scaler : StandardScaler
        Fitted (or passed through).
    vectorizer : TfidfVectorizer
        Fitted (or passed through).
    """
    if df.empty or set(["date", "merchant", "amount"]) - set(df.columns):
        raise ValueError("DataFrame must have columns: date, merchant, amount")

    dates = pd.to_datetime(df["date"], format="%Y-%m-%d")
    day_of_week = dates.dt.dayofweek.values.astype(np.float64).reshape(-1, 1)
    is_weekend = (day_of_week >= 5).astype(np.float64).reshape(-1, 1)

    amount = df["amount"].values.astype(np.float64).reshape(-1, 1)
    merchant = df["merchant"].fillna("").astype(str)

    if fit:
        scaler = StandardScaler()
        vectorizer = TfidfVectorizer(max_features=50, token_pattern=r"(?u)\b[a-zA-Z]{2,}\b", ngram_range=(1, 2), sublinear_tf=True)
        amount_scaled = scaler.fit_transform(amount)
        merchant_tfidf = vectorizer.fit_transform(merchant)
    else:
        if scaler is None or vectorizer is None:
            raise ValueError("When fit=False, scaler and vectorizer must be provided")
        amount_scaled = scaler.transform(amount)
        merchant_tfidf = vectorizer.transform(merchant)

    merchant_dense = np.asarray(merchant_tfidf.todense())
    merchant_dense = normalize(merchant_dense, norm='l2')
    amount_weighted = amount_scaled * 0.2
    dow_weighted = day_of_week * 0.1
    weekend_weighted = is_weekend * 0.1
    X = np.hstack([amount_weighted, merchant_dense, dow_weighted, weekend_weighted])
    return X, scaler, vectorizer
