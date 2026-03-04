"""Diagnostic: show which held-out rows are misclassified and why."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import random
from pipeline.ingest import load_and_clean
from pipeline.features import build_feature_matrix
from pipeline.cluster import _get_true_labels
from sklearn.cluster import KMeans

df = load_and_clean("synthetic_12mo.csv", bank="TD")
X, scaler, vectorizer = build_feature_matrix(df, fit=True)
kmeans = KMeans(n_clusters=12, random_state=42, n_init=20)
df = df.copy()
df["cluster_id"] = kmeans.fit_predict(X)
df["true_label"] = _get_true_labels(df)

labeled = df.reset_index(drop=True)
indices = labeled.index.tolist()
random.seed(42)
random.shuffle(indices)
mapping_indices = indices[:160]
eval_indices = indices[160:200]

cluster_to_category = {}
for cid in range(12):
    subset = labeled.loc[mapping_indices]
    subset = subset[subset["cluster_id"] == cid]
    if subset.empty:
        cluster_to_category[cid] = "Food & Dining"
        continue
    cluster_to_category[cid] = subset["true_label"].value_counts().index[0]

eval_df = labeled.loc[eval_indices].copy()
eval_df["predicted"] = eval_df["cluster_id"].map(cluster_to_category)
wrong = eval_df[eval_df["predicted"] != eval_df["true_label"]]

print(f"Total wrong: {len(wrong)} / 40\n")
print("Misclassified rows:")
for _, row in wrong.iterrows():
    print(f"  merchant={row['merchant']:<35} true={row['true_label']:<20} pred={row['predicted']:<20} cluster={row['cluster_id']}")

print("\nCategory breakdown of errors:")
print(wrong["true_label"].value_counts().to_string())
