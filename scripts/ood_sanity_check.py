import pandas as pd
import numpy as np
from scipy.spatial.distance import mahalanobis

preds = pd.read_csv("reports/tables/outer_test_predictions_agg.csv")
features = pd.read_csv("data/processed/with_inner_folds.csv", sep=";")

features = features.assign(row_index=lambda df: df.index)

target_col = features.columns[0]
exclude_cols = [target_col, "fold_outer", "fold_inner", "row_index"]
feature_cols = features.select_dtypes(include=[np.number]).columns
feature_cols = [c for c in feature_cols if c not in exclude_cols]

data = preds.merge(
    features[["row_index"] + feature_cols],
    on="row_index",
    how="left"
)

X_outer = data[feature_cols].values
X_train = features[feature_cols].values

mu = X_train.mean(axis=0)
cov = np.cov(X_train, rowvar=False)
cov_inv = np.linalg.pinv(cov) 

data["mahalanobis"] = [
    mahalanobis(x, mu, cov_inv) for x in X_outer
]

threshold = np.percentile(data["mahalanobis"], 90)
data["is_ood"] = data["mahalanobis"] >= threshold

T_low, T_high = 0.35, 0.55
data["undetermined"] = (
    (data["p"] >= T_low) &
    (data["p"] <= T_high)
)

summary = {
    "OOD_rate": data["is_ood"].mean(),
    "Undetermined_rate": data["undetermined"].mean(),
    "OOD_and_undetermined": (
        (data["is_ood"] & data["undetermined"]).mean()
    ),
    "Undetermined_that_are_OOD": (
        (data["is_ood"] & data["undetermined"]).sum() /
        max(1, data["undetermined"].sum())
    )
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv("reports/tables/ood_overlap.csv", index=False)

