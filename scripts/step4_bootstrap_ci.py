import os
import numpy as np
import pandas as pd

N_BINS = 10
N_BOOT = 1000
RANDOM_STATE = 42

IN_DIR = "reports/tables"
OUT_DIR = "reports/tables"
MODEL_TAG = "xgb_sigmoid"

rng = np.random.default_rng(RANDOM_STATE)

def brier_score(y_true, p):
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    return np.mean((p - y_true) ** 2)

def ece_quantile(y_true, p, n_bins=10):
    df = pd.DataFrame({"y_true": y_true, "p": p}).sort_values("p").reset_index(drop=True)
    n = len(df)

    df["bin"] = pd.qcut(df["p"], q=n_bins, labels=False, duplicates="drop")

    bins = (
        df.groupby("bin")
        .agg(mean_p=("p", "mean"), mean_y=("y_true", "mean"), n_bin=("y_true", "size"))
        .reset_index()
    )

    bins["weight"] = bins["n_bin"] / n
    ece = np.sum(bins["weight"] * np.abs(bins["mean_p"] - bins["mean_y"]))
    return float(ece)

def bootstrap_ci(y_true, p, n_boot=1000, alpha=0.05):
    y_true = np.asarray(y_true)
    p = np.asarray(p)
    n = len(y_true)

    briers = np.empty(n_boot, dtype=float)
    eces = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)  
        y_b = y_true[idx]
        p_b = p[idx]
        briers[b] = brier_score(y_b, p_b)
        eces[b] = ece_quantile(y_b, p_b, n_bins=N_BINS)

    lo = 100 * (alpha / 2)
    hi = 100 * (1 - alpha / 2)

    return {
        "brier_boot": briers,
        "ece_boot": eces,
        "brier_ci": (np.percentile(briers, lo), np.percentile(briers, hi)),
        "ece_ci": (np.percentile(eces, lo), np.percentile(eces, hi)),
    }

per_fold_rows = []
all_y = []
all_p = []

for i in range(1, 6):
    path = os.path.join(IN_DIR, f"preds_{MODEL_TAG}_fold{i}.csv")
    df = pd.read_csv(path)

    y = df["y_true"].astype(int).values
    p = df["p_cal"].astype(float).values

    all_y.append(y)
    all_p.append(p)

    brier = brier_score(y, p)
    ece = ece_quantile(y, p, n_bins=N_BINS)

    boot = bootstrap_ci(y, p, n_boot=N_BOOT)

    per_fold_rows.append({
        "fold": i,
        "brier": brier,
        "brier_ci_2_5": boot["brier_ci"][0],
        "brier_ci_97_5": boot["brier_ci"][1],
        "ece": ece,
        "ece_ci_2_5": boot["ece_ci"][0],
        "ece_ci_97_5": boot["ece_ci"][1],
        "n": len(df)
    })

per_fold_df = pd.DataFrame(per_fold_rows)
per_fold_df.to_csv(os.path.join(OUT_DIR, f"brier_ece_per_fold_{MODEL_TAG}.csv"), index=False)

Y_all = np.concatenate(all_y)
P_all = np.concatenate(all_p)

brier_pooled = brier_score(Y_all, P_all)
ece_pooled = ece_quantile(Y_all, P_all, n_bins=N_BINS)
boot_pooled = bootstrap_ci(Y_all, P_all, n_boot=N_BOOT)

brier_mean_fold = per_fold_df["brier"].mean()
ece_mean_fold = per_fold_df["ece"].mean()

summary = pd.DataFrame([
    {
        "scope": "pooled_test",
        "metric": "brier",
        "mean": brier_pooled,
        "ci_2_5": boot_pooled["brier_ci"][0],
        "ci_97_5": boot_pooled["brier_ci"][1],
        "n": len(Y_all)
    },
    {
        "scope": "pooled_test",
        "metric": "ece",
        "mean": ece_pooled,
        "ci_2_5": boot_pooled["ece_ci"][0],
        "ci_97_5": boot_pooled["ece_ci"][1],
        "n": len(Y_all)
    },
    {
        "scope": "mean_over_folds",
        "metric": "brier",
        "mean": brier_mean_fold,
        "ci_2_5": np.nan,
        "ci_97_5": np.nan,
        "n": int(per_fold_df["n"].sum())
    },
    {
        "scope": "mean_over_folds",
        "metric": "ece",
        "mean": ece_mean_fold,
        "ci_2_5": np.nan,
        "ci_97_5": np.nan,
        "n": int(per_fold_df["n"].sum())
    },
])

summary.to_csv(os.path.join(OUT_DIR, f"brier_ece_ci_{MODEL_TAG}.csv"), index=False)