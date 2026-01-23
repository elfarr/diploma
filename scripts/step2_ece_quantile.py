import pandas as pd
import numpy as np
import os

N_BINS = 10

def compute_ece_quantile(df, n_bins=10):
    df = df.sort_values("p_cal").reset_index(drop=True)
    N = len(df)

    df["bin"] = pd.qcut(
        df["p_cal"],
        q=n_bins,
        labels=False,
        duplicates="drop"
    )

    bin_stats = (
        df
        .groupby("bin")
        .agg(
            mean_p=("p_cal", "mean"),
            mean_y=("y_true", "mean"),
            n_bin=("y_true", "size")
        )
        .reset_index()
    )

    bin_stats["weight"] = bin_stats["n_bin"] / N
    bin_stats["abs_diff"] = (bin_stats["mean_p"] - bin_stats["mean_y"]).abs()
    bin_stats["ece_term"] = bin_stats["weight"] * bin_stats["abs_diff"]

    ece = bin_stats["ece_term"].sum()

    return ece, bin_stats

IN_DIR = "reports/tables"
OUT_DIR = "reports/tables"
MODEL_TAG = "xgb_sigmoid"

ece_rows = []

for i in range(1, 6):
    path = os.path.join(IN_DIR, f"preds_{MODEL_TAG}_fold{i}.csv")
    df = pd.read_csv(path)

    ece, bin_table = compute_ece_quantile(df, n_bins=10)
    bin_path = os.path.join(
        OUT_DIR,
        f"ece_bins_{MODEL_TAG}_fold{i}.csv"
    )
    bin_table.to_csv(bin_path, index=False)

    ece_rows.append({
        "fold": i,
        "ece": ece
    })

ece_df = pd.DataFrame(ece_rows)
ece_df.to_csv(
    os.path.join(OUT_DIR, f"ece_{MODEL_TAG}_per_fold.csv"),
    index=False
)
