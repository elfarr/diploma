import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

MODEL_TAG = "xgb_sigmoid"
IN_DIR = "reports/tables"
OUT_DIR = "reports/figures"
os.makedirs(OUT_DIR, exist_ok=True)

def plot_reliability(bin_df: pd.DataFrame, out_path: str, title: str):
    x = bin_df["mean_p"].values
    y = bin_df["mean_y"].values

    grid = np.linspace(0, 1, 200)

    plt.figure()
    plt.plot(grid, grid, linestyle="--")   
    plt.plot(x, y, marker="o")            
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Средняя предсказанная вероятность")
    plt.ylabel("Наблюдаемая доля благоприятного исхода")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

for i in range(1, 6):
    path = os.path.join(IN_DIR, f"ece_bins_{MODEL_TAG}_fold{i}.csv")
    bin_df = pd.read_csv(path)

    out_path = os.path.join(OUT_DIR, f"reliability_fold{i}.png")
    plot_reliability(
        bin_df,
        out_path,
        title=f"Идеальная калибровка (fold {i})"
    )

preds = []
for i in range(1, 6):
    p_path = os.path.join(IN_DIR, f"preds_{MODEL_TAG}_fold{i}.csv")
    preds.append(pd.read_csv(p_path))
pooled = pd.concat(preds, ignore_index=True)

pooled = pooled.sort_values("p_cal").reset_index(drop=True)
pooled["bin"] = pd.qcut(pooled["p_cal"], q=10, labels=False, duplicates="drop")

pooled_bins = (
    pooled.groupby("bin")
    .agg(mean_p=("p_cal", "mean"), mean_y=("y_true", "mean"), n_bin=("y_true", "size"))
    .reset_index()
)

out_path = os.path.join(OUT_DIR, "reliability_pooled.png")
plot_reliability(pooled_bins, out_path, title="Идеальная калибровка (pooled)")

