import os
import pandas as pd

SRC = "outer_test_predictions.csv"
OUT_DIR = "reports/tables"
MODEL_TAG = "xgb_sigmoid" 

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(SRC)

required = {"outer", "y_true", "y_pred_proba"}
missing = required - set(df.columns)
if missing:
    raise ValueError()

df["outer"] = df["outer"].astype(int)
df["y_true"] = df["y_true"].astype(int)

for i in range(1, 6):
    fold = df[df["outer"] == i][["y_true", "y_pred_proba"]].copy()
    if fold.empty:
        raise RuntimeError()

    fold = fold.rename(columns={"y_pred_proba": "p_cal"})

    out_path = os.path.join(OUT_DIR, f"preds_{MODEL_TAG}_fold{i}.csv")
    fold.to_csv(out_path, index=False)