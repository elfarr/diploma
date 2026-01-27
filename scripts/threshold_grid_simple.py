import os
import numpy as np
import pandas as pd

PRED_PATH = "reports/tables/outer_test_predictions_agg.csv"  
OUT_PATH = "reports/tables/grid_thresholds.csv"

t_low_values  = np.arange(0.20, 0.50 + 1e-9, 0.05)
t_high_values = np.arange(0.50, 0.80 + 1e-9, 0.05)

P_COL = "p"
Y_COL = "y_true"

df = pd.read_csv(PRED_PATH)

if P_COL not in df.columns and "y_pred_proba" in df.columns:
    df = df.rename(columns={"y_pred_proba": P_COL})

df = df.dropna(subset=[Y_COL, P_COL]).copy()
df[Y_COL] = df[Y_COL].astype(int)
df[P_COL] = df[P_COL].astype(float)

def calc_metrics_defined(y_true, y_pred):
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()

    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    ppv  = tp / (tp + fp) if (tp + fp) else np.nan
    npv  = tn / (tn + fn) if (tn + fn) else np.nan

    if (ppv is not np.nan) and (sens is not np.nan) and (ppv + sens):
        f1 = 2 * ppv * sens / (ppv + sens)
    else:
        f1 = np.nan

    return sens, spec, ppv, npv, f1

rows = []

N = len(df)

for t_low in t_low_values:
    for t_high in t_high_values:
        if not (t_low < t_high):
            continue

        p = df[P_COL].values
        y = df[Y_COL].values

        low_mask = p < t_low
        high_mask = p >= t_high
        und_mask = (~low_mask) & (~high_mask)

        rate_low = low_mask.mean()
        rate_high = high_mask.mean()
        rate_und = und_mask.mean()
        coverage = 1.0 - rate_und

        defined_mask = low_mask | high_mask
        y_def = y[defined_mask]

        pred_def = np.zeros_like(y_def)
        pred_def[p[defined_mask] >= t_high] = 1
        sens, spec, ppv, npv, f1 = calc_metrics_defined(y_def, pred_def) if len(y_def) else (np.nan,)*5

        rows.append({
            "T_low": float(t_low),
            "T_high": float(t_high),
            "coverage": float(coverage),
            "sens": float(sens) if sens == sens else np.nan,
            "spec": float(spec) if spec == spec else np.nan,
            "PPV": float(ppv) if ppv == ppv else np.nan,
            "NPV": float(npv) if npv == npv else np.nan,
            "F1": float(f1) if f1 == f1 else np.nan,
            "rate_low": float(rate_low),
            "rate_undetermined": float(rate_und),
            "rate_high": float(rate_high),
            "n_defined": int(defined_mask.sum())
        })

out = pd.DataFrame(rows)
out = out.sort_values(["coverage", "F1"], ascending=[False, False]).reset_index(drop=True)

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
out.to_csv(OUT_PATH, index=False)
