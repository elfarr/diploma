import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PRED_PATH = "outer_test_predictions.csv"
OUT_FIG = "reports/figs/dca_curves.png"
OUT_CSV = "reports/tables/dca_values.csv"

T_LOW = 0.35
T_HIGH = 0.55

P_COL = "p_cal"
Y_COL = "y_true"

pt_values = np.arange(0.10, 0.90 + 1e-9, 0.01)

df = pd.read_csv(PRED_PATH)

if P_COL not in df.columns and "y_pred_proba" in df.columns:
    df = df.rename(columns={"y_pred_proba": P_COL})

df = df.dropna(subset=[Y_COL, P_COL]).copy()
df[Y_COL] = df[Y_COL].astype(int)
df[P_COL] = df[P_COL].astype(float)

y = df[Y_COL].values
p = df[P_COL].values
N = len(df)

defined_mask = (p < T_LOW) | (p >= T_HIGH)
y_def = y[defined_mask]
p_def = p[defined_mask]

coverage = len(y_def) / N

def net_benefit(y_true, y_pred, pt):
    # NB = TP/N - FP/N * pt/(1-pt)
    if len(y_true) == 0:
        return np.nan

    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    w = pt / (1 - pt)

    return tp / len(y_true) - fp / len(y_true) * w

rows = []
nb_all = []
nb_def = []
nb_all_treat = []
nb_none = []

for pt in pt_values:
    pred_all = (p >= pt).astype(int)
    nb1 = net_benefit(y, pred_all, pt)
    pred_def = (p_def >= pt).astype(int)
    nb2 = net_benefit(y_def, pred_def, pt)
    nb3 = net_benefit(y, np.ones_like(y), pt)
    nb4 = 0.0

    nb_all.append(nb1)
    nb_def.append(nb2)
    nb_all_treat.append(nb3)
    nb_none.append(nb4)

    rows.append({
        "p_t": float(pt),
        "NB_model_all": nb1,
        "NB_model_defined": nb2,
        "NB_treat_all": nb3,
        "NB_treat_none": nb4
    })

out = pd.DataFrame(rows)

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
out.to_csv(OUT_CSV, index=False)

plt.figure(figsize=(9, 6))
plt.plot(pt_values, nb_all, label="Модель (все наблюдения)")
plt.plot(pt_values, nb_def, label="Модель (только определённые)")
plt.plot(pt_values, nb_all_treat, label="Лечить всех")
plt.plot(pt_values, nb_none, label="Не лечить никого")

plt.xlabel("Порог вмешательства pₜ")
plt.ylabel("Чистая польза (Net Benefit)")
plt.title("Decision Curve Analysis")
plt.grid(True, alpha=0.25)
plt.legend()
plt.tight_layout()

os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
plt.savefig(OUT_FIG, dpi=200)
plt.close()