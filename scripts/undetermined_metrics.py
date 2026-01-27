import pandas as pd
from sklearn.metrics import confusion_matrix

T_LOW = 0.35
T_HIGH = 0.55

df = pd.read_csv("reports/tables/outer_test_predictions_agg.csv")

df = df.rename(columns={"y_pred_proba": "p"})

def assign_zone(p):
    if p < T_LOW:
        return "low"
    elif p > T_HIGH:
        return "high"
    else:
        return "undetermined"

df["zone"] = df["p"].apply(assign_zone)

defined = df[df["zone"] != "undetermined"].copy()

coverage = len(defined) / len(df)
rate_undetermined = 1 - coverage

defined["y_pred"] = (defined["zone"] == "high").astype(int)

tn, fp, fn, tp = confusion_matrix(defined["y_true"], defined["y_pred"]).ravel()

sens = tp / (tp + fn) if (tp + fn) else 0.0
spec = tn / (tn + fp) if (tn + fp) else 0.0
ppv  = tp / (tp + fp) if (tp + fp) else 0.0
npv  = tn / (tn + fn) if (tn + fn) else 0.0
f1   = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0

out = pd.DataFrame([{
    "T_low": T_LOW,
    "T_high": T_HIGH,
    "coverage": coverage,
    "rate_undetermined": rate_undetermined,
    "sens": sens,
    "spec": spec,
    "PPV": ppv,
    "NPV": npv,
    "F1": f1,
    "n_total": len(df),
    "n_defined": len(defined),
    "n_undetermined": len(df) - len(defined)
}])

out.to_csv("reports/tables/undetermined_metrics.csv", index=False)
conf = pd.DataFrame(
    {"pred=1": [tp, fn], "pred=0": [fp, tn]},
    index=["y_true=1", "y_true=0"]
)
conf.to_csv("reports/tables/confusion_defined.csv")
