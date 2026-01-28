import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import shap

T_LOW, T_HIGH = 0.35, 0.55

PRED_PATH = "reports/tables/outer_test_predictions.csv"
DATA_PATH = "data/processed/with_inner_folds.csv"
SEP = ";"

OUT_TABLE = "reports/tables/patients_level_preds.csv"
OUT_SHAP_TOP = "reports/tables/shap_top.csv"
FIG_DIR = "reports/figs"

os.makedirs("reports/tables", exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

def status_by_risk(p_bad: float) -> str:
    if p_bad < T_LOW:
        return "низкий риск"
    if p_bad > T_HIGH:
        return "высокий риск"
    return "неопределённо"

def to_float_series(s: pd.Series) -> pd.Series:
    s2 = (
        s.astype(str)
        .str.replace(",", ".", regex=False)
        .str.replace(" ", "", regex=False)
    )
    return pd.to_numeric(s2, errors="coerce")

def logit_to_proba(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))

preds = pd.read_csv(PRED_PATH)
df = pd.read_csv(DATA_PATH, sep=SEP)

df = df.reset_index().rename(columns={"index": "row_index"})
target_col = "Исход"

p_patient = preds.groupby("row_index", as_index=False).agg(
    p_good=("y_pred_proba", "mean"),
    y_true=("y_true", "first"),
)

p_patient["p_bad"] = 1.0 - p_patient["p_good"]
p_patient["status"] = p_patient["p_bad"].apply(status_by_risk)

feature_cols = [c for c in df.columns if c not in [
    "row_index", target_col, "fold_outer", "fold_inner"
]]

patients = p_patient.merge(
    df[["row_index"] + feature_cols],
    on="row_index",
    how="left"
)

patients.to_csv(OUT_TABLE, index=False, encoding="utf-8-sig")
exclude_cols = {"row_index", "y_true", "p_good", "p_bad", "status"}
X_raw = patients[[c for c in patients.columns if c not in exclude_cols]].copy()
y = patients["y_true"].astype(int)

obj_cols = X_raw.select_dtypes(include="object").columns
for c in obj_cols:
    conv = to_float_series(X_raw[c])
    if conv.notna().mean() >= 0.5:
        X_raw[c] = conv

cat_cols = X_raw.select_dtypes(include="object").columns
num_cols = [c for c in X_raw.columns if c not in cat_cols]

for c in num_cols:
    X_raw[c] = pd.to_numeric(X_raw[c], errors="coerce")
    X_raw[c] = X_raw[c].fillna(X_raw[c].median())

X = pd.get_dummies(X_raw, columns=cat_cols, drop_first=False)
X = X.apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)

clf = XGBClassifier(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    random_state=42,
    eval_metric="logloss"
)
clf.fit(X, y)

background = X.sample(min(100, len(X)), random_state=42)
explainer = shap.Explainer(clf, background, feature_names=X.columns)
shap_values = explainer(X)

sv = shap_values.values
if sv.ndim == 3:
    sv = sv[:, :, 1] 

mean_abs = np.abs(sv).mean(axis=0)
shap_top = (
    pd.DataFrame({"feature": X.columns, "mean_abs_shap": mean_abs})
    .sort_values("mean_abs_shap", ascending=False)
)
shap_top.to_csv(OUT_SHAP_TOP, index=False, encoding="utf-8-sig")
row_to_pos = {ri: i for i, ri in enumerate(patients["row_index"])}

for j, r in enumerate(patients.sample(3, random_state=42)["row_index"], start=1):
    i = row_to_pos[r]

    p_good = patients.loc[patients["row_index"] == r, "p_good"].iloc[0]
    p_bad = 1 - p_good
    y_true = patients.loc[patients["row_index"] == r, "y_true"].iloc[0]
    status = patients.loc[patients["row_index"] == r, "status"].iloc[0]

    row_sv = pd.Series(sv[i], index=X.columns)
    up = row_sv.sort_values().head(3)
    down = row_sv.sort_values(ascending=False).head(3)

    lines = [
        f"Пациент (row_index = {r})",
        f"Истинный исход: y_true={y_true} (0=неблагоприятный, 1=благоприятный)",
        f"Вероятность благоприятного исхода (p_good): {p_good:.3f}",
        f"Риск неблагоприятного исхода (p_bad): {p_bad:.3f}",
        f"Режим решения: {status}",
        "",
        "Факторы, ПОВЫШАЮЩИЕ риск неблагоприятного исхода:",
    ]

    for f, v in up.items():
        lines.append(f"• {f}: вклад {abs(v):.3f}")

    lines.append("")
    lines.append("Факторы, СНИЖАЮЩИЕ риск неблагоприятного исхода:")

    for f, v in down.items():
        lines.append(f"• {f}: вклад {abs(v):.3f}")

    fig = plt.figure(figsize=(12, 6))
    plt.axis("off")
    fig.text(0.02, 0.98, "\n".join(lines), va="top", fontsize=12)
    plt.savefig(f"{FIG_DIR}/explain_case_{j}.png", dpi=200, bbox_inches="tight")
    plt.close()
