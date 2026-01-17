import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    brier_score_loss, make_scorer,
    roc_auc_score, average_precision_score,
    f1_score, confusion_matrix, precision_recall_curve, roc_curve
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

warnings.simplefilter("ignore", ConvergenceWarning)

RANDOM_STATE = 42

# данные
main_df = pd.read_csv("data/processed/with_inner_folds.csv", sep=";")
main_df["Исход"] = (
    main_df["Исход"]
    .astype(str).str.strip()
    .replace({"благоприятный": 1, "неблагоприятный": 0})
    .astype(int)
)

# скореры
brier_scorer = make_scorer(
    brier_score_loss,
    response_method="predict_proba",
    greater_is_better=False
)

cv3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

# функции
def clean_numeric(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c in X.columns:
        s = (
            X[c].astype(str)
            .str.replace(",", ".", regex=False)
            .str.replace(" ", "", regex=False)
        )
        X[c] = pd.to_numeric(s, errors="coerce")
    return X

def enough_for_isotonic(y: pd.Series) -> bool:
    n = len(y)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    return (n >= 200) and (pos >= 30) and (neg >= 30)

def safe_scale_pos_weight(y: pd.Series) -> float:
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    return (neg / pos) if pos > 0 else 1.0

def serialize_params(d: dict) -> str:
    return json.dumps(d, ensure_ascii=False)

def sens_spec(y_true, y_pred_bin):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    return sens, spec

def compute_ece(y_true, p, n_bins=10):
    df = pd.DataFrame({"y": y_true, "p": p})
    df = df.sort_values("p").reset_index(drop=True)
    df["bin"] = pd.qcut(df.index, q=n_bins, labels=False)
    ece = 0.0
    n = len(df)
    for b in range(n_bins):
        d = df[df["bin"] == b]
        if len(d) == 0:
            continue
        mean_p = d["p"].mean()
        freq = d["y"].mean()
        ece += (len(d) / n) * abs(mean_p - freq)
    return ece

# модели
def make_lr():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            random_state=RANDOM_STATE
        ))
    ])
LR_GRID = {"clf__C": [1e-4, 1e-3, 1e-2, 1e-1]}

def make_mlp():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            max_iter=2000,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=RANDOM_STATE
        ))
    ])
MLP_GRID = {"clf__hidden_layer_sizes": [(32,), (64, 32)], "clf__alpha": [1e-4, 1e-3]}

def make_xgb(scale_pos_weight: float):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", XGBClassifier(
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])
XGB_GRID = {"clf__max_depth": [3,4], "clf__n_estimators": [50,100], "clf__learning_rate": [0.05,0.1]}

# хранилища
rows_inner = []
rows_all = []
outer_candidates = []

outer_test_metrics = []
outer_test_preds = []

# outer цикл
for outer in range(1,6):
    outer_train_df = main_df[main_df["fold_outer"] != outer].copy()
    outer_test_df  = main_df[main_df["fold_outer"] == outer].copy()

    for k in range(3,13):
        features_all = [
    c for c in outer_train_df.columns 
    if c not in ["Исход","fold_outer","fold_inner"] 
    and pd.api.types.is_numeric_dtype(outer_train_df[c])
]
        features = features_all[:k]

        inner_fold_nums = outer_train_df["fold_inner"].unique()
        inner_results = []

        for inner in inner_fold_nums:
            inner_train_df = outer_train_df[outer_train_df["fold_inner"] != inner].copy()
            inner_valid_df = outer_train_df[outer_train_df["fold_inner"] == inner].copy()

            X_train = clean_numeric(inner_train_df[features])
            y_train = inner_train_df["Исход"].astype(int)
            X_valid = clean_numeric(inner_valid_df[features])
            y_valid = inner_valid_df["Исход"].astype(int)

            if y_train.nunique()<2 or y_valid.nunique()<2:
                continue

            base_results = []

            lr_gs = GridSearchCV(make_lr(), LR_GRID, scoring=brier_scorer, cv=cv3, n_jobs=-1)
            lr_gs.fit(X_train, y_train)
            lr_best = lr_gs.best_estimator_
            lr_pred = lr_best.predict_proba(X_valid)[:,1]
            lr_brier = brier_score_loss(y_valid, lr_pred)
            base_results.append(("logreg", lr_best, lr_gs.best_params_, lr_brier))

            spw = safe_scale_pos_weight(y_train)
            xgb_gs = GridSearchCV(make_xgb(spw), XGB_GRID, scoring=brier_scorer, cv=cv3, n_jobs=-1)
            xgb_gs.fit(X_train, y_train)
            xgb_best = xgb_gs.best_estimator_
            xgb_pred = xgb_best.predict_proba(X_valid)[:,1]
            xgb_brier = brier_score_loss(y_valid, xgb_pred)
            base_results.append(("xgb", xgb_best, xgb_gs.best_params_, xgb_brier))

            mlp_gs = GridSearchCV(make_mlp(), MLP_GRID, scoring=brier_scorer, cv=cv3, n_jobs=-1)
            mlp_gs.fit(X_train, y_train)
            mlp_best = mlp_gs.best_estimator_
            mlp_pred = mlp_best.predict_proba(X_valid)[:,1]
            mlp_brier = brier_score_loss(y_valid, mlp_pred)
            base_results.append(("mlp", mlp_best, mlp_gs.best_params_, mlp_brier))

            base_results.sort(key=lambda t: t[3])
            best_name, best_estimator, best_params, best_brier = base_results[0]
            candidates = [("none", best_estimator, best_brier)]
            platt = CalibratedClassifierCV(best_estimator, method="sigmoid", cv=3)
            platt.fit(X_train, y_train)
            platt_pred = platt.predict_proba(X_valid)[:,1]
            candidates.append(("sigmoid", platt, brier_score_loss(y_valid, platt_pred)))
            if enough_for_isotonic(y_train):
                iso = CalibratedClassifierCV(best_estimator, method="isotonic", cv=3)
                iso.fit(X_train, y_train)
                iso_pred = iso.predict_proba(X_valid)[:,1]
                candidates.append(("isotonic", iso, brier_score_loss(y_valid, iso_pred)))
            candidates.sort(key=lambda t: t[2])
            best_calib, best_obj, best_brier_final = candidates[0]

            inner_results.append({
                "outer": outer, "inner": inner, "k": k,
                "best_model": best_name,
                "best_params": best_params,
                "best_calibration": best_calib,
                "brier_valid": best_brier_final
            })

        if not inner_results:
            continue
        # берем средний Brier по inner folds
        df_inner = pd.DataFrame(inner_results)
        best_row = df_inner.groupby("best_model")["brier_valid"].mean().sort_values().head(1)
        best_model_name = best_row.index[0]
        best_row_full = df_inner[df_inner["best_model"]==best_model_name].iloc[0]
        model_name = best_row_full["best_model"]
        best_params = best_row_full["best_params"]
        calib = best_row_full["best_calibration"]

        X_train_outer = clean_numeric(outer_train_df[features])
        y_train_outer = outer_train_df["Исход"].astype(int)
        X_test_outer  = clean_numeric(outer_test_df[features])
        y_test_outer  = outer_test_df["Исход"].astype(int)

        if model_name=="logreg":
            est = make_lr(); est.set_params(**best_params)
        elif model_name=="mlp":
            est = make_mlp(); est.set_params(**best_params)
        elif model_name=="xgb":
            spw = safe_scale_pos_weight(y_train_outer)
            est = make_xgb(spw); est.set_params(**best_params)

        if calib=="none":
            est.fit(X_train_outer, y_train_outer)
            p = est.predict_proba(X_test_outer)[:,1]
        else:
            cal = CalibratedClassifierCV(est, method=calib, cv=3)
            cal.fit(X_train_outer, y_train_outer)
            p = cal.predict_proba(X_test_outer)[:,1]

        y_pred_bin = (p>=0.5).astype(int)
        brier = brier_score_loss(y_test_outer, p)
        roc   = roc_auc_score(y_test_outer, p)
        pr    = average_precision_score(y_test_outer, p)
        f1    = f1_score(y_test_outer, y_pred_bin)
        sens, spec = sens_spec(y_test_outer, y_pred_bin)
        ece = compute_ece(y_test_outer.values, p)

        outer_test_metrics.append({
            "outer": outer, "k": k,
            "model": model_name, "calibration": calib,
            "brier": brier, "roc_auc": roc, "pr_auc": pr,
            "f1": f1, "sensitivity": sens, "specificity": spec,
            "ece": ece
        })

        for idx, prob in zip(outer_test_df.index, p):
            outer_test_preds.append({"outer": outer, "row_index": idx,
                                     "y_true": int(main_df.loc[idx,"Исход"]),
                                     "y_pred_proba": float(prob)})

pd.DataFrame(outer_test_metrics).to_csv("outer_test_metrics.csv", index=False)
pd.DataFrame(outer_test_preds).to_csv("outer_test_predictions.csv", index=False)

df = pd.DataFrame(outer_test_metrics)
agg = df.groupby("k").agg(
    brier_mean=("brier","mean"), brier_sd=("brier","std"),
    roc_auc_mean=("roc_auc","mean"), roc_auc_sd=("roc_auc","std"),
    pr_auc_mean=("pr_auc","mean"), pr_auc_sd=("pr_auc","std"),
    f1_mean=("f1","mean"), f1_sd=("f1","std"),
    sens_mean=("sensitivity","mean"), sens_sd=("sensitivity","std"),
    spec_mean=("specificity","mean"), spec_sd=("specificity","std"),
    ece_mean=("ece","mean"), ece_sd=("ece","std")
).reset_index()
agg.to_csv("reports/tables/k_scan_metrics.csv", index=False)

preds = pd.DataFrame(outer_test_preds)
preds = preds.sort_values("y_pred_proba").reset_index(drop=True)
preds["bin"] = pd.qcut(preds.index, q=10, labels=False)
bins = [(d["y_pred_proba"].mean(), d["y_true"].mean()) for i,d in preds.groupby("bin")]
x,y = zip(*bins)
plt.figure()
plt.plot(x,y,"o")
plt.plot([0,1],[0,1],"--")
plt.xlabel("Mean predicted probability")
plt.ylabel("Observed frequency")
plt.title("Reliability diagram")
plt.savefig("reports/figures/reliability_plot.png")
plt.close()

# roc
plt.figure()
for outer, d in preds.groupby("outer"):
    fpr,tpr,_ = roc_curve(d["y_true"], d["y_pred_proba"])
    plt.plot(fpr,tpr, alpha=0.6)
plt.plot([0,1],[0,1],"--")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC curves (outer folds)")
plt.savefig("reports/figures/roc_by_model.png")
plt.close()

# pr
plt.figure()
for outer, d in preds.groupby("outer"):
    prec, rec,_ = precision_recall_curve(d["y_true"], d["y_pred_proba"])
    plt.plot(rec, prec, alpha=0.6)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("PR curves (outer folds)")
plt.savefig("reports/figures/pr_by_model.png")
plt.close()
