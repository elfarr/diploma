import json
import warnings
import numpy as np
import pandas as pd

from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import brier_score_loss, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV

from xgboost import XGBClassifier

warnings.simplefilter("ignore", ConvergenceWarning)

RANDOM_STATE = 42

main_df = pd.read_csv("data/processed/with_inner_folds.csv", sep=";")
results_df = pd.read_csv("nested_cv_results.csv")
results_df["selected_features"] = results_df["selected_features"].apply(eval)

main_df["Исход"] = (
    main_df["Исход"]
    .astype(str).str.strip()
    .replace({"благоприятный": 1, "неблагоприятный": 0})
    .astype(int)
)

brier_scorer = make_scorer(
    brier_score_loss,
    response_method="predict_proba",
    greater_is_better=False
)

cv3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

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

MLP_GRID = {
    "clf__hidden_layer_sizes": [(32,), (64, 32)],
    "clf__alpha": [1e-4, 1e-3]  
}

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

XGB_GRID = {
    "clf__max_depth": [3, 4],
    "clf__n_estimators": [50, 100],
    "clf__learning_rate": [0.05, 0.1]
}

rows_inner = []  
rows_all = []    

outer_candidates = []

HAS_SELECTOR = "selector" in results_df.columns
HAS_K = "k" in results_df.columns

for _, row in results_df.iterrows():
    outer = int(row["outer_fold"])
    inner = int(row["inner_fold"])
    features = list(row["selected_features"])

    outer_train_df = main_df[main_df["fold_outer"] != outer].copy()
    outer_test_df  = main_df[main_df["fold_outer"] == outer].copy()

    inner_train_df = outer_train_df[outer_train_df["fold_inner"] != inner].copy()
    inner_valid_df = outer_train_df[outer_train_df["fold_inner"] == inner].copy()

    X_train = clean_numeric(inner_train_df[features])
    y_train = inner_train_df["Исход"].astype(int)

    X_valid = clean_numeric(inner_valid_df[features])
    y_valid = inner_valid_df["Исход"].astype(int)

    if y_train.nunique() < 2 or y_valid.nunique() < 2:
        print(f"skip outer={outer} inner={inner}: one class in train/valid")
        continue

    selector = row["selector"] if HAS_SELECTOR else "unknown_selector"
    k = int(row["k"]) if HAS_K else -1
    base_results = []

    lr_gs = GridSearchCV(make_lr(), LR_GRID, scoring=brier_scorer, cv=cv3, n_jobs=-1)
    lr_gs.fit(X_train, y_train)
    lr_best = lr_gs.best_estimator_
    lr_pred = lr_best.predict_proba(X_valid)[:, 1]
    lr_brier = brier_score_loss(y_valid, lr_pred)
    base_results.append(("logreg", lr_best, lr_gs.best_params_, lr_brier))

    spw = safe_scale_pos_weight(y_train)
    xgb_gs = GridSearchCV(make_xgb(spw), XGB_GRID, scoring=brier_scorer, cv=cv3, n_jobs=-1)
    xgb_gs.fit(X_train, y_train)
    xgb_best = xgb_gs.best_estimator_
    xgb_pred = xgb_best.predict_proba(X_valid)[:, 1]
    xgb_brier = brier_score_loss(y_valid, xgb_pred)
    base_results.append(("xgb", xgb_best, xgb_gs.best_params_, xgb_brier))

    mlp_gs = GridSearchCV(make_mlp(), MLP_GRID, scoring=brier_scorer, cv=cv3, n_jobs=-1)
    mlp_gs.fit(X_train, y_train)
    mlp_best = mlp_gs.best_estimator_
    mlp_pred = mlp_best.predict_proba(X_valid)[:, 1]
    mlp_brier = brier_score_loss(y_valid, mlp_pred)
    base_results.append(("mlp", mlp_best, mlp_gs.best_params_, mlp_brier))

    for name, _, params, brier in base_results:
        rows_all.append({
            "outer": outer, "inner": inner, "selector": selector, "k": k,
            "features": json.dumps(features, ensure_ascii=False),
            "model": name, "calibration": "none",
            "params": serialize_params(params),
            "brier_valid": brier
        })

    base_results.sort(key=lambda t: t[3])  
    best_name, best_estimator, best_params, best_brier = base_results[0]

    candidates = [("none", best_estimator, best_brier)]
    platt = CalibratedClassifierCV(
        estimator=best_estimator,
        method="sigmoid",
        cv=3
    )
    platt.fit(X_train, y_train)
    platt_pred = platt.predict_proba(X_valid)[:, 1]
    platt_brier = brier_score_loss(y_valid, platt_pred)
    candidates.append(("platt", platt, platt_brier))
    rows_all.append({
        "outer": outer, "inner": inner, "selector": selector, "k": k,
        "features": json.dumps(features, ensure_ascii=False),
        "model": best_name, "calibration": "platt",
        "params": serialize_params(best_params),
        "brier_valid": platt_brier
    })

    if enough_for_isotonic(y_train):
        iso = CalibratedClassifierCV(
            estimator=best_estimator,
            method="isotonic",
            cv=3
        )
        iso.fit(X_train, y_train)
        iso_pred = iso.predict_proba(X_valid)[:, 1]
        iso_brier = brier_score_loss(y_valid, iso_pred)
        candidates.append(("isotonic", iso, iso_brier))
        rows_all.append({
            "outer": outer, "inner": inner, "selector": selector, "k": k,
            "features": json.dumps(features, ensure_ascii=False),
            "model": best_name, "calibration": "isotonic",
            "params": serialize_params(best_params),
            "brier_valid": iso_brier
        })
    else:
        rows_all.append({
            "outer": outer, "inner": inner, "selector": selector, "k": k,
            "features": json.dumps(features, ensure_ascii=False),
            "model": best_name, "calibration": "isotonic_skipped",
            "params": serialize_params(best_params),
            "brier_valid": np.nan
        })

    candidates.sort(key=lambda t: t[2])
    best_calib, best_obj, best_brier_final = candidates[0]

    rows_inner.append({
        "outer": outer, "inner": inner, "selector": selector, "k": k,
        "features": json.dumps(features, ensure_ascii=False),
        "best_model": best_name,
        "best_params": serialize_params(best_params),
        "best_calibration": best_calib,
        "brier_valid": best_brier_final
    })

    outer_candidates.append({
        "outer": outer, "inner": inner, "selector": selector, "k": k,
        "features": features,
        "best_model": best_name,
        "best_params": best_params,
        "best_calibration": best_calib,
        "brier_valid": best_brier_final
    })

    print(f"outer={outer} inner={inner} best={best_name}+{best_calib} brier={best_brier_final:.4f}")

df_all = pd.DataFrame(rows_all)
df_all.to_csv("inner_model_selection_with_calibration_all.csv", index=False)

df_inner = pd.DataFrame(rows_inner)
df_inner.to_csv("inner_model_selection_with_calibration_best.csv", index=False)

print("Saved:")
print(" - inner_model_selection_with_calibration_all.csv")
print(" - inner_model_selection_with_calibration_best.csv")


cand_df = pd.DataFrame(outer_candidates)

if cand_df.empty:
    raise RuntimeError()

cand_df["params_json"] = cand_df["best_params"].apply(lambda d: json.dumps(d, sort_keys=True))
cand_df["features_json"] = cand_df["features"].apply(lambda f: json.dumps(f, ensure_ascii=False))

group_cols = ["outer", "selector", "k", "best_model", "best_calibration", "params_json", "features_json"]
agg = (
    cand_df.groupby(group_cols, as_index=False)
    .agg(mean_brier_valid=("brier_valid", "mean"),
         n_inners=("brier_valid", "count"))
    .sort_values(["outer", "mean_brier_valid"])
)

best_per_outer = agg.groupby("outer", as_index=False).head(1).copy()
best_per_outer.to_csv("frozen_best_combo_per_outer.csv", index=False)
print("Saved: frozen_best_combo_per_outer.csv")

outer_metrics = []
outer_preds_rows = []

for _, r in best_per_outer.iterrows():
    outer = int(r["outer"])
    selector = r["selector"]
    k = int(r["k"])
    model_name = r["best_model"]
    calib = r["best_calibration"]
    params = json.loads(r["params_json"])
    features = json.loads(r["features_json"])

    outer_train_df = main_df[main_df["fold_outer"] != outer].copy()
    outer_test_df  = main_df[main_df["fold_outer"] == outer].copy()

    X_train = clean_numeric(outer_train_df[features])
    y_train = outer_train_df["Исход"].astype(int)

    X_test  = clean_numeric(outer_test_df[features])
    y_test  = outer_test_df["Исход"].astype(int)

    if model_name == "logreg":
        est = make_lr()
        est.set_params(**params)
    elif model_name == "mlp":
        est = make_mlp()
        est.set_params(**params)
    elif model_name == "xgb":
        spw = safe_scale_pos_weight(y_train)
        est = make_xgb(spw)
        est.set_params(**params)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    if calib == "none":
        est.fit(X_train, y_train)
        p = est.predict_proba(X_test)[:, 1]
    elif calib == "platt":
        cal = CalibratedClassifierCV(estimator=est, method="sigmoid", cv=3)
        cal.fit(X_train, y_train)
        p = cal.predict_proba(X_test)[:, 1]
    elif calib == "isotonic":
        cal = CalibratedClassifierCV(estimator=est, method="isotonic", cv=3)
        cal.fit(X_train, y_train)
        p = cal.predict_proba(X_test)[:, 1]
    else:
        raise ValueError(f"Unknown calibration: {calib}")

    brier = brier_score_loss(y_test, p)

    outer_metrics.append({
        "outer": outer,
        "selector": selector,
        "k": k,
        "model": model_name,
        "calibration": calib,
        "brier_outer_test": brier
    })

    for idx, prob in zip(outer_test_df.index.tolist(), p.tolist()):
        outer_preds_rows.append({
            "outer": outer,
            "row_index": idx,
            "y_true": int(main_df.loc[idx, "Исход"]),
            "y_pred_proba": float(prob)
        })

    print(f"FROZEN outer={outer}: {model_name}+{calib} | brier_outer_test={brier:.4f}")

pd.DataFrame(outer_metrics).to_csv("outer_test_metrics.csv", index=False)
pd.DataFrame(outer_preds_rows).to_csv("outer_test_predictions.csv", index=False)

print("Сохранено")
