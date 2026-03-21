from __future__ import annotations

import argparse
import ast
import itertools
import json
import logging
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, TypedDict

import numpy as np
import pandas as pd
try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - fallback for constrained environments
    yaml = None
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.pipeline.models import build_catboost, build_mlp, build_svm, build_xgboost
from src.pipeline.feature_engineering import (
    ENG_FP_SUBGROUP_SCORE,
    add_fp_signal_features,
    apply_risk_feature_mode,
    load_preprocess_config,
    normalize_numeric_string_columns,
    resolve_declared_feature_types,
)
from src.pipeline.preprocess import build_preprocess

LOGGER = logging.getLogger("ncv_runner")


def oversample_to_ratio(
    X: pd.DataFrame,
    y: pd.Series,
    target_pos_ratio: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    if not (0.0 < target_pos_ratio < 1.0):
        return X, y
    y_arr = y.to_numpy(dtype=int)
    pos_idx = np.where(y_arr == 1)[0]
    neg_idx = np.where(y_arr == 0)[0]
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)
    if n_pos == 0 or n_neg == 0:
        return X, y

    desired_pos = int(np.ceil((target_pos_ratio / (1.0 - target_pos_ratio)) * n_neg))
    if desired_pos <= n_pos:
        return X, y

    add_n = desired_pos - n_pos
    rng = np.random.default_rng(random_state)
    sampled = rng.choice(pos_idx, size=add_n, replace=True)
    all_idx = np.concatenate([np.arange(len(y_arr)), sampled])
    rng.shuffle(all_idx)
    return X.iloc[all_idx].reset_index(drop=True), y.iloc[all_idx].reset_index(drop=True)


def _supports_sample_weight(model_name: str) -> bool:
    return model_name in {"svm_rbf", "catboost", "xgboost"}


def _fit_estimator(
    estimator: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    sample_weight: np.ndarray | None = None,
) -> Any:
    if sample_weight is None:
        estimator.fit(X_train, y_train)
        return estimator
    estimator.fit(X_train, y_train, model__sample_weight=np.asarray(sample_weight, dtype=float))
    return estimator


def _compute_positive_class_weight(y_train: pd.Series, target_pos_ratio: float) -> np.ndarray:
    weights = np.ones(len(y_train), dtype=float)
    if not (0.0 < target_pos_ratio < 1.0):
        return weights
    y_arr = np.asarray(y_train, dtype=int)
    n_pos = int((y_arr == 1).sum())
    n_neg = int((y_arr == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return weights
    pos_weight = (float(target_pos_ratio) / (1.0 - float(target_pos_ratio))) * (float(n_neg) / float(n_pos))
    if pos_weight <= 1.0:
        return weights
    weights[y_arr == 1] = float(pos_weight)
    return weights


def _rank_to_unit_interval(values: np.ndarray) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=float))
    return series.rank(method="average", pct=True).to_numpy(dtype=float)


def _compute_fp_subgroup_multiplier(X_train: pd.DataFrame, y_train: pd.Series) -> np.ndarray:
    weights = np.ones(len(y_train), dtype=float)
    if ENG_FP_SUBGROUP_SCORE not in X_train.columns:
        return weights
    subgroup_score = pd.to_numeric(X_train[ENG_FP_SUBGROUP_SCORE], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    y_arr = np.asarray(y_train, dtype=int)
    weights[y_arr == 0] *= 1.0 + 0.22 * subgroup_score[y_arr == 0]
    return np.clip(weights, 1.0, 3.0)


def _compute_difficulty_multiplier(y_train: pd.Series, raw_score: np.ndarray) -> np.ndarray:
    y_arr = np.asarray(y_train, dtype=int)
    rank = _rank_to_unit_interval(raw_score)
    weights = np.ones(len(y_arr), dtype=float)
    neg_mask = y_arr == 0
    pos_mask = y_arr == 1
    weights[neg_mask] *= 1.0 + 1.25 * rank[neg_mask]
    weights[pos_mask] *= 1.0 + 0.85 * (1.0 - rank[pos_mask])
    return np.clip(weights, 1.0, 4.0)


def _augment_training_for_mlp(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    target_pos_ratio: float,
    raw_score: np.ndarray | None = None,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    X_fit, y_fit = oversample_to_ratio(
        X_train.reset_index(drop=True),
        y_train.reset_index(drop=True),
        target_pos_ratio=target_pos_ratio,
        random_state=random_state,
    )
    base = X_train.reset_index(drop=True)
    y_base = y_train.reset_index(drop=True)
    extra_idx: List[int] = []

    if ENG_FP_SUBGROUP_SCORE in base.columns:
        subgroup_score = pd.to_numeric(base[ENG_FP_SUBGROUP_SCORE], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        neg_mask = y_base.to_numpy(dtype=int) == 0
        extra_idx.extend(np.where(neg_mask & (subgroup_score >= 2.0))[0].tolist())

    if raw_score is not None:
        rank = _rank_to_unit_interval(raw_score)
        y_arr = y_base.to_numpy(dtype=int)
        extra_idx.extend(np.where((y_arr == 0) & (rank >= 0.8))[0].tolist())
        extra_idx.extend(np.where((y_arr == 1) & (rank <= 0.35))[0].tolist())

    if not extra_idx:
        return X_fit, y_fit

    extra_X = base.iloc[extra_idx].reset_index(drop=True)
    extra_y = y_base.iloc[extra_idx].reset_index(drop=True)
    X_out = pd.concat([X_fit, extra_X], axis=0, ignore_index=True)
    y_out = pd.concat([y_fit, extra_y], axis=0, ignore_index=True)
    return X_out, y_out


def fit_base_model_fp_aware(
    *,
    model_name: str,
    estimator: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    oversample_pos_ratio: float,
    random_state: int,
    difficulty_pass: bool,
) -> Any:
    y_reset = y_train.reset_index(drop=True)
    X_reset = X_train.reset_index(drop=True)

    if _supports_sample_weight(model_name):
        base_weights = _compute_positive_class_weight(y_reset, oversample_pos_ratio)
        subgroup_weights = _compute_fp_subgroup_multiplier(X_reset, y_reset)
        first_pass_weights = base_weights * subgroup_weights
        fitted = _fit_estimator(
            clone(estimator),
            X_reset,
            y_reset,
            sample_weight=first_pass_weights,
        )
        if not difficulty_pass:
            return fitted
        raw_score = _extract_raw_score(fitted, X_reset)
        difficulty_weights = _compute_difficulty_multiplier(y_reset, raw_score)
        final_weights = first_pass_weights * difficulty_weights
        return _fit_estimator(
            clone(estimator),
            X_reset,
            y_reset,
            sample_weight=final_weights,
        )

    X_fit, y_fit = _augment_training_for_mlp(
        X_reset,
        y_reset,
        target_pos_ratio=oversample_pos_ratio,
        raw_score=None,
        random_state=random_state,
    )
    fitted = _fit_estimator(clone(estimator), X_fit, y_fit, sample_weight=None)
    if not difficulty_pass:
        return fitted
    raw_score = _extract_raw_score(fitted, X_reset)
    X_fit2, y_fit2 = _augment_training_for_mlp(
        X_reset,
        y_reset,
        target_pos_ratio=oversample_pos_ratio,
        raw_score=raw_score,
        random_state=random_state + 1000,
    )
    return _fit_estimator(clone(estimator), X_fit2, y_fit2, sample_weight=None)


class MetricDict(TypedDict):
    roc_auc: float
    pr_auc: float
    brier: float
    ece: float
    sens: float
    spec: float
    tp: int
    fp: int
    tn: int
    fn: int
    fpr: float
    fnr: float


@dataclass
class FoldSelection:
    best_params: Dict[str, Any]
    best_calibrator: str
    decision_threshold: float
    best_inner_brier: float
    best_inner_fn: int
    best_inner_fp: int
    best_inner_fnr: float
    best_inner_fpr: float
    constraint_feasible: bool
    constraint_violation: float


@dataclass
class OuterFoldResult:
    outer_fold: int
    best_params: str
    best_calibrator: str
    decision_threshold: float
    constraint_feasible: bool
    constraint_violation: float
    roc_auc: float
    pr_auc: float
    brier: float
    ece: float
    sens: float
    spec: float
    fnr: float
    fpr: float
    tp: int
    fp: int
    tn: int
    fn: int


@dataclass
class CalibratedModelWrapper:
    model: Any
    calibrator: Any
    calibrator_type: str

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw_score = _extract_raw_score(self.model, X)
        return _apply_calibrator_to_raw_score(self.calibrator, self.calibrator_type, raw_score)


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def load_tabular(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)

    first_line = path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
    sep = ";" if ";" in first_line else ","
    for enc in ("utf-8", "utf-8-sig", "cp1251"):
        try:
            return pd.read_csv(path, sep=sep, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, sep=sep, encoding_errors="replace")


def load_grids_config(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        cfg = yaml.safe_load(text)
        if not isinstance(cfg, dict):
            raise ValueError()
        return cfg

    out: Dict[str, Any] = {}
    current: Dict[str, Any] | None = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue

        if not line.startswith(" ") and line.endswith(":"):
            key = line[:-1].strip()
            out[key] = {}
            current = out[key]
            continue

        if current is None:
            raise ValueError()

        if line.startswith("  ") and ":" in line:
            k, v = line.strip().split(":", 1)
            v_str = v.strip()
            if v_str == "":
                current[k] = {}
                continue
            val_norm = v_str.replace("true", "True").replace("false", "False")
            try:
                current[k] = ast.literal_eval(val_norm)
            except Exception:
                current[k] = v_str
            continue

        raise ValueError()

    return out


def infer_categorical_features(df: pd.DataFrame, features: Sequence[str]) -> List[str]:
    categorical: List[str] = []
    for col in features:
        series = df[col]
        if (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(series)
        ):
            categorical.append(col)
    return categorical


def compute_ece(y_true: Sequence[int], p_cal: Sequence[float], n_bins: int = 10) -> float:
    y_arr = np.asarray(y_true, dtype=float)
    p_arr = np.asarray(p_cal, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(p_arr, bin_edges[1:-1], right=False)

    ece = 0.0
    for bin_idx in range(n_bins):
        mask = bin_ids == bin_idx
        if not np.any(mask):
            continue
        acc = float(y_arr[mask].mean())
        conf = float(p_arr[mask].mean())
        weight = float(mask.mean())
        ece += abs(acc - conf) * weight
    return float(ece)


def assign_fixed_probability_bins(p_cal: Iterable[float], n_bins: int = 10) -> tuple[pd.Series, str]:
    p_series = pd.Series(p_cal).astype(float)
    fixed_bins = np.floor(p_series.to_numpy() * n_bins).astype(int)
    fixed_bins = np.clip(fixed_bins, 0, n_bins - 1)
    return pd.Series(fixed_bins, index=p_series.index, dtype=int), f"fixed_{1.0 / n_bins:.1f}"


def compute_bin_metrics_table(
    y_true: Iterable[Any],
    p_cal: Iterable[float],
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    y_series = pd.Series(y_true)
    p_series = pd.Series(p_cal).astype(float)
    if len(y_series) != len(p_series):
        raise ValueError()
    if len(y_series) == 0:
        raise ValueError()

    y_bin = to_binary_target(y_series)
    bins, binning_mode = assign_fixed_probability_bins(p_series, n_bins=10)

    tmp = pd.DataFrame({"y_true": y_bin, "p_cal": p_series, "bin": bins})
    n_total = len(tmp)
    rows: List[Dict[str, Any]] = []
    weighted_ece = 0.0

    for bin_id in sorted(tmp["bin"].unique().tolist()):
        part = tmp[tmp["bin"] == bin_id]
        n_bin = len(part)
        mean_y = float(part["y_true"].mean())
        mean_p = float(part["p_cal"].mean())
        brier_bin = float(((part["p_cal"] - part["y_true"]) ** 2).mean())
        ece_bin = float(abs(mean_y - mean_p))
        weighted_ece += ece_bin * (n_bin / n_total)
        rows.append(
            {
                "bin": int(bin_id),
                "n_bin": int(n_bin),
                "mean_y_true": mean_y,
                "mean_p_cal": mean_p,
                "brier_bin": brier_bin,
                "ece_bin": ece_bin,
            }
        )

    summary = {
        "n_total": int(n_total),
        "n_bins_actual": int(tmp["bin"].nunique()),
        "binning_mode": binning_mode,
        "ece_weighted": float(weighted_ece),
    }
    return pd.DataFrame(rows), summary


def compute_metrics(
    y_true: Iterable[Any],
    p_pred: Iterable[float],
    threshold: float = 0.5,
) -> MetricDict:
    y_series = pd.Series(y_true)
    p_series = pd.Series(p_pred)

    if len(y_series) != len(p_series):
        raise ValueError()
    if len(y_series) == 0:
        raise ValueError()
    if y_series.isna().any() or p_series.isna().any():
        raise ValueError()

    try:
        y_bin = y_series.astype(int)
    except Exception as exc:
        raise ValueError() from exc

    uniq = sorted(y_bin.unique().tolist())
    if len(uniq) != 2:
        raise ValueError()
    if uniq != [0, 1]:
        y_bin = y_bin.map({uniq[0]: 0, uniq[1]: 1}).astype(int)

    try:
        p_float = p_series.astype(float)
    except Exception as exc:
        raise ValueError() from exc

    if (p_float < 0).any() or (p_float > 1).any():
        raise ValueError()

    y_arr = y_bin.to_numpy(dtype=int)
    p_arr = p_float.to_numpy(dtype=float)

    try:
        pred_bin = (p_arr >= float(threshold)).astype(int)
        tp = int(((pred_bin == 1) & (y_arr == 1)).sum())
        fp = int(((pred_bin == 1) & (y_arr == 0)).sum())
        tn = int(((pred_bin == 0) & (y_arr == 0)).sum())
        fn = int(((pred_bin == 0) & (y_arr == 1)).sum())
        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
        fpr = float(fp / (tn + fp)) if (tn + fp) > 0 else float("nan")
        fnr = float(fn / (tp + fn)) if (tp + fn) > 0 else float("nan")
        return {
            "roc_auc": float(roc_auc_score(y_arr, p_arr)),
            "pr_auc": float(average_precision_score(y_arr, p_arr)),
            "brier": float(brier_score_loss(y_arr, p_arr)),
            "ece": float(compute_ece(y_arr, p_arr, n_bins=10)),
            "sens": sens,
            "spec": spec,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "fpr": fpr,
            "fnr": fnr,
        }
    except Exception as exc:
        raise ValueError() from exc


def select_model(
    p_svm: float,
    p_cat: float,
    p_mlp: float,
    competence: Dict[str, Any],
    fallback_model: str = "catboost",
) -> dict:
    p_avg = float((float(p_svm) + float(p_cat) + float(p_mlp)) / 3.0)
    bin_id = int(math.floor(p_avg * 10.0))
    bin_id = max(0, min(9, bin_id))

    model_names = ["svm_rbf", "catboost", "mlp"]
    available: List[tuple[str, float, float]] = []

    ece_map = competence.get("ece", {})
    brier_map = competence.get("brier", {})

    for model in model_names:
        ece_arr = ece_map.get(model)
        brier_arr = brier_map.get(model)
        if not isinstance(ece_arr, list) or bin_id >= len(ece_arr):
            continue
        if not isinstance(brier_arr, list) or bin_id >= len(brier_arr):
            continue

        ece_val = ece_arr[bin_id]
        brier_val = brier_arr[bin_id]

        if ece_val is None:
            continue
        if isinstance(ece_val, float) and math.isnan(ece_val):
            continue
        if brier_val is None or (isinstance(brier_val, float) and math.isnan(brier_val)):
            brier_val = float("inf")

        available.append((model, float(ece_val), float(brier_val)))

    if not available:
        return {"winner": fallback_model, "bin_id": bin_id, "p_avg": p_avg}

    winner = min(available, key=lambda x: (x[1], x[2]))[0]
    return {"winner": winner, "bin_id": bin_id, "p_avg": p_avg}


def aggregate_metric_mean(items: List[MetricDict]) -> Dict[str, float]:
    df = pd.DataFrame(items)
    return {
        "roc_auc_mean": float(df["roc_auc"].mean()),
        "pr_auc_mean": float(df["pr_auc"].mean()),
        "brier_mean": float(df["brier"].mean()),
        "ece_mean": float(df["ece"].mean()),
        "sens_mean": float(df["sens"].mean()),
        "spec_mean": float(df["spec"].mean()),
        "fn_mean": float(df["fn"].mean()),
        "fp_mean": float(df["fp"].mean()),
        "fnr_mean": float(df["fnr"].mean()),
        "fpr_mean": float(df["fpr"].mean()),
    }


def _confusion_at_threshold(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    pred = (p_pred >= float(threshold)).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    fnr = float(fn / (tp + fn)) if (tp + fn) > 0 else float("nan")
    fpr = float(fp / (tn + fp)) if (tn + fp) > 0 else float("nan")
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "sens": sens,
        "spec": spec,
        "fnr": fnr,
        "fpr": fpr,
    }


def pick_threshold_under_error_caps(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    max_fn: int,
    max_fp: int,
) -> Dict[str, Any]:
    if len(y_true) == 0:
        raise ValueError()

    thresholds = sorted(set([0.0, 1.0] + p_pred.tolist()))
    rows: List[Dict[str, Any]] = [
        _confusion_at_threshold(y_true=y_true, p_pred=p_pred, threshold=t) for t in thresholds
    ]

    feasible_rows = [r for r in rows if int(r["fn"]) <= int(max_fn) and int(r["fp"]) <= int(max_fp)]
    if feasible_rows:
        best = min(feasible_rows, key=lambda r: (int(r["fn"]) + int(r["fp"]), int(r["fp"]), int(r["fn"]), -float(r["threshold"])))
        best["feasible"] = True
        best["violation"] = 0.0
        return best

    fn_denom = max(int(max_fn), 1)
    fp_denom = max(int(max_fp), 1)
    for row in rows:
        fn_ratio = float(row["fn"]) / fn_denom
        fp_ratio = float(row["fp"]) / fp_denom
        row["violation"] = float(max(fn_ratio, fp_ratio))
        row["feasible"] = False

    best = min(
        rows,
        key=lambda r: (
            float(r["violation"]),
            int(r["fn"]) + int(r["fp"]),
            int(r["fn"]),
            int(r["fp"]),
            -float(r["threshold"]),
        ),
    )
    return best


def build_param_grid(model_name: str, raw_grid: Dict[str, Any]) -> List[Dict[str, Any]]:
    def _as_float_list(values: Iterable[Any]) -> List[float]:
        return [float(v) for v in values]

    def _as_int_list(values: Iterable[Any]) -> List[int]:
        return [int(v) for v in values]

    if model_name == "svm_rbf":
        keys = ["C", "gamma", "class_weight"]
        values = [
            _as_float_list(raw_grid.get("C", [1.0])),
            raw_grid.get("gamma", ["scale"]),
            raw_grid.get("class_weight", ["balanced"]),
        ]
    elif model_name == "catboost":
        keys = ["depth", "n_estimators", "learning_rate", "class_weights"]
        values = [
            _as_int_list(raw_grid.get("depth", [6])),
            _as_int_list(raw_grid.get("n_estimators", [500])),
            _as_float_list(raw_grid.get("learning_rate", [0.03])),
            raw_grid.get("class_weights", ["balanced"]),
        ]
    elif model_name == "xgboost":
        keys = [
            "n_estimators",
            "max_depth",
            "learning_rate",
            "min_child_weight",
            "subsample",
            "colsample_bytree",
            "reg_lambda",
            "gamma",
            "scale_pos_weight",
        ]
        values = [
            _as_int_list(raw_grid.get("n_estimators", [250])),
            _as_int_list(raw_grid.get("max_depth", [2])),
            _as_float_list(raw_grid.get("learning_rate", [0.05])),
            _as_float_list(raw_grid.get("min_child_weight", [4.0])),
            _as_float_list(raw_grid.get("subsample", [0.8])),
            _as_float_list(raw_grid.get("colsample_bytree", [0.7])),
            _as_float_list(raw_grid.get("reg_lambda", [6.0])),
            _as_float_list(raw_grid.get("gamma", [1.0])),
            _as_float_list(raw_grid.get("scale_pos_weight", [4.0])),
        ]
    elif model_name == "mlp":
        keys = ["layers", "dropout", "weight_decay", "early_stopping"]
        values = [
            raw_grid.get("layers", [[16]]),
            _as_float_list(raw_grid.get("dropout", [0.0])),
            _as_float_list(raw_grid.get("weight_decay", [1e-4])),
            [raw_grid.get("early_stopping", True)],
        ]
    else:
        raise ValueError()

    combos = [dict(zip(keys, vals)) for vals in itertools.product(*values)]
    if model_name != "mlp":
        return combos

    unique: Dict[tuple[Any, ...], Dict[str, Any]] = {}
    for combo in combos:
        key = (
            tuple(combo["layers"]),
            float(combo["weight_decay"]),
            bool(combo["early_stopping"]),
        )
        if key not in unique:
            unique[key] = combo
    deduped = list(unique.values())
    return deduped


def get_builder(model_name: str):
    if model_name == "svm_rbf":
        return build_svm
    if model_name == "catboost":
        return build_catboost
    if model_name == "xgboost":
        return build_xgboost
    if model_name == "mlp":
        return build_mlp
    raise ValueError()


def to_binary_target(y_raw: pd.Series) -> pd.Series:
    if (
        pd.api.types.is_object_dtype(y_raw)
        or pd.api.types.is_string_dtype(y_raw)
        or isinstance(y_raw.dtype, pd.CategoricalDtype)
    ):
        s = y_raw.astype(str).str.strip()
        s_lower = s.str.lower()
        common_map = {
            "1": 1,
            "0": 0,
            "true": 1,
            "false": 0,
            "yes": 1,
            "no": 0,
            "positive": 1,
            "negative": 0,
            "favorable": 1,
            "unfavorable": 0,
            "благоприятный": 0,
            "неблагоприятный": 1,
            "благоприятно": 0,
            "неблагоприятно": 1,
        }
        y_mapped = s_lower.map(common_map)
        if y_mapped.isna().any():
            uniq = sorted(s.dropna().unique().tolist())
            if len(uniq) != 2:
                raise ValueError()
            label_map = {uniq[0]: 0, uniq[1]: 1}
            return s.map(label_map).astype(int)
        return y_mapped.astype(int)

    y = y_raw.astype(int)
    uniq = sorted(y.unique().tolist())
    if len(uniq) != 2:
        raise ValueError()
    if uniq != [0, 1]:
        y = y.map({uniq[0]: 0, uniq[1]: 1}).astype(int)
    return y


def _extract_raw_score(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X))
        if proba.ndim != 2 or proba.shape[1] < 2:
            raise ValueError()
        return proba[:, 1].astype(float)
    if hasattr(model, "decision_function"):
        decision = model.decision_function(X)
        return np.asarray(decision, dtype=float).reshape(-1)
    raise ValueError()


def _apply_calibrator_to_raw_score(
    calibrator: Any,
    calibrator_type: str,
    raw_score: np.ndarray,
) -> np.ndarray:
    raw_arr = np.asarray(raw_score, dtype=float).reshape(-1)
    if calibrator_type == "none":
        return np.clip(raw_arr, 0.0, 1.0).astype(float)
    if calibrator_type == "platt":
        return calibrator.predict_proba(raw_arr.reshape(-1, 1))[:, 1].astype(float)
    if calibrator_type == "isotonic":
        return np.clip(calibrator.predict(raw_arr), 0.0, 1.0).astype(float)
    raise ValueError()


def _fit_raw_score_calibrator(
    raw_score: np.ndarray,
    y_true: pd.Series | np.ndarray,
    calibrator_type: str,
) -> Any:
    if calibrator_type == "none":
        return None
    y_arr = np.asarray(y_true, dtype=int).reshape(-1)
    raw_arr = np.asarray(raw_score, dtype=float).reshape(-1)
    if len(raw_arr) != len(y_arr):
        raise ValueError()
    if len(np.unique(y_arr)) < 2:
        raise ValueError()

    if calibrator_type == "platt":
        calibrator = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
        )
        calibrator.fit(raw_arr.reshape(-1, 1), y_arr)
        return calibrator
    if calibrator_type == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw_arr, y_arr)
        return calibrator
    raise ValueError()


def _split_training_for_search_calibration(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    calibration_frac: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    if len(X_train) != len(y_train):
        raise ValueError()
    if len(X_train) == 0:
        raise ValueError()

    y_arr = np.asarray(y_train, dtype=int)
    class_counts = np.bincount(y_arr, minlength=2)
    min_class = int(class_counts.min())
    if min_class < 2:
        X_fit = X_train.reset_index(drop=True)
        y_fit = y_train.reset_index(drop=True)
        return X_fit, y_fit, X_fit.copy(), y_fit.copy()

    frac = float(calibration_frac)
    frac = min(max(frac, 0.15), 0.4)
    frac = max(frac, 1.0 / max(min_class, 1))

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=frac, random_state=random_state)
    fit_idx, cal_idx = next(splitter.split(X_train, y_train))

    X_fit = X_train.iloc[fit_idx].reset_index(drop=True)
    y_fit = y_train.iloc[fit_idx].reset_index(drop=True)
    X_cal = X_train.iloc[cal_idx].reset_index(drop=True)
    y_cal = y_train.iloc[cal_idx].reset_index(drop=True)
    return X_fit, y_fit, X_cal, y_cal


def fit_calibrated_model(
    model_name: str,
    base_model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    calibrator_type: str,
    inner_cv: StratifiedKFold,
    oversample_pos_ratio: float,
) -> CalibratedModelWrapper:
    if calibrator_type not in {"none", "platt", "isotonic"}:
        raise ValueError()
    if len(X_train) != len(y_train):
        raise ValueError()
    if len(X_train) == 0:
        raise ValueError()

    if calibrator_type == "none":
        final_model = fit_base_model_fp_aware(
            model_name=model_name,
            estimator=base_model,
            X_train=X_train.reset_index(drop=True),
            y_train=y_train.reset_index(drop=True),
            oversample_pos_ratio=oversample_pos_ratio,
            random_state=4242,
            difficulty_pass=True,
        )
        return CalibratedModelWrapper(
            model=final_model,
            calibrator=None,
            calibrator_type="none",
        )

    LOGGER.info("Обучение калибратора (%s)", calibrator_type)
    y_arr = np.asarray(y_train, dtype=int)
    oof_raw = np.full(shape=len(y_arr), fill_value=np.nan, dtype=float)

    for fold_i, (tr_idx, va_idx) in enumerate(inner_cv.split(X_train, y_train), start=1):
        X_tr = X_train.iloc[tr_idx].reset_index(drop=True)
        y_tr = y_train.iloc[tr_idx].reset_index(drop=True)
        model_fold = fit_base_model_fp_aware(
            model_name=model_name,
            estimator=base_model,
            X_train=X_tr,
            y_train=y_tr,
            oversample_pos_ratio=oversample_pos_ratio,
            random_state=42 + fold_i,
            difficulty_pass=True,
        )
        oof_raw[va_idx] = _extract_raw_score(model_fold, X_train.iloc[va_idx])

    if np.isnan(oof_raw).any():
        raise RuntimeError()

    final_model = fit_base_model_fp_aware(
        model_name=model_name,
        estimator=base_model,
        X_train=X_train.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        oversample_pos_ratio=oversample_pos_ratio,
        random_state=4242,
        difficulty_pass=True,
    )

    if calibrator_type == "platt":
        calibrator = _fit_raw_score_calibrator(oof_raw, y_arr, calibrator_type="platt")
    else:
        calibrator = _fit_raw_score_calibrator(oof_raw, y_arr, calibrator_type="isotonic")

    return CalibratedModelWrapper(
        model=final_model,
        calibrator=calibrator,
        calibrator_type=calibrator_type,
    )


def select_best_params_and_calibrator(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    features: List[str],
    categorical: List[str],
    candidates: List[Dict[str, Any]],
    calib_methods: List[str],
    inner_cv: StratifiedKFold,
    oversample_pos_ratio: float,
    max_fn_allowed: int,
    max_fp_allowed: int,
    search_calibration_frac: float,
) -> tuple[FoldSelection, pd.DataFrame]:
    builder = get_builder(model_name)
    best_selection: FoldSelection | None = None
    best_key: tuple[float, ...] | None = None
    candidate_rows: List[Dict[str, Any]] = []

    for params in candidates:
        by_calibrator_metrics: Dict[str, List[MetricDict]] = {method: [] for method in calib_methods}
        by_calibrator_y: Dict[str, List[np.ndarray]] = {method: [] for method in calib_methods}
        by_calibrator_p: Dict[str, List[np.ndarray]] = {method: [] for method in calib_methods}

        for fold_i, (tr_idx, va_idx) in enumerate(inner_cv.split(X_train, y_train), start=1):
            X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]
            X_fit, y_fit, X_cal, y_cal = _split_training_for_search_calibration(
                X_train=X_tr,
                y_train=y_tr,
                calibration_frac=search_calibration_frac,
                random_state=140 + fold_i,
            )

            estimator_params = dict(params)
            if model_name == "svm_rbf":
                estimator_params["probability"] = False

            estimator = builder(estimator_params)
            if estimator is None:
                raise RuntimeError()

            base_estimator = Pipeline(
                steps=[
                    ("preprocess", build_preprocess(features=features, categorical=categorical)),
                    ("model", estimator),
                ]
            )
            base_estimator = fit_base_model_fp_aware(
                model_name=model_name,
                estimator=base_estimator,
                X_train=X_fit,
                y_train=y_fit,
                oversample_pos_ratio=oversample_pos_ratio,
                random_state=84 + fold_i,
                difficulty_pass=False,
            )
            raw_cal = _extract_raw_score(base_estimator, X_cal)
            raw_va = _extract_raw_score(base_estimator, X_va)

            for method in calib_methods:
                if method == "none":
                    if model_name == "svm_rbf":
                        continue
                    calibrator = None
                    p_cal = _apply_calibrator_to_raw_score(calibrator, method, raw_va)
                else:
                    calibrator = _fit_raw_score_calibrator(
                        raw_score=raw_cal,
                        y_true=y_cal,
                        calibrator_type=method,
                    )
                    p_cal = _apply_calibrator_to_raw_score(calibrator, method, raw_va)
                by_calibrator_metrics[method].append(compute_metrics(y_va, p_cal, threshold=0.5))
                by_calibrator_y[method].append(np.asarray(y_va, dtype=int))
                by_calibrator_p[method].append(np.asarray(p_cal, dtype=float))

        method_candidates: List[tuple[str, Dict[str, Any], tuple[float, ...]]] = []
        for method in calib_methods:
            if not by_calibrator_metrics[method]:
                continue
            method_metrics = aggregate_metric_mean(by_calibrator_metrics[method])
            y_oof = np.concatenate(by_calibrator_y[method], axis=0)
            p_oof = np.concatenate(by_calibrator_p[method], axis=0)

            best_thr_stats = pick_threshold_under_error_caps(
                y_true=y_oof,
                p_pred=p_oof,
                max_fn=max_fn_allowed,
                max_fp=max_fp_allowed,
            )
            method_record = {
                **method_metrics,
                "threshold": float(best_thr_stats["threshold"]),
                "tp": int(best_thr_stats["tp"]),
                "fp": int(best_thr_stats["fp"]),
                "tn": int(best_thr_stats["tn"]),
                "fn": int(best_thr_stats["fn"]),
                "sens": float(best_thr_stats["sens"]),
                "spec": float(best_thr_stats["spec"]),
                "fnr": float(best_thr_stats["fnr"]),
                "fpr": float(best_thr_stats["fpr"]),
                "constraint_feasible": bool(best_thr_stats["feasible"]),
                "constraint_violation": float(best_thr_stats["violation"]),
            }

            if method_record["constraint_feasible"]:
                method_key = (
                    0.0,
                    int(method_record["fn"]) + int(method_record["fp"]),
                    int(method_record["fp"]),
                    int(method_record["fn"]),
                    -float(method_record["pr_auc_mean"]),
                    float(method_record["brier_mean"]),
                    -float(method_record["threshold"]),
                )
            else:
                method_key = (
                    1.0,
                    float(method_record["constraint_violation"]),
                    int(method_record["fn"]) + int(method_record["fp"]),
                    int(method_record["fn"]),
                    int(method_record["fp"]),
                    -float(method_record["pr_auc_mean"]),
                    float(method_record["brier_mean"]),
                    -float(method_record["threshold"]),
                )
            method_candidates.append((method, method_record, method_key))

        method_candidates_sorted = sorted(method_candidates, key=lambda x: x[2])
        best_method, best_method_metrics, method_key = method_candidates_sorted[0]

        params_json = json.dumps(params, ensure_ascii=False, sort_keys=True)
        candidate_rows.append(
            {
                "model": model_name,
                "params": params_json,
                "best_calibrator": best_method,
                "threshold": float(best_method_metrics["threshold"]),
                "constraint_feasible": int(best_method_metrics["constraint_feasible"]),
                "constraint_violation": float(best_method_metrics["constraint_violation"]),
                "fn_inner_oof": int(best_method_metrics["fn"]),
                "fp_inner_oof": int(best_method_metrics["fp"]),
                "fnr_inner_oof": float(best_method_metrics["fnr"]),
                "fpr_inner_oof": float(best_method_metrics["fpr"]),
                "sens_inner_oof": float(best_method_metrics["sens"]),
                "spec_inner_oof": float(best_method_metrics["spec"]),
                "pr_auc_mean": float(best_method_metrics["pr_auc_mean"]),
                "brier_mean": float(best_method_metrics["brier_mean"]),
                "roc_auc_mean": float(best_method_metrics["roc_auc_mean"]),
                "ece_mean": float(best_method_metrics["ece_mean"]),
            }
        )

        if best_selection is None or best_key is None or method_key < best_key:
            best_selection = FoldSelection(
                best_params=params,
                best_calibrator=best_method,
                decision_threshold=float(best_method_metrics["threshold"]),
                best_inner_brier=float(best_method_metrics["brier_mean"]),
                best_inner_fn=int(best_method_metrics["fn"]),
                best_inner_fp=int(best_method_metrics["fp"]),
                best_inner_fnr=float(best_method_metrics["fnr"]),
                best_inner_fpr=float(best_method_metrics["fpr"]),
                constraint_feasible=bool(best_method_metrics["constraint_feasible"]),
                constraint_violation=float(best_method_metrics["constraint_violation"]),
            )
            best_key = method_key

    if best_selection is None:
        raise RuntimeError()
    candidates_df = pd.DataFrame(candidate_rows).sort_values(
        by=[
            "constraint_feasible",
            "constraint_violation",
            "fn_inner_oof",
            "fp_inner_oof",
            "pr_auc_mean",
            "brier_mean",
        ],
        ascending=[False, True, True, True, False, True],
    )
    return best_selection, candidates_df


def run_model_nested_cv(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    features: List[str],
    categorical: List[str],
    model_grid: Dict[str, Any],
    calib_methods: List[str],
    outer_cv: StratifiedKFold,
    inner_cv: StratifiedKFold,
    preds_dir: Path,
    oversample_pos_ratio: float,
    target_max_fn: int,
    target_max_fp: int,
    n_pos_ref: int,
    n_neg_ref: int,
    constraints_dir: Path,
    search_calibration_frac: float,
) -> List[OuterFoldResult]:
    builder = get_builder(model_name)
    candidates = build_param_grid(model_name, model_grid)
    if not candidates:
        raise ValueError()

    if model_name in {"catboost", "xgboost"} and builder({}) is None:
        raise RuntimeError()

    results: List[OuterFoldResult] = []
    selected_rows: List[Dict[str, Any]] = []
    pooled_tp = 0
    pooled_fp = 0
    pooled_tn = 0
    pooled_fn = 0
    for fold_idx, (tr_idx, te_idx) in enumerate(outer_cv.split(X, y), start=1):
        LOGGER.info("Модель %s: внешний фолд %s/5", model_name, fold_idx)
        X_train, X_test = X.iloc[tr_idx], X.iloc[te_idx]
        y_train, y_test = y.iloc[tr_idx], y.iloc[te_idx]
        n_pos_train = int((y_train == 1).sum())
        n_neg_train = int((y_train == 0).sum())

        max_fn_inner = int(math.ceil((target_max_fn * n_pos_train) / max(n_pos_ref, 1)))
        max_fp_inner = int(math.ceil((target_max_fp * n_neg_train) / max(n_neg_ref, 1)))

        selection, candidate_table = select_best_params_and_calibrator(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            features=features,
            categorical=categorical,
            candidates=candidates,
            calib_methods=calib_methods,
            inner_cv=inner_cv,
            oversample_pos_ratio=oversample_pos_ratio,
            max_fn_allowed=max_fn_inner,
            max_fp_allowed=max_fp_inner,
            search_calibration_frac=search_calibration_frac,
        )
        candidate_path = constraints_dir / f"ncv_constraint_candidates_{model_name}_fold{fold_idx}.csv"
        candidate_table.to_csv(candidate_path, index=False, encoding="utf-8")

        final_params = dict(selection.best_params)
        if model_name == "svm_rbf":
            final_params["probability"] = False

        final_estimator = builder(final_params)
        if final_estimator is None:
            raise RuntimeError()

        base_estimator = Pipeline(
            steps=[
                ("preprocess", build_preprocess(features=features, categorical=categorical)),
                ("model", final_estimator),
            ]
        )
        final_model = fit_calibrated_model(
            model_name=model_name,
            base_model=base_estimator,
            X_train=X_train,
            y_train=y_train,
            calibrator_type=selection.best_calibrator,
            inner_cv=inner_cv,
            oversample_pos_ratio=oversample_pos_ratio,
        )
        p_cal = final_model.predict_proba(X_test)
        fold_metrics = compute_metrics(y_test, p_cal, threshold=selection.decision_threshold)
        pooled_tp += int(fold_metrics["tp"])
        pooled_fp += int(fold_metrics["fp"])
        pooled_tn += int(fold_metrics["tn"])
        pooled_fn += int(fold_metrics["fn"])
        results.append(
            OuterFoldResult(
                outer_fold=fold_idx,
                best_params=json.dumps(selection.best_params, ensure_ascii=False),
                best_calibrator=selection.best_calibrator,
                decision_threshold=float(selection.decision_threshold),
                constraint_feasible=selection.constraint_feasible,
                constraint_violation=selection.constraint_violation,
                roc_auc=fold_metrics["roc_auc"],
                pr_auc=fold_metrics["pr_auc"],
                brier=fold_metrics["brier"],
                ece=fold_metrics["ece"],
                sens=fold_metrics["sens"],
                spec=fold_metrics["spec"],
                fnr=fold_metrics["fnr"],
                fpr=fold_metrics["fpr"],
                tp=fold_metrics["tp"],
                fp=fold_metrics["fp"],
                tn=fold_metrics["tn"],
                fn=fold_metrics["fn"],
            )
        )
        selected_rows.append(
            {
                "outer_fold": fold_idx,
                "best_params": json.dumps(selection.best_params, ensure_ascii=False),
                "best_calibrator": selection.best_calibrator,
                "decision_threshold": float(selection.decision_threshold),
                "constraint_feasible": int(selection.constraint_feasible),
                "constraint_violation": float(selection.constraint_violation),
                "inner_fn": int(selection.best_inner_fn),
                "inner_fp": int(selection.best_inner_fp),
                "inner_fnr": float(selection.best_inner_fnr),
                "inner_fpr": float(selection.best_inner_fpr),
            }
        )

        preds_path = preds_dir / f"preds_{model_name}_fold{fold_idx}.csv"
        pred_bin = (np.asarray(p_cal, dtype=float) >= float(selection.decision_threshold)).astype(int)
        pd.DataFrame(
            {
                "y_true": y_test.to_numpy(),
                "p_cal": p_cal,
                "pred_bin": pred_bin,
                "threshold": float(selection.decision_threshold),
            }
        ).to_csv(
            preds_path,
            index=False,
            encoding="utf-8",
        )

    selected_path = constraints_dir / f"ncv_constraint_selected_{model_name}.csv"
    pd.DataFrame(selected_rows).to_csv(selected_path, index=False, encoding="utf-8")
    summary_path = constraints_dir / f"ncv_constraint_summary_{model_name}.json"
    pooled_summary = {
        "model": model_name,
        "target_max_fn": int(target_max_fn),
        "target_max_fp": int(target_max_fp),
        "tp_total": int(pooled_tp),
        "fp_total": int(pooled_fp),
        "tn_total": int(pooled_tn),
        "fn_total": int(pooled_fn),
        "meets_target": bool(pooled_fn <= target_max_fn and pooled_fp <= target_max_fp),
    }
    summary_path.write_text(json.dumps(pooled_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument(
        "--config_features",
        default=Path("configs/kstar_features.json"),
        type=Path,
    )
    parser.add_argument(
        "--config_grids",
        default=Path("configs/grids.yaml"),
        type=Path,
    )
    parser.add_argument(
        "--oversample-pos-ratio",
        default=0.45,
        type=float,
    )
    parser.add_argument(
        "--target-max-fn",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--target-max-fp",
        default=7,
        type=int,
    )
    parser.add_argument(
        "--search-calibration-frac",
        default=0.25,
        type=float,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["svm_rbf", "catboost", "xgboost", "mlp"],
        default=None,
    )
    parser.add_argument(
        "--risk-feature-mode",
        type=str,
        default="relative_only",
    )
    args = parser.parse_args()

    setup_logging()
    LOGGER.info("Загрузка данных из %s", args.data)

    df = load_tabular(args.data)
    features_cfg = json.loads(args.config_features.read_text(encoding="utf-8"))
    grids_cfg = load_grids_config(args.config_grids)
    preprocess_path = Path("ml/inference_pack/preprocess.json")
    preprocess_cfg = load_preprocess_config(preprocess_path) if preprocess_path.exists() else None

    features = features_cfg.get("features", [])
    target = features_cfg.get("target", "target")
    if target not in df.columns:
        if preprocess_path.exists():
            fallback_target = preprocess_cfg.get("target_col") if preprocess_cfg is not None else None
        else:
            raise ValueError()

    feature_pool = apply_risk_feature_mode(features, args.risk_feature_mode)
    missing = [col for col in feature_pool if col not in df.columns]
    if missing:
        if preprocess_path.exists():
            raw_features = preprocess_cfg.get("raw_feature_cols", []) if preprocess_cfg is not None else []
            raw_features = apply_risk_feature_mode(raw_features, args.risk_feature_mode)
        else:
            raise ValueError()

    data = df[feature_pool + [target]].copy()
    declared_num_cols, declared_cat_cols = resolve_declared_feature_types(feature_pool, preprocess_cfg)
    if declared_num_cols:
        data = normalize_numeric_string_columns(data, declared_num_cols)
    data, engineered_cols = add_fp_signal_features(data)
    feature_pool = feature_pool + [col for col in engineered_cols if col in data.columns]
    data = data.dropna(subset=[target])
    X = data[feature_pool]
    y = to_binary_target(data[target])
    categorical = [col for col in declared_cat_cols if col in feature_pool]
    if not categorical:
        categorical = infer_categorical_features(X, feature_pool)
    LOGGER.info("Prepared data: %s rows, %s features", X.shape[0], X.shape[1])
    n_pos_ref = int((y == 1).sum())
    n_neg_ref = int((y == 0).sum())

    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    calib_methods = grids_cfg.get("calibration", {}).get("methods", ["platt", "isotonic"])

    tables_dir = Path("reports/tables")
    preds_dir = Path("reports/preds")
    constraints_dir = tables_dir / "constraints"
    tables_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)
    constraints_dir.mkdir(parents=True, exist_ok=True)

    model_to_table = {
        "svm_rbf": tables_dir / "ncv_results_svm.csv",
        "catboost": tables_dir / "ncv_results_cat.csv",
        "xgboost": tables_dir / "ncv_results_xgb.csv",
        "mlp": tables_dir / "ncv_results_mlp.csv",
    }

    requested_models = args.models or ["svm_rbf", "catboost", "xgboost", "mlp"]
    available_models: List[str] = []
    for model_name in requested_models:
        builder = get_builder(model_name)
        available_models.append(model_name)

    if not available_models:
        raise RuntimeError()

    for model_name in available_models:
        model_results = run_model_nested_cv(
            model_name=model_name,
            X=X,
            y=y,
            features=feature_pool,
            categorical=categorical,
            model_grid=grids_cfg.get(model_name, {}),
            calib_methods=calib_methods,
            outer_cv=outer_cv,
            inner_cv=inner_cv,
            preds_dir=preds_dir,
            oversample_pos_ratio=args.oversample_pos_ratio,
            target_max_fn=args.target_max_fn,
            target_max_fp=args.target_max_fp,
            n_pos_ref=n_pos_ref,
            n_neg_ref=n_neg_ref,
            constraints_dir=constraints_dir,
            search_calibration_frac=args.search_calibration_frac,
        )
        out_df = pd.DataFrame([asdict(row) for row in model_results])
        out_df.to_csv(model_to_table[model_name], index=False, encoding="utf-8")



if __name__ == "__main__":
    main()
